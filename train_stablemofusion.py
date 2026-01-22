import os
import json
import sys
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusers import DDPMScheduler

from options.vqdm.denoiser_option import get_denoiser_args_parser
import utils.utils_model as utils_model
from dataloader.dataset_TM_train import DATALoader, cycle
from dataloader import dataset_eval_t2m

from Evaluator_272.mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from Evaluator_272.mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

from utils.eval_trans import (
    SMPL,
    visualize_smpl_85,
    recover_from_local_rotation,
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
)

# StableMoFusion uses absolute imports internally, so add its root to sys.path.
STABLEMOFUSION_ROOT = os.path.join(os.path.dirname(__file__), "baselines", "stablemofusion_port")
if STABLEMOFUSION_ROOT not in sys.path:
    sys.path.insert(0, STABLEMOFUSION_ROOT)

from baselines.stablemofusion_port.stablemofusion.models import build_models
from baselines.stablemofusion_port.stablemofusion.models.gaussian_diffusion import DiffusePipeline

warnings.filterwarnings("ignore")


def masked_l2(pred, target, mask):
    mse = (pred - target).pow(2).mean(dim=-1)
    denom = mask.sum(dim=-1).clamp(min=1)
    loss = (mse * mask).sum(dim=-1) / denom
    return loss.mean()


@torch.no_grad()
def evaluation_stablemofusion(
    val_loader,
    pipeline,
    logger,
    evaluator,
    device=torch.device("cuda"),
    unit_length=4,
    prev_best_fid=None,
    prev_best_div=None,
    prev_best_rprecision_pred=None,
    prev_best_matching_score_pred=None,
    draw=False,
    vis_dir=None,
):
    textencoder, motionencoder = evaluator
    pipeline.model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = torch.tensor([0, 0, 0], device=device)
    R_precision = torch.tensor([0, 0, 0], device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    matching_score_pred = torch.tensor(0.0, device=device)

    nb_sample = torch.tensor(0, device=device)

    smpl_model = None
    if draw and vis_dir is not None:
        smpl_model = SMPL(model_path="./human_models/smpl")

    for batch in tqdm(val_loader, desc="Evaluating"):
        if len(batch) == 3:
            text, pose, m_length = batch
        else:
            text, pose, m_length = batch[:3]

        bs, seq = pose.shape[:2]
        num_joints = 22
        pose = pose.to(device).float()
        lengths = torch.as_tensor(m_length, device=device)
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1]), device=device)

        pred_list = pipeline.generate(list(text), lengths, batch_size=bs)
        pred_len = torch.zeros(bs, dtype=torch.long, device=device)
        for k in range(bs):
            pred = pred_list[k].to(device)
            length_k = min(int(lengths[k].item()), pred.shape[0], seq)
            if length_k > 0:
                pred_pose_eval[k, :length_k] = pred[:length_k]
            pred_len[k] = length_k

        if draw:
            try:
                for k in range(min(3, bs)):
                    length_k = int(pred_len[k].item())
                    gt_denorm = val_loader.dataset.inv_transform(
                        pose[k:k + 1, :length_k, :].detach().cpu().numpy()
                    )
                    pred_denorm = val_loader.dataset.inv_transform(
                        pred_pose_eval[k:k + 1, :length_k].detach().cpu().numpy()
                    )
                    visualize_smpl_85(
                        recover_from_local_rotation(gt_denorm.squeeze(0), num_joints),
                        smpl_model,
                        title="",
                        output_path=vis_dir,
                        name=f"gt_{k}",
                    )
                    visualize_smpl_85(
                        recover_from_local_rotation(pred_denorm.squeeze(0), num_joints),
                        smpl_model,
                        title="",
                        output_path=vis_dir,
                        name=f"pred_{k}",
                    )
            except Exception:
                pass
            draw = False

        et_pred, em_pred = textencoder(text).loc, motionencoder(pred_pose_eval, pred_len).loc
        et, em = textencoder(text).loc, motionencoder(pose, lengths).loc
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += torch.tensor(temp_R, device=device)
        matching_score_real += torch.tensor(temp_match, device=device)
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += torch.tensor(temp_R, device=device)
        matching_score_pred += torch.tensor(temp_match, device=device)
        nb_sample += et.shape[0]

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eval. :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, "
        f"Diversity Pred. {diversity:.4f}, R_precision Real. {R_precision_real}, "
        f"R_precision Pred. {R_precision}, MM-dist (matching_score) Real. {matching_score_real}, "
        f"MM-dist (matching_score) Pred. {matching_score_pred}"
    )
    logger.info(msg)

    best_fid_ret = fid
    if prev_best_fid is not None:
        try:
            prev_best_fid_val = float(prev_best_fid)
            if fid < prev_best_fid_val:
                logger.info(f"--> --> \t FID Improved from {prev_best_fid_val:.4f} to {fid:.4f} !!!")
            best_fid_ret = min(prev_best_fid_val, float(fid))
        except Exception:
            best_fid_ret = float(fid)

    best_rprec_ret = R_precision.clone() if isinstance(R_precision, torch.Tensor) else torch.tensor(R_precision)
    if prev_best_rprecision_pred is not None:
        try:
            prev_r = torch.as_tensor(prev_best_rprecision_pred, dtype=best_rprec_ret.dtype, device=best_rprec_ret.device)
            prev_r = prev_r.view(-1)[:3]
            for idx, k in enumerate([1, 2, 3]):
                if best_rprec_ret[idx].item() > prev_r[idx].item():
                    logger.info(
                        f"--> --> \t R_precision@{k} Pred Improved from {prev_r[idx].item():.4f} to {best_rprec_ret[idx].item():.4f} !!!"
                    )
            best_rprec_ret = torch.maximum(best_rprec_ret, prev_r)
        except Exception:
            best_rprec_ret = best_rprec_ret

    best_match_ret = matching_score_pred
    if prev_best_matching_score_pred is not None:
        try:
            prev_match_val = float(prev_best_matching_score_pred)
            if matching_score_pred.item() < prev_match_val:
                logger.info(
                    f"--> --> \t MM-dist Pred Improved from {prev_match_val:.4f} to {matching_score_pred.item():.4f} !!!"
                )
            best_match_ret = min(prev_match_val, matching_score_pred.item())
        except Exception:
            best_match_ret = (
                matching_score_pred.item() if isinstance(matching_score_pred, torch.Tensor) else float(matching_score_pred)
            )

    if prev_best_div is not None:
        try:
            prev_div_val = float(prev_best_div)
            if diversity > prev_div_val:
                logger.info(f"--> --> \t Diversity Pred Improved from {prev_div_val:.4f} to {diversity:.4f} !!!")
            best_diversity_ret = max(prev_div_val, float(diversity))
        except Exception:
            best_diversity_ret = float(diversity)
    else:
        best_diversity_ret = float(diversity)

    best_fid_ret = float(best_fid_ret)
    best_diversity_pred = float(best_diversity_ret)
    r1 = float(best_rprec_ret[0].item())
    r2 = float(best_rprec_ret[1].item())
    r3 = float(best_rprec_ret[2].item())
    best_match_ret = float(best_match_ret)

    return best_fid_ret, best_diversity_pred, r1, r2, r3, best_match_ret, logger


##### ---- Exp dirs ---- #####
args = get_denoiser_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

args_dir = os.path.join(args.out_dir, "args.json")
with open(args_dir, "w") as f:
    f.write(json.dumps(vars(args), indent=4, sort_keys=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### ---- Data ---- #####
args.latent_dir = os.path.join(
    f"./data/{'humanml3d_272' if args.dataset == 'humanml' else 'babel_272'}/motion_data"
)
args.latent_dim = 272

train_loader = DATALoader(
    args.dataset,
    args.batch_size,
    args.latent_dir,
    unit_length=args.unit_length,
    window_size=args.window_size,
    normalize=True,
)
train_loader_iter = cycle(train_loader)

val_loader = dataset_eval_t2m.DATALoader(
    args.dataset, True, 32, unit_length=args.unit_length, num_workers=8
)

logger.info(
    f"Dataset name: {args.dataset}, training on {args.train_split} ({len(train_loader.dataset)} samples), "
    f"evaluating on {args.eval_split} ({len(val_loader.dataset)} samples)"
)

##### ---- StableMoFusion model ---- #####
# Fill in StableMoFusion defaults if not provided by args.
args.dim_pose = getattr(args, "dim_pose", args.latent_dim)
args.text_latent_dim = getattr(args, "text_latent_dim", 256)
args.base_dim = getattr(args, "base_dim", 512)
args.dim_mults = getattr(args, "dim_mults", [2, 2, 2, 2])
args.time_dim = getattr(args, "time_dim", 512)
args.no_adagn = getattr(args, "no_adagn", False)
args.no_eff = getattr(args, "no_eff", False)
args.cond_mask_prob = getattr(args, "cond_mask_prob", 0.1)
args.diffusion_steps = getattr(args, "diffusion_steps", getattr(args, "diffusion_step", 1000))
args.beta_schedule = getattr(args, "beta_schedule", "linear")
args.prediction_type = getattr(args, "prediction_type", "sample")
args.diffuser_name = getattr(args, "diffuser_name", "ddpm")
args.num_inference_steps = getattr(args, "num_inference_steps", 50)

model = build_models(args)
model.to(device)
model.train()

noise_scheduler = DDPMScheduler(
    num_train_timesteps=args.diffusion_steps,
    beta_schedule=args.beta_schedule,
    variance_type="fixed_small",
    prediction_type=args.prediction_type,
    clip_sample=False,
)

pipeline = DiffusePipeline(
    opt=args,
    model=model,
    diffuser_name=args.diffuser_name,
    device=device,
    num_inference_steps=args.num_inference_steps,
    torch_dtype=torch.float32,
)

##### ---- Evaluator ---- #####
modelpath = "distilbert-base-uncased"

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = "checkpoints/evaluator/"
ckpt_path += "hml_epoch=99.ckpt" if args.dataset == "humanml" else "babel_epoch=69.ckpt"
print(f"Loading evaluator checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path)
textencoder_ckpt = {}
for k, v in ckpt["state_dict"].items():
    if k.split(".")[0] == "textencoder":
        name = k.replace("textencoder.", "")
        textencoder_ckpt[name] = v
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(device)

motionencoder_ckpt = {}
for k, v in ckpt["state_dict"].items():
    if k.split(".")[0] == "motionencoder":
        name = k.replace("motionencoder.", "")
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(device)

evaluator = [textencoder, motionencoder]

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Training ---- #####
nb_iter = 0
epoch_loss = {}
acc_list = []

best_fid = float("inf")
best_div = 0.0
best_top1, best_top2, best_top3 = 0.0, 0.0, 0.0
best_matching = float("inf")

for nb_iter in tqdm(
    range(1, args.total_iter + 1),
    desc=f"StableMoFusion training with dataset {args.dataset}",
    ncols=100,
    file=sys.stdout,
):
    batch = next(train_loader_iter)
    if len(batch) == 6:
        text, motion, m_tokens_len, _, _, _ = batch
    elif len(batch) == 4:
        text, motion, m_tokens_len, _ = batch
    else:
        text, motion, m_tokens_len = batch[:3]

    text = list(text)
    motion = motion.to(device).float()
    lengths = torch.as_tensor(m_tokens_len, device=device)

    x_start = motion
    B, T = x_start.shape[:2]
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float()

    real_noise = torch.randn_like(x_start)
    t = torch.randint(0, args.diffusion_steps, (B,), device=device)
    x_t = noise_scheduler.add_noise(x_start, real_noise, t)

    prediction = model(x_t, t, text=text)

    if args.prediction_type == "sample":
        target = x_start
    elif args.prediction_type == "epsilon":
        target = real_noise
    elif args.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(x_start, real_noise, t)
    else:
        raise ValueError(f"Unknown prediction_type: {args.prediction_type}")

    loss = masked_l2(prediction, target, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    acc_list.append(float(loss))
    if "loss" in epoch_loss:
        epoch_loss["loss"].append(loss.detach().cpu())
    else:
        epoch_loss["loss"] = [loss.detach().cpu()]

    if nb_iter % args.print_iter == 0:
        loss_msg = f"Denoiser step: {nb_iter}   "
        for key, value in epoch_loss.items():
            loss_msg += f"{key}: {round(float(sum(value) / len(value)), 4)}   "
        loss_msg += f"Loss: {round(sum(acc_list) / len(acc_list), 4)}"
        print("\033[A\033[A")
        print("\r\n", end="")
        logger.info(loss_msg)
        epoch_loss = {}
        acc_list = []

    if nb_iter % args.eval_iter == 0:
        prev_best_fid_local = best_fid
        eval_vis_dir = os.path.join(args.out_dir, "eval_vis", str(nb_iter))
        os.makedirs(eval_vis_dir, exist_ok=True)

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = (
            evaluation_stablemofusion(
                val_loader,
                pipeline,
                logger,
                evaluator,
                device=device,
                unit_length=args.unit_length,
                prev_best_fid=best_fid,
                prev_best_div=best_div,
                prev_best_rprecision_pred=[best_top1, best_top2, best_top3],
                prev_best_matching_score_pred=best_matching,
                draw=True,
                vis_dir=eval_vis_dir,
            )
        )

        if best_fid < prev_best_fid_local:
            save_dict = {
                "diffusion": model.state_dict(),
                "iter": nb_iter,
                "best_fid": best_fid,
                "best_top1": best_top1,
                "best_top2": best_top2,
                "best_top3": best_top3,
                "best_matching": best_matching,
            }
            if "scheduler" in locals():
                save_dict["scheduler"] = scheduler.state_dict()
            if "optimizer" in locals():
                save_dict["optimizer"] = optimizer.state_dict()
            torch.save(save_dict, os.path.join(args.out_dir, "best_fid.pth"))

        latest_save = {
            "diffusion": model.state_dict(),
            "iter": nb_iter,
            "best_fid": best_fid,
            "best_top1": best_top1,
            "best_top2": best_top2,
            "best_top3": best_top3,
            "best_matching": best_matching,
        }
        if "scheduler" in locals():
            latest_save["scheduler"] = scheduler.state_dict()
        if "optimizer" in locals():
            latest_save["optimizer"] = optimizer.state_dict()
        torch.save(latest_save, os.path.join(args.out_dir, "latest.pth"))
