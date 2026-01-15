import os
import json
import shutil
import sys

import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# from baselines.MDM.eval.eval_humanml import evaluation
from models.vqdm.vqvae import VQVAE
from models.vqdm.denoiser import VQDM
import utils.losses as losses 

# from options.vqvae_option import get_vqvae_args_parser
from options.vqdm.denoiser_option import get_denoiser_args_parser
import utils.utils_model as utils_model
from models.vqdm.modules.sampling import sample_from_logits

from dataloader.dataset_TM_train import DATALoader, cycle
from dataloader import dataset_eval_t2m as dataset_eval_t2m

from Evaluator_272.mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from Evaluator_272.mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

from utils.eval_trans import evaluation_vqdm

# from evaluate.eval_humanml import evaluation_denoiser
# from evaluate.eval_humanact12_uestc import evaluation_vqvae as evaluate_vqvae_humanact12_uestc
# from evaluate.eval_ntu import evaluation_vqvae as evaluate_vqvae_ntu

# from data_loaders.get_data import get_dataset_loader
# from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper, EvaluatorMDMWrapper_NTU
# from utils.word_vectorizer import WordVectorizer

from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
import models.tae as tae
import utils.losses as losses 
import options.option_tae as option_tae
import utils.utils_model as utils_model
from dataloader import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
import warnings

import sys
import importlib
if 'mld' not in sys.modules:
    try:
        sys.modules['mld'] = importlib.import_module('Evaluator_272.mld')
    except Exception:
        pass



warnings.filterwarnings('ignore')


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def pseudo_evaluation(eval_loader, vqvae, denoiser, prev_best_acc=0):
    correct = 0
    wrong = 0
    for _ in tqdm(eval_loader, desc=f"Evaluating accuracy", ncols=100, leave=False):
        batch = next(train_loader_iter)
        if len(batch) == 6:
            text, gt_motion, m_tokens_len, _, _, _ = batch
        else:
            text, gt_motion, m_tokens_len, _ = batch
        text = list(text)
        gt_motion = gt_motion.to(device).float()
        m_tokens_len = m_tokens_len.to(device)
        seq_len = gt_motion.shape[1]
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < m_tokens_len.unsqueeze(1)
        cond = {'y': {'text': text, 'mask': mask}}
        cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
        mask = cond['y']['mask']
        m_tokens_mask = mask[:, ::args.unit_length]
        gt_idxs = vqvae.encode(gt_motion)

        model_out = denoiser.sample(gt_idxs.shape, cond, device, filter_ratio=0)
        pred_idxs = model_out['content_token']
        correct += (gt_idxs[m_tokens_mask] == pred_idxs[m_tokens_mask]).sum().item()
        wrong += (gt_idxs[m_tokens_mask] != pred_idxs[m_tokens_mask]).sum().item()

    return correct / (correct + wrong)


##### ---- Exp dirs ---- #####
args = get_denoiser_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

args_dir = os.path.join(args.out_dir, "args.json")
with open(args_dir, "w") as f:
    f.write(json.dumps(vars(args), indent=4, sort_keys=True))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






##### ---- Without latent ---- #####
args.latent_dir = os.path.join(f"./data/{'humanml3d_272' if args.dataset == 'humanml' else 'babel_272'}/motion_data")
args.latent_dim = 272


##### ---- Dataloader ---- #####
train_loader = DATALoader(args.dataset, args.batch_size, args.latent_dir, unit_length=args.unit_length, window_size=args.window_size, normalize=True)
train_loader_iter = cycle(train_loader)

val_loader = dataset_eval_t2m.DATALoader(args.dataset, True, 32, unit_length=args.unit_length, num_workers=8)
val_loader_iter = cycle(val_loader)



logger.info(f'Dataset name: {args.dataset}, training on {args.train_split} ({len(train_loader.dataset)} samples), evaluating on {args.eval_split} ({len(val_loader.dataset)} samples)')


##### ---- Network ---- #####
vqvae = VQVAE(args, ## use args to define different parameters in different quantizers
            train_loader.dataset.inv_transform if args.dataset in ["kit", "humanml", "ntu"] else None,
            device,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)

if hasattr(train_loader.dataset, 'num_actions'):
    num_actions = train_loader.dataset.num_actions
else:
    num_actions = 1

## Denoiser input args
denoiser = VQDM(
    args, train_loader.dataset.inv_transform if args.dataset in ["kit", "humanml", "ntu"] else None, device,
    transformer_type=args.transformer_type,  # text2image or condition2image
    diffusion_step=args.diffusion_step,  # default 100
    cond_mode=args.cond_mode,  # action or text
    num_actions=num_actions,  # by dataset
    motion_length=args.window_size,  # by dataset
    unit_length=args.unit_length,  # default 4
    num_embed=args.nb_code,  # default 512, number of codes in codebook
    embed_dim=args.embed_dim,  # dimension of embedded conditions / indices
    hidden_dim=args.hidden_dim,

    # transformer options
    num_layers=args.num_layers,  # number of layers in denoising transformer
    num_heads=args.num_heads,  # number of heads in denoising transformer
    attn_type='selfcross' if args.transformer_type == 'text2image' else 'selfcondition',
    mlp_type=args.mlp_type,
    
    # condition embedding method
    action_emb_type=args.action_emb_type,  
        # layerwise (use class_type=adalayernorm in transformer) or single (use class_type=adalayernorm_mlp)
        # TODO: implement it for our setting: single condition embedding for both text and action
    text_emb_type='clip_seq' if args.transformer_type == 'text2image' else 'clip_vec',
    timestep_type=args.timestep_type,  # adalayernorm or adainsnorm
    clip_version='ViT-B/32'
)





##### ---- Evaluator ---- #####


modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = 'checkpoints/evaluator/'
ckpt_path += 'hml_epoch=99.ckpt' if args.dataset == 'humanml' else 'babel_epoch=69.ckpt'
print(f'Loading evaluator checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path)
# load textencoder
textencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "textencoder":
        name = k.replace("textencoder.", "")
        textencoder_ckpt[name] = v
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(device)

# load motionencoder
motionencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "motionencoder":
        name = k.replace("motionencoder.", "")
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(device)
#--------------------------------

evaluator = [textencoder, motionencoder]



##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(denoiser.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


if args.resume_pth is not None: 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    denoiser.load_state_dict(ckpt['net'], strict=True)
denoiser.train()
denoiser.to(device)

logger.info(f'loading VQVAE checkpoint from {args.pretrained_vqvae_pth}')
ckpt = torch.load(args.pretrained_vqvae_pth, map_location='cpu')
vqvae.load_state_dict(ckpt['net'], strict=True)

for p in vqvae.parameters():
    p.requires_grad = False

vqvae.eval()
vqvae.to(device)

assert os.path.isfile(args.pretrained_vqvae_pth)
logger.info(f'copying used VQVAE checkpoint into {os.path.join(args.out_dir, "vqvae.pth")}')
shutil.copy(args.pretrained_vqvae_pth, os.path.join(args.out_dir, "vqvae.pth"))






epoch_loss = {}
acc_list = []

nb_iter, avg_loss = 0, 0.

# Track best metrics across evaluations
best_fid = float('inf')
best_div = 0.0
best_top1, best_top2, best_top3 = 0.0, 0.0, 0.0
best_matching = float('inf')


for nb_iter in tqdm(range(1, args.total_iter + 1), desc=f"Denoiser training with dataset {args.dataset}", ncols=100, file=sys.stdout):   
    

    # breakpoint()

    batch = next(train_loader_iter)
    if len(batch) == 6:
        text, gt_motion, m_tokens_len, _, _, idxs = batch
    else:
        text, gt_motion, m_tokens_len, idxs = batch
    text = list(text)
    # Ensure all floating tensors are float32 to avoid dtype mismatches (e.g., Double vs Float)
    gt_motion = gt_motion.to(device).float()
    m_tokens_len = m_tokens_len.to(device)
    idxs = idxs.to(device)

    seq_len = gt_motion.shape[1]
    mask = torch.arange(seq_len, device=device).unsqueeze(0) < m_tokens_len.unsqueeze(1)
    cond = {'y': {'text': text, 'mask': mask}}
    cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
    mask = cond['y']['mask']
    m_tokens_mask = mask[:, ::args.unit_length]

    with torch.no_grad():
        gt_idxs = vqvae.encode(gt_motion).detach()

    model_out = denoiser(gt_idxs, cond, m_tokens_mask)

    # logits = model_out['logits'][:, :-1]
    # pred_idxs = sample_from_logits(logits, top_k=1)

    losses = {'vb': model_out['loss']}
    loss = model_out['loss']
    acc = float(model_out['acc'])
    acc_list.append(acc)

    for key, value in losses.items():
        if key in epoch_loss.keys():
            epoch_loss[key].append(value.mean().detach().cpu())
        else:
            epoch_loss[key] = [value.mean().detach().cpu()]

    # all_correct += (gt_idxs[m_tokens_mask] == pred_idxs[m_tokens_mask]).sum().item()
    # all_wrong += (gt_idxs[m_tokens_mask] != pred_idxs[m_tokens_mask]).sum().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if nb_iter % args.print_iter ==  0 :
        loss_msg = f"Denoiser step: {nb_iter}   "
        for key, value in epoch_loss.items():
            loss_msg += "{}: {}   ".format(key, round(float(sum(value)/len(value)), 4))
        loss_msg += f'Acc: {round(sum(acc_list)/len(acc_list) * 100, 4)}%'
        print ("\033[A\033[A")
        print("\r\n", end="")
        logger.info(loss_msg)
        epoch_loss = {}
        all_correct = 0
        all_wrong = 0


    if nb_iter % args.eval_iter==0 :
        # acc = pseudo_evaluation(eval_loader, vqvae, denoiser)
        # if prev_best_acc < acc:
        #     logger.info(f"-->  Accuracy improved from {round(prev_best_acc*100, 4)} to {round(acc*100, 4)}!!")
        #     prev_best_acc = acc
        #     torch.save({'net' : denoiser.state_dict()}, os.path.join(args.out_dir, 'net_best_acc.pth'))
        prev_best_fid_local = best_fid
        # Visualization directory for evaluation
        eval_vis_dir = os.path.join(args.out_dir, 'eval_vis', str(nb_iter))
        os.makedirs(eval_vis_dir, exist_ok=True)

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = evaluation_vqdm(
            val_loader,
            denoiser,
            vqvae,
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
        # Save best FID checkpoint if improved
        if best_fid < prev_best_fid_local:
            save_dict = {
                'diffusion': denoiser.state_dict(),
                'iter': nb_iter,
                'best_fid': best_fid,
                'best_top1': best_top1,
                'best_top2': best_top2,
                'best_top3': best_top3,
                'best_matching': best_matching,
            }
            if 'scheduler' in locals():
                try:
                    save_dict['scheduler'] = scheduler.state_dict()
                except Exception:
                    pass
            if 'optimizer' in locals():
                try:
                    save_dict['optimizer'] = optimizer.state_dict()
                except Exception:
                    pass
            torch.save(save_dict, os.path.join(args.out_dir, 'best_fid.pth'))
        # save 
        latest_save = {
            'diffusion': denoiser.state_dict(),
            'iter': nb_iter,
            'best_fid': best_fid,
            'best_top1': best_top1,
            'best_top2': best_top2,
            'best_top3': best_top3,
            'best_matching': best_matching,
        }
        if 'scheduler' in locals():
            try:
                latest_save['scheduler'] = scheduler.state_dict()
            except Exception:
                pass
        if 'optimizer' in locals():
            try:
                latest_save['optimizer'] = optimizer.state_dict()
            except Exception:
                pass
        torch.save(latest_save, os.path.join(args.out_dir, 'latest.pth'))