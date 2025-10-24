from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import copy

from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import PrimitiveSequenceDataset, WeightedPrimitiveSequenceDataset, WeightedPrimitiveSequenceDatasetV2
from data_loaders.humanml.data.dataset_hml3d import HML3dDataset
from utils.smpl_utils import get_smplx_param_from_6d
from pytorch3d import transforms
from diffusion.nn import mean_flat, sum_flat

debug = 0

@dataclass
class VAEArgs:
    arch: str = "all_encoder"
    ff_size: int = 1024
    num_layers: int = 5
    num_heads: int = 4
    dropout: float = 0.1
    normalize_before: bool = False
    activation: str = "gelu"
    position_embedding: str = "learned"
    latent_dim: tuple[int, int] = (1, 128)
    h_dim: int = 256

    nfeats: int = 0
    """feature dimension, will be auto filled"""

@dataclass
class DataArgs:
    cfg_path: str = "./config_files/config_hydra/motion_primitive/mp_h2_h8_r1.yaml"
    """motion primitive config file"""

    data_dir: str = "./data/seq_data"
    """processed dataset directory"""

    dataset: str = "mp_seq"
    """dataset name"""

    prob_static: float = 0.0
    enforce_gender: str = 'male'
    """enforce all data use the specified gender"""

    enforce_zero_beta: int = 1
    """enforce all data use zero shape parameters"""

    weight_scheme: str = 'uniform_samp:0.'
    """weighting schemes determining how motion primitives are sampled during training"""

    text_tolerance: float = 0.0
    """accept text labels in near future within some frames"""

    history_length: int = 0
    future_length: int = 0
    num_primitive: int = 0
    feature_dim: int = 0
    """auto filled"""

    body_type: str = 'smplx'
    """body type, 'smplx' or 'smplh'"""

@dataclass
class TrainArgs:
    learning_rate: float = 1e-4
    anneal_lr: int = 1
    batch_size: int = 128
    grad_clip: float = 1.0

    ema_decay: float = 0.999
    """exponential moving average decay"""
    use_amp: int = 0
    """use automatic mixed precision"""

    stage1_steps: int = 100000
    """training steps for stage 1 without rollout training"""
    stage2_steps: int = 100000
    """training steps for stage 2 with linearly increasing percent of rollout training"""
    stage3_steps: int = 100000
    """training steps for stage 3 with only rollout training"""

    weight_rec: float = 1.0  # vae only
    weight_kl: float = 1e-4  # vae only
    weight_smpl_joints_rec: float = 0.0
    weight_joints_consistency: float = 0.0
    weight_transl_delta: float = 0.0
    weight_orient_delta: float = 0.0
    weight_joints_delta: float = 0.0
    weight_latent_rec: float = 1.0  # denoiser only
    weight_feature_rec: float = 0.0  # denoiser only

    resume_checkpoint: str | None = None
    log_interval: int = 1000
    val_interval: int = 10000
    save_interval: int = 100000

    use_predicted_joints: int = 0
    """if set to 1, use predicted joints to rollout, otherwise use the regressed joints from smplx body model"""


@dataclass
class Args:
    train_args: TrainArgs = TrainArgs()
    model_args: VAEArgs = VAEArgs()
    data_args: DataArgs = DataArgs()

    exp_name: str = "mvae"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./mvae"

    track: int = 1
    wandb_project_name: str = "mld_vae"
    wandb_entity: str = "interaction"

class Trainer:
    def __init__(self, args: Args):
        self.args = args
        args.save_dir = Path('./mvae') / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        model_args = args.model_args
        data_args = args.data_args

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # load dataset
        if data_args.dataset == 'mp_seq_v2':
            dataset_class = WeightedPrimitiveSequenceDatasetV2
        elif data_args.dataset == 'hml3d':
            dataset_class = HML3dDataset
        else:
            dataset_class = WeightedPrimitiveSequenceDataset
        train_dataset = dataset_class(dataset_path=data_args.data_dir,
                                      dataset_name=data_args.dataset,
                                      cfg_path=data_args.cfg_path, prob_static=data_args.prob_static,
                                      enforce_gender=data_args.enforce_gender,
                                      enforce_zero_beta=data_args.enforce_zero_beta,
                                      body_type=data_args.body_type,
                                      split='train', device=device,
                                      weight_scheme=data_args.weight_scheme,
                                      )

        val_dataset = train_dataset
        # if 'text' in data_args.weight_scheme or 'samp:1' in data_args.weight_scheme:
        #     val_dataset = train_dataset
        # else:
        #     val_dataset = dataset_class(dataset_path=data_args.data_dir, dataset_name=data_args.dataset,
        #                                                    cfg_path=data_args.cfg_path, prob_static=data_args.prob_static,
        #                                                    enforce_gender=data_args.enforce_gender,
        #                                                    enforce_zero_beta=data_args.enforce_zero_beta,
        #                                                    split='val', device=device,
        #                                                    weight_scheme=data_args.weight_scheme,
        #                                                    )

        # get primitive configs
        data_args.history_length = train_dataset.history_length
        data_args.future_length = train_dataset.future_length
        data_args.num_primitive = train_dataset.num_primitive
        data_args.feature_dim = 0
        for k in train_dataset.motion_repr:
            data_args.feature_dim += train_dataset.motion_repr[k]
        model_args.nfeats = data_args.feature_dim

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)
        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
            wandb.run.log_code(root=".",
                               include_fn=lambda path, root: os.path.relpath(path, root).startswith("mld/") or
                                                             os.path.relpath(path, root).startswith("model/")
                               )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        print('model args:', asdict(model_args))
        model = AutoMldVae(
            **asdict(model_args),
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=train_args.learning_rate)
        start_step = 1
        if args.train_args.resume_checkpoint is not None:
            checkpoint = torch.load(args.train_args.resume_checkpoint)
            model_state_dict = checkpoint['model_state_dict']
            if 'latent_mean' not in model_state_dict:
                model_state_dict['latent_mean'] = torch.tensor(0)
            if 'latent_std' not in model_state_dict:
                model_state_dict['latent_std'] = torch.tensor(1)
            model.load_state_dict(model_state_dict)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['num_steps'] + 1
            print(f"Loading checkpoint from {args.train_args.resume_checkpoint} at step {start_step}")
        self.model_avg = None
        if args.train_args.ema_decay > 0:
            self.model_avg = copy.deepcopy(model)
            self.model_avg.eval()

        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.transf_rotmat = torch.eye(3, device=self.device).unsqueeze(0)
        self.transf_transl = torch.zeros(3, device=self.device).reshape(1, 1, 3)

    def calc_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent, dist):
        train_args = self.args.train_args
        model_kwargs = cond
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        terms = {}

        # kl loss
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
        kl_loss = kl_loss.mean()
        terms['kl_loss'] = kl_loss

        # reconstruction loss
        rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        terms['rec_loss'] = rec_loss

        """smplx consistency losses"""
        gt_motion_tensor = future_motion_gt
        pred_motion_tensor = future_motion_pred
        genders = model_kwargs['y']['gender']
        betas = model_kwargs['y']['betas']
        dataset = self.train_dataset
        primitive_utility = dataset.primitive_utility
        def get_smpl_body(motion_tensor, genders, betas):
            batch_size, num_frames, _ = motion_tensor.shape
            device = motion_tensor.device
            smpl_joints = []
            smpl_vertices = []
            joints = []
            for gender_name in ['female', 'male']:
                # body_model = body_model_male if gender_name == 'male' else body_model_female
                body_model = primitive_utility.get_smpl_model(gender=gender_name)
                gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
                sub_batch_size = len(gender_idx)
                if len(gender_idx) == 0:
                    continue
                # gender_betas = betas[gender_idx].unsqueeze(1).repeat(1, num_frames, 1).reshape(
                #     sub_batch_size * num_frames, -1)
                gender_betas = betas[gender_idx, history_length:, :].reshape(
                    sub_batch_size * num_frames, 10)
                gender_motion_tensor = motion_tensor[gender_idx, :, :]
                gender_motion_tensor = dataset.denormalize(gender_motion_tensor).reshape(
                    sub_batch_size * num_frames, -1)

                motion_dict = dataset.tensor_to_dict(gender_motion_tensor)
                motion_dict.update({'betas': gender_betas})
                joints.append(motion_dict['joints'].reshape(sub_batch_size, num_frames, 22, 3))
                smplx_param = get_smplx_param_from_6d(motion_dict)
                smplxout = body_model(return_verts=False, **smplx_param)
                smpl_joints.append(smplxout.joints[:, :22, :].reshape(sub_batch_size, num_frames, 22,
                                                                      3))  # [bs, nframes, 22, 3]
                # smpl_vertices.append(
                #     smplxout.vertices.reshape(sub_batch_size, num_frames, -1, 3))  # [bs, nframes, V, 3]

            smpl_joints = torch.cat(smpl_joints, dim=0)
            # smpl_vertices = torch.cat(smpl_vertices, dim=0)
            smpl_vertices = None
            joints = torch.cat(joints, dim=0)
            return {'smpl_joints': smpl_joints, 'smpl_vertices': smpl_vertices, 'joints': joints}

        with torch.no_grad():
            gt_result_dict = get_smpl_body(gt_motion_tensor, genders,
                                           betas)  # note that each batch is reordered according to gender. we assume the input batch is already sorted by gender, so the actual order does not change after this operation
        pred_result_dict = get_smpl_body(pred_motion_tensor, genders, betas)
        terms['smpl_joints_rec'] = self.rec_criterion(pred_result_dict['smpl_joints'], gt_result_dict['smpl_joints'])
        terms['joints_consistency'] = self.rec_criterion(pred_result_dict['joints'], pred_result_dict['smpl_joints'])
        # terms['smpl_vertices_rec'] = torch.zeros_like(terms['smpl_joints_rec'])

        """temporal delta loss"""
        pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)  # [B, F+1, D]
        pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
        pred_feature_dict = dataset.tensor_to_dict(pred_motion_tensor)
        pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
        pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
        pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
        calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
        calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
        pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['poses_6d'][:, :, :6])  # [B, T, 3, 3]
        calc_orient_delta_matrix = torch.matmul(pred_orient[:, 1:],
                                                pred_orient[:, :-1].permute(0, 1, 3, 2))
        calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
        terms["joints_delta"] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
        terms["transl_delta"] = self.rec_criterion(calc_transl_delta, pred_transl_delta)
        terms["orient_delta"] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)

        loss = train_args.weight_kl * kl_loss + train_args.weight_rec * rec_loss + \
               train_args.weight_smpl_joints_rec * terms['smpl_joints_rec'] + \
               train_args.weight_joints_consistency * terms['joints_consistency'] + \
               train_args.weight_joints_delta * terms["joints_delta"] + \
               train_args.weight_transl_delta * terms["transl_delta"] + \
               train_args.weight_orient_delta * terms["orient_delta"]
        terms['loss'] = loss
        return terms

    def train(self):
        model = self.model
        optimizer = self.optimizer
        args = self.args
        train_args = self.args.train_args
        writer = self.writer
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        model.train()
        total_steps = train_args.stage1_steps + train_args.stage2_steps + train_args.stage3_steps
        rest_steps = (total_steps - self.start_step) // self.train_dataset.num_primitive + 1
        rest_steps = rest_steps * self.train_dataset.num_primitive
        progress_bar = iter(tqdm(range(rest_steps)))
        self.step = self.start_step
        while self.step <= total_steps:
            # Annealing the rate if instructed to do so.
            if train_args.anneal_lr:
                frac = 1.0 - (self.step - 1.0) / total_steps
                lrnow = frac * train_args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                batch = self.train_dataset.get_batch(self.batch_size)
            last_primitive = None
            for primitive_idx in range(num_primitive):
                with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
                    future_motion_gt = motion_tensor[:, -future_length:, :]
                    history_motion = motion_tensor[:, :history_length, :]
                    if last_primitive is not None:
                        rollout_history = self.get_rollout_history(last_primitive, cond)
                        history_motion = rollout_history    # [B, H, D]

                    latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)
                    future_motion_pred = model.decode(latent, history_motion, nfuture=future_length)  # [B, F, D]

                    loss_dict = self.calc_loss(motion, cond, history_motion, future_motion_gt, future_motion_pred, latent, dist)
                    loss = loss_dict['loss']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
                optimizer.step()

                # update the average model using exponential moving average
                if train_args.ema_decay > 0:
                    for param, avg_param in zip(self.model.parameters(), self.model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(1.0, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.step % train_args.log_interval == 0:
                    for key in loss_dict:
                        writer.add_scalar(f"loss/{key}", loss_dict[key].item(), self.step)
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.step)

                if self.step % train_args.save_interval == 0 or self.step == total_steps:
                    self.save()

                if self.step % train_args.val_interval == 0 or self.step == total_steps:
                    self.validate()

                self.step += 1
                next(progress_bar)

    def get_primitive_batch(self, batch, primitive_idx):
        motion = batch[primitive_idx]['motion_tensor_normalized']  # [bs, D, 1, T]
        cond = {'y': {'text': batch[primitive_idx]['texts'],
                      'text_embedding': batch[primitive_idx]['text_embedding'],  # [bs, 512]
                      'gender': batch[primitive_idx]['gender'],
                      'betas': batch[primitive_idx]['betas'],  # [bs, T, 10]
                      'history_motion': batch[primitive_idx]['history_motion'],  # [bs, D, 1, T]
                      'history_mask': batch[primitive_idx]['history_mask'],
                      'history_length': batch[primitive_idx]['history_length'],
                      'future_length': batch[primitive_idx]['future_length']
                      }
                }
        return motion, cond

    def get_rollout_history(self, last_primitive, cond,
                            return_transform=False,
                            transf_rotmat=None, transf_transl=None
                            ):
        """update history motion seed, update global transform"""
        motion_tensor = last_primitive[:, -self.train_dataset.history_length:, :]  # [B, T, D]
        new_history_frames = self.train_dataset.denormalize(motion_tensor)
        primitive_utility = self.train_dataset.primitive_utility
        rollout_history = []
        genders = cond['y']['gender']
        new_transf_rotmat, new_transf_transl = [], []
        for gender_name in ['female', 'male']:
            gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
            if len(gender_idx) == 0:
                continue
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames[gender_idx])
            history_feature_dict.update(
                {
                    'transf_rotmat': self.transf_rotmat.repeat(len(gender_idx), 1, 1) if transf_rotmat is None else transf_rotmat[gender_idx],
                    'transf_transl': self.transf_transl.repeat(len(gender_idx), 1, 1) if transf_transl is None else transf_transl[gender_idx],
                    'gender': gender_name,
                    'betas': cond['y']['betas'][gender_idx, -self.train_dataset.history_length:, :],
                }
            )
            pelvis_delta = primitive_utility.calc_calibrate_offset({
                'betas': history_feature_dict['betas'][:, 0, :],  # [B, 10]
                'gender': gender_name,
            })
            history_feature_dict['pelvis_delta'] = pelvis_delta
            use_predicted_joints = getattr(self.args.train_args, 'use_predicted_joints', False)
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=use_predicted_joints)
            new_transf_rotmat.append(canonicalized_history_primitive_dict['transf_rotmat'])
            new_transf_transl.append(canonicalized_history_primitive_dict['transf_transl'])
            history_motion_tensor = primitive_utility.dict_to_tensor(blended_feature_dict)
            rollout_history.append(history_motion_tensor)

        rollout_history = torch.cat(rollout_history, dim=0)
        rollout_history = self.train_dataset.normalize(rollout_history)  # [B, T, D]
        # rollout_history = rollout_history.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T_history]

        if return_transform:
            return rollout_history, torch.cat(new_transf_rotmat, dim=0), torch.cat(new_transf_transl, dim=0)
        else:
            return rollout_history

    def get_latent_scale(self, model):
        """
        get the scale of the latent space
            model: model or model_avg
        """
        original_mode = model.training
        model.eval()

        train_args = self.args.train_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        with torch.no_grad():
            batch = self.train_dataset.get_batch(self.batch_size)
            primitive_idx = 0
            motion, cond = self.get_primitive_batch(batch, primitive_idx)
            motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
            future_motion_gt = motion_tensor[:, -future_length:, :]
            history_motion = motion_tensor[:, :history_length, :]

            latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)  # [1, B, D]
            all_mean = latent.mean()
            all_std = (latent - all_mean).pow(2).mean().sqrt()
            model.register_buffer("latent_mean", all_mean)
            model.register_buffer("latent_std", all_std)
            print(f"latent mean: {all_mean}, latent std: {all_std}")

        model.train(original_mode)

    def save(self):
        model = self.model if self.model_avg is None else self.model_avg
        print('save avg model:', self.model_avg is not None)
        self.get_latent_scale(model)
        checkpoint_path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({
            'num_steps': self.step,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")

    def validate(self):
        original_mode = self.model.training
        self.model.eval()

        model = self.model
        optimizer = self.optimizer
        args = self.args
        train_args = self.args.train_args
        writer = self.writer
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        with torch.no_grad():
            losses_dict = {}
            for _ in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                last_primitive = None
                for primitive_idx in range(num_primitive):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
                    future_motion_gt = motion_tensor[:, -future_length:, :]
                    history_motion = motion_tensor[:, :history_length, :]
                    if last_primitive is not None:
                        rollout_history = self.get_rollout_history(last_primitive, cond)
                        history_motion = rollout_history  # [B, H, D]

                    latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)
                    future_motion_pred = model.decode(latent, history_motion, nfuture=future_length)

                    loss_dict = self.calc_loss(motion, cond, history_motion, future_motion_gt, future_motion_pred,
                                               latent, dist)
                    for k, v in loss_dict.items():
                        if k not in losses_dict:
                            losses_dict[k] = []
                        losses_dict[k].append(v.detach())

                    if self.step > train_args.stage1_steps:
                        last_primitive = future_motion_pred.detach()
                    else:
                        last_primitive = None

        for k, v in losses_dict.items():
            losses_dict[k] = torch.stack(v).mean().item()
            self.writer.add_scalar(f"val_loss/{k}", losses_dict[k], self.step)
        self.model.train(original_mode)

    def close(self):
        self.writer.close()



if __name__ == "__main__":
    args = tyro.cli(Args)
    trainer = Trainer(args)
    trainer.train()
    trainer.close()

