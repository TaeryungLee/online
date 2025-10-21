"""Train original text to motion generation model with llama blocks, Two-Forward strategy and QK-Norm, using the motion latents encoded by the Causal TAE (trained in the first stage)."""

import os
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import json
 

from models.llama_model import LLaMAHF, LLaMAHFConfig
from dataloader.dataset_TM_train_roll import DATALoader, cycle
from dataloader import dataset_eval_t2m_roll as dataset_eval_t2m
import options.option_transformer as option_trans
import options.option_tae as option_tae
import utils.utils_model as utils_model
import warnings
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from models.latent import LatentSpaceVAE
from models.diffusion_roll import DiffusionRoll

from transformers import AutoTokenizer, AutoModel
from Evaluator_272.mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from Evaluator_272.mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict
from utils.eval_trans import evaluation_transformer_272_single, evaluation_transformer_272_roll, evaluation_transformer_272_roll_no_latent
from Evaluator_272 import mld
import sys
import importlib
if 'mld' not in sys.modules:
    try:
        sys.modules['mld'] = importlib.import_module('Evaluator_272.mld')
    except Exception:
        pass



warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# warm-up + cosine decay scheduler
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters, 
                                                  eta_min=min_lr)
        
    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'total_iters': self.total_iters,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']


args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)


##### ---- Device Setup ---- #####
comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Without latent ---- #####
args.latent_dir = os.path.join(f"./data/{'humanml3d_272' if args.dataname == 't2m_272' else 'babel_272'}/motion_data")
args.latent_dim = 272


##### ---- Dataloader ---- #####
train_loader = DATALoader(args.dataname, args.batch_size, args.latent_dir, unit_length=args.unit_length, window_size=args.window_size, normalize=True)
train_loader_iter = cycle(train_loader)

val_loader = dataset_eval_t2m.DATALoader(args.dataname, True, 32, unit_length=args.unit_length, num_workers=args.num_workers)
val_loader_iter = cycle(val_loader)


# ##### ---- Latent Model ---- #####
# clip_range = [-6, 6]
# net = LatentSpaceVAE(
#     cfg=args,
#     hidden_size=args.hidden_size,
#     depth=args.depth,
#     attn_window=args.attn_window,
#     n_heads=args.n_heads,
#     activation=args.activation,
#     norm=args.norm,
#     latent_dim=args.latent_dim,
#     clip_range=clip_range
# )
# net = net.to(comp_device)


diffusion = DiffusionRoll(args)

if args.resume is not None:
    print('loading transformer checkpoint from {}'.format(args.resume))
    ckpt = torch.load(args.resume, map_location='cpu')
    new_ckpt_diffusion = {}
    for key in ckpt['diffusion'].keys():
        if key.split('.')[0]=='module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_ckpt_diffusion[new_key] = ckpt['diffusion'][key]
    diffusion.load_state_dict(new_ckpt_diffusion, strict=True)
diffusion.train()
diffusion.to(comp_device)


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, diffusion, args.optimizer)
scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter//10, args.total_iter)




##### ---- Evaluator ---- #####


modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = 'checkpoints/evaluator/'
ckpt_path += 'hml_epoch=99.ckpt' if args.dataname == 't2m_272' else 'babel_epoch=69.ckpt'
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
textencoder.to(comp_device)

# load motionencoder
motionencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "motionencoder":
        name = k.replace("motionencoder.", "")
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)
#--------------------------------

evaluator = [textencoder, motionencoder]




##### ---- Training Loop ---- #####
nb_iter, avg_loss = 0, 0.

# Track best metrics across evaluations
best_fid = float('inf')
best_div = 0.0
best_top1, best_top2, best_top3 = 0.0, 0.0, 0.0
best_matching = float('inf')


while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    text, m_tokens, m_tokens_len, caption_enc, caption_enc_len, idxs = batch
    text = list(text)
    m_tokens, m_tokens_len, caption_enc, caption_enc_len, idxs = m_tokens.to(comp_device), m_tokens_len.to(comp_device), caption_enc.to(comp_device), caption_enc_len.to(comp_device), idxs.to(comp_device)


    loss, pred_xstart = diffusion(m_tokens, caption_enc, caption_enc_len, idxs)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(nb_iter)

    avg_loss = avg_loss + loss.item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss = avg_loss / args.print_iter
        writer.add_scalar('./Loss/train', avg_loss, nb_iter)
        writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss:.5f}"
        logger.info(msg)
        avg_loss = 0.


    if nb_iter % args.eval_iter == 0:
        prev_best_fid_local = best_fid
        # Visualization directory for evaluation
        eval_vis_dir = os.path.join(args.out_dir, 'eval_vis', str(nb_iter))
        os.makedirs(eval_vis_dir, exist_ok=True)
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = evaluation_transformer_272_roll_no_latent(
            val_loader,
            net,
            diffusion,
            logger,
            evaluator,
            device=comp_device,
            unit_length=args.unit_length,
            prev_best_fid=best_fid,
            prev_best_div=best_div,
            prev_best_rprecision_pred=[best_top1, best_top2, best_top3],
            prev_best_matching_score_pred=best_matching,
            draw=args.vis_eval,
            vis_dir=eval_vis_dir,
        )
        # Save best FID checkpoint if improved
        if best_fid < prev_best_fid_local:
            save_dict = {
                'diffusion': diffusion.state_dict(),
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
            'diffusion': diffusion.state_dict(),
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

                    

# no accelerator synchronization needed
