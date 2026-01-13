import os
import json
import sys

import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from models.vqdm.vqvae import VQVAE
import utils.losses as losses 

from options.vqdm.vqvae_option import get_vqvae_args_parser
# from options.eval_option import get_opt

import utils.utils_model as utils_model
# from evaluate.eval_humanml import evaluation_vqvae
# from evaluate.eval_humanact12_uestc import evaluation_vqvae as evaluate_vqvae_humanact12_uestc
# from evaluate.eval_ntu import evaluation_vqvae as evaluate_vqvae_ntu

# from data_loaders.get_data import get_dataset_loader
# from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper, EvaluatorMDMWrapper_NTU
# from utils.word_vectorizer import WordVectorizer

from torch.utils.tensorboard import SummaryWriter
import models.tae as tae
import utils.losses as losses 
import options.option_tae as option_tae
import utils.utils_model as utils_model
from dataloader import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')




from tqdm import tqdm
import warnings
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

##### ---- Exp dirs ---- #####
args = get_vqvae_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# w_vectorizer = WordVectorizer('./glove', 'our_vab')

# if args.dataset == 'kit': 
#     dataset_opt_path = 'dataset/kit_opt.txt'  
    
# elif args.dataset == 'humanml':
#     dataset_opt_path = 'dataset/humanml_opt.txt'
#     eval_wrapper = EvaluatorMDMWrapper(args.dataset, device)

# elif args.dataset == 'ntu':
#     dataset_opt_path = 'dataset/ntu_opt.txt'
#     eval_wrapper = EvaluatorMDMWrapper_NTU(args.dataset, device)



# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))

##### ---- Dataloader ---- #####
# train_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.window_size, unit_length=args.unit_length, split=args.train_split)
# eval_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=args.window_size, unit_length=args.unit_length, split=args.eval_split, drop_last=True, hml_mode='gt')


##### ---- Dataloader ---- #####
train_loader = dataset_tae.DATALoader(args.dataset,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=args.unit_length)

val_loader = dataset_eval_tae.DATALoader(args.dataset, False,
                                        32,
                                        unit_length=2**args.down_t)


logger.info(f'Dataset name: {args.dataset}, training on {args.train_split} ({len(train_loader.dataset)} samples), evaluating on {args.eval_split} ({len(val_loader.dataset)} samples)')

train_loader_iter = cycle(train_loader)


##### ---- Network ---- #####
net = VQVAE(args, ## use args to define different parameters in different quantizers
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

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  

Loss = losses.ReConsLoss(272)

##### ------ warm-up ------- #####
# epoch_loss = {}

# for nb_iter in tqdm(range(1, args.warm_up_iter), desc=f"Warm up VQVAE with dataset {args.dataset}", ncols=100, file=sys.stdout):
    
#     optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
#     gt_motion, cond = next(train_loader_iter)
#     # gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

#     gt_motion = gt_motion.to(device)
#     cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
#     B, J, D, T = gt_motion.shape
#     gt_motion = gt_motion.permute(0, 3, 1, 2).reshape(B, T, -1)
#     mask = cond['y']['mask'].squeeze()

#     pred_motion, losses, perplexity = net(gt_motion, mask, "train")
#     loss = losses['total_loss']

#     for key, value in losses.items():
#         if key in epoch_loss.keys():
#             epoch_loss[key].append(value.mean().detach().cpu())
#         else:
#             epoch_loss[key] = [value.mean().detach().cpu()]
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

    
#     if nb_iter % args.print_iter ==  0 :
#         loss_msg = f"Warm up step: {nb_iter}   "
#         for key, value in epoch_loss.items():
#             loss_msg += "{}: {}   ".format(key, round(float(sum(value)/len(value)), 4))
#         print ("\033[A\033[A")
#         print("\r\n", end="")
#         logger.info(loss_msg)
#         epoch_loss = {}

# breakpoint()

##### ---- Training ---- #####



# print("Running evaluation...")
# epoch_loss = {}
# if args.dataset in ["kit", "humanml"]:
#     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
#         evaluation_vqvae(args.out_dir, eval_loader, net, logger, None, 0, 
#                          best_fid=1000, best_iter=0, best_div=100, best_top1=0, 
#                          best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
# elif args.dataset == "ntu":
#     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
#         evaluate_vqvae_ntu(args.out_dir, eval_loader, net, logger, None, 0, 
#                            best_fid=1000, best_iter=0, best_div=100, best_top1=0, 
#                            best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
# if args.dataset in ["humanact12", "uestc"]:
#     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
#         evaluate_vqvae_humanact12_uestc(args, args.out_dir, eval_loader, net, logger, None, 0, 
#                                         best_fid=1000, best_iter=0, best_div=100, best_top1=0, 
#                                         best_top2=0, best_top3=0, best_matching=100, eval_wrapper=None)

# best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_vqdm_vae(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net, logger, writer, 0, best_iter=0, best_mpjpe=1000, device=device)
epoch_loss = {}
best_iter, best_mpjpe = 0, 1000
for nb_iter in tqdm(range(1, args.total_iter + 1), desc=f"VQVAE training with dataset {args.dataset}", ncols=100, file=sys.stdout):

    gt_motion = next(train_loader_iter)
    # gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

    gt_motion = gt_motion.to(device).float()
    # cond['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
    B, T, D = gt_motion.shape
    # gt_motion = gt_motion.permute(0, 3, 1, 2).reshape(B, T, -1)
    # mask = cond['y']['mask'].squeeze()
    mask = torch.ones(B, T).to(device).to(torch.bool)

    pred_motion, losses, perplexity = net(gt_motion, mask, "train")
    loss = losses['total_loss']

    for key, value in losses.items():
        if key in epoch_loss.keys():
            epoch_loss[key].append(value.mean().detach().cpu())
        else:
            epoch_loss[key] = [value.mean().detach().cpu()]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter ==  0 :
        loss_msg = f"VQVAE step: {nb_iter}   "
        for key, value in epoch_loss.items():
            loss_msg += "{}: {}   ".format(key, round(float(sum(value)/len(value)), 4))
        print ("\033[A\033[A")
        print("\r\n", end="")
        logger.info(loss_msg)
        epoch_loss = {}

    if nb_iter % args.eval_iter==0 :
        best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_vqdm_vae(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net, logger, writer, nb_iter, best_iter, best_mpjpe, device=device, draw=False)
        # if args.dataset in ["kit", "humanml"]:
        #     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        #         evaluation_vqvae(args.out_dir, eval_loader, net, logger, writer, nb_iter, best_fid, best_iter, 
        #                          best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        # elif args.dataset == "ntu":
        #     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        #         evaluate_vqvae_ntu(args.out_dir, eval_loader, net, logger, writer, nb_iter, best_fid, best_iter, 
        #                          best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        # if args.dataset in ["humanact12", "uestc"]:
        #     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        #         evaluate_vqvae_humanact12_uestc(args, args.out_dir, eval_loader, net, logger, writer, nb_iter, best_fid, best_iter, 
        #                          best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=None)

