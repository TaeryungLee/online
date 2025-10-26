import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.motionprimitive import MotionPrimitive
import utils.losses as losses 
import options.option_tae as option_tae
import utils.utils_model as utils_model
from dataloader import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')


##### ---- Device Setup ---- #####
comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.window_size = args.history + (args.future * 3)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')



##### ---- Rendering ---- #####




##### ---- Dataloader ---- #####
train_loader = dataset_tae.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=args.unit_length)

val_loader = dataset_eval_tae.DATALoader(args.dataname, False,
                                        32,
                                        unit_length=args.unit_length)

##### ---- Network ---- #####
clip_range = [-6, 6]

# net = tae.Causal_HumanTAE(
#                        hidden_size=args.hidden_size,
#                        down_t=args.down_t,
#                        stride_t=args.stride_t,
#                        depth=args.depth,
#                        dilation_growth_rate=args.dilation_growth_rate,
#                        activation='relu',
#                        latent_dim=args.latent_dim,
#                        clip_range=clip_range
#                        )

net = MotionPrimitive(cfg=args)

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt, strict=True)
net.train()
net.to(comp_device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

if args.num_gpus > 1:
    net = torch.nn.DataParallel(net)
    
train_loader_iter = dataset_tae.cycle(train_loader)

Loss = losses.ReConsLoss(motion_dim=272)

##### ------ warm-up ------- #####
avg_recons, avg_kl, avg_root, avg_seam = 0., 0., 0., 0.
for nb_iter in range(1, args.warm_up_iter+1):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float()
    history_motion_pred = None

    # 세 번의 forward pass를 하도록 수정
    for fwd_idx in range(3):
        gt_motion_fwd = gt_motion[:, fwd_idx * args.future: (fwd_idx + 1) * args.future + args.history]
        future_motion = gt_motion_fwd[:, args.history:]
        if fwd_idx == 0:
            history_motion = gt_motion_fwd[:, :args.history]
        else:
            history_motion = history_motion_pred

        pred_motion, mu, logvar = net(future_motion, history_motion)
        
        history_motion_pred = pred_motion[:, -args.history:].clone().detach()
        loss_motion = Loss(pred_motion, future_motion)
        
        loss_kl = Loss.forward_KL(mu, logvar)
        loss_root = Loss.forward_root(pred_motion, future_motion)
        loss_vel = Loss.forward_vel_loss(pred_motion, future_motion)
        loss_acc = Loss.forward_acc_loss(pred_motion, future_motion)
        # loss seam
        seam_vel = history_motion[:, -1] - pred_motion[:, 0]
        seam_vel_gt = gt_motion_fwd[:, args.history-1] - gt_motion_fwd[:, args.history]
        loss_seam = torch.mean(torch.abs(seam_vel - seam_vel_gt))

        loss = loss_motion + loss_kl * args.kl_loss + args.root_loss * loss_root + args.vel_loss * loss_vel + args.acc_loss * loss_acc + args.seam_loss * loss_seam

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_kl += loss_kl.item()
        avg_root += loss_root.item()
        avg_seam += loss_seam.item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter
        avg_root /= args.print_iter
        avg_seam /= args.print_iter
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Recons.  {avg_recons:.5f} \t KL. {avg_kl:.5f} \t Root. {avg_root:.5f} \t Seam. {avg_seam:.5f}")
        
        avg_recons, avg_kl, avg_root, avg_seam = 0., 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_kl, avg_root, avg_seam = 0., 0., 0., 0.



if args.num_gpus > 1:
    best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_motionprimitive_multi(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net.module, logger, writer, 0, best_iter=0, best_mpjpe=1000, device=comp_device)
else:
    best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_motionprimitive_multi(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net, logger, writer, 0, best_iter=0, best_mpjpe=1000, device=comp_device)

for nb_iter in range(1, args.total_iter + 1):
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float()
    history_motion_pred = None

    # 세 번의 forward pass를 하도록 수정
    for fwd_idx in range(3):
        gt_motion_fwd = gt_motion[:, fwd_idx * args.future: (fwd_idx + 1) * args.future + args.history]
        future_motion = gt_motion_fwd[:, args.history:]
        if fwd_idx == 0:
            history_motion = gt_motion_fwd[:, :args.history]
        else:
            history_motion = history_motion_pred

        pred_motion, mu, logvar = net(future_motion, history_motion)
        
        history_motion_pred = pred_motion[:, -args.history:].clone().detach()
        loss_motion = Loss(pred_motion, future_motion)
        
        loss_kl = Loss.forward_KL(mu, logvar)
        loss_root = Loss.forward_root(pred_motion, future_motion)
        loss_vel = Loss.forward_vel_loss(pred_motion, future_motion)
        loss_acc = Loss.forward_acc_loss(pred_motion, future_motion)
        # loss seam
        seam_vel = history_motion[:, -1] - pred_motion[:, 0]
        seam_vel_gt = gt_motion_fwd[:, args.history-1] - gt_motion_fwd[:, args.history]
        loss_seam = torch.mean(torch.abs(seam_vel - seam_vel_gt))

        loss = loss_motion + loss_kl * args.kl_loss + args.root_loss * loss_root + args.vel_loss * loss_vel + args.acc_loss * loss_acc + args.seam_loss * loss_seam

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_kl += loss_kl.item()
        avg_root += loss_root.item()
        avg_seam += loss_seam.item()

    scheduler.step()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter
        avg_root /= args.print_iter
        avg_seam /= args.print_iter
        writer.add_scalar('./Train/Recon_loss', avg_recons, nb_iter)
        writer.add_scalar('./Train/KL', avg_kl, nb_iter)
        writer.add_scalar('./Train/Root_loss', avg_root, nb_iter)
        writer.add_scalar('./Train/Seam_loss', avg_seam, nb_iter)
        writer.add_scalar('./Train/LR', current_lr, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Recons.  {avg_recons:.5f} \t KL. {avg_kl:.5f} \t Root. {avg_root:.5f} \t Seam. {avg_seam:.5f}")
        
        avg_recons, avg_kl, avg_root, avg_seam = 0., 0., 0., 0.

    if nb_iter % args.eval_iter==0:
        if args.num_gpus > 1:
            best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_motionprimitive_multi(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net.module, logger, writer, nb_iter, best_iter, best_mpjpe, device=comp_device)
        else:
            best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_motionprimitive_multi(args.out_dir, os.path.join(args.out_dir, str(nb_iter)), val_loader, net, logger, writer, nb_iter, best_iter, best_mpjpe, device=comp_device)
