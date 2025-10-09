import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json
from models.latent import LatentSpaceVAE
import options.option_tae as option_tae
import utils.utils_model as utils_model
from dataloader.dataset_tae_tokenizer import DATALoader
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader = DATALoader(args.dataname, unit_length=args.unit_length)

clip_range = [-6, 6]

net = LatentSpaceVAE(
    cfg=args,
    hidden_size=args.hidden_size,
    depth=args.depth,
    attn_window=args.attn_window,
    n_heads=args.n_heads,
    activation=args.activation,
    norm=args.norm,
    latent_dim=args.latent_dim,
    clip_range=clip_range
)

args.resume_pth = os.path.join(args.out_dir, args.resume_pth)
logger.info('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
state_dict = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
net.load_state_dict(state_dict, strict=True)
net.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)


##### ---- get reference end latent ---- #####
# reference_end_pose = torch.zeros(1, 4, 272).to(device)   # impossible pose prior
# reference_end_latent = net.encode(reference_end_pose).squeeze(0)  # (4, latent_dim)
# np.save(f'reference_end_latent_{args.dataname}.npy', reference_end_latent.detach().cpu().numpy())

os.makedirs(args.latent_dir, exist_ok = True)

for batch in tqdm(train_loader):
    pose, name = batch
    pose = pose.to(device).float()  # (1, T, 272)
    latent = net.encode(pose).squeeze(0)  # (T, latent_dim)
    # latent = torch.cat([latent, reference_end_latent], dim=0)  # append 4-frame ref
    np.save(pjoin(args.latent_dir, name[0] +'.npy'), latent.detach().cpu().numpy())
