import os
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import sys
import options.option_transformer_orig as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from data.humanml3d_272 import dataset_eval_tae
import models.tae as tae
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd()) 

comp_device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
val_loader = dataset_eval_tae.DATALoader(args.dataname, False, 32)

##### ---- Network ---- #####
clip_range = [-30,20]

net = tae.Causal_HumanTAE(
                       hidden_size=args.hidden_size,
                       down_t=args.down_t,
                       stride_t=args.stride_t,
                       depth=args.depth,
                       dilation_growth_rate=args.dilation_growth_rate,
                       activation='relu',
                       latent_dim=args.latent_dim,
                       clip_range=clip_range
                       )

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(comp_device)

# load evaluator:
from Evaluator_272.mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = '../checkpoints/evaluator/'
ckpt_path += 'hml_epoch=99.ckpt' if args.dataname == 't2m_272' else 'babel_epoch=69.ckpt'
print(f'Loading evaluator checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path)
motionencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split(".")[0] == "motionencoder":
        name = k.replace("motionencoder.", "")
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)
#--------------------------------

evaluator = [None, motionencoder]

fid = []
mpjpe = []

best_fid, best_mpjpe, writer, logger = eval_trans.evaluation_tae_single(args.out_dir, val_loader, net, logger, writer, evaluator=evaluator, device=comp_device)
fid.append(best_fid)
mpjpe.append(best_mpjpe)

logger.info('final result:')
logger.info(f'fid: {fid}')
logger.info(f'mpjpe: {mpjpe} (mm)')
