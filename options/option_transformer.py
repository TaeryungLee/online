import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='options',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataname', type=str, default='t2m_272', help='dataset directory') 
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for training. ')
    parser.add_argument('--latent_dir', type=str, default='latents/t2m_latents', help='latent directory')
    parser.add_argument('--unit_length', type=int, default=1, help='unit length')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--overlap-size', type=int, default=8, help='overlap size')
    parser.add_argument('--dim_pose', type=int, default=272, help='dimension of pose')


    parser.add_argument("--latent-model-pth", type=str, default=None, help='resume pth for causal TAE')
    parser.add_argument("--resume", type=str, default=None, help='resume denoiser pth')
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp', help='name of the experiment, will create a file inside out-dir')

    # Latent Model Parameters
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--latent_dim', default=128, type=int, help='latent dimension')
    parser.add_argument("--depth", type=int, default=6, help="depth of the network")
    parser.add_argument("--attn-window", type=int, default=3, help="attention window")
    parser.add_argument("--n-heads", type=int, default=8, help="number of heads")
    parser.add_argument('--norm', type=str, default='ln', help='normalization function')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument("--decoder-conv-mlp", type=bool, default=True, help='whether use conv mlp for decoder')
    parser.add_argument("--encoder", type=str, default='transformer', help="encoder")

    # Denoiser Parameters
    parser.add_argument('--num_diffusion_head_layers', type=int, default=9, help='number of diffusion head layers')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--decay-option',default='all', type=str, choices=['all'], help='weight decay option')
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='optimizer')

    parser.add_argument('--num_timesteps', type=int, default=10, help='number of timesteps')
    parser.add_argument('--sigma_data', type=float, default=0.5, help='sigma data')
    parser.add_argument('--sigma_min', type=float, default=0.002, help='sigma min')
    parser.add_argument('--sigma_max', type=float, default=80.0, help='sigma max')
    parser.add_argument('--rho', type=float, default=-10.0, help='rho')
    parser.add_argument('--rho_init', type=float, default=7.0, help='rho init')
    parser.add_argument('--heun_churn', type=float, default=0.0, help='heun churn')

    parser.add_argument('--denoiser_block', type=int, default=5)
    parser.add_argument('--denoiser_num_layers', type=int, default=12, help='number of layers')
    parser.add_argument('--denoiser_num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--denoiser_hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--denoiser_dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--denoiser_norm', type=str, default='ln', help='normalization')
    parser.add_argument('--denoiser_activation', type=str, default='gelu', help='activation')
    parser.add_argument('--denoiser_conv_mlp', type=bool, default=True, help='conv mlp')
    parser.add_argument('--denoiser_ff_mult', type=int, default=4, help='ff mult')
    parser.add_argument('--text_dim', type=int, default=1024, help='text dimension')
    parser.add_argument('--cfg', type=float, default=2.0, help='classifier free guidance scale (1.0 = off)')



    # Others

    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs')
    parser.add_argument('--total-iter', default=5000000, type=int, help='number of total iterations to run')
    parser.add_argument('--eval-iter', default=50000, type=int, help='evaluation frequency')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')

    parser.add_argument('--text', type=str, default='A man is jogging around.')
    parser.add_argument('--mode', type=str, default='rot', choices=['pos', 'rot'], help='recover mode, pos: position, rot: rotation')
    parser.add_argument('--vis-eval', type=bool, default=True, help='visualize evaluation')


    return parser.parse_args()