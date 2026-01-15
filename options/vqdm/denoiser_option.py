import argparse

def get_denoiser_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader (identical to vqvae)
    parser.add_argument('--dataset', type=str, default='humanml', help='dataset directory')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--eval_split", default='test', type=str,
                       help="Which split to evaluate on during training.")
    parser.add_argument("--train_split", default='train', type=str,
                       help="Which split to train.")

    ## optimization
    parser.add_argument('--total_iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr_scheduler', default=[250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=4.5e-2, type=float, help='weight decay')
    parser.add_argument('--num_samples', default=1000, type=int, help='num samples for evaluation.')
    
    ## VQVAE placeholders (not used in denoiser training)
    parser.add_argument("--lambda_commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--lambda_param_vel', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--lambda_joint', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--lambda_hml_joint', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--lambda_joint_vel', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--lambda_fc', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--lambda_gumbel_kl', type=float, default=5e-4, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    ## vqvae arch (identical to vqvae)
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')
    
    ## quantizer (identical to vqvae)
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset', 'gumbel'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## denoiser arch
    parser.add_argument('--lambda_auxiliary', type=float, default=1e-4, help='hyper-parameter for the velocity loss')
    parser.add_argument("--transformer_type", type=str, default='text2image', choices = ['condition2image', 'text2image'])
    parser.add_argument("--diffusion_step", type=int, default=100)
    parser.add_argument("--skip_step", type=int, default=1)
    parser.add_argument("--cond_mode", type=str, default='text', choices = ['action', 'text'])
    parser.add_argument("--mlp_type", type=str, default='conv_mlp', choices = ['conv_mlp', 'fc'])
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--action_emb_type", type=str, default='single', choices = ['single', 'layerwise'])
    parser.add_argument("--timestep_type", type=str, default='adalayernorm', choices = ['adalayernorm', 'adainsnorm'])

    ## resume  
    parser.add_argument("--resume_pth", type=str, default=None, help='resume pth for denoiser')
    parser.add_argument("--pretrained_vqvae_pth", type=str, required=True, help='resume pth for denoiser')
    
    ## output directory 
    parser.add_argument('--out_dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results_dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual_name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp_name', type=str, default='dev', help='name of the experiment, will create a file inside out-dir')



    ## other
    parser.add_argument('--print_iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis_gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb_vis', default=20, type=int, help='nb of visualizations')

    args = parser.parse_args()

    if args.dataset in ['humanact12', 'uestc']:
        args.window_size = 60
    
    elif args.dataset in ['ntu']:
        args.window_size = 200
    
    elif args.dataset in ['humanml', 'kit']:
        args.window_size = 300

    args.lr_scheduler = [int(args.total_iter * 0.9)]

    return args