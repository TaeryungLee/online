



CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 64 --exp-name latent_depth12_dim64_win16_kl --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl
CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64_win16 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl_loss 1e-4

CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 64 --exp-name latent_depth12_dim64_win16_kl --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl
CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64_win16_kl --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl



# CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 128 --exp-name latent_depth12_dim128_win16 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8
# CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 128 --exp-name latent_depth16_dim128_win16 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8




# CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 128 --exp-name latent_depth12_dim128_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8
# CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 64 --exp-name latent_depth12_dim64_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8


# CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8
# CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 128 --exp-name latent_depth16_dim128_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8






CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64_win16 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl_loss 1e-4
CUDA_VISIBLE_DEVICES=1 python train_causal_TAE.py --exp-name causal_TAE --kl_loss 1e-4
CUDA_VISIBLE_DEVICES=2 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name conv1d_encoder --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl_loss 1e-4 --encoder conv1d
CUDA_VISIBLE_DEVICES=3 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name transformer_encoder --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8 --kl_loss 1e-4 --encoder transformer


# 실험 6개
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win32_dim64_nl16 --hidden_size 512 --depth 16 --latent_dim 64 --attn-window 32 --n-heads 8 --kl_loss 1e-4 
CUDA_VISIBLE_DEVICES=2 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win8_dim64_nl16 --hidden_size 512 --depth 16 --latent_dim 64 --attn-window 8 --n-heads 8 --kl_loss 1e-4 

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim64_nl16 --hidden_size 512 --depth 16 --latent_dim 64 --attn-window 4 --n-heads 8 --kl_loss 1e-4 
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win16_dim64_nl8 --hidden_size 512 --depth 8 --latent_dim 64 --attn-window 16 --n-heads 8 --kl_loss 1e-4 

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win16_dim64_nl12 --hidden_size 512 --depth 12 --latent_dim 64 --attn-window 16 --n-heads 8 --kl_loss 1e-4 
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win16_dim64_nl20 --hidden_size 512 --depth 20 --latent_dim 64 --attn-window 16 --n-heads 8 --kl_loss 1e-4 


CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim64_nl8 --hidden_size 512 --depth 8 --latent_dim 64 --attn-window 4 --n-heads 8 --kl_loss 1e-4 
CUDA_VISIBLE_DEVICES=3 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 


# anchor
python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl8 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win2_dim128_nl8 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 2 --n-heads 8 --kl_loss 1e-4
CUDA_VISIBLE_DEVICES=2 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl6 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_kl5e-4 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 5e-4
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_kl5e-5 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 5e-5

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_kl1e-5 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-5
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_vel0.05_acc0.01 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.05 --acc_loss 0.01


CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_vel0.1_acc0.05 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.1 --acc_loss 0.05
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8_vel0.5_acc0.1 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl6_vel0.5_acc0.5_dec_conv --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp

python get_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --decoder-conv-mlp --resume-pth checkpoints/latent/humanml3d/net_best_mpjpe.pth


CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv_single --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1


# CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv_single --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
# CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win4_dim128_nl6_vel0.5_acc0.1_dec_conv --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win4_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1



CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer.py --dataname t2m_272 --exp-name denoiser01_ln8_lr1e-4_smax2.0 --denoiser_block 1 --denoiser_num_layers 8 --lr 1e-4 --sigma_max 2.0


CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser01_ln16 --denoiser_block 1 --denoiser_num_layers 16 --lr 2e-4
CUDA_VISIBLE_DEVICES=1 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser01_ln18 --denoiser_block 1 --denoiser_num_layers 18 --lr 2e-4
CUDA_VISIBLE_DEVICES=2 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser01_ln20 --denoiser_block 1 --denoiser_num_layers 20 --lr 2e-4
CUDA_VISIBLE_DEVICES=3 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser01_ln22 --denoiser_block 1 --denoiser_num_layers 22 --lr 2e-4

CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser01_ln24 --denoiser_block 1 --denoiser_num_layers 24 --lr 2e-4

torchrun --nproc_per_node=4 train_t2m_initializer_ddp.py --num_workers 50 --batch_size 256 --exp-name denoiser01_ln16_ddp --denoiser_block 2 --denoiser_num_layers 16 --lr 2e-4
torchrun --nproc_per_node=1 train_t2m_initializer_ddp.py --num_workers 50 --batch_size 256 --exp-name denoiser01_ln16_ddp --denoiser_block 2 --denoiser_num_layers 16 --lr 2e-4

CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser02_ln12 --denoiser_block 2 --denoiser_num_layers 12 --lr 2e-4
CUDA_VISIBLE_DEVICES=1 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser03_ln12 --denoiser_block 3 --denoiser_num_layers 12 --lr 2e-4
CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser04_ln12 --denoiser_block 4 --denoiser_num_layers 12 --lr 2e-4
CUDA_VISIBLE_DEVICES=1 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser05_ln12 --denoiser_block 5 --denoiser_num_layers 12 --lr 2e-4

CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer_no_latent.py --num_workers 50 --dataname t2m_272 --exp-name denoiser05_ln12_no_latent --denoiser_block 5 --denoiser_num_layers 12 --lr 2e-4





CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4
CUDA_VISIBLE_DEVICES=1 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl8 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 8
CUDA_VISIBLE_DEVICES=2 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl12 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 12
CUDA_VISIBLE_DEVICES=3 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl16 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 16


CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl8_step15 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 8 --num_timesteps 15
CUDA_VISIBLE_DEVICES=1 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl8_step20 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 8 --num_timesteps 20


CUDA_VISIBLE_DEVICES=0 python train_t2m_initializer_no_latent.py --num_workers 50 --dataname t2m_272 --exp-name denoiser05_ln12_text_pe_no_latent --denoiser_block 5 --denoiser_num_layers 12 --lr 2e-4 --eval-iter 10 --batch-size 32
CUDA_VISIBLE_DEVICES=1 python train_t2m_initializer.py --num_workers 50 --dataname t2m_272 --exp-name denoiser05_ln12_text_pe --denoiser_block 5 --denoiser_num_layers 12 --lr 2e-4 --eval-iter 10 --batch-size 32


CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl1_t1 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 1 --num_timesteps 1
CUDA_VISIBLE_DEVICES=1 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl2_t2 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 2 --num_timesteps 2
CUDA_VISIBLE_DEVICES=2 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t4 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 4
CUDA_VISIBLE_DEVICES=3 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t4 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 4

CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t1 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 1
CUDA_VISIBLE_DEVICES=1 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl8_t8 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 8 --num_timesteps 8


CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl1_t1 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 1 --num_timesteps 1 --resume best_fid.pth --eval-iter 1
CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl2_t2 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 2 --num_timesteps 2 --resume best_fid.pth --eval-iter 1
CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t4 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 4 --resume best_fid.pth --eval-iter 1
CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t4 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 4 --resume best_fid.pth --eval-iter 1

CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl4_t1 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 4 --num_timesteps 1 --resume best_fid.pth --eval-iter 1
CUDA_VISIBLE_DEVICES=0 python train_t2m_diffusion.py --num_workers 50 --dataname t2m_272 --exp-name denoiser06_roll_ln12_ovl8_t8 --denoiser_block 6 --denoiser_num_layers 12 --lr 2e-4 --overlap-size 8 --num_timesteps 8 --resume best_fid.pth --eval-iter 1



python train_motionprimitive.py --dataname t2m_272 --exp-name motionprimitive_dim512_nl9_dim512 --hidden_size 512 --depth 9 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --window-size 10


CUDA_VISIBLE_DEVICES=0 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim512_nl8_dim512_f8_h2 --hidden_size 512 --depth 8 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 8 --history 2
CUDA_VISIBLE_DEVICES=1 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim512_nl12_dim512_f8_h2 --hidden_size 512 --depth 12 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 8 --history 2
CUDA_VISIBLE_DEVICES=2 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim512_nl16_dim512_f8_h2 --hidden_size 512 --depth 16 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 8 --history 2
CUDA_VISIBLE_DEVICES=3 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim512_nl12_dim512_f4_h2 --hidden_size 512 --depth 12 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 4 --history 2

CUDA_VISIBLE_DEVICES=0 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim512_nl12_dim512_f4_h4 --hidden_size 512 --depth 12 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 4 --history 4
CUDA_VISIBLE_DEVICES=1 python train_cvae.py --dataname t2m_272 --exp-name cvae_hdim1024_nl12_dim512_f8_h2 --hidden_size 1024 --depth 12 --latent_dim 512 --n-heads 8 --kl_loss 1e-4 --future 8 --history 2


