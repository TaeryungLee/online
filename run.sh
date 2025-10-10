



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

python get_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win4_dim128_nl8 --hidden_size 512 --depth 8 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 


CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv_single --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_272 --encoder transformer --exp-name latent_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1


# CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv_single --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
# CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1_dec_conv --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win3_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 3 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1

CUDA_VISIBLE_DEVICES=0 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win4_dim128_nl6_vel0.5_acc0.1_dec_conv --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1 --decoder-conv-mlp
CUDA_VISIBLE_DEVICES=1 python train_latent.py --dataname t2m_babel_272 --encoder transformer --exp-name latent_babel_trans_enc_win4_dim128_nl6_vel0.5_acc0.1 --hidden_size 512 --depth 6 --latent_dim 128 --attn-window 4 --n-heads 8 --kl_loss 1e-4 --vel_loss 0.5 --acc_loss 0.1




