



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

