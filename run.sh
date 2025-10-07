CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 128 --exp-name latent_depth12_dim128 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8
CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 64 --exp-name latent_depth12_dim64 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8

CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8
CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 128 --exp-name latent_depth16_dim128 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 16 --n-heads 8




CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 128 --exp-name latent_depth12_dim128_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8
CUDA_VISIBLE_DEVICES=0 python train_latent.py --hidden_size 512 --depth 12 --latent_dim 64 --exp-name latent_depth12_dim64_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8


CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 64 --exp-name latent_depth16_dim64_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8
CUDA_VISIBLE_DEVICES=1 python train_latent.py --hidden_size 512 --depth 16 --latent_dim 128 --exp-name latent_depth16_dim128_win32 --dataname t2m_272 --num_gpus 1 --activation gelu --norm ln --attn-window 32 --n-heads 8

