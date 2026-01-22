from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace
from typing import List, Optional, Sequence, Union

import os
import numpy as np
import torch

try:
    from accelerate.utils import set_seed as _set_seed  # type: ignore
except Exception:
    _set_seed = None  # type: ignore

def set_seed(seed: int) -> None:
    """Seed python/np/torch (and accelerate if available)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _set_seed is not None:
        try:
            _set_seed(seed)
        except Exception:
            pass




def patch_clip_layernorm_for_fp16() -> None:
    """Apply the README-recommended CLIP LayerNorm fp16 fix without editing site-packages.

    If the `clip` package layout differs, this is a no-op.
    """
    try:
        import clip  # type: ignore
        import torch as _torch
        from torch import nn as _nn

        # openai-clip defines LayerNorm in clip/model.py and exposes as clip.model.LayerNorm
        if not hasattr(clip, "model") or not hasattr(clip.model, "LayerNorm"):
            return

        base = clip.model.LayerNorm

        class PatchedLayerNorm(base):  # type: ignore
            """Handle fp16 activations with fp32 weights safely."""

            def forward(self, x: _torch.Tensor):
                if getattr(self, "weight", None) is not None and self.weight.dtype == _torch.float32:
                    orig_type = x.dtype
                    ret = super().forward(x.to(_torch.float32))
                    return ret.to(orig_type)
                return super().forward(x)

        clip.model.LayerNorm = PatchedLayerNorm  # type: ignore
    except Exception:
        return


@dataclass
class GenerationResult:
    """Convenience container."""
    motions_norm: List[torch.Tensor]         # list of (T, D) normalized feature sequences
    joints_xyz: Optional[List[np.ndarray]] = None  # list of (T, J, 3) in meters(?) after recover_from_ric
    texts: Optional[List[str]] = None
    lengths: Optional[List[int]] = None


class StableMoFusionPipeline:
    """Thin wrapper around the original StableMoFusion code, made import/path-safe.

    - Loads `opt.txt` using the repository's parser (get_opt).
    - Builds T2MUnet and loads `latest.tar` (EMA by default).
    - Uses Diffusers schedulers for sampling.
    """

    def __init__(self, opt: Namespace, model: torch.nn.Module, pipeline):
        self.opt = opt
        self.model = model
        self.pipeline = pipeline

    @classmethod
    def from_opt_path(
        cls,
        opt_path: str,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        diffuser_name: str = "dpmsolver",
        num_inference_steps: int = 25,
        use_ema: bool = True,
        patch_clip_fp16: bool = True,
    ) -> "StableMoFusionPipeline":
        if patch_clip_fp16 and dtype == torch.float16:
            patch_clip_layernorm_for_fp16()

        opt = Namespace()
        # populate from opt.txt + derive meta/model dirs
        get_opt(opt, opt_path)

        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device
        opt.device = device_str

        model = build_models(opt)
        ckpt_path = os.path.join(opt.model_dir, "latest.tar")
        load_model_weights(model, ckpt_path, use_ema=use_ema, device=device_str)

        # Choose pipeline: footskate-aware pipeline can still run without footskate_cleanup
        pipe_cls = DiffusePipelineFoot if hasattr(opt, "footskate_cleanup") else DiffusePipelineBase
        pipeline = pipe_cls(
            opt=opt,
            model=model,
            diffuser_name=diffuser_name,
            device=device_str,
            num_inference_steps=num_inference_steps,
            torch_dtype=dtype,
        )
        return cls(opt=opt, model=model, pipeline=pipeline)

    def generate(
        self,
        texts: Sequence[str],
        motion_length_seconds: Optional[float] = None,
        motion_lengths_frames: Optional[Sequence[int]] = None,
        seed: int = 0,
        batch_size: int = 32,
        footskate_cleanup: bool = False,
        return_joints_xyz: bool = True,
        temporal_filter_sigma: float = 1.0,
    ) -> GenerationResult:
        if motion_lengths_frames is None:
            if motion_length_seconds is None:
                raise ValueError("Provide either motion_length_seconds or motion_lengths_frames.")
            motion_lengths_frames = [int(round(motion_length_seconds * float(self.opt.fps))) for _ in texts]

        lengths = [int(x) for x in motion_lengths_frames]
        set_seed(seed)

        m_lens = torch.LongTensor(lengths).to(self.opt.device)
        motions = self.pipeline.generate(
            list(texts),
            m_lens,
            batch_size=batch_size,
            footskate_cleanup=footskate_cleanup,
        )

        joints_xyz = None
        if return_joints_xyz:
            mean = np.load(os.path.join(self.opt.meta_dir, "mean.npy"))
            std = np.load(os.path.join(self.opt.meta_dir, "std.npy"))
            joints_xyz = []
            for mot in motions:
                mot_np = mot.detach().cpu().numpy() * std + mean
                mot_t = torch.from_numpy(mot_np).float()
                xyz = recover_from_ric(mot_t, self.opt.joints_num)  # (T, J, 3)
                # put on floor
                floor_h = xyz.min(dim=0)[0].min(dim=0)[0][1]
                xyz[:, :, 1] -= floor_h
                xyz_np = xyz.numpy()
                if temporal_filter_sigma and temporal_filter_sigma > 0:
                    xyz_np = motion_temporal_filter(xyz_np, sigma=temporal_filter_sigma)
                joints_xyz.append(xyz_np)

        return GenerationResult(
            motions_norm=motions,
            joints_xyz=joints_xyz,
            texts=list(texts),
            lengths=lengths,
        )
