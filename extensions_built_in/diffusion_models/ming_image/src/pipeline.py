"""Transformer call convention and a preview sampler for Ming-Image.

Timesteps: the toolkit counts 0..1000 with 1000 = pure noise; the Z-Image DiT
takes t in [0, 1] with 1 = clean and predicts `clean - noise`, so the call is
`transformer(x, 1 - t/1000, ...)` and the toolkit's velocity is the negated
output. Negative conditioning is all zeros (both streams), as in the reference
implementation; guidance 1.0 (no CFG) is the recommended setting.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from toolkit.samplers.custom_flowmatch_sampler import calculate_shift

# Z-Image's dynamic shift (0.5 at 256 tokens -> 1.15 at 4096) below 1024^2;
# the reference pipeline pins mu at 1.35 for 1024^2 and above instead of
# extrapolating the line
SHIFT_BASE_SEQ_LEN = 256
SHIFT_MAX_SEQ_LEN = 4096
SHIFT_BASE = 0.5
SHIFT_MAX = 1.15
SHIFT_MAX_LARGE = 1.35


def ming_shift_params(image_seq_len: int):
    """(max_shift, max_image_seq_len) the reference uses for this many image tokens."""
    if image_seq_len >= SHIFT_MAX_SEQ_LEN:
        return SHIFT_MAX_LARGE, image_seq_len
    return SHIFT_MAX, SHIFT_MAX_SEQ_LEN


def ming_shift_mu(image_seq_len: int) -> float:
    max_shift, max_seq_len = ming_shift_params(image_seq_len)
    return calculate_shift(image_seq_len, SHIFT_BASE_SEQ_LEN, max_seq_len, SHIFT_BASE, max_shift)

VAE_SCALE_FACTOR = 8
# 8 for the VAE, 2 for the DiT's 2x2 latent patches
PIXELS_PER_TOKEN = VAE_SCALE_FACTOR * 2


# The direct-VLM condition (`cap_feats_2`) is the only variable-length part of
# the DiT sequence: every caption length otherwise produces a distinct set of
# activation shapes, and the CUDA caching allocator keeps a block per shape it
# has ever seen — reserved memory climbs step after step until a small card
# OOMs (measured: +7.6GB over 24 steps on the 16GB RTX 4060 Ti). Zero-padding
# the direct stream to one shared length pins the sequence length to
# `image tokens + query tokens + this constant`; the transformer masks the
# padded slots out of attention, so outputs are unchanged. 256 matches the
# fixed query length; requests longer than this fall back to the next
# multiple of DIRECT_COND_PAD_MULTIPLE.
DIRECT_COND_PAD_LEN = 256
DIRECT_COND_PAD_MULTIPLE = 32


def _pad_direct_embeds(
    direct_embeds: List[Optional[torch.Tensor]],
) -> Tuple[List[Optional[torch.Tensor]], Optional[List[int]]]:
    """Zero-pad per-item direct conditions to one shared length. Returns the
    padded list plus each item's valid length (None when nothing is padded)."""
    lens = [d.shape[0] if d is not None else 0 for d in direct_embeds]
    if not lens or all(n == 0 for n in lens):
        return direct_embeds, None
    target = DIRECT_COND_PAD_LEN
    max_len = max(lens)
    if max_len > target:
        target = -(-max_len // DIRECT_COND_PAD_MULTIPLE) * DIRECT_COND_PAD_MULTIPLE
    if all(n == target for n in lens):
        return direct_embeds, None
    padded, valid = [], []
    for d, n in zip(direct_embeds, lens):
        valid.append(n)
        if d is None or n == target:
            padded.append(d)
            continue
        pad = torch.zeros((target - n, d.shape[1]), dtype=d.dtype, device=d.device)
        padded.append(torch.cat([d, pad], dim=0))
    return padded, valid


def run_transformer(
    transformer,
    latents: torch.Tensor,  # (B, C, h, w) noisy target latents
    timestep: torch.Tensor,  # (B,) toolkit scale, 0..1000
    query_embeds: List[torch.Tensor],  # per item (256, cap_feat_dim)
    direct_embeds: List[torch.Tensor],  # per item (T, dim)
    ref_latents: Optional[List[Optional[torch.Tensor]]] = None,  # per item (C, h, w) or None
    direct_lens: Optional[List[int]] = None,  # pass when direct_embeds are pre-padded
    **kwargs,
) -> torch.Tensor:
    """Returns the flow-matching velocity `noise - clean` as `(B, C, h, w)`."""
    batch_size = latents.shape[0]
    dtype = latents.dtype
    x = list(latents.unsqueeze(2).unbind(dim=0))
    ref_x = None
    if ref_latents is not None and any(r is not None for r in ref_latents):
        ref_x = [None if r is None else r.unsqueeze(1).to(dtype) for r in ref_latents]
    if direct_lens is None:
        direct_embeds, direct_lens = _pad_direct_embeds(direct_embeds)
    tau = 1.0 - timestep.to(latents.device, dtype=torch.float32) / 1000.0
    if tau.dim() == 0:
        tau = tau.unsqueeze(0)
    if tau.shape[0] != batch_size:
        tau = tau.expand(batch_size)
    out = transformer(
        x,
        tau.to(dtype),
        [q.to(latents.device, dtype) for q in query_embeds],
        cap_feats_2=[d.to(latents.device, dtype) for d in direct_embeds],
        ref_x=ref_x,
        direct_lens=direct_lens,
        return_dict=False,
        **kwargs,
    )[0]
    # (C, 1, h, w) per item -> (B, C, h, w); the model predicts clean - noise
    return -torch.stack([o[:, 0] for o in out], dim=0)


class MingImagePipeline:
    """Minimal flow-matching sampler for ai-toolkit's preview generation."""

    def __init__(self, model):
        # `model` is the MingImageModel, so its encode/decode and config are reused
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds=None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 12,
        guidance_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        reference_latents: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device, dtype = model.device_torch, model.torch_dtype
        transformer = model.transformer

        latent_height = height // VAE_SCALE_FACTOR
        latent_width = width // VAE_SCALE_FACTOR
        channels = transformer.config.in_channels

        if latents is None:
            # randn_tensor, not torch.randn: the caller's generator may be a cpu one
            latents = randn_tensor(
                (1, channels, latent_height, latent_width),
                generator=generator,
                device=torch.device(device),
                dtype=torch.float32,
            )
        latents = latents.to(device, dtype=dtype)

        scheduler = model.get_train_scheduler()
        # the reference pipeline spaces `steps` sigmas from 1 down to 0 and
        # evaluates the model at sigma 0 too (a no-op step); skip that one
        if num_inference_steps > 1:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps)[:-1]
        else:
            sigmas = np.array([1.0])
        mu = ming_shift_mu((height // PIXELS_PER_TOKEN) * (width // PIXELS_PER_TOKEN))
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        scheduler.set_begin_index(0)

        query = list(conditional_embeds.text_embeds)
        direct = list(conditional_embeds.direct_embeds)
        # the reference negative is zero conditioning, whatever the negative prompt says
        do_cfg = guidance_scale > 1.0
        query_neg = [torch.zeros_like(q) for q in query]
        direct_neg = [torch.zeros_like(d) for d in direct]

        for timestep in scheduler.timesteps:
            t = timestep.expand(latents.shape[0]).to(device)
            noise_pred = run_transformer(
                transformer, latents, t, query, direct, ref_latents=reference_latents
            )
            if do_cfg:
                uncond_pred = run_transformer(
                    transformer, latents, t, query_neg, direct_neg, ref_latents=reference_latents
                )
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)

            latents = scheduler.step(
                noise_pred.to(torch.float32),
                timestep,
                latents.to(torch.float32),
                return_dict=False,
            )[0].to(dtype)

        return model.decode_to_images(latents)
