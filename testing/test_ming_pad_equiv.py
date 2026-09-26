"""Numerical equivalence test: direct-condition zero-padding + masking must not
change any output value of the Ming DiT (small random model, CPU, fp32)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from extensions_built_in.diffusion_models.ming_image.src.transformer import (
    MingImageTransformer2DModel as MingTransformer2DModel,
)
from extensions_built_in.diffusion_models.ming_image.src.pipeline import (
    run_transformer,
    _pad_direct_embeds,
)

torch.manual_seed(0)

# tiny config: head_dim 16 = axes_dims sum, dim 128, 2+1 layers
model = MingTransformer2DModel(
    in_channels=4,
    dim=128,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=8,
    n_kv_heads=8,
    cap_feat_dim=32,
    axes_dims=(6, 6, 4),
    axes_lens=(2048, 512, 512),
    patch_size=2,
)
model.eval()

B = 1
C, F_, H, W = 4, 1, 16, 16  # latent 16x16 -> 8x8=64 image tokens
latents = torch.randn(B, C, H, W)
t = torch.tensor([500.0])

cap_feat_dim, dim = 32, 128
query = [torch.randn(256, cap_feat_dim)]  # fixed 256 query tokens, as cached

def call(directs, extra=None):
    with torch.no_grad():
        return run_transformer(
            model, latents, t, [q for q in query], [d.clone() for d in directs], **(extra or {})
        )

# 1) baseline: raw variable-length directs (old behaviour path -> direct_lens=None inside)
d1 = [torch.randn(26, dim)]
d2 = [torch.randn(195, dim)]
out_raw_short = call(d1)
out_raw_long = call(d2)

# 2) padded: pad the SAME tensors to 256 by hand and pass direct_lens, bypassing _pad_direct_embeds
def pad_manual(d, target=256):
    pad = torch.zeros((target - d.shape[0], d.shape[1]))
    return torch.cat([d, pad], dim=0)

with torch.no_grad():
    out_pad_short = run_transformer(
        model, latents, t, [q for q in query], [pad_manual(d1[0])], direct_lens=[26]
    )
    out_pad_long = run_transformer(
        model, latents, t, [q for q in query], [pad_manual(d2[0])], direct_lens=[195]
    )

def maxdiff(a, b):
    return (a - b).abs().max().item()

print("raw short  vs padded short :", maxdiff(out_raw_short, out_pad_short))
print("raw long   vs padded long  :", maxdiff(out_raw_long, out_pad_long))

# 3) _pad_direct_embeds sanity
p, v = _pad_direct_embeds([d1[0], d2[0]])
assert p[0].shape == p[1].shape == (256, dim), (p[0].shape, p[1].shape)
assert v == [26, 195], v
p2, v2 = _pad_direct_embeds([torch.randn(300, dim)])
assert p2[0].shape[0] == 320 and v2 == [300], (p2[0].shape, v2)  # falls back to next multiple of 32
p3, v3 = _pad_direct_embeds([torch.randn(256, dim)])
assert v3 is None and p3[0].shape[0] == 256  # already exact -> no padding, no mask
print("pad helper: OK")

# 4) batch of two with different lengths (cross-item + internal padding)
lat2 = torch.randn(2, C, H, W)
d_a = torch.randn(40, dim)
d_b = torch.randn(150, dim)
with torch.no_grad():
    out_batch_pad = run_transformer(
        model, lat2, torch.tensor([500.0, 600.0]),
        [query[0], query[0]], [pad_manual(d_a), pad_manual(d_b)], direct_lens=[40, 150],
    )    # reference: run each item alone, unpadded
    out_a = run_transformer(model, lat2[:1], torch.tensor([500.0]), [query[0]], [d_a])
    out_b = run_transformer(model, lat2[1:2], torch.tensor([600.0]), [query[0]], [d_b])
print("batch item0 vs single:", maxdiff(out_batch_pad[0:1], out_a))
print("batch item1 vs single:", maxdiff(out_batch_pad[1:2], out_b))

tol = 1e-4
ok = all(x < tol for x in [
    maxdiff(out_raw_short, out_pad_short),
    maxdiff(out_raw_long, out_pad_long),
    maxdiff(out_batch_pad[0:1], out_a),
    maxdiff(out_batch_pad[1:2], out_b),
])
print("EQUIVALENT" if ok else "MISMATCH", f"(tol {tol})")
sys.exit(0 if ok else 1)
