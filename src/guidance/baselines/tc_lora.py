"""TC-LoRA baseline.

PAPER
    Minkyoung Cho, Ruben Ohana, Christian Jacobsen, Adityan Jothi, Min-Hung Chen,
    Z. Morley Mao, Ethem F. Can.
    "TC-LoRA: Temporally Modulated Conditional LoRA for Adaptive Diffusion Control."
    NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE).
    arXiv:2510.09561.  Project page: https://minkyoungcho.github.io/tc-lora/

CODE PROVENANCE
    The authors have not released code. Checked 2026-09-28: the arXiv listing has no
    code link and the project page links only Paper / arXiv / BibTeX. No third-party
    reimplementation was found either.

    This file is therefore OUR OWN reimplementation, written from the paper's
    Equation 1, Appendix A (hypernetwork architecture) and Appendix B (training
    objective). No code was copied from anywhere. Report it as a reimplementation,
    not as the authors' method reproduced.

WHAT WE FOLLOW FROM THE PAPER
    - Eq. 1:  W' = W + B(i, t, y) A(i, t, y), with A and B produced on the fly and
      never trained directly.
    - App. A: a SINGLE hypernetwork shared across all adapted layers, conditioned on
      four things -- diffusion timestep, input condition, target layer identity, and
      layer type.
    - App. A: sinusoidal timestep embedding, 64-dim.
    - App. A: layer-ID encoder with residual connections producing a 128-dim
      embedding from (layer depth, layer type).
    - App. A: trunk of residual blocks with skip connections from early and
      intermediate stages added to the output projection.
    - App. A: the final layer generating B is zero-initialised, so the adapted model
      is identical to the frozen base model at the start of training.
    - App. B: trained end-to-end with the ordinary diffusion loss; the hypernetwork
      is the only trainable component and the backbone stays frozen.

WHAT WE CHANGED, AND WHY
    1. Backbone. They adapt the linear projections inside the self- and cross-
       attention blocks of a DiT (Cosmos-Predict1). We have a convolutional U-Net
       for images and an MLP for cell states, so we adapt nn.Conv2d and nn.Linear.
       Conv adapters are applied as two 1x1 convolutions with per-sample weights via
       grouped convolution, which assumes the base conv preserves spatial size.
    2. Condition encoder. Their condition is a depth map, encoded by the base
       model's pretrained autoencoder and then a 3-layer MLP to 1024 dims. Our
       condition is already a vector (a frozen text embedding), so we keep the
       3-layer MLP and drop the autoencoder. Default width is 256, not 1024,
       because our backbones are roughly three orders of magnitude smaller.
    3. Layer type. Their type vector distinguishes self- vs cross-attention and
       query vs key vs value. We have no attention, so type is Linear vs Conv2d.
    4. Shared output head across differently shaped layers. Their adapted layers are
       attention projections of near-uniform width, so one output head fits all. Our
       U-Net channel widths vary (64 to 256), so the head emits factors at the
       maximum width and each layer slices the prefix it needs. The layer-ID
       embedding is what distinguishes layers.
    5. Scale. They train 3 days on 8x H100 with 251M trainable hypernetwork
       parameters. Ours is a few hundred thousand. Any comparison must say so.

Run this file directly to execute the self-checks at the bottom.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_TYPES = ("linear", "conv2d")


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features. (B,) -> (B, dim). App. A uses dim = 64."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    """Pre-norm residual MLP block. App. A's 'multi-scale residual blocks'."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class LayerIDEncoder(nn.Module):
    """(layer depth, layer type) -> dense embedding. App. A, 128-dim with residuals."""

    def __init__(self, n_layers, out_dim=128):
        super().__init__()
        self.depth = nn.Embedding(n_layers, out_dim)
        self.kind = nn.Embedding(len(LAYER_TYPES), out_dim)
        self.res = ResBlock(out_dim)

    def forward(self, idx, kind):
        return self.res(self.depth(idx) + self.kind(kind))


class HyperNet(nn.Module):
    """One hypernetwork, shared across every adapted layer (App. A).

    Context vector = [ time (64) ; condition (cond_hidden) ; layer id+type (128) ].

    Emits A and B at the maximum adapted width; each layer slices what it needs
    (deviation 4 in the module docstring).

    The B head is zero-initialised and the A head is not. That asymmetry is what
    makes the adapter an exact no-op at initialisation while keeping it trainable:
    the gradient with respect to the B head is A, which is nonzero. Zero-
    initialising both would strand the adapter at zero forever.
    """

    def __init__(self, n_layers, d_in_max, d_out_max, cond_dim, rank=4,
                 time_dim=64, id_dim=128, width=256, n_blocks=3):
        super().__init__()
        self.rank, self.d_in_max, self.d_out_max = rank, d_in_max, d_out_max
        self.time_dim = time_dim

        # App. A: condition goes through a 3-layer MLP (their autoencoder step is
        # dropped -- see deviation 2).
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width),
        )
        self.layer_enc = LayerIDEncoder(n_layers, id_dim)
        self.stem = nn.Linear(time_dim + width + id_dim, width)
        self.blocks = nn.ModuleList(ResBlock(width) for _ in range(n_blocks))

        # App. A: multi-range skips from early and intermediate stages into the
        # output projection.
        self.skips = nn.ModuleList(nn.Linear(width, width) for _ in range(n_blocks))

        self.head_a = nn.Linear(width, rank * d_in_max)
        self.head_b = nn.Linear(width, d_out_max * rank)
        nn.init.zeros_(self.head_b.weight)
        nn.init.zeros_(self.head_b.bias)

    def forward(self, t, y, idx, kind):
        """t (B,), y (B,cond_dim), idx (L,), kind (L,) -> A (B,L,r,d_in_max),
        B (B,L,d_out_max,r)."""
        b, n_layers = t.shape[0], idx.shape[0]

        ctx = torch.cat([timestep_embedding(t, self.time_dim), self.cond_mlp(y)], dim=-1)
        ctx = ctx[:, None, :].expand(b, n_layers, -1)
        lay = self.layer_enc(idx, kind)[None].expand(b, n_layers, -1)

        h = self.stem(torch.cat([ctx, lay], dim=-1))
        acc = 0.0
        for block, skip in zip(self.blocks, self.skips):
            h = block(h)
            acc = acc + skip(h)
        h = h + acc

        a = self.head_a(h).view(b, n_layers, self.rank, self.d_in_max)
        bb = self.head_b(h).view(b, n_layers, self.d_out_max, self.rank)
        return a, bb


class _Adapted(nn.Module):
    """Base for a wrapped layer. `ab` is filled per forward pass, then cleared."""

    def __init__(self, base, scale):
        super().__init__()
        self.base = base
        self.scale = scale
        self.ab = None
        for p in self.base.parameters():
            p.requires_grad_(False)


class AdaptedLinear(_Adapted):
    kind = "linear"

    def __init__(self, base: nn.Linear, scale=1.0):
        super().__init__(base, scale)
        self.shape = (base.out_features, base.in_features)

    def forward(self, x):
        out = self.base(x)
        if self.ab is None:
            return out
        a, b = self.ab                                   # (B,r,d_in), (B,d_out,r)
        h = torch.einsum("b...i,bri->b...r", x, a)
        return out + self.scale * torch.einsum("b...r,bor->b...o", h, b)


class AdaptedConv2d(_Adapted):
    """LoRA as two 1x1 convolutions with per-sample weights (deviation 1).

    Grouped convolution gives each batch element its own kernel. Assumes the base
    convolution preserves spatial size, which holds for the padded 3x3 blocks in our
    U-Net; downsampling happens in pooling layers, which are not adapted.
    """

    kind = "conv2d"

    def __init__(self, base: nn.Conv2d, scale=1.0):
        super().__init__(base, scale)
        if base.stride != (1, 1):
            raise ValueError("AdaptedConv2d assumes stride 1; see deviation 1")
        self.shape = (base.out_channels, base.in_channels)

    def forward(self, x):
        out = self.base(x)
        if self.ab is None:
            return out
        a, b = self.ab
        n, c_in, h, w = x.shape
        r, c_out = a.shape[1], b.shape[1]
        down = F.conv2d(x.reshape(1, n * c_in, h, w), a.reshape(n * r, c_in, 1, 1), groups=n)
        up = F.conv2d(down, b.reshape(n * c_out, r, 1, 1), groups=n)
        return out + self.scale * up.reshape(n, c_out, h, w)


def _wrap(model: nn.Module, scale: float, skip_names=()):
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip_names):
                continue
            if isinstance(child, nn.Linear):
                new = AdaptedLinear(child, scale)
            elif isinstance(child, nn.Conv2d) and child.stride == (1, 1):
                new = AdaptedConv2d(child, scale)
            else:
                continue
            setattr(module, child_name, new)
            wrapped.append(new)
    return wrapped


class TCLoRA(nn.Module):
    """Frozen backbone plus the shared hypernetwork that adapts its weights.

        model = TCLoRA(unet, cond_dim=512)
        eps   = model(x_t, t, y=text_embedding, cond_ids=labels)

    Arguments after `y` are forwarded to the backbone unchanged.
    """

    def __init__(self, backbone: nn.Module, cond_dim: int, rank=4, scale=1.0,
                 time_dim=64, id_dim=128, width=256, n_blocks=3, skip_names=()):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.adapted = _wrap(backbone, scale, skip_names)
        if not self.adapted:
            raise ValueError("no adaptable Linear or stride-1 Conv2d layers found")

        d_out_max = max(m.shape[0] for m in self.adapted)
        d_in_max = max(m.shape[1] for m in self.adapted)
        self.hyper = HyperNet(len(self.adapted), d_in_max, d_out_max, cond_dim,
                              rank, time_dim, id_dim, width, n_blocks)

        self.register_buffer("_idx", torch.arange(len(self.adapted)), persistent=False)
        self.register_buffer(
            "_kind", torch.tensor([LAYER_TYPES.index(m.kind) for m in self.adapted]),
            persistent=False,
        )

    def forward(self, x, t, y, *args, **kwargs):
        a, b = self.hyper(t, y, self._idx, self._kind)
        for i, layer in enumerate(self.adapted):
            d_out, d_in = layer.shape
            layer.ab = (a[:, i, :, :d_in], b[:, i, :d_out, :])   # slice, deviation 4
        try:
            return self.backbone(x, t, *args, **kwargs)
        finally:
            for layer in self.adapted:
                layer.ab = None

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    class ToyMLP(nn.Module):
        def __init__(self, d=50):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d + 1, 128), nn.SiLU(), nn.Linear(128, d))

        def forward(self, x, t):
            return self.net(torch.cat([x, t[:, None]], dim=-1))

    class ToyUNet(nn.Module):
        def __init__(self, c=16):
            super().__init__()
            self.a = nn.Conv2d(3, c, 3, padding=1)
            self.b = nn.Conv2d(c, 3, 3, padding=1)

        def forward(self, x, t):
            return self.b(F.silu(self.a(x)))

    for name, base, x in [
        ("mlp ", ToyMLP(), torch.randn(4, 50)),
        ("unet", ToyUNet(), torch.randn(4, 3, 32, 32)),
    ]:
        t, y = torch.rand(4), torch.randn(4, 16)
        ref = base(x, t).detach().clone()                      # before wrapping

        model = TCLoRA(base, cond_dim=16, rank=4)

        out0 = model(x, t, y)
        assert torch.allclose(out0, ref, atol=1e-6), (out0 - ref).abs().max().item()
        print(f"{name}: no-op at init, matches the frozen backbone")

        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert all(n.startswith("hyper.") for n in trainable), sorted(trainable)[:3]
        print(f"{name}: {model.n_trainable():,} trainable params, all in the hypernetwork")

        model(x, t, y).sum().backward()
        g = model.hyper.head_b.weight.grad
        assert g is not None and g.abs().sum() > 0, "B head received no gradient"
        print(f"{name}: B head has nonzero gradient")

        with torch.no_grad():
            model.hyper.head_b.weight.normal_(0, 0.02)
        assert not torch.allclose(model(x, t, y), model(x, t, torch.randn(4, 16)))
        assert not torch.allclose(model(x, t, y), model(x, torch.rand(4), y))
        print(f"{name}: output varies with both the condition and the timestep\n")
