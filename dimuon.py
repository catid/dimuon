"""PyTorch implementation of the recovered Dimuon update.

Each parameter is one LoRA factor: [m, n] or [experts, m, n]. Store fused
projections as one parameter. Defaults: FP16 Polar Express, BF16 momentum.
"""
from __future__ import annotations

import math

import torch

_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
PE_COEFFICIENTS = tuple((a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _RAW)
_DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}


def _polar_express(z: torch.Tensor, precision: str) -> torch.Tensor:
    """z: [E, m, n] in fp32/fp64. Returns the 5-step Polar Express output in z's dtype."""
    transpose = z.shape[-2] > z.shape[-1]
    x = z.mT if transpose else z
    norm = torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
    work = _DTYPES[precision]
    guard = 1e-7 if precision == "fp16" else 0.0
    x = (x / (norm * 1.01 + guard)).to(work)
    for a, b, c in PE_COEFFICIENTS:
        gram = x @ x.mT
        middle = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, middle, x, beta=a, alpha=1.0)
    x = x.to(z.dtype)
    return x.mT if transpose else x


def _conditioner(m_old: torch.Tensor, dim: int, damping: float) -> torch.Tensor:
    energy = (m_old * m_old).sum(dim=dim)                      # [E, k]
    peak = energy.amax(dim=-1, keepdim=True)
    cond = (energy / peak.clamp_min(torch.finfo(energy.dtype).tiny) + damping) ** -0.25
    return torch.where(peak > 0, cond, torch.ones_like(cond))  # identity when the old momentum is zero


class Dimuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, grad_clip_norm: float = 0.0,
                 precision: str = "fp16", damping: float = 1e-4, state_dtype: torch.dtype = torch.float32,
                 momentum_dtype: torch.dtype | None = torch.bfloat16):
        """state_dtype: arithmetic dtype for momentum, conditioners and normalisation.
        momentum_dtype: dtype the momentum is KEPT in between steps (bf16 matches the server; None = state_dtype).
        The new momentum is computed in state_dtype and used unrounded for the current step; only the stored
        copy is rounded."""
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if precision not in _DTYPES:
            raise ValueError("precision must be fp16, fp32 or fp64")
        super().__init__(params, {"lr": lr, "beta1": beta1})
        self.grad_clip_norm, self.precision, self.damping, self.state_dtype = grad_clip_norm, precision, damping, state_dtype
        self.momentum_dtype = momentum_dtype or state_dtype
        self.grouped = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for g in self.param_groups for p in g["params"]]
        grads = [p.grad for p in params if p.grad is not None]
        if not grads and not self.state:
            return loss
        total = torch.sqrt(sum((g.to(self.state_dtype) ** 2).sum() for g in grads)) if grads else None
        scale = 1.0
        if self.grad_clip_norm > 0 and total is not None and float(total) > 0:
            scale = min(1.0, self.grad_clip_norm / float(total))
        for group in self.param_groups:
            lr, beta1 = group["lr"], group["beta1"]
            # Group equal-shaped factors into one batched Polar Express call.  A
            # LoRA adapter often has dozens of identical factors; batching keeps
            # the five small GEMMs from being launched once per factor.
            groups = {}
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None and "momentum" not in state:
                    continue
                key = (tuple(p.shape), p.device, p.dtype)
                groups.setdefault(key, []).append((p, state))
            for _, entries in groups.items():
                ps, states = zip(*entries)
                gs = torch.stack([torch.zeros_like(p, dtype=self.state_dtype) if p.grad is None else p.grad.to(self.state_dtype) for p in ps])
                olds = torch.stack([torch.zeros_like(gs[i]) if "momentum" not in st else st["momentum"].to(self.state_dtype) for i, st in enumerate(states)])
                # Stack adds a leading factor dimension. For ordinary LoRA
                # factors this is already the [F,m,n] batch expected by the
                # Polar Express batched GEMMs. For an expert/fused [e,m,n]
                # parameter, flatten F and e temporarily because baddbmm
                # accepts a three-dimensional batch only.
                if ps[0].dim() == 2:
                    g3, old3 = gs, olds
                    restore = None
                elif ps[0].dim() == 3:
                    f, e, m, n = gs.shape
                    g3, old3 = gs.reshape(f * e, m, n), olds.reshape(f * e, m, n)
                    restore = (f, e, m, n)
                else:
                    raise ValueError("Dimuon expects 2-D or 3-D factors")
                left = _conditioner(old3, -1, self.damping)
                right = _conditioner(old3, -2, self.damping)
                m_new = beta1 * old3 + (1.0 - beta1) * scale * g3
                stored = m_new.to(torch.float32).to(self.momentum_dtype) if self.momentum_dtype == torch.bfloat16 and m_new.dtype == torch.float64 else m_new.to(self.momentum_dtype)
                stored_for_state = stored.reshape(restore) if restore is not None else stored
                for i, st in enumerate(states):
                    st["momentum"] = stored_for_state[i].clone()
                inner = left[..., :, None] * m_new * right[..., None, :]
                live = torch.linalg.vector_norm(inner, dim=(-2, -1)) > 0
                if not bool(live.any()):
                    continue
                # Polar Express is written for a batch of matrices [B,m,n].
                # Flatten any expert axis (and the factor batch) for the small
                # GEMMs, then restore the grouped shape.  Conditioning remains
                # per factor and per expert because it was computed before this
                # flattening.
                flat_inner = inner.reshape(-1, inner.shape[-2], inner.shape[-1])
                polar = _polar_express(flat_inner, self.precision).reshape_as(inner)
                d = left[..., :, None] * polar * right[..., None, :]
                rank = min(d.shape[-2], d.shape[-1])
                target = lr * math.sqrt(rank) * (rank / 32) ** 0.175
                dn = torch.linalg.vector_norm(d, dim=(-2, -1), keepdim=True)
                live_mask = live[(...,) + (None,) * (d.ndim - live.ndim)]
                delta = torch.where(live_mask, -target * d / dn.clamp_min(torch.finfo(d.dtype).tiny), torch.zeros_like(d))
                if restore is not None:
                    delta = delta.reshape(restore)
                for i, p in enumerate(ps):
                    p.add_(delta[i].reshape(p.shape).to(p.dtype))
        return loss


class GroupedDimuon(Dimuon):
    """Dimuon with the same update batched across equal-shaped LoRA factors.

    The recovered rule is factor-wise, but every FFN LoRA A (and B) factor in
    a dense transformer has the same shape.  Stacking those factors turns the
    five Polar Express iterations into two batched GEMM streams and removes
    dozens of Python/kernel launches.  The update is algebraically identical
    apart from normal floating-point associativity.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for g in self.param_groups for p in g["params"]]
        grads = [p.grad for p in params if p.grad is not None]
        if not grads and not self.state:
            return loss
        total = torch.sqrt(sum((g.to(self.state_dtype) ** 2).sum() for g in grads)) if grads else None
        scale = 1.0
        if self.grad_clip_norm > 0 and total is not None and float(total) > 0:
            scale = min(1.0, self.grad_clip_norm / float(total))
        groups = {}
        for p in params:
            if p.grad is None and "momentum" not in self.state[p]:
                continue
            key = (tuple(p.shape), p.device, p.dtype)
            groups.setdefault(key, []).append(p)
        for group in self.param_groups:
            lr, beta1 = group["lr"], group["beta1"]
            for key, ps in groups.items():
                shape = ps[0].shape
                gs = torch.stack([
                    torch.zeros_like(p, dtype=self.state_dtype) if p.grad is None
                    else p.grad.to(self.state_dtype) for p in ps
                ])
                old = torch.stack([
                    torch.zeros_like(p, dtype=self.state_dtype)
                    if "momentum" not in self.state[p]
                    else self.state[p]["momentum"].to(self.state_dtype)
                    for p in ps
                ])
                g3 = gs if gs.dim() == 4 else gs.unsqueeze(1) if gs.dim() == 3 else gs
                # For ordinary 2-D factors stack gives [E,m,n]; for 3-D
                # expert factors it gives [E,experts,m,n], which the original
                # optimizer does not support as one item. Keep LoRA 2-D path
                # strict and fail loudly for accidental higher-rank tensors.
                if gs.dim() != 3:
                    raise ValueError("GroupedDimuon expects 2-D LoRA factors")
                left = _conditioner(old, -1, self.damping)
                right = _conditioner(old, -2, self.damping)
                m_new = beta1 * old + (1.0 - beta1) * scale * gs
                for p, m in zip(ps, m_new):
                    if self.momentum_dtype == torch.bfloat16 and m.dtype == torch.float64:
                        self.state[p]["momentum"] = m.to(torch.float32).to(torch.bfloat16)
                    else:
                        self.state[p]["momentum"] = m.to(self.momentum_dtype)
                inner = left[..., :, None] * m_new * right[..., None, :]
                d = left[..., :, None] * _polar_express(inner, self.precision) * right[..., None, :]
                rank = min(d.shape[-2], d.shape[-1])
                target = lr * math.sqrt(rank) * (rank / 32) ** 0.175
                dn = torch.linalg.vector_norm(d, dim=(-2, -1), keepdim=True)
                delta = -target * d / dn.clamp_min(torch.finfo(d.dtype).tiny)
                for p, u in zip(ps, delta):
                    p.add_(u.to(p.dtype))
        return loss
