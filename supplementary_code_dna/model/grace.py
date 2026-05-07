"""
GRACE (GRAdient Constrained sEquence design)
─────────────────────────────────────────────
Null-space projected gradient ascent for multi-oracle constrained optimization.

Problem:
    Maximize f₁(sequence) subject to f₂, f₃, ..., fₙ remaining unchanged.

Solution:
    At each step, find the direction d* in parameter space that:
      (a) maximally improves f₁ (positive projection onto g₁)
      (b) does not change any fₙ to first order (orthogonal to gₙ)

Math:
    gₙ = ∇_θ fₙ               gradient of oracle n w.r.t. SeqProp parameters
    J_c = stack([g₂, ..., gₙ])  constraint Jacobian, shape (N-1, D)
    G_c = J_c J_c^T              Gram matrix,         shape (N-1, N-1)
    a   = J_c g₁                 projections of g₁ onto constraint gradients
    μ*  = G_c⁻¹ a               Lagrange multipliers
    d*  = g₁ - J_c^T μ*         projected gradient

Geometric meaning:
    d* is the component of g₁ that lies in the null space of J_c.
    Moving in direction d* improves f₁ while leaving all fₙ unchanged (to first order).

When d* ≈ 0:
    g₁ is in the row space of J_c — improving f₁ necessarily disturbs a constraint.
    This is a biologically meaningful signal: the objectives are fundamentally coupled
    at the current sequence.

Reference:
    Linder et al. (Fast SeqProp paper) + standard constrained optimization theory.
    See also: Désidéri (2012) for MGDA, Sener & Koltun (2018) for multi-task learning.
"""

import torch
import warnings
from typing import List, Tuple, Dict, Any


def compute_grace_direction(
    g_target:       torch.Tensor,
    g_constraints:  List[torch.Tensor],
    regularization: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute the GRACE null-space projected gradient direction.

    Args:
        g_target:       (D,) flattened gradient of target oracle f₁ w.r.t. all SeqProp
                        parameters (logits + gamma + beta, concatenated).
        g_constraints:  list of (D,) flattened gradients for constraint oracles f₂..fₙ.
                        Empty list → unconstrained gradient ascent.
        regularization: ridge term added to G_c diagonal.
                        Prevents singularity when two constraint gradients are nearly parallel.
                        Increase if you see NaN gradients.

    Returns:
        d_star:  (D,) projected gradient direction.
                 Moving SeqProp parameters by η·d_star improves f₁ while
                 leaving all constraint oracles unchanged (to first order).

        info:    dict with diagnostic information:
                   "gram_matrix"      : (N-1, N-1) Gram matrix G_c (CPU tensor)
                   "conflict_scores"  : list of cosine similarities <g_n, g₁> / (‖g_n‖·‖g₁‖)
                                        Positive = constraint and target agree at this step.
                                        Negative = fundamental conflict; d* will be small.
                   "projected_norm"   : ‖d*‖  — approaches 0 when objectives are fully coupled
                   "original_norm"    : ‖g₁‖
                   "projection_ratio" : ‖d*‖ / ‖g₁‖ — fraction of gradient that survives projection
    """
    D = g_target.shape[0]
    device = g_target.device

    # ── Unconstrained case ────────────────────────────────────────────────────
    if len(g_constraints) == 0:
        return g_target.clone(), {
            "gram_matrix":      None,
            "conflict_scores":  [],
            "projected_norm":   g_target.norm().item(),
            "original_norm":    g_target.norm().item(),
            "projection_ratio": 1.0,
        }

    # ── Build constraint Jacobian J_c ─────────────────────────────────────────
    # J_c[n] = g_{n+1}   (gradient of constraint oracle n+1)
    J_c = torch.stack(g_constraints, dim=0)    # (N-1, D)
    N   = J_c.shape[0]

    # ── Gram matrix G_c = J_c J_c^T ──────────────────────────────────────────
    # G_c[m, n] = <g_{m+1}, g_{n+1}>  (inner product between constraint gradients)
    G_c = J_c @ J_c.T                          # (N-1, N-1)
    G_c = G_c + regularization * torch.eye(N, device=device, dtype=G_c.dtype)

    # ── Projection coefficients: a[n] = <g_{n+1}, g₁> ────────────────────────
    a = J_c @ g_target                         # (N-1,)

    # ── Solve G_c μ* = a  (tiny N×N system, negligible cost) ─────────────────
    try:
        mu_star = torch.linalg.solve(G_c, a)   # (N-1,)
    except torch.linalg.LinAlgError:
        warnings.warn(
            "GRACE: G_c is singular even after regularization. "
            "Using least-squares fallback. Consider increasing `regularization`."
        )
        mu_star = torch.linalg.lstsq(G_c, a.unsqueeze(1)).solution.squeeze(1)

    # ── Projected gradient d* = g₁ - J_c^T μ* ───────────────────────────────
    # This removes the component of g₁ in the row space of J_c.
    # What remains (d*) lies in the null space of J_c.
    d_star = g_target - J_c.T @ mu_star       # (D,)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    g1_norm_raw = g_target.norm().item()
    g1_norm = max(g1_norm_raw, 1e-9)
    conflict_scores = []
    for n in range(N):
        gc_norm = J_c[n].norm().item()
        denom = max(gc_norm * g1_norm, 1e-9)
        conflict_scores.append((J_c[n] @ g_target).item() / denom)
    projected_norm = d_star.norm().item()

    info = {
        "gram_matrix":      G_c.detach().cpu(),
        "conflict_scores":  conflict_scores,
        "projected_norm":   projected_norm,
        "original_norm":    g1_norm_raw,
        "projection_ratio": projected_norm / g1_norm,
        "mu_star":          mu_star.detach().cpu(),
    }

    return d_star, info


def flatten_gradients(
    grads:  Tuple[torch.Tensor, ...],
    params: List[torch.Tensor],
) -> torch.Tensor:
    """
    Flatten a tuple of per-parameter gradients into a single vector.

    If a gradient is None (parameter didn't contribute to the loss),
    a zero vector of the appropriate size is used.

    Args:
        grads:  tuple from torch.autograd.grad(...)
        params: corresponding list of parameters (used to get shapes)

    Returns:
        flat: (D,) concatenated gradient vector
    """
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            parts.append(g.flatten())
    return torch.cat(parts)


def apply_grace_gradient(
    params:  List[torch.Tensor],
    d_star:  torch.Tensor,
    negate:  bool = True,
) -> None:
    """
    Write d_star back into the .grad fields of SeqProp parameters.

    The optimizer does: param -= lr * param.grad
    We want: param += lr * d_star  (ascent in GRACE direction)
    So we set: param.grad = -d_star

    Args:
        params:  list of SeqProp parameters (from seqprop.parameters())
        d_star:  (D,) GRACE projected gradient
        negate:  True → set grad = -d_star (for gradient *descent* optimizers)
                 False → set grad = d_star  (for manual gradient *ascent*)
    """
    idx = 0
    sign = -1.0 if negate else 1.0

    for param in params:
        n = param.numel()
        chunk = d_star[idx : idx + n].reshape(param.shape)

        if param.grad is None:
            param.grad = sign * chunk.clone()
        else:
            param.grad.copy_(sign * chunk)

        idx += n
