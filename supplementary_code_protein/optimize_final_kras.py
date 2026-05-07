"""
optimize_final_kras.py — KRAS optimizer with real decoded-sequence tracking.

This is a KRAS-specific version of the newer optimization pipeline used for
DNA/mRNA experiments. It keeps the original GRACE / ALM optimization in the
soft sequence space, but it also:

1. scores the discrete starting sequence in eval mode,
2. tracks real decoded-sequence oracle values every step,
3. optionally rejects updates that worsen decoded real constraint violation,
4. saves best feasible decoded sequences and best feasible sampled sequences.

Example:
  python optimize_final_kras.py \
    --sequence TEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM \
    --target PI3 \
    --constraints SOS abundance \
    --ckpts PI3:trainings/CW_RAS_full_profile/kras_models/PI3/model.pt \
            SOS:trainings/CW_RAS_full_profile/kras_models/SOS/model.pt \
            abundance:trainings/CW_RAS_full_profile/kras_models/abundance/model.pt \
    --mode grace_lagrangian \
    --constraint_eps 0.1 --constraint_type absolute \
    --decoded_constraint_guard \
    --out results/kras_real/pi3_vs_sos_abundance/grace_lagrangian.json
"""

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.grace import compute_grace_direction, flatten_gradients
from oracles.esm2_backbone import ESM2SoftBackbone
from oracles.kras import KRASOracle
from utils.encoding import (
    AA_ALPHABET,
    AA_TO_IDX,
    IDX_TO_AA,
    NUM_AAS,
    expected_hamming,
    hamming_distance,
    sequence_to_onehot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES = [
    "unconstrained",
    "grace_only",
    "lagrangian_only",
    "grace_lagrangian",
    "simulated_annealing",
    "botorch_baseline",
    "directed_evolution",
    "adalead_baseline",
]


class NegatedOracle(nn.Module):
    """Wrap an oracle so maximizing this wrapper means minimizing the base oracle."""

    def __init__(self, base_oracle: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.base_oracle = base_oracle
        self.name = name or f"decrease_{getattr(base_oracle, 'name', 'oracle')}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.base_oracle(x)


class ProteinSeqPropTau(nn.Module):
    """Protein SeqProp with temperature-aware sampling for smoother early optimization."""

    def __init__(
        self,
        L: int,
        init_sequence: str = None,
        init_logit_scale: float = 8.0,
        init_gamma: float = 1.0,
    ):
        super().__init__()
        self.L = L
        logits = torch.zeros(NUM_AAS, L)

        if init_sequence is not None:
            if len(init_sequence) != L:
                raise ValueError(f"init_sequence length {len(init_sequence)} != L={L}")
            for i, aa in enumerate(init_sequence):
                if aa not in AA_TO_IDX:
                    raise ValueError(f"Unknown amino acid '{aa}' at position {i}")
                logits[AA_TO_IDX[aa], i] = init_logit_scale

        logits += 0.01 * torch.randn_like(logits)
        self.logits = nn.Parameter(logits)
        self.gamma = nn.Parameter(torch.tensor([init_gamma]))
        self.beta = nn.Parameter(torch.zeros(NUM_AAS))
        self._inst_norm = nn.InstanceNorm1d(NUM_AAS, affine=False, eps=1e-5)

    def normalized_logits(self) -> torch.Tensor:
        z = self._inst_norm(self.logits.unsqueeze(0)).squeeze(0)
        return self.gamma * z + self.beta.unsqueeze(1)

    def probabilities(self, tau: float = 1.0) -> torch.Tensor:
        return F.softmax(self.normalized_logits() / tau, dim=0)

    def st_sample(self, K: int, tau: float = 1.0):
        P = self.probabilities(tau=tau)
        samples = []
        for _ in range(K):
            c = torch.multinomial(P.T, num_samples=1).squeeze(1)
            hard = F.one_hot(c, num_classes=NUM_AAS).T.float()
            samples.append(hard + (P - P.detach()))
        return samples, P

    def decode(self, method: str = "argmax") -> str:
        P = self.probabilities()
        if method == "argmax":
            idx = P.argmax(dim=0)
        elif method == "sample":
            idx = torch.multinomial(P.T, 1).squeeze(1)
        else:
            raise ValueError(method)
        return "".join(IDX_TO_AA[i.item()] for i in idx)

    def entropy(self) -> torch.Tensor:
        P = self.probabilities()
        return -(P * (P + 1e-9).log()).sum(dim=0).mean()


@dataclass
class OptimConfig:
    mode: str = "grace_lagrangian"
    n_steps: int = 500
    K: int = 8
    lr: float = 1e-2
    hamming_lambda: float = 0.05
    constraint_eps: float = 0.1
    constraint_type: str = "absolute"   # "absolute" | "percent"
    alm_rho: float = 10.0
    alm_dual_lr: float = 0.02
    alm_curriculum_start_steps: int = 0
    alm_curriculum_ramp_steps: int = 0
    lambda_max: float = 100.0
    lambda_init: float = 0.0
    grad_balance: bool = True
    grace_reg: float = 1e-6
    log_every: int = 10
    n_decode_samples: int = 5000
    decoded_constraint_guard: bool = False
    decoded_guard_tol: float = 0.0
    sa_lambda_penalty: float = 10.0
    sa_temp_init: float = 1.0
    sa_temp_final: float = 1e-2
    sa_min_mutations: int = 1
    sa_max_mutations: int = 3
    bo_n_init: int = 12
    bo_candidate_pool: int = 96
    bo_refit_every: int = 25
    bo_train_window: int = 48
    bo_num_restarts: int = 4
    bo_raw_samples: int = 64
    bo_random_mutations_min: int = 1
    bo_random_mutations_max: int = 4
    adalead_budget: int = 4000
    adalead_population_size: int = 128
    adalead_num_leaders: int = 16
    adalead_mutants_per_leader: int = 16
    adalead_min_mutations: int = 1
    adalead_max_mutations: int = 3
    de_top_positions: int = 3
    de_beam_width: int = 1
    de_lambda_penalty: float = 10.0
    tau_start: float = 0.5
    tau_end: float = 0.05
    tau_warmup_steps: int = 400
    device: str = "cpu"


def canonicalize_protein(seq: str) -> str:
    seq = seq.strip().upper()
    bad = [aa for aa in seq if aa not in AA_TO_IDX]
    if bad:
        raise ValueError(f"Unsupported amino acids: {sorted(set(bad))}; supported={AA_ALPHABET}")
    return seq


def propose_point_mutation_protein(seq: str, n_mutations: int = 1) -> str:
    n_mutations = max(1, min(n_mutations, len(seq)))
    chars = list(seq)
    for idx in random.sample(range(len(seq)), k=n_mutations):
        current = chars[idx]
        choices = [aa for aa in AA_ALPHABET if aa != current]
        chars[idx] = random.choice(choices)
    return "".join(chars)


def sequence_to_flat_onehot_protein(seq: str) -> torch.Tensor:
    return sequence_to_onehot(seq).reshape(-1)


def annealing_temperature(step: int, n_steps: int, temp_init: float, temp_final: float) -> float:
    if n_steps <= 1:
        return temp_final
    frac = step / (n_steps - 1)
    return temp_init * ((temp_final / temp_init) ** frac)


def compute_violation(
    y_c: torch.Tensor,
    c_0: float,
    eps: float,
    constraint_type: str = "absolute",
) -> torch.Tensor:
    abs_drift = (y_c - c_0).abs()
    if constraint_type == "percent":
        return F.relu(abs_drift / (abs(c_0) + 1e-4) - eps)
    return F.relu(abs_drift - eps)


def is_within_tolerance(
    val: float,
    c_0: float,
    eps: float,
    constraint_type: str = "absolute",
) -> bool:
    abs_drift = abs(val - c_0)
    if constraint_type == "percent":
        return abs_drift / (abs(c_0) + 1e-4) <= eps
    return abs_drift <= eps


def scalar_violation(
    val: float,
    c_0: float,
    eps: float,
    constraint_type: str = "absolute",
) -> float:
    abs_drift = abs(val - c_0)
    if constraint_type == "percent":
        return max(0.0, abs_drift / (abs(c_0) + 1e-4) - eps)
    return max(0.0, abs_drift - eps)


def total_violation(
    vals: Dict[str, float],
    c_initial: Dict[str, float],
    eps: float,
    constraint_type: str = "absolute",
) -> float:
    return sum(
        scalar_violation(vals[name], c_initial[name], eps, constraint_type)
        for name in vals
    )


@torch.no_grad()
def score_protein_sequence(
    seq: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    device: str,
) -> tuple[float, Dict[str, float]]:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    target = float(target_oracle(x).item())
    constraints = {oracle.name: float(oracle(x).item()) for oracle in constraint_oracles}
    return target, constraints


def compute_target_gradient_protein(
    seq: str,
    target_oracle: nn.Module,
    device: str,
) -> torch.Tensor:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    x.requires_grad_(True)
    target = target_oracle(x)
    target.sum().backward()
    return x.grad.detach().squeeze(0).cpu()


def compute_constrained_gradient_protein(
    seq: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    device: str,
    regularization: float = 1e-6,
) -> torch.Tensor:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    x.requires_grad_(True)

    target = target_oracle(x)
    g_target = torch.autograd.grad(
        target.sum(),
        x,
        retain_graph=bool(constraint_oracles),
        create_graph=False,
        allow_unused=False,
    )[0].detach().reshape(-1)

    if not constraint_oracles:
        return g_target.view(NUM_AAS, -1).cpu()

    g_constraints = []
    for i, oracle in enumerate(constraint_oracles):
        score = oracle(x)
        g_c = torch.autograd.grad(
            score.sum(),
            x,
            retain_graph=i < len(constraint_oracles) - 1,
            create_graph=False,
            allow_unused=False,
        )[0].detach().reshape(-1)
        g_constraints.append(g_c)

    d_star, _ = compute_grace_direction(
        g_target=g_target,
        g_constraints=g_constraints,
        regularization=regularization,
    )
    return d_star.view(NUM_AAS, -1).cpu()


def propose_gradient_mutants_protein(
    seq: str,
    gradient: torch.Tensor,
    top_positions: int,
    seen_sequences: set[str],
) -> list[tuple[str, Dict[str, Any]]]:
    seq = canonicalize_protein(seq)
    current_idx = [AA_TO_IDX[aa] for aa in seq]
    ranked_mutations: list[tuple[float, int, int]] = []
    for pos, cur in enumerate(current_idx):
        for alt in range(NUM_AAS):
            if alt == cur:
                continue
            delta = float((gradient[alt, pos] - gradient[cur, pos]).item())
            ranked_mutations.append((delta, pos, alt))
    ranked_mutations.sort(key=lambda x: x[0], reverse=True)

    proposals: list[tuple[str, Dict[str, Any]]] = []
    used_positions: set[int] = set()
    for delta, pos, alt in ranked_mutations:
        if pos in used_positions:
            continue
        chars = list(seq)
        chars[pos] = IDX_TO_AA[alt]
        cand = "".join(chars)
        if cand in seen_sequences:
            continue
        proposals.append(
            (
                cand,
                {
                    "position": pos,
                    "from": seq[pos],
                    "to": IDX_TO_AA[alt],
                    "delta": delta,
                },
            )
        )
        used_positions.add(pos)
        if len(proposals) >= top_positions:
            break
    return proposals


def summarize_relaxed_state(
    seqprop: ProteinSeqPropTau,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    X_orig: torch.Tensor,
    K: int,
) -> tuple[float, Dict[str, float], float, float]:
    with torch.no_grad():
        samples, P = seqprop.st_sample(K)
        batch = torch.stack(samples, dim=0)
        t_val = float(target_oracle(batch).mean().item())
        c_vals = {oracle.name: float(oracle(batch).mean().item()) for oracle in constraint_oracles}
        h_val = float(expected_hamming(P, X_orig).item())
        ent_val = float(seqprop.entropy().item())
    return t_val, c_vals, h_val, ent_val


@torch.no_grad()
def load_sequence_into_seqprop(
    seqprop: ProteinSeqPropTau,
    seq: str,
    logit_scale: float = 8.0,
) -> None:
    seq = canonicalize_protein(seq)
    logits = torch.zeros_like(seqprop.logits)
    for i, aa in enumerate(seq):
        logits[AA_TO_IDX[aa], i] = logit_scale
    logits += 0.01 * torch.randn_like(logits)
    seqprop.logits.copy_(logits)
    seqprop.gamma.fill_(1.0)
    seqprop.beta.zero_()


def optimize_simulated_annealing(
    sequence: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_protein(sequence)
    L = len(sequence)

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    current_seq = sequence
    current_target, current_constraints = score_protein_sequence(
        current_seq,
        target_oracle,
        constraint_oracles,
        device,
    )
    c_initial = {target_oracle.name: current_target, **current_constraints}
    real_c_initial = dict(c_initial)
    current_violation = total_violation(
        current_constraints,
        real_c_initial,
        config.constraint_eps,
        config.constraint_type,
    )
    current_objective = current_target - config.sa_lambda_penalty * current_violation

    history: Dict[str, Any] = {
        "scores": {target_oracle.name: [current_target], **{o.name: [current_constraints[o.name]] for o in all_oracles}},
        "real_scores": {target_oracle.name: [current_target], **{o.name: [current_constraints[o.name]] for o in all_oracles}},
        "rel_drifts": {o.name: [0.0] for o in all_oracles},
        "lambdas": {o.name: [] for o in constraint_oracles},
        "hamming": [0],
        "entropy": [],
        "proj_ratio": [],
        "sequences": [(0, current_seq)],
        "decoded_sequences": [current_seq],
        "decoded_guard_rejected": [],
        "decoded_guard_violation": [current_violation],
        "temperature": [config.sa_temp_init],
        "constraint_eps": config.constraint_eps,
        "constraint_type": config.constraint_type,
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
    }

    best_objective_seq = current_seq
    best_objective_value = current_objective
    best_objective_score = current_target
    best_objective_constraints = dict(current_constraints)
    best_objective_step = 0

    best_feasible_sequence = current_seq if all(
        is_within_tolerance(current_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type)
        for o in constraint_oracles
    ) else None
    best_feasible_score = current_target if best_feasible_sequence is not None else float("-inf")
    best_feasible_step = 0 if best_feasible_sequence is not None else None

    logger.info(
        f"step={0:4d} | target={current_target:+.3f} | "
        + "  ".join(f"{k}={v:.3f}(Δ0.00)" for k, v in current_constraints.items())
        + f" | hamming=0.0 | T={config.sa_temp_init:.4f} | viol={current_violation:.4f} | accepted=1"
    )

    for step in range(1, config.n_steps + 1):
        temperature = annealing_temperature(
            step, config.n_steps, config.sa_temp_init, config.sa_temp_final
        )
        proposal_mutations = random.randint(config.sa_min_mutations, config.sa_max_mutations)
        cand_seq = propose_point_mutation_protein(current_seq, n_mutations=proposal_mutations)
        cand_target, cand_constraints = score_protein_sequence(
            cand_seq,
            target_oracle,
            constraint_oracles,
            device,
        )
        cand_violation = total_violation(
            cand_constraints,
            real_c_initial,
            config.constraint_eps,
            config.constraint_type,
        )
        cand_objective = cand_target - config.sa_lambda_penalty * cand_violation

        delta = cand_objective - current_objective
        accept = delta >= 0 or random.random() < np.exp(delta / max(temperature, 1e-12))
        if accept:
            current_seq = cand_seq
            current_target = cand_target
            current_constraints = cand_constraints
            current_violation = cand_violation
            current_objective = cand_objective

        rel_drifts = {
            oracle.name: abs(current_constraints[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4)
            for oracle in all_oracles
        }

        history["scores"][target_oracle.name].append(current_target)
        history["real_scores"][target_oracle.name].append(current_target)
        for oracle in all_oracles:
            history["scores"][oracle.name].append(current_constraints[oracle.name])
            history["real_scores"][oracle.name].append(current_constraints[oracle.name])
            history["rel_drifts"][oracle.name].append(rel_drifts[oracle.name])
        history["hamming"].append(hamming_distance(sequence, current_seq))
        history["decoded_sequences"].append(current_seq)
        history["decoded_guard_rejected"].append(False)
        history["decoded_guard_violation"].append(current_violation)
        history["temperature"].append(temperature)

        if step % config.log_every == 0:
            c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in current_constraints.items())
            logger.info(
                f"step={step:4d} | target={current_target:+.3f} | {c_str} | "
                f"hamming={hamming_distance(sequence, current_seq):.1f} | "
                f"proposal_k={proposal_mutations} | T={temperature:.4f} | "
                f"viol={current_violation:.4f} | accepted={1 if accept else 0}"
            )
        if step % (config.log_every * 5) == 0:
            history["sequences"].append((step, current_seq))

        if current_objective > best_objective_value:
            best_objective_seq = current_seq
            best_objective_value = current_objective
            best_objective_score = current_target
            best_objective_constraints = dict(current_constraints)
            best_objective_step = step

        if all(
            is_within_tolerance(current_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type)
            for o in constraint_oracles
        ) and current_target > best_feasible_score:
            best_feasible_sequence = current_seq
            best_feasible_score = current_target
            best_feasible_step = step

    final_seq = best_feasible_sequence or best_objective_seq
    n_mutations = hamming_distance(sequence, final_seq)
    final_vals = best_objective_constraints if final_seq == best_objective_seq else {
        **score_protein_sequence(final_seq, target_oracle, constraint_oracles, device)[1]
    }

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | "
        f"Best penalized objective: {best_objective_value:.4f}"
    )
    logger.info(
        f"Target {target_oracle.name}: {c_initial[target_oracle.name]:.3f} -> "
        f"{best_objective_score:.3f} (best objective step: {best_objective_step})"
    )
    if best_feasible_sequence is not None:
        logger.info(
            f"Best feasible sequence (step {best_feasible_step}): "
            f"{best_feasible_sequence} | target={best_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_feasible_sequence)}"
        )
    else:
        logger.info("Best feasible sequence: none found")
    for oracle in all_oracles:
        logger.info(
            f"Oracle {oracle.name}: initial={c_initial[oracle.name]:.3f} "
            f"final={final_vals[oracle.name]:.3f} "
            f"rel_drift={abs(final_vals[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4):.2f}"
        )

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "target_oracle": target_oracle.name,
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [],
            "final_lambdas": {oracle.name: 0.0 for oracle in constraint_oracles},
            "argmax_sequence": final_seq,
            "argmax_n_mutations": n_mutations,
            "best_sampled_sequence": None,
            "best_sampled_target_score": None,
            "best_sampled_n_mutations": None,
            "best_sampled_feasible_sequence": None,
            "best_sampled_feasible_target_score": None,
            "best_sampled_feasible_step": None,
            "best_sampled_feasible_n_mutations": None,
            "sampled_feasible_count": 0,
            "n_decode_samples": 0,
            "best_hard_feasible_sequence": best_feasible_sequence,
            "best_hard_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_hard_feasible_step": best_feasible_step,
            "best_real_feasible_sequence": best_feasible_sequence,
            "best_real_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_real_feasible_step": best_feasible_step,
            "best_real_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_objective_sequence": best_objective_seq,
            "best_objective_value": best_objective_value,
            "best_objective_step": best_objective_step,
        }
    )
    return history


def optimize_botorch_baseline(
    sequence: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    config: OptimConfig,
) -> Dict[str, Any]:
    try:
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            "BoTorch baseline requested, but botorch/gpytorch are not installed in this environment."
        ) from exc

    device = config.device
    sequence = canonicalize_protein(sequence)
    L = len(sequence)
    dim = NUM_AAS * L

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    start_target, start_constraints = score_protein_sequence(
        sequence,
        target_oracle,
        constraint_oracles,
        device,
    )
    c_initial = {target_oracle.name: start_target, **start_constraints}
    real_c_initial = dict(c_initial)
    logger.info(f"Initial oracle values (discrete start, eval mode): {c_initial}")

    history: Dict[str, Any] = {
        "scores": {target_oracle.name: [start_target], **{o.name: [start_constraints[o.name]] for o in all_oracles}},
        "real_scores": {target_oracle.name: [start_target], **{o.name: [start_constraints[o.name]] for o in all_oracles}},
        "rel_drifts": {o.name: [0.0] for o in all_oracles},
        "lambdas": {o.name: [] for o in constraint_oracles},
        "hamming": [0],
        "entropy": [0.0],
        "proj_ratio": [1.0],
        "sequences": [(0, sequence)],
        "decoded_sequences": [sequence],
        "decoded_guard_rejected": [False],
        "decoded_guard_violation": [
            total_violation(start_constraints, real_c_initial, config.constraint_eps, config.constraint_type)
        ],
        "constraint_eps": config.constraint_eps,
        "constraint_type": config.constraint_type,
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
    }

    train_X: List[torch.Tensor] = [sequence_to_flat_onehot_protein(sequence).to(dtype=torch.double)]
    train_Y: List[List[float]] = [[start_target] + [start_constraints[o.name] for o in all_oracles]]
    seen_sequences = {sequence}

    best_overall_sequence = sequence
    best_overall_score = start_target
    best_overall_step = 0

    is_start_feasible = all(
        is_within_tolerance(
            start_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type
        )
        for o in constraint_oracles
    )
    best_feasible_sequence = sequence if is_start_feasible else None
    best_feasible_score = start_target if is_start_feasible else float("-inf")
    best_feasible_step = 0 if is_start_feasible else None
    acqf = None
    last_refit_step = -1

    def append_history(step: int, seq: str, target: float, constraints: Dict[str, float], source: str) -> None:
        rel_drifts = {
            o.name: abs(constraints[o.name] - c_initial[o.name]) / (abs(c_initial[o.name]) + 1e-4)
            for o in all_oracles
        }
        violation = total_violation(constraints, real_c_initial, config.constraint_eps, config.constraint_type)
        history["scores"][target_oracle.name].append(target)
        history["real_scores"][target_oracle.name].append(target)
        for o in all_oracles:
            history["scores"][o.name].append(constraints[o.name])
            history["real_scores"][o.name].append(constraints[o.name])
            history["rel_drifts"][o.name].append(rel_drifts[o.name])
        for o in constraint_oracles:
            history["lambdas"][o.name].append(0.0)
        history["hamming"].append(hamming_distance(sequence, seq))
        history["entropy"].append(0.0)
        history["proj_ratio"].append(1.0)
        history["decoded_sequences"].append(seq)
        history["decoded_guard_rejected"].append(False)
        history["decoded_guard_violation"].append(violation)

        if step % config.log_every == 0:
            c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in constraints.items())
            logger.info(
                f"step={step:4d} | target={target:+.3f} | {c_str} | "
                f"hamming={hamming_distance(sequence, seq):.1f} | source={source} | viol={violation:.4f}"
            )
        if step % (config.log_every * 5) == 0:
            history["sequences"].append((step, seq))

    def random_novel_sequence(anchor: Optional[str] = None) -> str:
        anchor = anchor or sequence
        max_attempts = 128
        for _ in range(max_attempts):
            n_mut = random.randint(config.bo_random_mutations_min, config.bo_random_mutations_max)
            cand = propose_point_mutation_protein(anchor, n_mutations=n_mut)
            if cand not in seen_sequences:
                return cand
        for _ in range(max_attempts):
            cand = propose_point_mutation_protein(best_overall_sequence, n_mutations=1)
            if cand not in seen_sequences:
                return cand
        return sequence

    def fit_botorch_model() -> Optional[Any]:
        window = min(len(train_X), config.bo_train_window)
        train_X_tensor = torch.stack(train_X[-window:], dim=0)
        train_Y_tensor = torch.tensor(train_Y[-window:], dtype=torch.double)
        try:
            model = SingleTaskGP(
                train_X=train_X_tensor,
                train_Y=train_Y_tensor,
                input_transform=Normalize(d=dim),
                outcome_transform=Standardize(m=train_Y_tensor.shape[-1]),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except Exception as exc:
            logger.warning(f"BoTorch GP fitting failed, falling back to random proposal: {exc}")
            return None

        feasible_targets = []
        for row in train_Y[-window:]:
            row_constraints = {o.name: row[1 + idx] for idx, o in enumerate(all_oracles)}
            if all(
                is_within_tolerance(
                    row_constraints[o.name],
                    real_c_initial[o.name],
                    config.constraint_eps,
                    config.constraint_type,
                )
                for o in constraint_oracles
            ):
                feasible_targets.append(row[0])
        best_f = max(feasible_targets) if feasible_targets else max(row[0] for row in train_Y[-window:])

        objective = GenericMCObjective(lambda samples, X=None: samples[..., 0])
        constraints = [
            (
                lambda samples, idx=1 + i, c0=real_c_initial[o.name], eps=config.constraint_eps:
                (samples[..., idx] - c0).abs() - eps
            )
            for i, o in enumerate(constraint_oracles)
        ]

        try:
            return qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                objective=objective,
                constraints=constraints or None,
            )
        except Exception as exc:
            logger.warning(f"BoTorch acquisition construction failed, falling back to random proposal: {exc}")
            return None

    def sample_candidate_pool() -> List[str]:
        pool: List[str] = []
        anchors = [sequence, best_overall_sequence]
        if best_feasible_sequence is not None:
            anchors.append(best_feasible_sequence)
        max_attempts = max(config.bo_candidate_pool * 10, 512)
        attempts = 0
        while len(pool) < config.bo_candidate_pool and attempts < max_attempts:
            attempts += 1
            anchor = random.choice(anchors)
            if random.random() < 0.15:
                n_mut = max(config.bo_random_mutations_max, 6)
            else:
                n_mut = random.randint(config.bo_random_mutations_min, config.bo_random_mutations_max)
            cand = propose_point_mutation_protein(anchor, n_mutations=n_mut)
            if cand not in seen_sequences and cand not in pool:
                pool.append(cand)
        if not pool:
            pool.append(random_novel_sequence())
        return pool

    def suggest_sequence_via_bo(acquisition: Any) -> Optional[str]:
        pool = sample_candidate_pool()
        pool_X = torch.stack(
            [sequence_to_flat_onehot_protein(seq).to(dtype=torch.double) for seq in pool],
            dim=0,
        )
        try:
            with torch.no_grad():
                acq_vals = acquisition(pool_X.unsqueeze(1)).reshape(-1)
            order = torch.argsort(acq_vals, descending=True)
            for idx in order.tolist():
                seq = pool[idx]
                if seq not in seen_sequences:
                    return seq
        except Exception as exc:
            logger.warning(f"BoTorch acquisition evaluation failed, falling back to random proposal: {exc}")
        return None

    logger.info(
        f"BoTorch baseline: using {min(config.bo_n_init, config.n_steps + 1)} total warm-start points "
        f"(including wild-type), then constrained qLogEI over a discrete mutation pool "
        f"(pool={config.bo_candidate_pool}, refit_every={config.bo_refit_every}, "
        f"train_window={config.bo_train_window})."
    )

    for step in range(1, config.n_steps + 1):
        if step < config.bo_n_init:
            cand_seq = random_novel_sequence()
            source = "random_init"
        else:
            should_refit = (acqf is None) or ((step - last_refit_step) >= config.bo_refit_every)
            if should_refit:
                acqf = fit_botorch_model()
                last_refit_step = step
            cand_seq = suggest_sequence_via_bo(acqf) if acqf is not None else None
            source = "botorch_pool"
            if cand_seq is None or cand_seq in seen_sequences:
                cand_seq = random_novel_sequence(best_overall_sequence)
                source = "random_fallback"

        cand_target, cand_constraints = score_protein_sequence(
            cand_seq,
            target_oracle,
            constraint_oracles,
            device,
        )
        seen_sequences.add(cand_seq)
        train_X.append(sequence_to_flat_onehot_protein(cand_seq).to(dtype=torch.double))
        train_Y.append([cand_target] + [cand_constraints[o.name] for o in all_oracles])

        append_history(step, cand_seq, cand_target, cand_constraints, source)

        if cand_target > best_overall_score:
            best_overall_score = cand_target
            best_overall_sequence = cand_seq
            best_overall_step = step

        if all(
            is_within_tolerance(
                cand_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type
            )
            for o in constraint_oracles
        ) and cand_target > best_feasible_score:
            best_feasible_sequence = cand_seq
            best_feasible_score = cand_target
            best_feasible_step = step

    final_seq = best_feasible_sequence or best_overall_sequence
    n_mutations = hamming_distance(sequence, final_seq)

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | "
        f"Best observed target: {best_overall_score:.4f}"
    )
    logger.info(
        f"Target {target_oracle.name}: {c_initial[target_oracle.name]:.3f} -> "
        f"{best_overall_score:.3f} (best observed step: {best_overall_step})"
    )
    if best_feasible_sequence is not None:
        logger.info(
            f"Best feasible sequence (step {best_feasible_step}): "
            f"{best_feasible_sequence} | target={best_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_feasible_sequence)}"
        )
    else:
        logger.info("Best feasible sequence: none found")

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "target_oracle": target_oracle.name,
            "constraint_oracles": [o.name for o in constraint_oracles],
            "monitor_oracles": [],
            "final_lambdas": {o.name: 0.0 for o in constraint_oracles},
            "argmax_sequence": best_overall_sequence,
            "argmax_n_mutations": hamming_distance(sequence, best_overall_sequence),
            "best_sampled_sequence": best_overall_sequence,
            "best_sampled_target_score": best_overall_score,
            "best_sampled_n_mutations": hamming_distance(sequence, best_overall_sequence),
            "best_sampled_feasible_sequence": best_feasible_sequence,
            "best_sampled_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_sampled_feasible_step": best_feasible_step,
            "best_sampled_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "sampled_feasible_count": sum(
                1
                for row in train_Y
                if all(
                    is_within_tolerance(
                        row[1 + i],
                        real_c_initial[o.name],
                        config.constraint_eps,
                        config.constraint_type,
                    )
                    for i, o in enumerate(constraint_oracles)
                )
            ),
            "n_decode_samples": 0,
            "best_hard_feasible_sequence": best_feasible_sequence,
            "best_hard_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_hard_feasible_step": best_feasible_step,
            "best_real_feasible_sequence": best_feasible_sequence,
            "best_real_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_real_feasible_step": best_feasible_step,
            "best_real_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_objective_sequence": best_overall_sequence,
            "best_objective_value": best_overall_score,
            "best_objective_step": best_overall_step,
        }
    )
    return history


def optimize_directed_evolution(
    sequence: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_protein(sequence)
    L = len(sequence)

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    start_target, start_constraints = score_protein_sequence(
        sequence,
        target_oracle,
        constraint_oracles,
        device,
    )
    c_initial = {target_oracle.name: start_target, **start_constraints}
    real_c_initial = dict(c_initial)
    logger.info(f"Initial oracle values (discrete start, eval mode): {c_initial}")

    history: Dict[str, Any] = {
        "scores": {target_oracle.name: [start_target], **{o.name: [start_constraints[o.name]] for o in all_oracles}},
        "real_scores": {target_oracle.name: [start_target], **{o.name: [start_constraints[o.name]] for o in all_oracles}},
        "rel_drifts": {o.name: [0.0] for o in all_oracles},
        "lambdas": {o.name: [] for o in constraint_oracles},
        "hamming": [0],
        "entropy": [0.0],
        "proj_ratio": [1.0],
        "sequences": [(0, sequence)],
        "decoded_sequences": [sequence],
        "decoded_guard_rejected": [False],
        "decoded_guard_violation": [
            total_violation(start_constraints, real_c_initial, config.constraint_eps, config.constraint_type)
        ],
        "constraint_eps": config.constraint_eps,
        "constraint_type": config.constraint_type,
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
    }

    # `directed_evolution` is kept as a compatibility alias, but the
    # implementation is now a standard AdaLead-style discrete oracle baseline.
    # Search is unconstrained during evolution; feasibility is only used for
    # reporting best feasible target under the same oracle-evaluation budget.
    total_budget = max(1, config.adalead_budget)
    population_size = max(1, config.adalead_population_size)
    num_leaders = max(1, min(config.adalead_num_leaders, population_size))
    mutants_per_leader = max(1, config.adalead_mutants_per_leader)
    min_mut = max(1, config.adalead_min_mutations)
    max_mut = max(min_mut, config.adalead_max_mutations)
    seen_sequences = {sequence}

    best_objective_sequence = sequence
    best_objective_value = start_target
    best_objective_score = start_target
    best_objective_step = 0

    is_start_feasible = all(
        is_within_tolerance(
            start_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type
        )
        for o in constraint_oracles
    )
    best_feasible_sequence = sequence if is_start_feasible else None
    best_feasible_score = start_target if is_start_feasible else float("-inf")
    best_feasible_step = 0 if is_start_feasible else None

    population = [
        {
            "seq": sequence,
            "target": start_target,
            "constraints": start_constraints,
        }
    ]

    logger.info(
        "AdaLead baseline: "
        f"budget={total_budget}, population={population_size}, leaders={num_leaders}, "
        f"mutants_per_leader={mutants_per_leader}, mutation_size=[{min_mut},{max_mut}]."
    )

    def append_history(step: int, seq: str, target: float, constraints: Dict[str, float], source: str) -> None:
        rel_drifts = {
            o.name: abs(constraints[o.name] - c_initial[o.name]) / (abs(c_initial[o.name]) + 1e-4)
            for o in all_oracles
        }
        violation = total_violation(constraints, real_c_initial, config.constraint_eps, config.constraint_type)
        history["scores"][target_oracle.name].append(target)
        history["real_scores"][target_oracle.name].append(target)
        for o in all_oracles:
            history["scores"][o.name].append(constraints[o.name])
            history["real_scores"][o.name].append(constraints[o.name])
            history["rel_drifts"][o.name].append(rel_drifts[o.name])
        for o in constraint_oracles:
            history["lambdas"][o.name].append(0.0)
        history["hamming"].append(hamming_distance(sequence, seq))
        history["entropy"].append(0.0)
        history["proj_ratio"].append(1.0)
        history["decoded_sequences"].append(seq)
        history["decoded_guard_rejected"].append(False)
        history["decoded_guard_violation"].append(violation)

        if step % config.log_every == 0:
            c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in constraints.items())
            logger.info(
                f"step={step:4d} | target={target:+.3f} | {c_str} | "
                f"hamming={hamming_distance(sequence, seq):.1f} | source={source} | viol={violation:.4f}"
            )
        if step % (config.log_every * 5) == 0:
            history["sequences"].append((step, seq))

    def update_bests(step_idx: int, cand_seq: str, cand_target: float, cand_constraints: Dict[str, float]) -> None:
        nonlocal best_objective_sequence, best_objective_value, best_objective_score, best_objective_step
        nonlocal best_feasible_sequence, best_feasible_score, best_feasible_step
        if cand_target > best_objective_value:
            best_objective_sequence = cand_seq
            best_objective_value = cand_target
            best_objective_score = cand_target
            best_objective_step = step_idx
        if all(
            is_within_tolerance(
                cand_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type
            )
            for o in constraint_oracles
        ) and cand_target > best_feasible_score:
            best_feasible_sequence = cand_seq
            best_feasible_score = cand_target
            best_feasible_step = step_idx

    def sample_unique_mutant(parent_seq: str) -> Optional[str]:
        max_attempts = 128
        for _ in range(max_attempts):
            n_mut = random.randint(min_mut, max_mut)
            cand = propose_point_mutation_protein(parent_seq, n_mutations=n_mut)
            if cand not in seen_sequences:
                return cand
        return None

    step = 0
    update_bests(0, sequence, start_target, start_constraints)

    # Initial population counts toward the same oracle-evaluation budget.
    while len(population) < population_size and (step + 1) < total_budget:
        cand_seq = sample_unique_mutant(sequence)
        if cand_seq is None:
            break
        cand_target, cand_constraints = score_protein_sequence(
            cand_seq,
            target_oracle,
            constraint_oracles,
            device,
        )
        step += 1
        seen_sequences.add(cand_seq)
        append_history(step, cand_seq, cand_target, cand_constraints, "adalead_init")
        population.append(
            {
                "seq": cand_seq,
                "target": cand_target,
                "constraints": cand_constraints,
            }
        )
        update_bests(step, cand_seq, cand_target, cand_constraints)

    per_round_evals = num_leaders * mutants_per_leader
    max_rounds = max(0, (total_budget - (step + 1)) // per_round_evals)
    rounds_completed = 0
    logger.info(
        f"AdaLead initialized with {len(population)} sequences after {step} evaluations; "
        f"running up to {max_rounds} full rounds."
    )

    for round_idx in range(max_rounds):
        leaders = sorted(population, key=lambda item: item["target"], reverse=True)[:num_leaders]
        offspring: list[Dict[str, Any]] = []
        for leader in leaders:
            for _ in range(mutants_per_leader):
                if step >= total_budget:
                    break
                cand_seq = sample_unique_mutant(leader["seq"])
                if cand_seq is None:
                    continue
                cand_target, cand_constraints = score_protein_sequence(
                    cand_seq,
                    target_oracle,
                    constraint_oracles,
                    device,
                )
                step += 1
                seen_sequences.add(cand_seq)
                append_history(step, cand_seq, cand_target, cand_constraints, f"adalead_r{round_idx + 1}")
                record = {
                    "seq": cand_seq,
                    "target": cand_target,
                    "constraints": cand_constraints,
                }
                offspring.append(record)
                update_bests(step, cand_seq, cand_target, cand_constraints)

        combined: Dict[str, Dict[str, Any]] = {item["seq"]: item for item in population}
        for item in offspring:
            prev = combined.get(item["seq"])
            if prev is None or item["target"] > prev["target"]:
                combined[item["seq"]] = item
        population = sorted(combined.values(), key=lambda item: item["target"], reverse=True)[:population_size]
        rounds_completed += 1

    if (step + 1) < total_budget:
        logger.info(
            f"AdaLead ended early after {step + 1}/{total_budget} oracle evaluations "
            f"(unique mutant proposals exhausted before another full round)."
        )

    final_seq = best_feasible_sequence or best_objective_sequence
    n_mutations = hamming_distance(sequence, final_seq)

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | "
        f"Best observed target: {best_objective_value:.4f}"
    )
    logger.info(
        f"Target {target_oracle.name}: {c_initial[target_oracle.name]:.3f} -> "
        f"{best_objective_score:.3f} (best objective step: {best_objective_step})"
    )
    if best_feasible_sequence is not None:
        logger.info(
            f"Best feasible sequence (step {best_feasible_step}): "
            f"{best_feasible_sequence} | target={best_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_feasible_sequence)}"
        )
    else:
        logger.info("Best feasible sequence: none found")

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "target_oracle": target_oracle.name,
            "constraint_oracles": [o.name for o in constraint_oracles],
            "monitor_oracles": [],
            "final_lambdas": {o.name: 0.0 for o in constraint_oracles},
            "argmax_sequence": best_objective_sequence,
            "argmax_n_mutations": hamming_distance(sequence, best_objective_sequence),
            "best_sampled_sequence": best_objective_sequence,
            "best_sampled_target_score": best_objective_score,
            "best_sampled_n_mutations": hamming_distance(sequence, best_objective_sequence),
            "best_sampled_feasible_sequence": best_feasible_sequence,
            "best_sampled_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_sampled_feasible_step": best_feasible_step,
            "best_sampled_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "sampled_feasible_count": sum(
                1
                for step_idx in range(len(history["real_scores"][target_oracle.name]))
                if all(
                    is_within_tolerance(
                        history["real_scores"][o.name][step_idx],
                        real_c_initial[o.name],
                        config.constraint_eps,
                        config.constraint_type,
                    )
                    for o in constraint_oracles
                )
            ),
            "n_decode_samples": 0,
            "best_hard_feasible_sequence": best_feasible_sequence,
            "best_hard_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_hard_feasible_step": best_feasible_step,
            "best_real_feasible_sequence": best_feasible_sequence,
            "best_real_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_real_feasible_step": best_feasible_step,
            "best_real_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_objective_sequence": best_objective_sequence,
            "best_objective_value": best_objective_value,
            "best_objective_step": best_objective_step,
            "oracle_budget": total_budget,
            "oracle_evaluations_used": step + 1,
            "adalead_population_size": population_size,
            "adalead_num_leaders": num_leaders,
            "adalead_mutants_per_leader": mutants_per_leader,
            "adalead_mutation_size": [min_mut, max_mut],
            "adalead_rounds_completed": rounds_completed,
        }
    )
    return history


def optimize(
    sequence: str,
    target_oracle: nn.Module,
    constraint_oracles: List[nn.Module],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_protein(sequence)
    L = len(sequence)
    use_grace = config.mode in ("grace_only", "grace_lagrangian")
    use_alm = config.mode in ("lagrangian_only", "grace_lagrangian")

    seqprop = ProteinSeqPropTau(L=L, init_sequence=sequence).to(device)
    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    X_orig = sequence_to_onehot(sequence).to(device)
    optimizer = torch.optim.Adam(seqprop.parameters(), lr=config.lr)
    all_params = list(seqprop.parameters())

    with torch.no_grad():
        X_start = sequence_to_onehot(sequence).unsqueeze(0).to(device)
        # Soft baseline: K samples at tau=0.5 (same approach as original).
        # This keeps c_initial in the same "soft" frame as the soft oracle evaluations,
        # so |soft_score - c_initial| is correctly calibrated.
        s0, _ = seqprop.st_sample(config.K, tau=0.5)
        b0 = torch.stack(s0, dim=0)
        c_initial = {o.name: o(b0).mean().item() for o in [target_oracle] + all_oracles}
        # Real baseline uses one-hot for accurate discrete tracking.
        real_c_initial = {o.name: o(X_start).item() for o in [target_oracle] + all_oracles}

    logger.info(f"Initial oracle values — soft: {c_initial} | real: {real_c_initial}")

    lambdas = {o.name: config.lambda_init for o in constraint_oracles}
    
    # Ensure our real trackers use the real baseline!
    current_real_seq = sequence
    current_real_target = real_c_initial[target_oracle.name]
    current_real_constraints = {o.name: real_c_initial[o.name] for o in constraint_oracles}
    current_real_violation = total_violation(
        current_real_constraints,
        real_c_initial,
        config.constraint_eps,
        config.constraint_type,
    )

    all_names = [target_oracle.name] + [o.name for o in all_oracles]
    history: Dict[str, Any] = {
        "scores": {n: [] for n in all_names},
        "real_scores": {n: [] for n in all_names},
        "rel_drifts": {o.name: [] for o in all_oracles},
        "lambdas": {o.name: [] for o in constraint_oracles},
        "hamming": [],
        "entropy": [],
        "proj_ratio": [],
        "sequences": [],
        "decoded_sequences": [],
        "decoded_guard_rejected": [],
        "decoded_guard_violation": [],
        "constraint_eps": config.constraint_eps,
        "constraint_type": config.constraint_type,
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
    }
    history["real_scores"][target_oracle.name].append(current_real_target)
    for oracle in all_oracles:
        history["real_scores"][oracle.name].append(real_c_initial[oracle.name])

    best_hard_feasible_score = float("-inf")
    best_hard_feasible_sequence = None
    best_hard_feasible_step = None
    tau_start = config.tau_start
    tau_end = config.tau_end
    tau_warmup_steps = config.tau_warmup_steps  # reach tau_end by this step, then hold fixed
    high_constraint_mode = len(constraint_oracles) >= 3
    local_refine_every = 20 if not high_constraint_mode else 25
    local_refine_top_positions = 5
    local_refine_min_gain = 0.02
    phase_switch_step = (
        min(200, max(1, config.n_steps // 3))
        if not high_constraint_mode
        else min(400, max(1, config.n_steps // 2))
    )
    if high_constraint_mode:
        tau_warmup_steps = max(tau_warmup_steps, phase_switch_step + 200)

    curriculum_start = max(0, config.alm_curriculum_start_steps)
    curriculum_ramp = max(0, config.alm_curriculum_ramp_steps)
    if use_alm and constraint_oracles and curriculum_start > 0:
        logger.info(
            "ALM curriculum enabled: "
            f"penalty off for first {curriculum_start} steps, "
            f"then annealed over {curriculum_ramp} steps."
        )

    for step in range(config.n_steps):
        optimizer.zero_grad()

        param_snapshot = None
        optimizer_snapshot = None
        if config.decoded_constraint_guard and constraint_oracles:
            param_snapshot = [p.detach().clone() for p in all_params]
            optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        # Piecewise schedule: exponential decay during warmup, then hold at tau_end.
        # Reaching tau_end early means soft≈real for all remaining steps → real feasibility.
        if step < tau_warmup_steps:
            current_tau = tau_start * (tau_end / tau_start) ** (step / max(1, tau_warmup_steps - 1))
        else:
            current_tau = tau_end
        samples, P = seqprop.st_sample(config.K, tau=current_tau)
        batch = torch.stack(samples, dim=0)

        y_target = target_oracle(batch).mean()
        y_constraints = [o(batch).mean() for o in constraint_oracles]
        hamming = expected_hamming(P, X_orig)

        violations = [
            compute_violation(y_c, c_initial[o.name], config.constraint_eps, config.constraint_type)
            for o, y_c in zip(constraint_oracles, y_constraints)
        ]

        g_target = flatten_gradients(
            torch.autograd.grad(
                y_target,
                all_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            ),
            all_params,
        )

        


        if use_grace and constraint_oracles:
            # 1. Calculate constraint gradients ONCE
            g_constraints_flat = [
                flatten_gradients(
                    torch.autograd.grad(
                        y_c,
                        all_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    ),
                    all_params,
                )
                for y_c in y_constraints
            ]
            
            # 2. Calculate dynamic GRACE regularization
            current_reg = config.grace_reg
            if step < 100:
                reg_start = 0.1
                frac = step / 99.0
                current_reg = reg_start * ((config.grace_reg / reg_start) ** frac)

            # 3. Call compute_grace_direction
            d_star, grace_info = compute_grace_direction(
                g_target=g_target,
                g_constraints=g_constraints_flat,
                regularization=current_reg, 
            )
        else:
            d_star = g_target
            grace_info = {"projection_ratio": 1.0}








        effective_alm_rho = config.alm_rho
        effective_alm_dual_lr = config.alm_dual_lr
        if high_constraint_mode and step < phase_switch_step:
            effective_alm_rho = 0.5 * config.alm_rho
            effective_alm_dual_lr = 0.25 * config.alm_dual_lr

        # Optional ALM curriculum: keep the constraint "walls" soft early so
        # the optimizer can first climb the objective landscape, then
        # gradually tighten feasibility pressure.
        curriculum_scale = 1.0
        if curriculum_start > 0:
            if step < curriculum_start:
                curriculum_scale = 0.0
            elif curriculum_ramp > 0:
                curriculum_scale = min(1.0, (step - curriculum_start + 1) / curriculum_ramp)
        effective_alm_rho = effective_alm_rho * curriculum_scale
        effective_alm_dual_lr = effective_alm_dual_lr * curriculum_scale

        g_alm = None
        if use_alm:
            alm_penalty = torch.tensor(0.0, device=device)
            for oracle, viol in zip(constraint_oracles, violations):
                lam = lambdas[oracle.name]
                alm_penalty = alm_penalty + lam * viol + (effective_alm_rho / 2) * viol ** 2
            if alm_penalty.grad_fn is not None:
                g_alm = flatten_gradients(
                    torch.autograd.grad(
                        alm_penalty,
                        all_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    ),
                    all_params,
                )

        current_hamming_lambda = 0.0 if step < phase_switch_step else config.hamming_lambda
        g_hamming = flatten_gradients(
            torch.autograd.grad(
                current_hamming_lambda * hamming,
                all_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ),
            all_params,
        )

        if g_alm is not None and config.grad_balance:
            dir_norm = d_star.norm().clamp(min=1e-8)
            alm_norm = g_alm.norm().clamp(min=1e-8)
            g_alm_scaled = g_alm * torch.clamp(dir_norm / alm_norm, max=1.0)
        else:
            g_alm_scaled = g_alm

        final_g = g_hamming - d_star
        if g_alm_scaled is not None:
            final_g = final_g + g_alm_scaled

        offset = 0
        for p in all_params:
            numel = p.numel()
            if p.requires_grad:
                p.grad = final_g[offset:offset + numel].view(p.shape).clone()
            offset += numel

        optimizer.step()

        with torch.no_grad():
            cand_seq = seqprop.decode("argmax")
            cand_oh = sequence_to_onehot(cand_seq).unsqueeze(0).to(device)
            cand_target = target_oracle(cand_oh).item()
            cand_c_vals = {o.name: o(cand_oh).item() for o in constraint_oracles}
            cand_real_violation = total_violation(
                cand_c_vals,
                real_c_initial,
                config.constraint_eps,
                config.constraint_type,
            )
            rejected = False

            if (
                config.decoded_constraint_guard
                and constraint_oracles
                and cand_real_violation > current_real_violation + config.decoded_guard_tol
            ):
                rejected = True
                for p, saved in zip(all_params, param_snapshot):
                    p.data.copy_(saved)
                if optimizer_snapshot is not None:
                    optimizer.load_state_dict(optimizer_snapshot)

                cand_seq = current_real_seq
                cand_target = current_real_target
                cand_c_vals = dict(current_real_constraints)
                cand_real_violation = current_real_violation

                t_val, c_vals, h_val, ent_val = summarize_relaxed_state(
                    seqprop,
                    target_oracle,
                    constraint_oracles,
                    X_orig,
                    config.K,
                )
                pr = history["proj_ratio"][-1] if history["proj_ratio"] else grace_info["projection_ratio"]
            else:
                current_real_seq = cand_seq
                current_real_target = cand_target
                current_real_constraints = dict(cand_c_vals)
                current_real_violation = cand_real_violation

                t_val = y_target.item()
                c_vals = {o.name: y.item() for o, y in zip(constraint_oracles, y_constraints)}
                h_val = hamming.item()
                ent_val = seqprop.entropy().item()
                pr = grace_info["projection_ratio"]

                if (
                    use_grace
                    and constraint_oracles
                    and step >= phase_switch_step
                    and step > 0
                    and step % local_refine_every == 0
                ):
                    with torch.enable_grad():
                        projected_grad = compute_constrained_gradient_protein(
                            cand_seq,
                            target_oracle,
                            constraint_oracles,
                            device,
                            regularization=current_reg if use_grace else config.grace_reg,
                        )
                    local_candidates = propose_gradient_mutants_protein(
                        cand_seq,
                        projected_grad,
                        local_refine_top_positions,
                        seen_sequences=set(),
                    )
                    best_local = None
                    for local_seq, _meta in local_candidates:
                        local_oh = sequence_to_onehot(local_seq).unsqueeze(0).to(device)
                        local_target = target_oracle(local_oh).item()
                        local_c_vals = {o.name: o(local_oh).item() for o in constraint_oracles}
                        if not all(
                            is_within_tolerance(
                                local_c_vals[o.name],
                                real_c_initial[o.name],
                                config.constraint_eps,
                                config.constraint_type,
                            )
                            for o in constraint_oracles
                        ):
                            continue
                        if best_local is None or local_target > best_local[1]:
                            best_local = (local_seq, local_target, local_c_vals)

                    if best_local is not None and (
                        best_local[1] > cand_target + local_refine_min_gain or cand_real_violation > 0.0
                    ):
                        cand_seq, cand_target, cand_c_vals = best_local
                        current_real_seq = cand_seq
                        current_real_target = cand_target
                        current_real_constraints = dict(cand_c_vals)
                        current_real_violation = total_violation(
                            cand_c_vals,
                            real_c_initial,
                            config.constraint_eps,
                            config.constraint_type,
                        )
                        load_sequence_into_seqprop(seqprop, cand_seq)
                        optimizer.state.clear()
                        rejected = False
                        t_val = cand_target
                        c_vals = dict(cand_c_vals)
                        h_val = float(hamming_distance(sequence, cand_seq))
                        ent_val = 0.0
                        pr = 1.0

            # Update Lagrange multipliers using REAL decoded violation so the dual
            # variable responds to actual constraint satisfaction, not the soft
            # oracle proxy (which can be miscalibrated by 2x or more).
            if use_alm:
                for oracle in constraint_oracles:
                    real_viol = scalar_violation(
                        cand_c_vals[oracle.name],
                        real_c_initial[oracle.name],
                        config.constraint_eps,
                        config.constraint_type,
                    )
                    lambdas[oracle.name] = min(
                        config.lambda_max,
                        max(0.0, lambdas[oracle.name] + effective_alm_dual_lr * real_viol),
                    )

            rel_drifts = {
                o.name: abs(c_vals[o.name] - c_initial[o.name]) / (abs(c_initial[o.name]) + 1e-4)
                for o in all_oracles
            }

            history["scores"][target_oracle.name].append(t_val)
            history["real_scores"][target_oracle.name].append(cand_target)
            for oracle in all_oracles:
                history["scores"][oracle.name].append(c_vals[oracle.name])
                history["real_scores"][oracle.name].append(cand_c_vals[oracle.name])
                history["rel_drifts"][oracle.name].append(rel_drifts[oracle.name])
            for oracle in constraint_oracles:
                history["lambdas"][oracle.name].append(lambdas[oracle.name])
            history["hamming"].append(h_val)
            history["entropy"].append(ent_val)
            history["proj_ratio"].append(pr)
            history["decoded_sequences"].append(cand_seq)
            history["decoded_guard_rejected"].append(rejected)
            history["decoded_guard_violation"].append(cand_real_violation)

            if step % config.log_every == 0:
                c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in c_vals.items())
                lam_str = " ".join(f"λ_{k}={v:.2f}" for k, v in lambdas.items()) if lambdas else ""
                logger.info(
                    f"step={step:4d} | target={t_val:+.3f} | {c_str} | "
                    f"hamming={h_val:.1f} | entropy={ent_val:.3f} | pr={pr:.3f}"
                    + (f" | real_viol={cand_real_violation:.4f}" if constraint_oracles else "")
                    + (" | rejected=1" if rejected else "")
                    + (f" | {lam_str}" if lam_str else "")
                )

            if step % (config.log_every * 5) == 0:
                history["sequences"].append((step, cand_seq))

            if all(
                is_within_tolerance(
                    cand_c_vals[o.name],
                    real_c_initial[o.name],
                    config.constraint_eps,
                    config.constraint_type,
                )
                for o in constraint_oracles
            ):
                if cand_target > best_hard_feasible_score:
                    best_hard_feasible_score = cand_target
                    best_hard_feasible_sequence = cand_seq
                    best_hard_feasible_step = step

    logger.info(f"Decoding: sampling {config.n_decode_samples} sequences...")
    best_seq, best_score = sequence, float("-inf")
    best_sampled_sequence, best_sampled_score = None, float("-inf")
    best_sampled_feasible_sequence, best_sampled_feasible_score = None, float("-inf")
    best_sampled_feasible_step = None
    sampled_feasible_count = 0

    with torch.no_grad():
        for _ in range(config.n_decode_samples):
            cand = seqprop.decode("sample")
            cand_oh = sequence_to_onehot(cand).unsqueeze(0).to(device)
            score = target_oracle(cand_oh).item()

            if score > best_sampled_score:
                best_sampled_score = score
                best_sampled_sequence = cand
            if score > best_score:
                best_score = score
                best_seq = cand

            cand_c_vals = {o.name: o(cand_oh).item() for o in constraint_oracles}
            if all(
                is_within_tolerance(
                    cand_c_vals[o.name],
                    real_c_initial[o.name],
                    config.constraint_eps,
                    config.constraint_type,
                )
                for o in constraint_oracles
            ):
                sampled_feasible_count += 1
                if score > best_sampled_feasible_score:
                    best_sampled_feasible_score = score
                    best_sampled_feasible_sequence = cand
                    best_sampled_feasible_step = config.n_steps
                if score > best_hard_feasible_score:
                    best_hard_feasible_score = score
                    best_hard_feasible_sequence = cand
                    best_hard_feasible_step = config.n_steps

        argmax_seq = seqprop.decode("argmax")
        argmax_oh = sequence_to_onehot(argmax_seq).unsqueeze(0).to(device)
        argmax_score = target_oracle(argmax_oh).item()
        argmax_c_vals = {o.name: o(argmax_oh).item() for o in constraint_oracles}
        if argmax_score > best_score:
            best_score, best_seq = argmax_score, argmax_seq
        if all(
            is_within_tolerance(
                argmax_c_vals[o.name],
                real_c_initial[o.name],
                config.constraint_eps,
                config.constraint_type,
            )
            for o in constraint_oracles
        ):
            if argmax_score > best_hard_feasible_score:
                best_hard_feasible_score = argmax_score
                best_hard_feasible_sequence = argmax_seq
                best_hard_feasible_step = config.n_steps

    final_seq = best_sampled_feasible_sequence or best_hard_feasible_sequence or best_seq
    n_mutations = hamming_distance(sequence, final_seq)

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | Final target: {best_score:.4f}"
    )
    logger.info(
        f"Target {target_oracle.name}: {c_initial[target_oracle.name]:.3f} -> "
        f"{history['scores'][target_oracle.name][-1]:.3f} "
        f"(best decoded: {best_score:.4f})"
    )
    if best_hard_feasible_sequence is not None:
        logger.info(
            f"Best hard-feasible decoded sequence (step {best_hard_feasible_step}): "
            f"{best_hard_feasible_sequence} | score={best_hard_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_hard_feasible_sequence)}"
        )
    else:
        logger.info("Best hard-feasible decoded sequence: none found")
    for oracle in all_oracles:
        logger.info(
            f"Oracle {oracle.name}: initial={c_initial[oracle.name]:.3f} -> "
            f"{history['scores'][oracle.name][-1]:.3f}"
        )

    history.update({
        "original_sequence": sequence,
        "final_sequence": final_seq,
        "n_mutations": n_mutations,
        "target_oracle": target_oracle.name,
        "constraint_oracles": [o.name for o in constraint_oracles],
        "monitor_oracles": [],
        "final_lambdas": lambdas,
        "argmax_sequence": argmax_seq,
        "argmax_n_mutations": hamming_distance(sequence, argmax_seq),
        "best_sampled_sequence": best_sampled_sequence,
        "best_sampled_target_score": None if best_sampled_sequence is None else best_sampled_score,
        "best_sampled_n_mutations": None if best_sampled_sequence is None else hamming_distance(sequence, best_sampled_sequence),
        "best_sampled_feasible_sequence": best_sampled_feasible_sequence,
        "best_sampled_feasible_target_score": None if best_sampled_feasible_sequence is None else best_sampled_feasible_score,
        "best_sampled_feasible_step": best_sampled_feasible_step,
        "best_sampled_feasible_n_mutations": None if best_sampled_feasible_sequence is None else hamming_distance(sequence, best_sampled_feasible_sequence),
        "sampled_feasible_count": sampled_feasible_count,
        "n_decode_samples": config.n_decode_samples,
        "best_hard_feasible_sequence": best_hard_feasible_sequence,
        "best_hard_feasible_target_score": None if best_hard_feasible_sequence is None else best_hard_feasible_score,
        "best_hard_feasible_step": best_hard_feasible_step,
        "best_real_feasible_sequence": best_hard_feasible_sequence,
        "best_real_feasible_target_score": None if best_hard_feasible_sequence is None else best_hard_feasible_score,
        "best_real_feasible_step": best_hard_feasible_step,
        "best_real_feasible_n_mutations": None if best_hard_feasible_sequence is None else hamming_distance(sequence, best_hard_feasible_sequence),
    })
    return history


def build_kras_oracle(name: str, ckpt_map: Dict[str, str], backbone: ESM2SoftBackbone) -> nn.Module:
    if name.startswith("min:") or name.startswith("neg:"):
        _, base_name = name.split(":", 1)
        base = build_kras_oracle(base_name, ckpt_map, backbone)
        return NegatedOracle(base, name=f"decrease_{getattr(base, 'name', base_name)}")
    if name not in ckpt_map:
        raise ValueError(f"Missing checkpoint for oracle '{name}'. Available: {sorted(ckpt_map)}")
    return KRASOracle.load(ckpt_map[name], backbone, oracle_name=name)


def parse_args():
    p = argparse.ArgumentParser(description="KRAS real-score GRACE/ALM optimizer")
    p.add_argument("--sequence", required=True)
    p.add_argument("--target", required=True, help="Target oracle name, e.g. PI3 or RAF")
    p.add_argument("--constraints", nargs="*", default=[], help="Constraint oracle names")
    p.add_argument("--mode", default="grace_lagrangian", choices=MODES)
    p.add_argument("--ckpts", nargs="*", default=[], metavar="NAME:PATH")
    p.add_argument("--esm_model_ckpts", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hamming_lam", type=float, default=0.005)
    p.add_argument("--constraint_eps", type=float, default=0.1)
    p.add_argument("--constraint_type", default="absolute", choices=["absolute", "percent"])
    p.add_argument("--alm_rho", type=float, default=10.0)
    p.add_argument("--alm_dual_lr", type=float, default=0.02)
    p.add_argument("--alm_curriculum_start_steps", type=int, default=0)
    p.add_argument("--alm_curriculum_ramp_steps", type=int, default=0)
    p.add_argument("--lambda_max", type=float, default=100.0)
    p.add_argument("--lambda_init", type=float, default=0.0)
    p.add_argument("--no_grad_balance", action="store_true")
    p.add_argument("--decoded_constraint_guard", action="store_true")
    p.add_argument("--decoded_guard_tol", type=float, default=0.0)
    p.add_argument("--n_decode_samples", type=int, default=5000)
    p.add_argument("--sa_lambda_penalty", type=float, default=10.0)
    p.add_argument("--sa_temp_init", type=float, default=1.0)
    p.add_argument("--sa_temp_final", type=float, default=1e-2)
    p.add_argument("--sa_min_mutations", type=int, default=1)
    p.add_argument("--sa_max_mutations", type=int, default=3)
    p.add_argument("--bo_n_init", type=int, default=12)
    p.add_argument("--bo_candidate_pool", type=int, default=96)
    p.add_argument("--bo_refit_every", type=int, default=25)
    p.add_argument("--bo_train_window", type=int, default=48)
    p.add_argument("--bo_num_restarts", type=int, default=4)
    p.add_argument("--bo_raw_samples", type=int, default=64)
    p.add_argument("--bo_random_mutations_min", type=int, default=1)
    p.add_argument("--bo_random_mutations_max", type=int, default=4)
    p.add_argument("--adalead_budget", type=int, default=4000)
    p.add_argument("--adalead_population_size", type=int, default=128)
    p.add_argument("--adalead_num_leaders", type=int, default=16)
    p.add_argument("--adalead_mutants_per_leader", type=int, default=16)
    p.add_argument("--adalead_min_mutations", type=int, default=1)
    p.add_argument("--adalead_max_mutations", type=int, default=3)
    p.add_argument("--de_top_positions", type=int, default=3)
    p.add_argument("--de_beam_width", type=int, default=1)
    p.add_argument("--de_lambda_penalty", type=float, default=10.0)
    p.add_argument("--out", default="results/kras_real/optimized.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sequence = canonicalize_protein(args.sequence)
    ckpt_map: Dict[str, str] = {}
    for entry in args.ckpts:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--ckpts entries must be NAME:PATH, got '{entry}'")
        ckpt_map[parts[0]] = parts[1]

    backbone = ESM2SoftBackbone(model_name=args.esm_model_ckpts, freeze=True)
    target_oracle = build_kras_oracle(args.target, ckpt_map, backbone)
    constraint_oracles = [build_kras_oracle(name, ckpt_map, backbone) for name in (args.constraints or []) if name != "none"]

    logger.info(f"Mode: {args.mode} | Target: {target_oracle.name} | Constraints: {[o.name for o in constraint_oracles]}")

    config = OptimConfig(
        mode=args.mode,
        n_steps=args.steps,
        K=args.K,
        lr=args.lr,
        hamming_lambda=args.hamming_lam,
        constraint_eps=args.constraint_eps,
        constraint_type=args.constraint_type,
        alm_rho=args.alm_rho,
        alm_dual_lr=args.alm_dual_lr,
        alm_curriculum_start_steps=args.alm_curriculum_start_steps,
        alm_curriculum_ramp_steps=args.alm_curriculum_ramp_steps,
        lambda_max=args.lambda_max,
        lambda_init=args.lambda_init,
        grad_balance=not args.no_grad_balance,
        decoded_constraint_guard=args.decoded_constraint_guard,
        decoded_guard_tol=args.decoded_guard_tol,
        n_decode_samples=args.n_decode_samples,
        sa_lambda_penalty=args.sa_lambda_penalty,
        sa_temp_init=args.sa_temp_init,
        sa_temp_final=args.sa_temp_final,
        sa_min_mutations=args.sa_min_mutations,
        sa_max_mutations=args.sa_max_mutations,
        bo_n_init=args.bo_n_init,
        bo_candidate_pool=args.bo_candidate_pool,
        bo_refit_every=args.bo_refit_every,
        bo_train_window=args.bo_train_window,
        bo_num_restarts=args.bo_num_restarts,
        bo_raw_samples=args.bo_raw_samples,
        bo_random_mutations_min=args.bo_random_mutations_min,
        bo_random_mutations_max=args.bo_random_mutations_max,
        adalead_budget=args.adalead_budget,
        adalead_population_size=args.adalead_population_size,
        adalead_num_leaders=args.adalead_num_leaders,
        adalead_mutants_per_leader=args.adalead_mutants_per_leader,
        adalead_min_mutations=args.adalead_min_mutations,
        adalead_max_mutations=args.adalead_max_mutations,
        de_top_positions=args.de_top_positions,
        de_beam_width=args.de_beam_width,
        de_lambda_penalty=args.de_lambda_penalty,
        device=args.device,
    )

    if args.mode == "simulated_annealing":
        results = optimize_simulated_annealing(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    elif args.mode == "botorch_baseline":
        results = optimize_botorch_baseline(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    elif args.mode in ("directed_evolution", "adalead_baseline"):
        results = optimize_directed_evolution(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    else:
        results = optimize(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
