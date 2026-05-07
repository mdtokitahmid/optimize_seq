"""
optimize_dna.py — GRACE / ALM optimizer for cell-type-selective DNA enhancer design.

Goal: increase expression in a target cell type (hepG2 / k562 / sknsh)
      while keeping expression in the other two cell types near baseline.

Oracles: CNN regressors trained per cell type in Big_Oracles/DNA/cnn_dna_models/.
Backbone: No ESM — the DNA CNN takes (B, 4, L) one-hot directly.

Example:
  python optimize_dna.py \\
    --sequence ACGT... \\
    --target k562 \\
    --constraints hepG2 sknsh \\
    --mode grace_lagrangian \\
    --steps 500 --K 8 --lr 1e-2 \\
    --constraint_eps 0.5 \\
    --out data/results/dna/k562_vs_others/grace_lagrangian.json
"""

import argparse
import copy
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DNA_MODEL_DIR = DATA_ROOT / "Big_Oracles" / "DNA"
CODE_DNA_DIR = ROOT / "Big_Oracles" / "DNA"
if str(CODE_DNA_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DNA_DIR))

from train_cnn_dna import DNARegressor

from model.grace import compute_grace_direction, flatten_gradients

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES  = [
    "unconstrained",
    "grace_only",
    "lagrangian_only",
    "grace_lagrangian",
    "simulated_annealing",
    "botorch_baseline",
    "directed_evolution",
]
TASKS  = ["hepG2", "k562", "sknsh"]
DNA_ALPHABET = "ACGT"
DNA_TO_IDX   = {b: i for i, b in enumerate(DNA_ALPHABET)}
IDX_TO_DNA   = {i: b for i, b in enumerate(DNA_ALPHABET)}
NUM_BASES    = 4


# ── Sequence utilities ────────────────────────────────────────────────────────

def canonicalize(seq: str) -> str:
    seq = seq.upper()
    bad = [b for b in seq if b not in DNA_TO_IDX]
    if bad:
        raise ValueError(f"Unsupported bases: {sorted(set(bad))}")
    return seq

def sequence_to_onehot(seq: str) -> torch.Tensor:
    seq = canonicalize(seq)
    x = torch.zeros(NUM_BASES, len(seq), dtype=torch.float32)
    for i, b in enumerate(seq):
        x[DNA_TO_IDX[b], i] = 1.0
    return x

def onehot_to_sequence(x: torch.Tensor) -> str:
    """x: (4, L) — argmax decoding."""
    return "".join(IDX_TO_DNA[i.item()] for i in x.argmax(dim=0))


def flat_onehot_to_sequence(x: torch.Tensor) -> str:
    """x: (4 * L,) or (4, L) continuous encoding -> argmax DNA sequence."""
    if x.ndim == 1:
        x = x.view(NUM_BASES, -1)
    return "".join(IDX_TO_DNA[i.item()] for i in x.argmax(dim=0))


def sequence_to_flat_onehot(seq: str) -> torch.Tensor:
    return sequence_to_onehot(seq).reshape(-1)

def expected_hamming(P: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
    return (1.0 - (P * X_orig).sum(dim=0)).sum()

def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def propose_point_mutation(seq: str, n_mutations: int = 1) -> str:
    n_mutations = max(1, min(n_mutations, len(seq)))
    chars = list(seq)
    for idx in random.sample(range(len(seq)), k=n_mutations):
        current = chars[idx]
        choices = [b for b in DNA_ALPHABET if b != current]
        chars[idx] = random.choice(choices)
    return "".join(chars)


# ── SeqProp for DNA ───────────────────────────────────────────────────────────

class DNASeqProp(nn.Module):
    """Soft differentiable DNA sequence parameterization (straight-through)."""

    def __init__(self, L: int, init_sequence: str = None, init_logit_scale: float = 4.0):
        super().__init__()
        self.L = L
        logits = torch.zeros(NUM_BASES, L)
        if init_sequence is not None:
            seq = canonicalize(init_sequence)
            assert len(seq) == L, f"init_sequence length {len(seq)} != L={L}"
            for i, b in enumerate(seq):
                logits[DNA_TO_IDX[b], i] = init_logit_scale
        logits += 0.01 * torch.randn_like(logits)
        self.logits = nn.Parameter(logits)
        self.gamma  = nn.Parameter(torch.ones(1))
        self.beta   = nn.Parameter(torch.zeros(NUM_BASES))
        self._inst_norm = nn.InstanceNorm1d(NUM_BASES, affine=False, eps=1e-5)

    def probabilities(self, tau: float = 1.0) -> torch.Tensor:
        z = self._inst_norm(self.logits.unsqueeze(0)).squeeze(0)
        return F.softmax((self.gamma * z + self.beta.unsqueeze(1)) / tau, dim=0)

    def st_sample(self, K: int, tau: float = 1.0):
        P = self.probabilities(tau=tau)
        samples = []
        for _ in range(K):
            c    = torch.multinomial(P.T, num_samples=1).squeeze(1)
            hard = F.one_hot(c, num_classes=NUM_BASES).T.float()
            samples.append(hard + (P - P.detach()))   # straight-through
        return samples, P

    def decode(self, method: str = "argmax") -> str:
        P = self.probabilities()
        if method == "argmax":
            idx = P.argmax(dim=0)
        elif method == "sample":
            idx = torch.multinomial(P.T, 1).squeeze(1)
        else:
            raise ValueError(method)
        return "".join(IDX_TO_DNA[i.item()] for i in idx)

    def entropy(self) -> torch.Tensor:
        P = self.probabilities()
        return -(P * (P + 1e-9).log()).sum(dim=0).mean()


# ── DNA Oracle ────────────────────────────────────────────────────────────────

class DNAOracle(nn.Module):
    """Wraps a trained DNARegressor + denormalization."""

    def __init__(self, name: str, model: DNARegressor, mean: float, std: float):
        super().__init__()
        self.name  = name
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, L) — returns (B,) raw score."""
        return self.model(x) * self.std + self.mean

    @classmethod
    def load(cls, task: str, model_dir: Path) -> "DNAOracle":
        ckpt_path = model_dir / task / "best_model.pt"
        norm_path = model_dir / task / "normalization.npy"
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DNARegressor()
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        norm = np.load(norm_path)
        mean, std = float(norm[0]), float(norm[1])
        logger.info(f"[DNAOracle:{task}] loaded from {ckpt_path}  "
                    f"(val spearman={ckpt.get('metrics', {}).get('spearman', float('nan')):.4f})")
        return cls(task, model, mean, std)


# ── Violation ─────────────────────────────────────────────────────────────────

def compute_violation(y_c: torch.Tensor, c_0: float, eps: float,
                      constraint_type: str = "absolute") -> torch.Tensor:
    """
    Compute constraint violation as a non-negative scalar.
      absolute: relu(|y_c - c_0| - eps)
      percent:  relu(|y_c - c_0| / (|c_0| + 1e-4) - eps)   [eps is a fraction, e.g. 0.05 for 5%]
    """
    abs_drift = (y_c - c_0).abs()
    if constraint_type == "percent":
        return F.relu(abs_drift / (abs(c_0) + 1e-4) - eps)
    return F.relu(abs_drift - eps)


def is_within_tolerance(val: float, c_0: float, eps: float,
                        constraint_type: str = "absolute") -> bool:
    """Scalar feasibility check matching compute_violation."""
    abs_drift = abs(val - c_0)
    if constraint_type == "percent":
        return abs_drift / (abs(c_0) + 1e-4) <= eps
    return abs_drift <= eps


def scalar_violation(val: float, c_0: float, eps: float,
                     constraint_type: str = "absolute") -> float:
    abs_drift = abs(val - c_0)
    if constraint_type == "percent":
        return max(0.0, abs_drift / (abs(c_0) + 1e-4) - eps)
    return max(0.0, abs_drift - eps)


def total_violation(vals: Dict[str, float], c_initial: Dict[str, float], eps: float,
                    constraint_type: str = "absolute") -> float:
    return sum(
        scalar_violation(vals[name], c_initial[name], eps, constraint_type)
        for name in vals
    )


def summarize_relaxed_state(
    seqprop: "DNASeqProp",
    target_oracle: "DNAOracle",
    constraint_oracles: List["DNAOracle"],
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


# ── Optimizer ─────────────────────────────────────────────────────────────────

@dataclass
class OptimConfig:
    mode:            str   = "grace_lagrangian"
    n_steps:         int   = 500
    K:               int   = 8
    lr:              float = 1e-2
    hamming_lambda:  float = 0.05
    constraint_eps:  float = 0.05
    constraint_type: str  = "percent"   # "absolute" | "percent"
    alm_rho:         float = 10.0
    alm_dual_lr:     float = 0.02
    lambda_max:      float = 100.0
    lambda_init:     float = 0.0
    grad_balance:    bool  = True
    grace_reg:       float = 1e-6
    log_every:       int   = 10
    n_decode_samples: int  = 200
    decoded_constraint_guard: bool = False
    decoded_guard_tol: float = 0.0
    sa_lambda_penalty: float = 10.0
    sa_temp_init: float = 1.0
    sa_temp_final: float = 1e-2
    sa_min_mutations: int = 1
    sa_max_mutations: int = 3
    bo_n_init: int = 16
    bo_num_restarts: int = 10
    bo_raw_samples: int = 256
    bo_candidate_pool: int = 256
    bo_refit_every: int = 10
    bo_train_window: int = 128
    bo_random_mutations_min: int = 1
    bo_random_mutations_max: int = 3
    de_top_positions: int = 3
    de_beam_width: int = 1
    de_lambda_penalty: float = 10.0
    device:          str   = "cpu"


def annealing_temperature(step: int, n_steps: int, temp_init: float, temp_final: float) -> float:
    if n_steps <= 1:
        return temp_final
    frac = step / (n_steps - 1)
    return temp_init * ((temp_final / temp_init) ** frac)


@torch.no_grad()
def score_dna_sequence(
    seq: str,
    target_oracle: DNAOracle,
    constraint_oracles: List[DNAOracle],
    device: str,
) -> tuple[float, Dict[str, float]]:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    target = float(target_oracle(x).item())
    constraints = {oracle.name: float(oracle(x).item()) for oracle in constraint_oracles}
    return target, constraints


def compute_target_gradient(
    seq: str,
    target_oracle: DNAOracle,
    device: str,
) -> torch.Tensor:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    x.requires_grad_(True)
    target = target_oracle(x)
    target.sum().backward()
    return x.grad.detach().squeeze(0).cpu()


def propose_gradient_mutants(
    seq: str,
    gradient: torch.Tensor,
    top_positions: int,
    seen_sequences: set[str],
) -> list[tuple[str, Dict[str, Any]]]:
    seq = canonicalize(seq)
    current_idx = [DNA_TO_IDX[b] for b in seq]
    ranked_mutations: list[tuple[float, int, int]] = []
    for pos, cur in enumerate(current_idx):
        for alt in range(NUM_BASES):
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
        chars[pos] = IDX_TO_DNA[alt]
        cand = "".join(chars)
        if cand in seen_sequences:
            continue
        proposals.append(
            (
                cand,
                {
                    "position": pos,
                    "from": seq[pos],
                    "to": IDX_TO_DNA[alt],
                    "delta": delta,
                },
            )
        )
        used_positions.add(pos)
        if len(proposals) >= top_positions:
            break
    return proposals


def optimize_simulated_annealing(
    sequence: str,
    target_oracle: DNAOracle,
    constraint_oracles: List[DNAOracle],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize(sequence)
    L = len(sequence)

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    current_seq = sequence
    current_target, current_constraints = score_dna_sequence(
        current_seq, target_oracle, constraint_oracles, device
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
        + f" | hamming=0.0 | T={config.sa_temp_init:.4f} | accepted=1"
    )

    for step in range(1, config.n_steps + 1):
        temperature = annealing_temperature(
            step, config.n_steps, config.sa_temp_init, config.sa_temp_final
        )
        proposal_mutations = random.randint(config.sa_min_mutations, config.sa_max_mutations)
        cand_seq = propose_point_mutation(current_seq, n_mutations=proposal_mutations)
        cand_target, cand_constraints = score_dna_sequence(
            cand_seq, target_oracle, constraint_oracles, device
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
        **score_dna_sequence(final_seq, target_oracle, constraint_oracles, device)[1]
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
    target_oracle: DNAOracle,
    constraint_oracles: List[DNAOracle],
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
    sequence = canonicalize(sequence)
    L = len(sequence)
    dim = NUM_BASES * L

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    start_target, start_constraints = score_dna_sequence(
        sequence, target_oracle, constraint_oracles, device
    )
    c_initial = {target_oracle.name
    : start_target, **start_constraints}
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

    train_X: List[torch.Tensor] = [sequence_to_flat_onehot(sequence).to(dtype=torch.double)]
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
    gp_model = None
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

    def random_novel_sequence() -> str:
        max_attempts = 64
        for _ in range(max_attempts):
            n_mut = random.randint(config.bo_random_mutations_min, config.bo_random_mutations_max)
            cand = propose_point_mutation(sequence, n_mutations=n_mut)
            if cand not in seen_sequences:
                return cand
        for _ in range(max_attempts):
            cand = propose_point_mutation(best_overall_sequence, n_mutations=1)
            if cand not in seen_sequences:
                return cand
        return sequence

    def fit_botorch_model() -> tuple[Optional[Any], Optional[Any]]:
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
            return None, None

        feasible_targets = []
        for row in train_Y[-window:]:
            row_constraints = {
                o.name: row[1 + idx]
                for idx, o in enumerate(all_oracles)
            }
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
            acquisition = qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                objective=objective,
                constraints=constraints or None,
            )
        except Exception as exc:
            logger.warning(f"BoTorch acquisition construction failed, falling back to random proposal: {exc}")
            return None, None
        return model, acquisition

    def sample_candidate_pool() -> list[str]:
        pool: list[str] = []
        anchors = [sequence]
        if best_overall_sequence is not None:
            anchors.append(best_overall_sequence)
        if best_feasible_sequence is not None:
            anchors.append(best_feasible_sequence)

        max_attempts = max(config.bo_candidate_pool * 8, 256)
        attempts = 0
        while len(pool) < config.bo_candidate_pool and attempts < max_attempts:
            attempts += 1
            anchor = random.choice(anchors)
            if random.random() < 0.15:
                n_mut = max(config.bo_random_mutations_max, 4)
            else:
                n_mut = random.randint(config.bo_random_mutations_min, config.bo_random_mutations_max)
            cand = propose_point_mutation(anchor, n_mutations=n_mut)
            if cand not in seen_sequences and cand not in pool:
                pool.append(cand)
        if not pool:
            pool.append(random_novel_sequence())
        return pool

    def suggest_sequence_via_bo(acquisition: Any) -> Optional[str]:
        pool = sample_candidate_pool()
        pool_X = torch.stack(
            [sequence_to_flat_onehot(seq).to(dtype=torch.double) for seq in pool],
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
                gp_model, acqf = fit_botorch_model()
                last_refit_step = step
            cand_seq = suggest_sequence_via_bo(acqf) if acqf is not None else None
            source = "botorch_pool"
            if cand_seq is None or cand_seq in seen_sequences:
                cand_seq = random_novel_sequence()
                source = "random_fallback"

        cand_target, cand_constraints = score_dna_sequence(
            cand_seq, target_oracle, constraint_oracles, device
        )
        seen_sequences.add(cand_seq)
        train_X.append(sequence_to_flat_onehot(cand_seq).to(dtype=torch.double))
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
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [],
            "final_lambdas": {oracle.name: 0.0 for oracle in constraint_oracles},
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
    target_oracle: DNAOracle,
    constraint_oracles: List[DNAOracle],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize(sequence)
    L = len(sequence)

    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    start_target, start_constraints = score_dna_sequence(
        sequence, target_oracle, constraint_oracles, device
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

    def objective(target: float, constraints: Dict[str, float]) -> float:
        violation = total_violation(
            constraints,
            real_c_initial,
            config.constraint_eps,
            config.constraint_type,
        )
        return target - config.de_lambda_penalty * violation

    start_objective = objective(start_target, start_constraints)
    seen_sequences = {sequence}

    best_objective_sequence = sequence
    best_objective_value = start_objective
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

    beam = [
        {
            "seq": sequence,
            "target": start_target,
            "constraints": start_constraints,
            "objective": start_objective,
        }
    ]

    logger.info(
        f"Directed evolution baseline: beam_width={config.de_beam_width}, "
        f"top_positions={config.de_top_positions}, lambda_penalty={config.de_lambda_penalty:.3f}. "
        f"Each real candidate evaluation counts as one optimization step."
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

    step = 0
    while step < config.n_steps:
        offspring: list[Dict[str, Any]] = []
        for parent in beam:
            if step >= config.n_steps:
                break
            grad = compute_target_gradient(parent["seq"], target_oracle, device)
            proposals = propose_gradient_mutants(
                parent["seq"], grad, config.de_top_positions, seen_sequences
            )
            if not proposals:
                for _ in range(config.de_top_positions):
                    cand = propose_point_mutation(parent["seq"], n_mutations=1)
                    if cand not in seen_sequences:
                        proposals.append((cand, {"position": None}))
                    if len(proposals) >= config.de_top_positions:
                        break

            for cand_seq, meta in proposals:
                if step >= config.n_steps:
                    break
                cand_target, cand_constraints = score_dna_sequence(
                    cand_seq, target_oracle, constraint_oracles, device
                )
                cand_objective = objective(cand_target, cand_constraints)
                step += 1
                seen_sequences.add(cand_seq)
                append_history(
                    step,
                    cand_seq,
                    cand_target,
                    cand_constraints,
                    "grad_pos" if meta.get("position") is not None else "random_fallback",
                )
                offspring.append(
                    {
                        "seq": cand_seq,
                        "target": cand_target,
                        "constraints": cand_constraints,
                        "objective": cand_objective,
                    }
                )

                if cand_objective > best_objective_value:
                    best_objective_sequence = cand_seq
                    best_objective_value = cand_objective
                    best_objective_score = cand_target
                    best_objective_step = step

                if all(
                    is_within_tolerance(
                        cand_constraints[o.name], real_c_initial[o.name], config.constraint_eps, config.constraint_type
                    )
                    for o in constraint_oracles
                ) and cand_target > best_feasible_score:
                    best_feasible_sequence = cand_seq
                    best_feasible_score = cand_target
                    best_feasible_step = step

        pool = beam + offspring
        pool.sort(key=lambda item: item["objective"], reverse=True)
        next_beam: list[Dict[str, Any]] = []
        used: set[str] = set()
        for item in pool:
            if item["seq"] in used:
                continue
            next_beam.append(item)
            used.add(item["seq"])
            if len(next_beam) >= config.de_beam_width:
                break
        beam = next_beam or beam

    final_seq = best_feasible_sequence or best_objective_sequence
    n_mutations = hamming_distance(sequence, final_seq)

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | "
        f"Best objective: {best_objective_value:.4f}"
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
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [],
            "final_lambdas": {oracle.name: 0.0 for oracle in constraint_oracles},
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
        }
    )
    return history


def optimize(
    sequence:           str,
    target_oracle:      DNAOracle,
    constraint_oracles: List[DNAOracle],
    config:             OptimConfig,
) -> Dict[str, Any]:
    device    = config.device
    sequence  = canonicalize(sequence)
    L         = len(sequence)
    use_grace = config.mode in ("grace_only", "grace_lagrangian")
    use_alm   = config.mode in ("lagrangian_only", "grace_lagrangian")

    seqprop   = DNASeqProp(L=L, init_sequence=sequence).to(device)
    target_oracle = target_oracle.to(device).eval()   # eval mode: use running BatchNorm stats
    all_oracles   = list(constraint_oracles)
    for o in all_oracles:
        o.to(device).eval()                           # eval mode: consistent with re-evaluation

    X_orig    = sequence_to_onehot(sequence).to(device)
    optimizer = torch.optim.Adam(seqprop.parameters(), lr=config.lr)
    all_params = list(seqprop.parameters())

    # Initial values — scored on the DISCRETE start sequence in eval mode.
    # This matches exactly what the re-evaluation script computes, so
    # feasibility tracked here (|discrete_cand - c_initial| <= eps) will
    # be consistent with post-hoc evaluation.
    with torch.no_grad():
        X_start   = sequence_to_onehot(sequence).unsqueeze(0).to(device)
        c_initial = {o.name: o(X_start).item() for o in [target_oracle] + all_oracles}
    logger.info(f"Initial oracle values (discrete start, eval mode): {c_initial}")

    lambdas = {o.name: config.lambda_init for o in constraint_oracles}
    real_c_initial = dict(c_initial)
    current_real_seq = sequence
    current_real_target = c_initial[target_oracle.name]
    current_real_constraints = {o.name: c_initial[o.name] for o in constraint_oracles}
    current_real_violation = total_violation(
        current_real_constraints,
        real_c_initial,
        config.constraint_eps,
        config.constraint_type,
    )
    all_names = [target_oracle.name] + [o.name for o in all_oracles]
    history: Dict[str, Any] = {
        "scores":     {n: [] for n in all_names},
        "real_scores": {n: [] for n in all_names},
        "rel_drifts": {o.name: [] for o in all_oracles},
        "lambdas":    {o.name: [] for o in constraint_oracles},
        "hamming":    [],
        "entropy":    [],
        "proj_ratio": [],
        "sequences":       [],
        "decoded_sequences": [],
        "decoded_guard_rejected": [],
        "decoded_guard_violation": [],
        "constraint_eps":  config.constraint_eps,
        "constraint_type": config.constraint_type,
        "c_initial":  c_initial,
        "real_c_initial": real_c_initial,
        "mode":       config.mode,
    }
    history["real_scores"][target_oracle.name].append(current_real_target)
    for o in all_oracles:
        history["real_scores"][o.name].append(real_c_initial[o.name])

    best_hard_feasible_score    = float("-inf")
    best_hard_feasible_sequence = None
    best_hard_feasible_step     = None
    tau_start = 1.0
    tau_end = 0.1

    for step in range(config.n_steps):
        optimizer.zero_grad()

        param_snapshot = None
        optimizer_snapshot = None
        if config.decoded_constraint_guard and constraint_oracles:
            param_snapshot = [p.detach().clone() for p in all_params]
            optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        current_tau = tau_start * (tau_end / tau_start) ** (step / max(1, config.n_steps - 1))
        samples, P = seqprop.st_sample(config.K, tau=current_tau)
        batch      = torch.stack(samples, dim=0)

        y_target      = target_oracle(batch).mean()
        y_constraints = [o(batch).mean() for o in constraint_oracles]
        hamming       = expected_hamming(P, X_orig)

        violations = [
            compute_violation(y_c, c_initial[o.name], config.constraint_eps,
                              config.constraint_type)
            for o, y_c in zip(constraint_oracles, y_constraints)
        ]

        # Target gradient
        g_target = flatten_gradients(
            torch.autograd.grad(y_target, all_params,
                                retain_graph=True, create_graph=False, allow_unused=True),
            all_params,
        )

        # GRACE projection
        if use_grace and constraint_oracles:
            g_constraints_flat = [
                flatten_gradients(
                    torch.autograd.grad(y_c, all_params,
                                        retain_graph=True, create_graph=False, allow_unused=True),
                    all_params,
                )
                for y_c in y_constraints
            ]
            d_star, grace_info = compute_grace_direction(
                g_target=g_target,
                g_constraints=g_constraints_flat,
                regularization=config.grace_reg,
            )
        else:
            d_star     = g_target
            grace_info = {"projection_ratio": 1.0}

        # ALM gradient
        g_alm = None
        if use_alm:
            alm_penalty = torch.tensor(0.0, device=device)
            for o, viol in zip(constraint_oracles, violations):
                lam = lambdas[o.name]
                alm_penalty = alm_penalty + lam * viol + (config.alm_rho / 2) * viol ** 2
            if alm_penalty.grad_fn is not None:
                g_alm_tuple = torch.autograd.grad(
                    alm_penalty, all_params,
                    retain_graph=True, create_graph=False, allow_unused=True,
                )
                g_alm = flatten_gradients(g_alm_tuple, all_params)

        # Hamming gradient
        g_hamming = flatten_gradients(
            torch.autograd.grad(
                config.hamming_lambda * hamming, all_params,
                retain_graph=False, create_graph=False, allow_unused=True,
            ),
            all_params,
        )

        # Scale ALM (only down, never inflate small violations)
        if g_alm is not None and config.grad_balance:
            dir_norm  = d_star.norm().clamp(min=1e-8)
            alm_norm  = g_alm.norm().clamp(min=1e-8)
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

        # Dual update
        if use_alm:
            with torch.no_grad():
                for o, viol in zip(constraint_oracles, violations):
                    lambdas[o.name] = min(
                        config.lambda_max,
                        max(0.0, lambdas[o.name] + config.alm_dual_lr * viol.item())
                    )

        # Logging
        with torch.no_grad():
            cand_seq    = seqprop.decode("argmax")
            cand_oh     = sequence_to_onehot(cand_seq).unsqueeze(0).to(device)
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

                t_val  = y_target.item()
                c_vals = {o.name: y.item() for o, y in zip(constraint_oracles, y_constraints)}
                h_val   = hamming.item()
                ent_val = seqprop.entropy().item()
                pr      = grace_info["projection_ratio"]

            rel_drifts = {
                o.name: abs(c_vals[o.name] - c_initial[o.name]) / (abs(c_initial[o.name]) + 1e-4)
                for o in all_oracles
            }

            history["scores"][target_oracle.name].append(t_val)
            history["real_scores"][target_oracle.name].append(cand_target)
            for o in all_oracles:
                history["scores"][o.name].append(c_vals[o.name])
                history["real_scores"][o.name].append(cand_c_vals[o.name])
                history["rel_drifts"][o.name].append(rel_drifts[o.name])
            for o in constraint_oracles:
                history["lambdas"][o.name].append(lambdas[o.name])
            history["hamming"].append(h_val)
            history["entropy"].append(ent_val)
            history["proj_ratio"].append(pr)
            history["decoded_sequences"].append(cand_seq)
            history["decoded_guard_rejected"].append(rejected)
            history["decoded_guard_violation"].append(cand_real_violation)

            if step % config.log_every == 0:
                c_str = "  ".join(
                    f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in c_vals.items()
                )
                lam_str = " ".join(f"λ_{k}={v:.1f}" for k, v in lambdas.items()) if lambdas else ""
                logger.info(
                    f"step={step:4d} | target={t_val:+.3f} | {c_str} | "
                    f"hamming={h_val:.1f} | pr={pr:.3f}"
                    + (f" | real_viol={cand_real_violation:.4f}" if constraint_oracles else "")
                    + (" | rejected=1" if rejected else "")
                    + (f" | {lam_str}" if lam_str else "")
                )

            if step % (config.log_every * 5) == 0:
                history["sequences"].append((step, cand_seq))

            if all(is_within_tolerance(cand_c_vals[o.name], real_c_initial[o.name],
                                       config.constraint_eps, config.constraint_type)
                   for o in constraint_oracles):
                if cand_target > best_hard_feasible_score:
                    best_hard_feasible_score    = cand_target
                    best_hard_feasible_sequence = cand_seq
                    best_hard_feasible_step     = step

    # Final decoding: best-of-N sampling
    logger.info(f"Decoding: sampling {config.n_decode_samples} sequences...")
    best_seq, best_score = sequence, float("-inf")
    best_sampled_sequence, best_sampled_score = None, float("-inf")
    best_sampled_feasible_sequence, best_sampled_feasible_score = None, float("-inf")
    best_sampled_feasible_step = None
    sampled_feasible_count = 0
    with torch.no_grad():
        for _ in range(config.n_decode_samples):
            cand    = seqprop.decode("sample")
            cand_oh = sequence_to_onehot(cand).unsqueeze(0).to(device)
            score   = target_oracle(cand_oh).item()
            if score > best_sampled_score:
                best_sampled_score = score
                best_sampled_sequence = cand
            if score > best_score:
                best_score = score
                best_seq   = cand
            cand_c_vals = {o.name: o(cand_oh).item() for o in constraint_oracles}
            if all(is_within_tolerance(cand_c_vals[o.name], real_c_initial[o.name],
                                       config.constraint_eps, config.constraint_type)
                   for o in constraint_oracles):
                sampled_feasible_count += 1
                if score > best_sampled_feasible_score:
                    best_sampled_feasible_score = score
                    best_sampled_feasible_sequence = cand
                    best_sampled_feasible_step = config.n_steps
                if score > best_hard_feasible_score:
                    best_hard_feasible_score    = score
                    best_hard_feasible_sequence = cand
                    best_hard_feasible_step     = config.n_steps

        # Also consider argmax
        argmax_seq    = seqprop.decode("argmax")
        argmax_oh     = sequence_to_onehot(argmax_seq).unsqueeze(0).to(device)
        argmax_score  = target_oracle(argmax_oh).item()
        argmax_c_vals = {o.name: o(argmax_oh).item() for o in constraint_oracles}
        if argmax_score > best_score:
            best_score, best_seq = argmax_score, argmax_seq
        if all(is_within_tolerance(argmax_c_vals[o.name], real_c_initial[o.name],
                                   config.constraint_eps, config.constraint_type)
               for o in constraint_oracles):
            if argmax_score > best_hard_feasible_score:
                best_hard_feasible_score    = argmax_score
                best_hard_feasible_sequence = argmax_seq
                best_hard_feasible_step     = config.n_steps

    final_seq = best_sampled_feasible_sequence or best_hard_feasible_sequence or best_seq
    n_mutations = hamming_distance(sequence, final_seq)
    logger.info(f"Mode: {config.mode} | Mutations: {n_mutations}/{L} | "
                f"Final target: {best_score:.4f}")
    for o in [target_oracle] + all_oracles:
        logger.info(f"  {o.name}: {c_initial[o.name]:.3f} → "
                    f"{history['scores'][o.name][-1]:.3f}")

    history.update({
        "original_sequence":           sequence,
        "final_sequence":              final_seq,
        "n_mutations":                 n_mutations,
        "target_oracle":               target_oracle.name,
        "constraint_oracles":          [o.name for o in constraint_oracles],
        "monitor_oracles":             [],
        "final_lambdas":               lambdas,
        "argmax_sequence":             argmax_seq,
        "argmax_n_mutations":          hamming_distance(sequence, argmax_seq),
        "best_sampled_sequence":       best_sampled_sequence,
        "best_sampled_target_score":   None if best_sampled_sequence is None else best_sampled_score,
        "best_sampled_n_mutations":    None if best_sampled_sequence is None else hamming_distance(sequence, best_sampled_sequence),
        "best_sampled_feasible_sequence": best_sampled_feasible_sequence,
        "best_sampled_feasible_target_score": None if best_sampled_feasible_sequence is None else best_sampled_feasible_score,
        "best_sampled_feasible_step":  best_sampled_feasible_step,
        "best_sampled_feasible_n_mutations": None if best_sampled_feasible_sequence is None else hamming_distance(sequence, best_sampled_feasible_sequence),
        "sampled_feasible_count":      sampled_feasible_count,
        "n_decode_samples":            config.n_decode_samples,
        "best_hard_feasible_sequence": best_hard_feasible_sequence,
        "best_hard_feasible_target_score": None if best_hard_feasible_sequence is None
                                           else best_hard_feasible_score,
        "best_hard_feasible_step":     best_hard_feasible_step,
        "best_real_feasible_sequence": best_hard_feasible_sequence,
        "best_real_feasible_target_score": None if best_hard_feasible_sequence is None
                                           else best_hard_feasible_score,
        "best_real_feasible_step":     best_hard_feasible_step,
        "best_real_feasible_n_mutations": None if best_hard_feasible_sequence is None
                                          else hamming_distance(sequence, best_hard_feasible_sequence),
    })
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRACE + ALM optimizer for cell-type-selective DNA design")
    p.add_argument("--sequence",       required=True)
    p.add_argument("--target",         required=True, choices=TASKS)
    p.add_argument("--constraints",    nargs="*", default=[],
                   help="Cell types to keep constrained (e.g. hepG2 sknsh)")
    p.add_argument("--mode",           default="grace_lagrangian", choices=MODES)
    p.add_argument("--model_dir",      default=str(DATA_ROOT / "Big_Oracles" / "DNA" / "cnn_dna_models"))
    p.add_argument("--steps",          type=int,   default=500)
    p.add_argument("--K",              type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-2)
    p.add_argument("--hamming_lam",    type=float, default=0.05)
    p.add_argument("--constraint_eps",  type=float, default=0.05,
                   help="Constraint drift tolerance (fraction if percent, raw units if absolute)")
    p.add_argument("--constraint_type", default="percent", choices=["absolute", "percent"],
                   help="How constraint_eps is interpreted: 'percent' (e.g. 0.05=5%%) or 'absolute'")
    p.add_argument("--alm_rho",        type=float, default=10.0)
    p.add_argument("--alm_dual_lr",    type=float, default=0.02)
    p.add_argument("--lambda_max",     type=float, default=100.0)
    p.add_argument("--sa_lambda_penalty", type=float, default=10.0)
    p.add_argument("--sa_temp_init",   type=float, default=1.0)
    p.add_argument("--sa_temp_final",  type=float, default=1e-2)
    p.add_argument("--sa_min_mutations", type=int, default=1)
    p.add_argument("--sa_max_mutations", type=int, default=3)
    p.add_argument("--bo_n_init",      type=int, default=16)
    p.add_argument("--bo_num_restarts", type=int, default=10)
    p.add_argument("--bo_raw_samples", type=int, default=256)
    p.add_argument("--bo_candidate_pool", type=int, default=256)
    p.add_argument("--bo_refit_every", type=int, default=10)
    p.add_argument("--bo_train_window", type=int, default=128)
    p.add_argument("--bo_random_mutations_min", type=int, default=1)
    p.add_argument("--bo_random_mutations_max", type=int, default=3)
    p.add_argument("--de_top_positions", type=int, default=3)
    p.add_argument("--de_beam_width", type=int, default=1)
    p.add_argument("--de_lambda_penalty", type=float, default=10.0)
    p.add_argument("--decoded_constraint_guard", action="store_true",
                   help="Reject updates that worsen decoded-sequence real constraint violation.")
    p.add_argument("--decoded_guard_tol", type=float, default=0.0,
                   help="Allow this much increase in decoded violation before rejecting a step.")
    p.add_argument("--n_decode_samples", type=int, default=200)
    p.add_argument("--out",            default="results/dna/optimized.json")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Mode: {args.mode} | Target: {args.target} | Constraints: {args.constraints}")

    model_dir = Path(args.model_dir)
    target_oracle      = DNAOracle.load(args.target, model_dir)
    constraint_oracles = [DNAOracle.load(t, model_dir) for t in (args.constraints or [])]

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
        lambda_max=args.lambda_max,
        sa_lambda_penalty=args.sa_lambda_penalty,
        sa_temp_init=args.sa_temp_init,
        sa_temp_final=args.sa_temp_final,
        sa_min_mutations=args.sa_min_mutations,
        sa_max_mutations=args.sa_max_mutations,
        bo_n_init=args.bo_n_init,
        bo_num_restarts=args.bo_num_restarts,
        bo_raw_samples=args.bo_raw_samples,
        bo_candidate_pool=args.bo_candidate_pool,
        bo_refit_every=args.bo_refit_every,
        bo_train_window=args.bo_train_window,
        bo_random_mutations_min=args.bo_random_mutations_min,
        bo_random_mutations_max=args.bo_random_mutations_max,
        de_top_positions=args.de_top_positions,
        de_beam_width=args.de_beam_width,
        de_lambda_penalty=args.de_lambda_penalty,
        decoded_constraint_guard=args.decoded_constraint_guard,
        decoded_guard_tol=args.decoded_guard_tol,
        n_decode_samples=args.n_decode_samples,
        device=args.device,
    )

    if args.mode == "simulated_annealing":
        results = optimize_simulated_annealing(
            sequence=args.sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    elif args.mode == "botorch_baseline":
        results = optimize_botorch_baseline(
            sequence=args.sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    elif args.mode == "directed_evolution":
        results = optimize_directed_evolution(
            sequence=args.sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )
    else:
        results = optimize(
            sequence=args.sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            config=config,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
