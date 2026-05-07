"""
optimize_final_real.py — generic real-score protein optimizer with GRACE / ALM / SA.

This is the real decoded-sequence counterpart of optimize_final.py. It keeps the
soft-sequence GRACE / ALM optimization, but additionally:

1. scores the discrete starting sequence in eval mode,
2. tracks real decoded-sequence oracle values every step,
3. optionally rejects updates that worsen decoded real constraint violation,
4. saves best feasible decoded sequences and best feasible sampled sequences,
5. supports simulated annealing as an ablation baseline.

Checkpoint types are auto-detected from --ckpts exactly like optimize_final.py:
  - ThermostabilityOracle
  - KRASOracle
  - FineTunedESMOracle
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
from optimize_final_kras import optimize_botorch_baseline, optimize_directed_evolution
from oracles.esm2_backbone import ESM2SoftBackbone
from oracles.finetuned import FineTunedESMOracle
from oracles.kras import KRASOracle
from oracles.thermostability import ThermostabilityOracle
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
    de_top_positions: int = 3
    de_beam_width: int = 1
    de_lambda_penalty: float = 10.0
    adalead_budget: int = 4000
    adalead_population_size: int = 128
    adalead_num_leaders: int = 16
    adalead_mutants_per_leader: int = 16
    adalead_min_mutations: int = 1
    adalead_max_mutations: int = 3
    tau_start: float = 0.5
    tau_end: float = 0.05
    init_logit_scale: float = 8.0
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

    seqprop = ProteinSeqPropTau(L=L, init_sequence=sequence, init_logit_scale=config.init_logit_scale).to(device)
    target_oracle = target_oracle.to(device).eval()
    all_oracles = list(constraint_oracles)
    for oracle in all_oracles:
        oracle.to(device).eval()

    X_orig = sequence_to_onehot(sequence).to(device)
    optimizer = torch.optim.Adam(seqprop.parameters(), lr=config.lr)
    all_params = list(seqprop.parameters())

    with torch.no_grad():
        X_start = sequence_to_onehot(sequence).unsqueeze(0).to(device)
        real_c_initial = {o.name: o(X_start).item() for o in [target_oracle] + all_oracles}
        # Use the real discrete baseline as the ALM anchor so the Lagrangian is
        # calibrated to actual feasibility thresholds. The tau=0.5 soft baseline
        # was 2x lower than the real score for GB1/stability, making the dual
        # variable track a completely different constraint than intended.
        c_initial = dict(real_c_initial)

    logger.info(f"Initial oracle values (discrete start / ALM anchor): {real_c_initial}")

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

    for step in range(config.n_steps):
        optimizer.zero_grad()

        param_snapshot = None
        optimizer_snapshot = None
        if config.decoded_constraint_guard and constraint_oracles:
            param_snapshot = [p.detach().clone() for p in all_params]
            optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        current_tau = tau_start * (tau_end / tau_start) ** (step / max(1, config.n_steps - 1))
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








        g_alm = None
        if use_alm:
            alm_penalty = torch.tensor(0.0, device=device)
            for oracle, viol in zip(constraint_oracles, violations):
                lam = lambdas[oracle.name]
                alm_penalty = alm_penalty + lam * viol + (config.alm_rho / 2) * viol ** 2
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

        g_hamming = flatten_gradients(
            torch.autograd.grad(
                config.hamming_lambda * hamming,
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

            # Update dual variables from REAL decoded violation so lambda
            # grows whenever the actual discrete sequence violates constraints,
            # not based on the soft-oracle proxy.
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
                        max(0.0, lambdas[oracle.name] + config.alm_dual_lr * real_viol),
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


def _detect_ckpt_type(path: str) -> str:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "head_state_dict" in ckpt:
        return "thermo"
    if "model_state_dict" in ckpt and "normalisation" in ckpt:
        return "kras"
    if "model" in ckpt:
        return "finetuned"
    raise ValueError(f"Unrecognised checkpoint format in {path}. Keys: {list(ckpt.keys())}")


def build_oracle_from_ckpt(
    name: str,
    path: str,
    backbone: ESM2SoftBackbone,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    pooling: str = "mean",
) -> nn.Module:
    kind = _detect_ckpt_type(path)
    if kind == "thermo":
        return ThermostabilityOracle.load(path, backbone, oracle_name=name)
    if kind == "kras":
        return KRASOracle.load(path, backbone, oracle_name=name)
    return FineTunedESMOracle.load(
        path,
        oracle_name=name,
        model_name=model_name,
        pooling=pooling,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generic real-score protein GRACE/ALM/SA optimizer")
    p.add_argument("--sequence", required=True)
    p.add_argument("--target", required=True, help="Target oracle name")
    p.add_argument("--constraints", nargs="*", default=[], help="Constraint oracle names")
    p.add_argument("--mode", default="grace_lagrangian", choices=MODES)
    p.add_argument(
        "--ckpts",
        nargs="*",
        default=[],
        metavar="NAME:PATH[:MODEL[:POOLING]]",
        help="Oracle checkpoints. Format: name:path[:esm_model[:pooling]]",
    )
    p.add_argument("--esm_model_ckpts", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hamming_lam", type=float, default=0.005)
    p.add_argument("--constraint_eps", type=float, default=0.1)
    p.add_argument("--constraint_type", default="absolute", choices=["absolute", "percent"])
    p.add_argument("--alm_rho", type=float, default=10.0)
    p.add_argument("--alm_dual_lr", type=float, default=0.02)
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
    p.add_argument("--de_top_positions", type=int, default=3)
    p.add_argument("--de_beam_width", type=int, default=1)
    p.add_argument("--de_lambda_penalty", type=float, default=10.0)
    p.add_argument("--adalead_budget", type=int, default=4000)
    p.add_argument("--adalead_population_size", type=int, default=128)
    p.add_argument("--adalead_num_leaders", type=int, default=16)
    p.add_argument("--adalead_mutants_per_leader", type=int, default=16)
    p.add_argument("--adalead_min_mutations", type=int, default=1)
    p.add_argument("--adalead_max_mutations", type=int, default=3)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.05)
    p.add_argument("--init_logit_scale", type=float, default=8.0)
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
    ckpt_model_map: Dict[str, str] = {}
    ckpt_pooling_map: Dict[str, str] = {}
    for entry in args.ckpts:
        parts = entry.split(":")
        if len(parts) < 2:
            raise ValueError(f"--ckpts entries must be name:path, got '{entry}'")
        name, path = parts[0], parts[1]
        ckpt_map[name] = path
        if len(parts) >= 3:
            ckpt_model_map[name] = parts[2]
        if len(parts) >= 4:
            ckpt_pooling_map[name] = parts[3]

    backbone = ESM2SoftBackbone(model_name=args.esm_model_ckpts, freeze=True)
    def build_oracle_for(name: str) -> nn.Module:
        if name == "none":
            raise ValueError("'none' should be filtered before calling build_oracle_for")
        if name.startswith("min:") or name.startswith("neg:"):
            _, base_name = name.split(":", 1)
            base = build_oracle_for(base_name)
            return NegatedOracle(base, name=f"decrease_{getattr(base, 'name', base_name)}")
        if name not in ckpt_map:
            raise ValueError(f"Missing checkpoint for oracle '{name}'. Available: {sorted(ckpt_map)}")
        return build_oracle_from_ckpt(
            name,
            ckpt_map[name],
            backbone,
            model_name=ckpt_model_map.get(name, args.esm_model_ckpts),
            pooling=ckpt_pooling_map.get(name, "mean"),
        )

    target_oracle = build_oracle_for(args.target)
    constraint_oracles = [build_oracle_for(name) for name in (args.constraints or []) if name != "none"]

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
        de_top_positions=args.de_top_positions,
        de_beam_width=args.de_beam_width,
        de_lambda_penalty=args.de_lambda_penalty,
        adalead_budget=args.adalead_budget,
        adalead_population_size=args.adalead_population_size,
        adalead_num_leaders=args.adalead_num_leaders,
        adalead_mutants_per_leader=args.adalead_mutants_per_leader,
        adalead_min_mutations=args.adalead_min_mutations,
        adalead_max_mutations=args.adalead_max_mutations,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        init_logit_scale=args.init_logit_scale,
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
