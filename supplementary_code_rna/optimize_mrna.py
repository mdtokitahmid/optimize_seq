"""
optimize_mrna.py — GRACE / ALM optimizer for 5' UTR design.

Default use case:
  increase MRL while keeping MFE close to the starting sequence.

Examples:
  python optimize_mrna.py \
    --sequence ACGAUGCAUAGCAGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGACC \
    --target mrl \
    --constraints mfe \
    --mode grace_lagrangian \
    --steps 300 \
    --constraint_eps 0.05 \
    --out results/mrna_opt.json

  python optimize_mrna.py \
    --sequence ACGAUGCAUAGCAGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGACC \
    --target mrl \
    --constraints mfe unpaired_frac five_prime_frac n_stems \
    --mode grace_lagrangian \
    --constraint_eps 0.1 \
    --out results/mrna_multi_constraint.json
"""

import argparse
import copy
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.grace import compute_grace_direction, flatten_gradients


ROOT = Path(__file__).resolve().parent
MRNA_ORACLE_DIR = ROOT / "Big_Oracles" / "mRNA" / "oracles"
if str(MRNA_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(MRNA_ORACLE_DIR))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES = ["unconstrained", "grace_only", "lagrangian_only", "grace_lagrangian", "simulated_annealing", "directed_evolution"]
RNA_ALPHABET = "ACGU"
RNA_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
IDX_TO_RNA = {i: nt for i, nt in enumerate(RNA_ALPHABET)}
NUM_NTS = 4
MODEL_SEQ_LEN = 50
ORACLE_NAMES = ["mrl", "mfe", "unpaired_frac", "five_prime_frac", "n_stems", "struct_score"]
CONSTRAINT_ORACLE_NAMES = ["mrl", "mfe", "unpaired_frac", "five_prime_frac", "n_stems"]
DEFAULT_MONITORS = ["mfe", "unpaired_frac", "five_prime_frac", "n_stems", "struct_score"]

MRL_SCALER_MEAN = 4.6509
MRL_SCALER_STD = 1.0540


@dataclass
class OptimConfig:
    mode: str = "grace_lagrangian"
    n_steps: int = 300
    K: int = 8
    lr: float = 1e-2
    hamming_lambda: float = 0.05
    constraint_eps: float = 0.05
    constraint_eps_pct: float | None = None
    alm_rho: float = 1.0
    alm_dual_lr: float = 0.01
    lambda_max: float = 50.0
    lambda_init: float = 0.0
    grad_balance: bool = True
    grace_reg: float = 1e-6
    log_every: int = 10
    n_decode_samples: int = 200
    sa_lambda_penalty: float = 10.0
    sa_temp_init: float = 1.0
    sa_temp_final: float = 1e-2
    de_top_positions: int = 3
    de_beam_width: int = 1
    de_lambda_penalty: float = 10.0
    decoded_constraint_guard: bool = False
    decoded_guard_tol: float = 0.0
    device: str = "cpu"


def canonicalize_sequence(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    bad = [nt for nt in seq if nt not in RNA_ALPHABET]
    if bad:
        raise ValueError(f"Sequence contains unsupported characters: {sorted(set(bad))}")
    return seq


def sequence_to_onehot(seq: str) -> torch.Tensor:
    seq = canonicalize_sequence(seq)
    x = torch.zeros(NUM_NTS, len(seq), dtype=torch.float32)
    for i, nt in enumerate(seq):
        x[RNA_TO_IDX[nt], i] = 1.0
    return x


def expected_hamming(P: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
    return (1.0 - (P * X_orig).sum(dim=0)).sum()


def hamming_distance(seq_a: str, seq_b: str) -> int:
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequences must have equal length for Hamming distance.")
    return sum(a != b for a, b in zip(seq_a, seq_b))


def pad_batch_for_oracle(x: torch.Tensor, model_seq_len: int = MODEL_SEQ_LEN) -> torch.Tensor:
    """
    Left-pad soft/hard one-hot RNA sequences to the oracle model length.

    Input:
      x: (B, 4, L)
    Output:
      (B, 4, model_seq_len)
    """
    seq_len = x.shape[-1]
    if seq_len > model_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds model length {model_seq_len}")
    if seq_len == model_seq_len:
        return x
    pad_left = model_seq_len - seq_len
    return F.pad(x, (pad_left, 0))


class RNASeqProp(nn.Module):
    def __init__(self, L: int, init_sequence: str = None, init_logit_scale: float = 4.0):
        super().__init__()
        self.L = L
        init_logits = self._make_init_logits(L, init_sequence, init_logit_scale)
        self.logits = nn.Parameter(init_logits)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(NUM_NTS))
        self._inst_norm = nn.InstanceNorm1d(NUM_NTS, affine=False, eps=1e-5)

    @staticmethod
    def _make_init_logits(L: int, sequence: str, scale: float) -> torch.Tensor:
        logits = torch.zeros(NUM_NTS, L)
        if sequence is not None:
            sequence = canonicalize_sequence(sequence)
            if len(sequence) != L:
                raise ValueError(f"init_sequence length {len(sequence)} != L={L}")
            for i, nt in enumerate(sequence):
                logits[RNA_TO_IDX[nt], i] = scale
        logits += 0.01 * torch.randn_like(logits)
        return logits

    def normalized_logits(self) -> torch.Tensor:
        z = self._inst_norm(self.logits.unsqueeze(0)).squeeze(0)
        return self.gamma * z + self.beta.unsqueeze(1)

    def probabilities(self, tau: float = 1.0) -> torch.Tensor:
            # Divide the normalized logits by tau before softmaxing
            return F.softmax(self.normalized_logits() / tau, dim=0)

    def st_sample(self, K: int, tau: float = 1.0):
        # Pass tau into the probabilities calculation
        P = self.probabilities(tau=tau)
        samples = []
        for _ in range(K):
            c = torch.multinomial(P.T, num_samples=1).squeeze(1)
            Y_hard = F.one_hot(c, num_classes=NUM_NTS).T.float()
            Y_st = Y_hard + (P - P.detach())
            samples.append(Y_st)
        return samples, P

    def decode(self, method: str = "argmax") -> str:
        P = self.probabilities()
        if method == "argmax":
            indices = P.argmax(dim=0)
        elif method == "sample":
            indices = torch.multinomial(P.T, 1).squeeze(1)
        else:
            raise ValueError(f"decode method must be 'argmax' or 'sample', got {method}")
        return "".join(IDX_TO_RNA[i.item()] for i in indices)

    def entropy(self) -> torch.Tensor:
        P = self.probabilities()
        H = -(P * (P + 1e-9).log()).sum(dim=0)
        return H.mean()


class RNAOracle(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MRLOracle(RNAOracle):
    def __init__(self, model: nn.Module):
        super().__init__("mrl")
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = pad_batch_for_oracle(x).permute(0, 2, 1)
        scaled = self.model(x_seq)
        return scaled * MRL_SCALER_STD + MRL_SCALER_MEAN


class StructureOracle(RNAOracle):
    def __init__(self, model: nn.Module, norm_stats, property_name: str):
        super().__init__(property_name)
        self.model = model
        self.property_name = property_name
        mfe_mean, mfe_std, stems_max = norm_stats
        self.register_buffer("mfe_mean", torch.tensor(float(mfe_mean), dtype=torch.float32))
        self.register_buffer("mfe_std", torch.tensor(float(mfe_std), dtype=torch.float32))
        self.register_buffer("stems_max", torch.tensor(float(stems_max), dtype=torch.float32))

    def _decode_outputs(self, x: torch.Tensor):
        raw = self.model(pad_batch_for_oracle(x).permute(0, 2, 1))
        mfe = raw[:, 0] * self.mfe_std + self.mfe_mean
        unpaired_frac = torch.clamp(raw[:, 1], 0.0, 1.0)
        five_prime_frac = torch.clamp(raw[:, 2], 0.0, 1.0)
        n_stems = torch.clamp(raw[:, 3] * self.stems_max, min=0.0)
        return mfe, unpaired_frac, five_prime_frac, n_stems

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mfe, unpaired_frac, five_prime_frac, n_stems = self._decode_outputs(x)

        if self.property_name == "mfe":
            return mfe
        if self.property_name == "unpaired_frac":
            return unpaired_frac
        if self.property_name == "five_prime_frac":
            return five_prime_frac
        if self.property_name == "n_stems":
            return n_stems
        if self.property_name == "struct_score":
            mfe_target = -8.0
            mfe_sigma = 6.0
            s_mfe = torch.exp(-0.5 * ((mfe - mfe_target) / mfe_sigma) ** 2)
            s_unpaired = torch.clamp(unpaired_frac / 0.80, max=1.0)
            s_stems = torch.exp(-0.7 * torch.clamp(n_stems - 1.0, min=0.0))
            s_five_prime = torch.clamp(five_prime_frac / 0.80, max=1.0)
            return (s_mfe * s_unpaired * s_stems * s_five_prime).pow(0.25)

        raise ValueError(f"Unknown structure property {self.property_name}")


def compute_violation(
    y_c: torch.Tensor,
    c_0: float,
    eps_abs: float | None = None,
    eps_pct: float | None = None,
) -> torch.Tensor:
    if eps_pct is not None:
        denom = abs(c_0) + 1e-4
        return F.relu((y_c - c_0).abs() / denom - (eps_pct / 100.0))
    if eps_abs is None:
        raise ValueError("Either eps_abs or eps_pct must be provided.")
    return F.relu((y_c - c_0).abs() - eps_abs)


def candidate_is_feasible(
    constraint_vals: Dict[str, float],
    c_initial: Dict[str, float],
    eps_abs: float | None = None,
    eps_pct: float | None = None,
) -> bool:
    if eps_pct is not None:
        return all(
            abs(val - c_initial[name]) / (abs(c_initial[name]) + 1e-4) <= (eps_pct / 100.0)
            for name, val in constraint_vals.items()
        )
    if eps_abs is None:
        raise ValueError("Either eps_abs or eps_pct must be provided.")
    return all(abs(val - c_initial[name]) <= eps_abs for name, val in constraint_vals.items())


def scalar_violation(
    val: float,
    c_0: float,
    eps_abs: float | None = None,
    eps_pct: float | None = None,
) -> float:
    if eps_pct is not None:
        return max(0.0, abs(val - c_0) / (abs(c_0) + 1e-4) - (eps_pct / 100.0))
    if eps_abs is None:
        raise ValueError("Either eps_abs or eps_pct must be provided.")
    return max(0.0, abs(val - c_0) - eps_abs)


def total_violation(
    constraint_vals: Dict[str, float],
    c_initial: Dict[str, float],
    eps_abs: float | None = None,
    eps_pct: float | None = None,
) -> float:
    return sum(
        scalar_violation(
            val=constraint_vals[name],
            c_0=c_initial[name],
            eps_abs=eps_abs,
            eps_pct=eps_pct,
        )
        for name in constraint_vals
    )


def score_sequence_with_oracles(
    seq: str,
    target_oracle: RNAOracle,
    constraint_oracles: List[RNAOracle],
    monitor_oracles: List[RNAOracle],
    device: str,
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        target_val = float(target_oracle(x).item())
        constraint_vals = {oracle.name: float(oracle(x).item()) for oracle in constraint_oracles}
        monitor_vals = {oracle.name: float(oracle(x).item()) for oracle in monitor_oracles}
    return target_val, constraint_vals, monitor_vals


def summarize_relaxed_state(
    seqprop: "RNASeqProp",
    target_oracle: "RNAOracle",
    constraint_oracles: List["RNAOracle"],
    monitor_oracles: List["RNAOracle"],
    X_orig: torch.Tensor,
    K: int,
) -> tuple[float, Dict[str, float], Dict[str, float], float, float]:
    with torch.no_grad():
        samples, P = seqprop.st_sample(K)
        batch = torch.stack(samples, dim=0)
        t_val = float(target_oracle(batch).mean().item())
        c_vals = {oracle.name: float(oracle(batch).mean().item()) for oracle in constraint_oracles}
        m_vals = {oracle.name: float(oracle(batch).mean().item()) for oracle in monitor_oracles}
        h_val = float(expected_hamming(P, X_orig).item())
        ent_val = float(seqprop.entropy().item())
    return t_val, c_vals, m_vals, h_val, ent_val


def propose_single_mutation(seq: str) -> str:
    pos = random.randrange(len(seq))
    current = seq[pos]
    choices = [nt for nt in RNA_ALPHABET if nt != current]
    new_nt = random.choice(choices)
    return seq[:pos] + new_nt + seq[pos + 1:]


def compute_target_gradient_rna(
    seq: str,
    target_oracle: RNAOracle,
    device: str,
) -> torch.Tensor:
    x = sequence_to_onehot(seq).unsqueeze(0).to(device)
    x.requires_grad_(True)
    target = target_oracle(x)
    target.sum().backward()
    return x.grad.detach().squeeze(0).cpu()


def propose_gradient_mutants_rna(
    seq: str,
    gradient: torch.Tensor,
    top_positions: int,
    seen_sequences: set[str],
) -> list[tuple[str, Dict[str, Any]]]:
    seq = canonicalize_sequence(seq)
    current_idx = [RNA_TO_IDX[nt] for nt in seq]
    ranked_mutations: list[tuple[float, int, int]] = []
    for pos, cur in enumerate(current_idx):
        for alt in range(NUM_NTS):
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
        chars[pos] = IDX_TO_RNA[alt]
        cand = "".join(chars)
        if cand in seen_sequences:
            continue
        proposals.append(
            (
                cand,
                {
                    "position": pos,
                    "from": seq[pos],
                    "to": IDX_TO_RNA[alt],
                    "delta": delta,
                },
            )
        )
        used_positions.add(pos)
        if len(proposals) >= top_positions:
            break
    return proposals


def annealing_temperature(step: int, n_steps: int, temp_init: float, temp_final: float) -> float:
    if n_steps <= 1:
        return temp_final
    frac = step / (n_steps - 1)
    return float(temp_init * ((temp_final / temp_init) ** frac))


def optimize_simulated_annealing(
    sequence: str,
    target_oracle: RNAOracle,
    constraint_oracles: List[RNAOracle],
    monitor_oracles: List[RNAOracle],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_sequence(sequence)
    L = len(sequence)
    all_oracles = constraint_oracles + monitor_oracles

    target_oracle = target_oracle.to(device)
    for oracle in all_oracles:
        oracle.to(device)

    start_target, start_constraints, start_monitors = score_sequence_with_oracles(
        sequence, target_oracle, constraint_oracles, monitor_oracles, device
    )
    c_initial = {target_oracle.name: start_target, **start_constraints, **start_monitors}
    logger.info(f"Initial oracle values: {c_initial}")

    current_seq = sequence
    current_target = start_target
    current_constraints = dict(start_constraints)
    current_monitors = dict(start_monitors)
    current_violation = total_violation(
        current_constraints,
        c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    )
    current_objective = current_target - config.sa_lambda_penalty * current_violation

    best_objective_seq = current_seq
    best_objective_score = current_target
    best_objective_value = current_objective
    best_objective_constraints = dict(current_constraints)
    best_objective_monitors = dict(current_monitors)
    best_objective_step = 0

    best_feasible_sequence = current_seq if candidate_is_feasible(
        current_constraints, c_initial, eps_abs=config.constraint_eps, eps_pct=config.constraint_eps_pct
    ) else None
    best_feasible_score = current_target if best_feasible_sequence is not None else float("-inf")
    best_feasible_step = 0 if best_feasible_sequence is not None else None

    tracked_names = [target_oracle.name] + [oracle.name for oracle in all_oracles]
    history: Dict[str, Any] = {
        "scores": {name: [] for name in tracked_names},
        "real_scores": {name: [] for name in tracked_names},
        "rel_drifts": {oracle.name: [] for oracle in all_oracles},
        "lambdas": {oracle.name: [] for oracle in constraint_oracles},
        "hamming": [],
        "entropy": [],
        "proj_ratio": [],
        "sequences": [],
        "c_initial": c_initial,
        "real_c_initial": dict(c_initial),
        "mode": config.mode,
        "objective": [],
        "temperature": [],
        "accepted": [],
    }

    # Log the true starting state as step 0 so history, plots, and summary agree.
    start_all_vals = {**current_constraints, **current_monitors}
    start_rel_drifts = {
        oracle.name: abs(start_all_vals[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4)
        for oracle in all_oracles
    }
    history["scores"][target_oracle.name].append(current_target)
    history["real_scores"][target_oracle.name].append(current_target)
    for oracle in all_oracles:
        history["scores"][oracle.name].append(start_all_vals[oracle.name])
        history["real_scores"][oracle.name].append(start_all_vals[oracle.name])
        history["rel_drifts"][oracle.name].append(start_rel_drifts[oracle.name])
    for oracle in constraint_oracles:
        history["lambdas"][oracle.name].append(0.0)
    history["hamming"].append(0.0)
    history["entropy"].append(float("nan"))
    history["proj_ratio"].append(1.0)
    history["objective"].append(current_objective)
    history["temperature"].append(config.sa_temp_init)
    history["accepted"].append(True)
    history["sequences"].append((0, current_seq))

    logger.info(
        f"step={0:4d} | target={current_target:+.3f} | "
        + "  ".join(f"{k}={v:.3f}(Δ{start_rel_drifts[k]:.2f})" for k, v in start_all_vals.items())
        + f" | obj={current_objective:+.3f} | viol={current_violation:.4f} | "
        + f"T={config.sa_temp_init:.4f} | accepted=1"
    )

    for step in range(1, config.n_steps):
        temperature = annealing_temperature(
            step, config.n_steps, config.sa_temp_init, config.sa_temp_final
        )
        cand_seq = propose_single_mutation(current_seq)
        cand_target, cand_constraints, cand_monitors = score_sequence_with_oracles(
            cand_seq, target_oracle, constraint_oracles, monitor_oracles, device
        )
        cand_violation = total_violation(
            cand_constraints,
            c_initial,
            eps_abs=config.constraint_eps,
            eps_pct=config.constraint_eps_pct,
        )
        cand_objective = cand_target - config.sa_lambda_penalty * cand_violation

        delta = cand_objective - current_objective
        accept = delta >= 0.0
        if not accept and temperature > 0.0:
            accept = random.random() < float(np.exp(delta / max(temperature, 1e-8)))

        if accept:
            current_seq = cand_seq
            current_target = cand_target
            current_constraints = cand_constraints
            current_monitors = cand_monitors
            current_violation = cand_violation
            current_objective = cand_objective

        all_vals = {**current_constraints, **current_monitors}
        rel_drifts = {
            oracle.name: abs(all_vals[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4)
            for oracle in all_oracles
        }

        history["scores"][target_oracle.name].append(current_target)
        history["real_scores"][target_oracle.name].append(current_target)
        for oracle in all_oracles:
            history["scores"][oracle.name].append(all_vals[oracle.name])
            history["real_scores"][oracle.name].append(all_vals[oracle.name])
            history["rel_drifts"][oracle.name].append(rel_drifts[oracle.name])
        for oracle in constraint_oracles:
            history["lambdas"][oracle.name].append(0.0)
        history["hamming"].append(float(hamming_distance(sequence, current_seq)))
        history["entropy"].append(float("nan"))
        history["proj_ratio"].append(1.0)
        history["objective"].append(current_objective)
        history["temperature"].append(temperature)
        history["accepted"].append(bool(accept))

        if step % config.log_every == 0:
            c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in all_vals.items())
            logger.info(
                f"step={step:4d} | target={current_target:+.3f} | {c_str} | "
                f"obj={current_objective:+.3f} | viol={current_violation:.4f} | "
                f"T={temperature:.4f} | accepted={int(accept)}"
            )

        if step % (config.log_every * 5) == 0:
            history["sequences"].append((step, current_seq))

        if current_objective > best_objective_value:
            best_objective_value = current_objective
            best_objective_seq = current_seq
            best_objective_score = current_target
            best_objective_constraints = dict(current_constraints)
            best_objective_monitors = dict(current_monitors)
            best_objective_step = step

        if candidate_is_feasible(
            current_constraints,
            c_initial,
            eps_abs=config.constraint_eps,
            eps_pct=config.constraint_eps_pct,
        ) and current_target > best_feasible_score:
            best_feasible_sequence = current_seq
            best_feasible_score = current_target
            best_feasible_step = step

    final_seq = best_objective_seq
    n_mutations = hamming_distance(sequence, final_seq)
    final_all_vals = {**best_objective_constraints, **best_objective_monitors}

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations} / {L} | "
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
            f"final={final_all_vals[oracle.name]:.3f} "
            f"rel_drift={abs(final_all_vals[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4):.2f}"
        )

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "constraint_eps": config.constraint_eps,
            "constraint_eps_pct": config.constraint_eps_pct,
            "constraint_type": "percent" if config.constraint_eps_pct is not None else "absolute",
            "target_oracle": target_oracle.name,
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [oracle.name for oracle in monitor_oracles],
            "final_lambdas": {oracle.name: 0.0 for oracle in constraint_oracles},
            "argmax_sequence": final_seq,
            "argmax_n_mutations": n_mutations,
            "best_feasible_sequence": best_feasible_sequence,
            "best_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_feasible_step": best_feasible_step,
            "best_feasible_n_mutations": None
            if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_hard_feasible_sequence": best_feasible_sequence,
            "best_hard_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_hard_feasible_step": best_feasible_step,
            "best_hard_feasible_n_mutations": None
            if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_objective_sequence": best_objective_seq,
            "best_objective_value": best_objective_value,
            "best_objective_step": best_objective_step,
        }
    )
    return history


def optimize_directed_evolution(
    sequence: str,
    target_oracle: RNAOracle,
    constraint_oracles: List[RNAOracle],
    monitor_oracles: List[RNAOracle],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_sequence(sequence)
    L = len(sequence)
    all_oracles = constraint_oracles + monitor_oracles

    target_oracle = target_oracle.to(device)
    for oracle in all_oracles:
        oracle.to(device)

    start_target, start_constraints, start_monitors = score_sequence_with_oracles(
        sequence, target_oracle, constraint_oracles, monitor_oracles, device
    )
    c_initial = {target_oracle.name: start_target, **start_constraints, **start_monitors}
    real_c_initial = dict(c_initial)
    logger.info(f"Initial discrete oracle values: {c_initial}")

    def objective(target: float, constraints: Dict[str, float]) -> float:
        violation = total_violation(
            constraints,
            real_c_initial,
            eps_abs=config.constraint_eps,
            eps_pct=config.constraint_eps_pct,
        )
        return target - config.de_lambda_penalty * violation

    tracked_names = [target_oracle.name] + [oracle.name for oracle in all_oracles]
    history: Dict[str, Any] = {
        "scores": {name: [] for name in tracked_names},
        "real_scores": {name: [] for name in tracked_names},
        "rel_drifts": {oracle.name: [] for oracle in all_oracles},
        "lambdas": {oracle.name: [] for oracle in constraint_oracles},
        "hamming": [],
        "entropy": [],
        "proj_ratio": [],
        "sequences": [],
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
        "decoded_guard_rejected": [],
        "decoded_guard_violation": [],
        "constraint_eps": config.constraint_eps,
        "constraint_eps_pct": config.constraint_eps_pct,
        "constraint_type": "percent" if config.constraint_eps_pct is not None else "absolute",
        "objective": [],
    }

    current_all = {**start_constraints, **start_monitors}
    start_rel_drifts = {
        oracle.name: abs(current_all[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4)
        for oracle in all_oracles
    }
    start_violation = total_violation(
        start_constraints,
        real_c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    )
    start_objective = objective(start_target, start_constraints)

    history["scores"][target_oracle.name].append(start_target)
    history["real_scores"][target_oracle.name].append(start_target)
    for oracle in all_oracles:
        history["scores"][oracle.name].append(current_all[oracle.name])
        history["real_scores"][oracle.name].append(current_all[oracle.name])
        history["rel_drifts"][oracle.name].append(start_rel_drifts[oracle.name])
    for oracle in constraint_oracles:
        history["lambdas"][oracle.name].append(0.0)
    history["hamming"].append(0.0)
    history["entropy"].append(0.0)
    history["proj_ratio"].append(1.0)
    history["decoded_guard_rejected"].append(False)
    history["decoded_guard_violation"].append(start_violation)
    history["objective"].append(start_objective)
    history["sequences"].append((0, sequence))

    best_objective_sequence = sequence
    best_objective_value = start_objective
    best_objective_score = start_target
    best_objective_step = 0

    feasible_start = candidate_is_feasible(
        start_constraints,
        real_c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    )
    best_feasible_sequence = sequence if feasible_start else None
    best_feasible_score = start_target if feasible_start else float("-inf")
    best_feasible_step = 0 if feasible_start else None

    beam = [
        {
            "seq": sequence,
            "target": start_target,
            "constraints": dict(start_constraints),
            "monitors": dict(start_monitors),
            "objective": start_objective,
        }
    ]
    seen_sequences = {sequence}

    logger.info(
        f"Directed evolution baseline: beam_width={config.de_beam_width}, "
        f"top_positions={config.de_top_positions}, lambda_penalty={config.de_lambda_penalty:.3f}. "
        f"Each real candidate evaluation counts as one optimization step."
    )

    step = 0
    while step < config.n_steps:
        offspring: list[Dict[str, Any]] = []
        for parent in beam:
            if step >= config.n_steps:
                break
            grad = compute_target_gradient_rna(parent["seq"], target_oracle, device)
            proposals = propose_gradient_mutants_rna(
                parent["seq"], grad, config.de_top_positions, seen_sequences
            )
            if not proposals:
                for _ in range(config.de_top_positions):
                    cand = propose_single_mutation(parent["seq"])
                    if cand not in seen_sequences:
                        proposals.append((cand, {"position": None}))
                    if len(proposals) >= config.de_top_positions:
                        break

            for cand_seq, meta in proposals:
                if step >= config.n_steps:
                    break
                cand_target, cand_constraints, cand_monitors = score_sequence_with_oracles(
                    cand_seq, target_oracle, constraint_oracles, monitor_oracles, device
                )
                cand_objective = objective(cand_target, cand_constraints)
                step += 1
                seen_sequences.add(cand_seq)

                cand_all = {**cand_constraints, **cand_monitors}
                rel_drifts = {
                    oracle.name: abs(cand_all[oracle.name] - c_initial[oracle.name]) / (abs(c_initial[oracle.name]) + 1e-4)
                    for oracle in all_oracles
                }
                violation = total_violation(
                    cand_constraints,
                    real_c_initial,
                    eps_abs=config.constraint_eps,
                    eps_pct=config.constraint_eps_pct,
                )

                history["scores"][target_oracle.name].append(cand_target)
                history["real_scores"][target_oracle.name].append(cand_target)
                for oracle in all_oracles:
                    history["scores"][oracle.name].append(cand_all[oracle.name])
                    history["real_scores"][oracle.name].append(cand_all[oracle.name])
                    history["rel_drifts"][oracle.name].append(rel_drifts[oracle.name])
                for oracle in constraint_oracles:
                    history["lambdas"][oracle.name].append(0.0)
                history["hamming"].append(float(hamming_distance(sequence, cand_seq)))
                history["entropy"].append(0.0)
                history["proj_ratio"].append(1.0)
                history["decoded_guard_rejected"].append(False)
                history["decoded_guard_violation"].append(violation)
                history["objective"].append(cand_objective)

                if step % config.log_every == 0:
                    c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in cand_all.items())
                    logger.info(
                        f"step={step:4d} | target={cand_target:+.3f} | {c_str} | "
                        f"hamming={hamming_distance(sequence, cand_seq):.1f} | "
                        f"source={'grad_pos' if meta.get('position') is not None else 'random_fallback'} | "
                        f"viol={violation:.4f}"
                    )
                if step % (config.log_every * 5) == 0:
                    history["sequences"].append((step, cand_seq))

                offspring.append(
                    {
                        "seq": cand_seq,
                        "target": cand_target,
                        "constraints": dict(cand_constraints),
                        "monitors": dict(cand_monitors),
                        "objective": cand_objective,
                    }
                )

                if cand_objective > best_objective_value:
                    best_objective_sequence = cand_seq
                    best_objective_value = cand_objective
                    best_objective_score = cand_target
                    best_objective_step = step

                if candidate_is_feasible(
                    cand_constraints,
                    real_c_initial,
                    eps_abs=config.constraint_eps,
                    eps_pct=config.constraint_eps_pct,
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
        f"Mode: {config.mode} | Mutations: {n_mutations} / {L} | "
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

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "target_oracle": target_oracle.name,
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [oracle.name for oracle in monitor_oracles],
            "final_lambdas": {oracle.name: 0.0 for oracle in constraint_oracles},
            "argmax_sequence": best_objective_sequence,
            "argmax_n_mutations": hamming_distance(sequence, best_objective_sequence),
            "best_feasible_sequence": best_feasible_sequence,
            "best_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_feasible_step": best_feasible_step,
            "best_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_hard_feasible_sequence": best_feasible_sequence,
            "best_hard_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_hard_feasible_step": best_feasible_step,
            "best_hard_feasible_n_mutations": None if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_objective_sequence": best_objective_sequence,
            "best_objective_value": best_objective_value,
            "best_objective_step": best_objective_step,
            "n_decode_samples": 0,
            "sampled_feasible_count": sum(
                1
                for i in range(len(history["real_scores"][target_oracle.name]))
                if candidate_is_feasible(
                    {oracle.name: history["real_scores"][oracle.name][i] for oracle in constraint_oracles},
                    real_c_initial,
                    eps_abs=config.constraint_eps,
                    eps_pct=config.constraint_eps_pct,
                )
            ),
        }
    )
    return history


def optimize(
    sequence: str,
    target_oracle: RNAOracle,
    constraint_oracles: List[RNAOracle],
    monitor_oracles: List[RNAOracle],
    config: OptimConfig,
) -> Dict[str, Any]:
    device = config.device
    sequence = canonicalize_sequence(sequence)
    L = len(sequence)
    use_grace = config.mode in ("grace_only", "grace_lagrangian")
    use_alm = config.mode in ("lagrangian_only", "grace_lagrangian")

    seqprop = RNASeqProp(L=L, init_sequence=sequence).to(device)
    target_oracle = target_oracle.to(device)
    all_oracles = constraint_oracles + monitor_oracles
    for oracle in all_oracles:
        oracle.to(device)

    X_orig = sequence_to_onehot(sequence).to(device)
    optimizer = torch.optim.Adam(seqprop.parameters(), lr=config.lr)
    all_params = list(seqprop.parameters())

    with torch.no_grad():
        # 1. Evaluate the REAL, discrete starting sequence
        real_start_target, real_start_constraints, real_start_monitors = score_sequence_with_oracles(
            sequence, target_oracle, constraint_oracles, monitor_oracles, device
        )
        
        # 2. Set c_initial to the REAL discrete scores!
        c_initial = {
            target_oracle.name: real_start_target,
            **real_start_constraints,
            **real_start_monitors,
        }
        
        # Keep real_c_initial identical for tracking purposes
        real_c_initial = dict(c_initial)
        
    logger.info(f"Initial discrete oracle values: {c_initial}")

    lambdas = {oracle.name: config.lambda_init for oracle in constraint_oracles}
    current_real_seq = sequence
    current_real_target = real_start_target
    current_real_constraints = dict(real_start_constraints)
    current_real_monitors = dict(real_start_monitors)
    current_real_violation = total_violation(
        current_real_constraints,
        real_c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    )
    best_feasible_sequence = None
    best_feasible_score = float("-inf")
    best_feasible_step = None
    best_hard_feasible_sequence = sequence if candidate_is_feasible(
        real_start_constraints,
        real_c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    ) else None
    best_hard_feasible_score = real_start_target if best_hard_feasible_sequence is not None else float("-inf")
    best_hard_feasible_step = 0 if best_hard_feasible_sequence is not None else None
    tracked_names = [target_oracle.name] + [oracle.name for oracle in all_oracles]
    history: Dict[str, Any] = {
        "scores": {name: [] for name in tracked_names},
        "real_scores": {name: [] for name in tracked_names},
        "rel_drifts": {oracle.name: [] for oracle in all_oracles},
        "lambdas": {oracle.name: [] for oracle in constraint_oracles},
        "hamming": [],
        "entropy": [],
        "proj_ratio": [],
        "sequences": [],
        "c_initial": c_initial,
        "real_c_initial": real_c_initial,
        "mode": config.mode,
        "decoded_guard_rejected": [],
        "decoded_guard_violation": [],
    }

    history["real_scores"][target_oracle.name].append(real_start_target)
    for oracle in all_oracles:
        history["real_scores"][oracle.name].append(real_c_initial[oracle.name])
    tau_start = 1.0
    tau_end = 0.1

    for step in range(config.n_steps):
        optimizer.zero_grad()

        param_snapshot = None
        optimizer_snapshot = None
        if config.decoded_constraint_guard and constraint_oracles:
            param_snapshot = [p.detach().clone() for p in all_params]
            optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        # Calculate the decaying temperature
        current_tau = tau_start * (tau_end / tau_start) ** (step / max(1, config.n_steps - 1))

        # Pass it into your sampler!
        samples, P = seqprop.st_sample(config.K, tau=current_tau)
        batch = torch.stack(samples, dim=0)

        y_target = target_oracle(batch).mean()
        y_constraints = [oracle(batch).mean() for oracle in constraint_oracles]
        y_monitors = [oracle(batch).mean() for oracle in monitor_oracles]
        hamming = expected_hamming(P, X_orig)

        violations = [
            compute_violation(
                y_c,
                c_initial[oracle.name],
                eps_abs=config.constraint_eps,
                eps_pct=config.constraint_eps_pct,
            )
            for oracle, y_c in zip(constraint_oracles, y_constraints)
        ]

        g_target = flatten_gradients(
            torch.autograd.grad(
                y_target, all_params, retain_graph=True, create_graph=False, allow_unused=True
            ),
            all_params,
        )

        if use_grace and constraint_oracles:
            g_constraints_flat = [
                flatten_gradients(
                    torch.autograd.grad(
                        y_c, all_params, retain_graph=True, create_graph=False, allow_unused=True
                    ),
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
            d_star = g_target
            grace_info = {"projection_ratio": 1.0, "conflict_scores": []}

        g_alm = None
        if use_alm:
            alm_penalty = torch.tensor(0.0, device=device)
            for oracle, viol in zip(constraint_oracles, violations):
                lam = lambdas[oracle.name]
                alm_penalty = alm_penalty + lam * viol + (config.alm_rho / 2) * viol ** 2
            if alm_penalty.requires_grad or alm_penalty.grad_fn is not None:
                g_alm_tuple = torch.autograd.grad(
                    alm_penalty, all_params, retain_graph=True, create_graph=False, allow_unused=True
                )
                g_alm = flatten_gradients(g_alm_tuple, all_params)

        g_hamming_tuple = torch.autograd.grad(
            config.hamming_lambda * hamming,
            all_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        g_hamming = flatten_gradients(g_hamming_tuple, all_params)

        if g_alm is not None and config.grad_balance:
            dir_norm = d_star.norm().clamp(min=1e-8)
            alm_norm = g_alm.norm().clamp(min=1e-8)
            g_alm_scaled = g_alm * (dir_norm / alm_norm)
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

        if use_alm:
            with torch.no_grad():
                for oracle, viol in zip(constraint_oracles, violations):
                    lambdas[oracle.name] = min(
                        config.lambda_max,
                        max(0.0, lambdas[oracle.name] + config.alm_dual_lr * viol.item()),
                    )

        with torch.no_grad():
            cand_seq = seqprop.decode("argmax")
            cand_onehot = sequence_to_onehot(cand_seq).unsqueeze(0).to(device)
            cand_target = target_oracle(cand_onehot).item()
            cand_constraint_vals = {
                oracle.name: oracle(cand_onehot).item() for oracle in constraint_oracles
            }
            cand_monitor_vals = {
                oracle.name: oracle(cand_onehot).item() for oracle in monitor_oracles
            }
            cand_real_violation = total_violation(
                cand_constraint_vals,
                real_c_initial,
                eps_abs=config.constraint_eps,
                eps_pct=config.constraint_eps_pct,
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
                cand_constraint_vals = dict(current_real_constraints)
                cand_monitor_vals = dict(current_real_monitors)
                cand_real_violation = current_real_violation

                t_val, c_vals, m_vals, h_val, ent_val = summarize_relaxed_state(
                    seqprop,
                    target_oracle,
                    constraint_oracles,
                    monitor_oracles,
                    X_orig,
                    config.K,
                )
                pr = history["proj_ratio"][-1] if history["proj_ratio"] else grace_info["projection_ratio"]
            else:
                current_real_seq = cand_seq
                current_real_target = cand_target
                current_real_constraints = dict(cand_constraint_vals)
                current_real_monitors = dict(cand_monitor_vals)
                current_real_violation = cand_real_violation

                t_val = y_target.item()
                c_vals = {oracle.name: y.item() for oracle, y in zip(constraint_oracles, y_constraints)}
                m_vals = {oracle.name: y.item() for oracle, y in zip(monitor_oracles, y_monitors)}
                h_val = hamming.item()
                ent_val = seqprop.entropy().item()
                pr = grace_info["projection_ratio"]

            all_vals = {**c_vals, **m_vals}

            rel_drifts = {
                oracle.name: abs(all_vals[oracle.name] - c_initial[oracle.name])
                / (abs(c_initial[oracle.name]) + 1e-4)
                for oracle in all_oracles
            }

            history["scores"][target_oracle.name].append(t_val)
            history["real_scores"][target_oracle.name].append(cand_target)
            for oracle in all_oracles:
                history["scores"][oracle.name].append(all_vals[oracle.name])
            for oracle in all_oracles:
                if oracle.name in cand_constraint_vals:
                    history["real_scores"][oracle.name].append(cand_constraint_vals[oracle.name])
                else:
                    history["real_scores"][oracle.name].append(cand_monitor_vals[oracle.name])
            for oracle in all_oracles:
                history["rel_drifts"][oracle.name].append(rel_drifts[oracle.name])
            for oracle in constraint_oracles:
                history["lambdas"][oracle.name].append(lambdas[oracle.name])
            history["hamming"].append(h_val)
            history["entropy"].append(ent_val)
            history["proj_ratio"].append(pr)
            history["decoded_guard_rejected"].append(rejected)
            history["decoded_guard_violation"].append(cand_real_violation)

            if step % config.log_every == 0:
                c_str = "  ".join(f"{k}={v:.3f}(Δ{rel_drifts[k]:.2f})" for k, v in all_vals.items())
                lam_str = " ".join(f"λ_{k}={v:.2f}" for k, v in lambdas.items()) if lambdas else ""
                logger.info(
                    f"step={step:4d} | target={t_val:+.3f} | {c_str} | "
                    f"hamming={h_val:.1f} | pr={pr:.3f}"
                    + (f" | real_viol={cand_real_violation:.4f}" if constraint_oracles else "")
                    + (" | rejected=1" if rejected else "")
                    + (f" | {lam_str}" if lam_str else "")
                )

            if step % (config.log_every * 5) == 0:
                history["sequences"].append((step, cand_seq))
            if candidate_is_feasible(
                c_vals,
                c_initial,
                eps_abs=config.constraint_eps,
                eps_pct=config.constraint_eps_pct,
            ):
                if t_val > best_feasible_score:
                    best_feasible_score = t_val
                    best_feasible_sequence = cand_seq
                    best_feasible_step = step
            if candidate_is_feasible(
                cand_constraint_vals,
                real_c_initial,
                eps_abs=config.constraint_eps,
                eps_pct=config.constraint_eps_pct,
            ):
                if cand_target > best_hard_feasible_score:
                    best_hard_feasible_score = cand_target
                    best_hard_feasible_sequence = cand_seq
                    best_hard_feasible_step = step

    n_decode = config.n_decode_samples
    logger.info(f"Decoding: sampling {n_decode} sequences from final distribution...")

    best_seq, best_score = sequence, float("-inf")
    best_sampled_sequence, best_sampled_score = None, float("-inf")
    best_sampled_feasible_sequence, best_sampled_feasible_score = None, float("-inf")
    best_sampled_feasible_step = None
    sampled_feasible_count = 0
    with torch.no_grad():
        for _ in range(n_decode):
            cand = seqprop.decode("sample")
            cand_onehot = sequence_to_onehot(cand).unsqueeze(0).to(device)
            cand_score = target_oracle(cand_onehot).item()
            cand_constraint_vals = {
                oracle.name: oracle(cand_onehot).item() for oracle in constraint_oracles
            }
            if cand_score > best_sampled_score:
                best_sampled_score = cand_score
                best_sampled_sequence = cand
            if cand_score > best_score:
                best_score = cand_score
                best_seq = cand
            if candidate_is_feasible(
                cand_constraint_vals,
                real_c_initial,
                eps_abs=config.constraint_eps,
                eps_pct=config.constraint_eps_pct,
            ):
                sampled_feasible_count += 1
                if cand_score > best_sampled_feasible_score:
                    best_sampled_feasible_score = cand_score
                    best_sampled_feasible_sequence = cand
                    best_sampled_feasible_step = config.n_steps
                if cand_score > best_hard_feasible_score:
                    best_hard_feasible_score = cand_score
                    best_hard_feasible_sequence = cand
                    best_hard_feasible_step = config.n_steps

    argmax_seq = seqprop.decode("argmax")
    argmax_onehot = sequence_to_onehot(argmax_seq).unsqueeze(0).to(device)
    argmax_score = target_oracle(argmax_onehot).item()
    argmax_constraint_vals = {
        oracle.name: oracle(argmax_onehot).item() for oracle in constraint_oracles
    }
    if argmax_score > best_score:
        best_score = argmax_score
        best_seq = argmax_seq
    if candidate_is_feasible(
        argmax_constraint_vals,
        real_c_initial,
        eps_abs=config.constraint_eps,
        eps_pct=config.constraint_eps_pct,
    ):
        if argmax_score > best_hard_feasible_score:
            best_hard_feasible_score = argmax_score
            best_hard_feasible_sequence = argmax_seq
            best_hard_feasible_step = config.n_steps

    final_seq = best_sampled_feasible_sequence or best_hard_feasible_sequence or best_seq
    n_mutations = hamming_distance(sequence, final_seq)

    logger.info("-" * 60)
    logger.info(
        f"Mode: {config.mode} | Mutations: {n_mutations} / {L} | "
        f"Final target score: {best_score:.4f}"
    )
    logger.info(
        f"Target {target_oracle.name}: {c_initial.get(target_oracle.name, '?')} -> "
        f"{history['scores'][target_oracle.name][-1]:.3f} (best decoded: {best_score:.4f})"
    )
    if best_feasible_sequence is not None:
        logger.info(
            f"Best feasible decoded sequence (step {best_feasible_step}): "
            f"{best_feasible_sequence} | score={best_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_feasible_sequence)}"
        )
    else:
        logger.info("Best feasible decoded sequence: none found")
    if best_hard_feasible_sequence is not None:
        logger.info(
            f"Best hard-feasible decoded sequence (step {best_hard_feasible_step}): "
            f"{best_hard_feasible_sequence} | score={best_hard_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_hard_feasible_sequence)}"
        )
    else:
        logger.info("Best hard-feasible decoded sequence: none found")
    if best_sampled_feasible_sequence is not None:
        logger.info(
            f"Best sampled feasible sequence (final sampling): "
            f"{best_sampled_feasible_sequence} | score={best_sampled_feasible_score:.4f} | "
            f"mutations={hamming_distance(sequence, best_sampled_feasible_sequence)} | "
            f"feasible_samples={sampled_feasible_count}/{n_decode}"
        )
    else:
        logger.info(f"Best sampled feasible sequence: none found ({sampled_feasible_count}/{n_decode} feasible)")
    for oracle in all_oracles:
        logger.info(
            f"Oracle {oracle.name}: initial={c_initial[oracle.name]:.3f} "
            f"final={history['scores'][oracle.name][-1]:.3f} "
            f"rel_drift={history['rel_drifts'][oracle.name][-1]:.2f}"
            + (f" final_λ={lambdas[oracle.name]:.2f}" if oracle.name in lambdas else "")
        )

    history.update(
        {
            "original_sequence": sequence,
            "final_sequence": final_seq,
            "n_mutations": n_mutations,
            "constraint_eps": config.constraint_eps,
            "constraint_eps_pct": config.constraint_eps_pct,
            "constraint_type": "percent" if config.constraint_eps_pct is not None else "absolute",
            "target_oracle": target_oracle.name,
            "constraint_oracles": [oracle.name for oracle in constraint_oracles],
            "monitor_oracles": [oracle.name for oracle in monitor_oracles],
            "final_lambdas": lambdas,
            "argmax_sequence": argmax_seq,
            "argmax_n_mutations": hamming_distance(sequence, argmax_seq),
            "best_sampled_sequence": best_sampled_sequence,
            "best_sampled_target_score": None if best_sampled_sequence is None else best_sampled_score,
            "best_sampled_n_mutations": None
            if best_sampled_sequence is None else hamming_distance(sequence, best_sampled_sequence),
            "best_sampled_feasible_sequence": best_sampled_feasible_sequence,
            "best_sampled_feasible_target_score": None
            if best_sampled_feasible_sequence is None else best_sampled_feasible_score,
            "best_sampled_feasible_step": best_sampled_feasible_step,
            "best_sampled_feasible_n_mutations": None
            if best_sampled_feasible_sequence is None else hamming_distance(sequence, best_sampled_feasible_sequence),
            "sampled_feasible_count": sampled_feasible_count,
            "n_decode_samples": n_decode,
            "best_feasible_sequence": best_feasible_sequence,
            "best_feasible_target_score": None if best_feasible_sequence is None else best_feasible_score,
            "best_feasible_step": best_feasible_step,
            "best_feasible_n_mutations": None
            if best_feasible_sequence is None else hamming_distance(sequence, best_feasible_sequence),
            "best_hard_feasible_sequence": best_hard_feasible_sequence,
            "best_hard_feasible_target_score": None
            if best_hard_feasible_sequence is None else best_hard_feasible_score,
            "best_hard_feasible_step": best_hard_feasible_step,
            "best_hard_feasible_n_mutations": None
            if best_hard_feasible_sequence is None else hamming_distance(sequence, best_hard_feasible_sequence),
        }
    )
    return history


def build_oracle(name: str, shared_models: Dict[str, Any]) -> RNAOracle:
    if name == "mrl":
        return MRLOracle(shared_models["mrl_model"])
    if name in {"mfe", "unpaired_frac", "five_prime_frac", "n_stems", "struct_score"}:
        return StructureOracle(shared_models["structure_model"], shared_models["structure_stats"], name)
    if name in {"structure", "stability"}:
        return StructureOracle(shared_models["structure_model"], shared_models["structure_stats"], "mfe")
    raise ValueError(f"Unknown mRNA oracle '{name}'. Supported: {ORACLE_NAMES + ['structure', 'stability']}")


def normalize_constraint_name(name: str) -> str:
    if name in {"structure", "stability"}:
        return "mfe"
    if name == "struct_score":
        raise ValueError(
            "`struct_score` is not allowed as a constraint in optimize_mrna.py. "
            "Use `mfe` if you want to preserve structure."
        )
    return name


def load_shared_mrl_model(model_path: str, device: str):
    try:
        from predict_mrl import load_mrl_model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Failed to import the MRL oracle. This script needs the mRNA oracle "
            "dependencies installed, especially `h5py` for loading the .hdf5 model."
        ) from e
    return load_mrl_model(model_path, device)


def load_shared_structure_model(model_path: str, device: str):
    from predict_structure import load_structure_model
    return load_structure_model(model_path, device)


def parse_args():
    p = argparse.ArgumentParser(description="GRACE + ALM optimizer for mRNA 5' UTR design")
    p.add_argument("--sequence", required=True, help=f"RNA/DNA sequence of length 1-{MODEL_SEQ_LEN}")
    p.add_argument("--target", default="mrl", choices=ORACLE_NAMES)
    p.add_argument(
        "--constraints",
        nargs="*",
        default=["mfe"],
        help="Constraint oracles. Aliases 'structure' and 'stability' map to mfe. "
             "`struct_score` is intentionally disallowed as a constraint.",
    )
    p.add_argument(
        "--monitors",
        nargs="*",
        default=None,
        help="Tracked but unconstrained oracles. If omitted, all remaining structure metrics are tracked.",
    )
    p.add_argument("--mode", default="grace_lagrangian", choices=MODES)

    p.add_argument("--mrl_model", default=str(MRNA_ORACLE_DIR / "models" / "main_MRL_model.hdf5"))
    p.add_argument(
        "--structure_model", default=str(MRNA_ORACLE_DIR / "models" / "structure_surrogate.pt")
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hamming_lam", type=float, default=0.05)
    p.add_argument(
        "--constraint_eps",
        type=float,
        default=0.05,
        help="Absolute oracle drift tolerance.",
    )
    p.add_argument(
        "--constraint_eps_pct",
        type=float,
        default=None,
        help="Relative drift tolerance in percent. If set, this is used instead of --constraint_eps "
             "and matches the feasibility notion used by plot_ablation.py.",
    )
    p.add_argument("--alm_rho", type=float, default=1.0)
    p.add_argument("--alm_dual_lr", type=float, default=0.01)
    p.add_argument("--lambda_max", type=float, default=50.0)
    p.add_argument("--lambda_init", type=float, default=0.0)
    p.add_argument("--sa_lambda_penalty", type=float, default=10.0)
    p.add_argument("--sa_temp_init", type=float, default=1.0)
    p.add_argument("--sa_temp_final", type=float, default=1e-2)
    p.add_argument("--de_top_positions", type=int, default=3)
    p.add_argument("--de_beam_width", type=int, default=1)
    p.add_argument("--de_lambda_penalty", type=float, default=10.0)
    p.add_argument("--grace_reg", type=float, default=1e-6)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--no_grad_balance", action="store_true")
    p.add_argument(
        "--decoded_constraint_guard",
        action="store_true",
        help="Reject gradient updates that worsen decoded-sequence real constraint violation.",
    )
    p.add_argument(
        "--decoded_guard_tol",
        type=float,
        default=0.0,
        help="Allow this much increase in decoded real violation before rejecting a step.",
    )
    p.add_argument("--n_decode_samples", type=int, default=200)
    p.add_argument("--out", default="results/mrna_optimized.json")
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
    print(f"Using device: {args.device}")

    sequence = canonicalize_sequence(args.sequence)
    if not (1 <= len(sequence) <= MODEL_SEQ_LEN):
        raise ValueError(
            f"mRNA optimizer currently supports sequence lengths 1-{MODEL_SEQ_LEN}, "
            f"got length {len(sequence)}"
        )

    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Mode: {args.mode}")

    shared_models = {}
    shared_models["mrl_model"] = load_shared_mrl_model(args.mrl_model, args.device)
    structure_model, structure_stats = load_shared_structure_model(args.structure_model, args.device)
    shared_models["structure_model"] = structure_model
    shared_models["structure_stats"] = structure_stats

    constraint_names = [normalize_constraint_name(n) for n in (args.constraints or [])]
    if args.monitors is None:
        used = {args.target, *constraint_names}
        monitor_names = [name for name in DEFAULT_MONITORS if name not in used]
    else:
        monitor_names = [
            "struct_score" if n in {"structure", "stability"} else n for n in (args.monitors or [])
        ]

    target_oracle = build_oracle(args.target, shared_models)
    constraint_oracles = [build_oracle(name, shared_models) for name in constraint_names if name != "none"]
    monitor_oracles = [build_oracle(name, shared_models) for name in monitor_names if name != "none"]

    logger.info(f"Target:      {target_oracle.name}")
    logger.info(f"Constrained: {[oracle.name for oracle in constraint_oracles]}")
    logger.info(f"Monitored:   {[oracle.name for oracle in monitor_oracles]}")

    config = OptimConfig(
        mode=args.mode,
        n_steps=args.steps,
        K=args.K,
        lr=args.lr,
        hamming_lambda=args.hamming_lam,
        constraint_eps=args.constraint_eps,
        constraint_eps_pct=args.constraint_eps_pct,
        alm_rho=args.alm_rho,
        alm_dual_lr=args.alm_dual_lr,
        lambda_max=args.lambda_max,
        lambda_init=args.lambda_init,
        sa_lambda_penalty=args.sa_lambda_penalty,
        sa_temp_init=args.sa_temp_init,
        sa_temp_final=args.sa_temp_final,
        de_top_positions=args.de_top_positions,
        de_beam_width=args.de_beam_width,
        de_lambda_penalty=args.de_lambda_penalty,
        grad_balance=not args.no_grad_balance,
        grace_reg=args.grace_reg,
        log_every=args.log_every,
        decoded_constraint_guard=args.decoded_constraint_guard,
        decoded_guard_tol=args.decoded_guard_tol,
        n_decode_samples=args.n_decode_samples,
        device=args.device,
    )

    if args.mode == "simulated_annealing":
        results = optimize_simulated_annealing(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            monitor_oracles=monitor_oracles,
            config=config,
        )
    elif args.mode == "directed_evolution":
        results = optimize_directed_evolution(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            monitor_oracles=monitor_oracles,
            config=config,
        )
    else:
        results = optimize(
            sequence=sequence,
            target_oracle=target_oracle,
            constraint_oracles=constraint_oracles,
            monitor_oracles=monitor_oracles,
            config=config,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
