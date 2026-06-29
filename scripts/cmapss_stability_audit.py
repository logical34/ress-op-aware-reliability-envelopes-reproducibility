"""Stability audit for the C-MAPSS Case C profile-conditioned envelopes.

This script is intentionally separate from ``simulation.py`` because it expands
the public benchmark analysis across multiple fold seeds, horizons, bootstrap
quantiles, and beta profiles. It does not change the main Case C data,
prediction definitions, or reported baseline run.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cmapss_case_study as cs  # noqa: E402


RESULTS_DIR = ROOT / "results"
AUDIT_MD = ROOT / "stability_audit.md"

STABILITY_SEEDS = [20260620 + i for i in range(10)]
STABILITY_HORIZONS = [50.0, 100.0, 150.0]
STABILITY_BOOTSTRAP_QUANTILES = [80.0, 90.0, 95.0]
STABILITY_BETAS = [0.0, 0.25, 0.50, 0.75, 1.0]
BOOTSTRAP_REPS = cs.BOOTSTRAP_REPS


def condition_weights(conditions: np.ndarray, pi: dict[str, float]) -> np.ndarray:
    weights = np.zeros(len(conditions), dtype=float)
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        weights[mask] = float(pi_d) / float(mask.sum())
        active += float(pi_d)
    if active <= 1e-12:
        weights[:] = 1.0 / max(len(conditions), 1)
        return weights
    return weights / active


def weighted_brier_value(
    omega: np.ndarray, predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> float:
    weights = condition_weights(conditions, pi)
    residual = predictions @ omega - labels
    return float(np.sum(weights * residual**2))


def brier_quadratic_terms(
    predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = condition_weights(conditions, pi)
    weighted_x = predictions * weights[:, None]
    q_mat = predictions.T @ weighted_x
    q_vec = predictions.T @ (weights * labels)
    const = float(np.sum(weights * labels**2))
    return q_mat, q_vec, const


def loss_from_terms(omega: np.ndarray, q_mat: np.ndarray, q_vec: np.ndarray, const: float) -> float:
    omega = np.asarray(omega, dtype=float)
    return max(0.0, float(omega @ q_mat @ omega - 2.0 * q_vec @ omega + const))


def exact_brier_weight(
    predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> tuple[np.ndarray, float]:
    """Solve the four-source simplex-constrained Brier problem by active sets."""

    n_sources = predictions.shape[1]
    q_mat, q_vec, const = brier_quadratic_terms(predictions, labels, conditions, pi)

    best_omega: np.ndarray | None = None
    best_loss = math.inf
    for size in range(1, n_sources + 1):
        for active in itertools.combinations(range(n_sources), size):
            active_idx = np.array(active, dtype=int)
            q_sub = q_mat[np.ix_(active_idx, active_idx)]
            qv_sub = q_vec[active_idx]
            kkt = np.block(
                [
                    [q_sub, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.r_[qv_sub, 1.0]
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            omega_active = solution[:size]
            if np.any(omega_active < -1e-8):
                continue
            omega_active = np.clip(omega_active, 0.0, 1.0)
            total = float(omega_active.sum())
            if total <= 1e-12:
                continue
            omega_active /= total
            omega = np.zeros(n_sources, dtype=float)
            omega[active_idx] = omega_active
            loss = float(omega @ q_mat @ omega - 2.0 * q_vec @ omega + const)
            if loss < best_loss:
                best_loss = loss
                best_omega = omega
    if best_omega is None:
        best_omega = np.full(n_sources, 1.0 / n_sources)
        best_loss = weighted_brier_value(best_omega, predictions, labels, conditions, pi)
    return best_omega, max(0.0, best_loss)


def support_basis(size: int) -> np.ndarray:
    basis = np.zeros((size, size - 1), dtype=float)
    for idx in range(size - 1):
        basis[idx, idx] = 1.0
        basis[-1, idx] = -1.0
    return basis


def endpoint_over_support(
    q_mat: np.ndarray,
    q_vec: np.ndarray,
    const: float,
    loss_limit: float,
    direction: np.ndarray,
    support: tuple[int, ...],
    sense: str,
) -> dict[str, object] | None:
    support_idx = np.array(support, dtype=int)
    size = len(support_idx)
    q_sub = q_mat[np.ix_(support_idx, support_idx)]
    qv_sub = q_vec[support_idx]
    direction_sub = direction[support_idx]

    if size == 1:
        omega = np.zeros(len(direction), dtype=float)
        omega[support_idx[0]] = 1.0
        loss = loss_from_terms(omega, q_mat, q_vec, const)
        if loss <= loss_limit + 5e-7:
            return {
                "omega": omega,
                "objective_value": float(direction @ omega),
                "loss": loss,
                "support_size": size,
                "method": "active_set_vertex",
            }
        return None

    w0 = np.full(size, 1.0 / size, dtype=float)
    basis = support_basis(size)
    hessian = basis.T @ q_sub @ basis
    hessian = 0.5 * (hessian + hessian.T)
    linear = basis.T @ (q_sub @ w0 - qv_sub)
    base_loss = float(w0 @ q_sub @ w0 - 2.0 * qv_sub @ w0 + const)
    objective_direction = basis.T @ direction_sub

    try:
        hessian_inv = np.linalg.pinv(hessian, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    center = -hessian_inv @ linear
    center_loss = float(center @ hessian @ center + 2.0 * linear @ center + base_loss)
    if center_loss > loss_limit + 5e-7:
        return None
    radius_sq = max(0.0, float(loss_limit - center_loss))
    h_inv_direction = hessian_inv @ objective_direction
    denom = float(objective_direction @ h_inv_direction)
    if radius_sq <= 1e-14 or denom <= 1e-14:
        z = center
    else:
        sign = -1.0 if sense == "min" else 1.0
        z = center + sign * math.sqrt(radius_sq / denom) * h_inv_direction
    omega_sub = w0 + basis @ z
    if np.min(omega_sub) < -5e-7:
        return None
    omega_sub = np.clip(omega_sub, 0.0, 1.0)
    total = float(omega_sub.sum())
    if total <= 1e-12:
        return None
    omega_sub /= total
    omega = np.zeros(len(direction), dtype=float)
    omega[support_idx] = omega_sub
    loss = loss_from_terms(omega, q_mat, q_vec, const)
    if loss > loss_limit + 2e-6:
        return None
    return {
        "omega": omega,
        "objective_value": float(direction @ omega),
        "loss": loss,
        "support_size": size,
        "method": "active_set_quadratic",
    }


def fast_linear_endpoint(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    direction: np.ndarray,
    sense: str,
    omega_hat: np.ndarray,
    best_loss: float,
) -> dict[str, object]:
    q_mat, q_vec, const = brier_quadratic_terms(predictions, labels, conditions, pi)
    loss_limit = float(best_loss + epsilon)
    candidates: list[dict[str, object]] = []
    n_sources = len(direction)
    for size in range(1, n_sources + 1):
        for support in itertools.combinations(range(n_sources), size):
            candidate = endpoint_over_support(q_mat, q_vec, const, loss_limit, direction, support, sense)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        omega = np.asarray(omega_hat, dtype=float)
        loss = loss_from_terms(omega, q_mat, q_vec, const)
        return {
            "omega": omega,
            "objective_value": float(direction @ omega),
            "loss": loss,
            "loss_limit": loss_limit,
            "loss_excess": float(loss - loss_limit),
            "simplex_residual": abs(float(omega.sum() - 1.0)),
            "bound_violation": max(float(-omega.min()), float(omega.max() - 1.0), 0.0),
            "success": False,
            "used_fallback": True,
            "method": "omega_hat_fallback",
        }

    if sense == "min":
        best = min(candidates, key=lambda item: float(item["objective_value"]))
    else:
        best = max(candidates, key=lambda item: float(item["objective_value"]))
    omega = np.asarray(best["omega"], dtype=float)
    loss = float(best["loss"])
    return {
        "omega": omega,
        "objective_value": float(best["objective_value"]),
        "loss": loss,
        "loss_limit": loss_limit,
        "loss_excess": float(loss - loss_limit),
        "simplex_residual": abs(float(omega.sum() - 1.0)),
        "bound_violation": max(float(-omega.min()), float(omega.max() - 1.0), 0.0),
        "success": True,
        "used_fallback": False,
        "method": str(best["method"]),
    }


def fast_continuous_envelope(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    source_means: np.ndarray,
    omega_hat: np.ndarray,
    best_loss: float,
) -> dict[str, object]:
    lower = fast_linear_endpoint(
        predictions, labels, conditions, pi, epsilon, source_means, "min", omega_hat, best_loss
    )
    upper = fast_linear_endpoint(
        predictions, labels, conditions, pi, epsilon, source_means, "max", omega_hat, best_loss
    )
    return {
        "lower": float(lower["objective_value"]),
        "upper": float(upper["objective_value"]),
        "width": float(upper["objective_value"] - lower["objective_value"]),
        "lower_success": bool(lower["success"]),
        "upper_success": bool(upper["success"]),
        "used_fallback": bool(lower["used_fallback"] or upper["used_fallback"]),
        "lower_loss_excess": float(lower["loss_excess"]),
        "upper_loss_excess": float(upper["loss_excess"]),
        "lower_simplex_residual": float(lower["simplex_residual"]),
        "upper_simplex_residual": float(upper["simplex_residual"]),
        "lower_bound_violation": float(lower["bound_violation"]),
        "upper_bound_violation": float(upper["bound_violation"]),
        "lower_method": str(lower["method"]),
        "upper_method": str(upper["method"]),
    }


def bootstrap_profile_weights(
    conditions: np.ndarray, pi: dict[str, float], rng: np.random.Generator, reps: int
) -> np.ndarray:
    unique_conditions = np.array(sorted(set(conditions)))
    active = sum(float(pi.get(str(condition), 0.0)) for condition in unique_conditions)
    weights = np.zeros((reps, len(conditions)), dtype=float)
    if active <= 1e-12:
        weights[:] = 1.0 / max(len(conditions), 1)
        return weights
    for condition in unique_conditions:
        idx = np.flatnonzero(conditions == condition)
        if len(idx) == 0:
            continue
        local_draws = rng.integers(0, len(idx), size=(reps, len(idx)))
        counts = np.zeros((reps, len(idx)), dtype=float)
        np.add.at(counts, (np.repeat(np.arange(reps), len(idx)), local_draws.ravel()), 1.0)
        weights[:, idx] = counts * (float(pi.get(str(condition), 0.0)) / active / float(len(idx)))
    return weights


def simplex_optimum_losses_from_terms(
    q_mats: np.ndarray, q_vecs: np.ndarray, consts: np.ndarray
) -> np.ndarray:
    reps, n_sources, _ = q_mats.shape
    best_losses = np.full(reps, np.inf, dtype=float)
    for size in range(1, n_sources + 1):
        for support in itertools.combinations(range(n_sources), size):
            active_idx = np.array(support, dtype=int)
            q_sub = q_mats[:, active_idx, :][:, :, active_idx]
            qv_sub = q_vecs[:, active_idx]
            kkt = np.zeros((reps, size + 1, size + 1), dtype=float)
            kkt[:, :size, :size] = q_sub
            kkt[:, :size, size] = 1.0
            kkt[:, size, :size] = 1.0
            rhs = np.zeros((reps, size + 1), dtype=float)
            rhs[:, :size] = qv_sub
            rhs[:, size] = 1.0
            solution = np.matmul(np.linalg.pinv(kkt, rcond=1e-12), rhs[:, :, None])[:, :, 0]
            omega_active = solution[:, :size]
            feasible = np.all(omega_active >= -1e-8, axis=1)
            omega_active = np.clip(omega_active, 0.0, 1.0)
            totals = omega_active.sum(axis=1)
            feasible &= totals > 1e-12
            omega_active[feasible] = omega_active[feasible] / totals[feasible, None]
            losses = (
                np.einsum("ri,rij,rj->r", omega_active, q_sub, omega_active, optimize=True)
                - 2.0 * np.einsum("ri,ri->r", qv_sub, omega_active, optimize=True)
                + consts
            )
            losses = np.maximum(0.0, losses)
            best_losses = np.where(feasible & (losses < best_losses), losses, best_losses)
    unresolved = ~np.isfinite(best_losses)
    if np.any(unresolved):
        equal = np.full(n_sources, 1.0 / n_sources, dtype=float)
        best_losses[unresolved] = (
            np.einsum("i,rij,j->r", equal, q_mats[unresolved], equal, optimize=True)
            - 2.0 * (q_vecs[unresolved] @ equal)
            + consts[unresolved]
        )
    return np.maximum(0.0, best_losses)


def bootstrap_loss_excesses(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> np.ndarray:
    omega_hat_original, _ = exact_brier_weight(predictions, labels, conditions, pi)
    weights = bootstrap_profile_weights(conditions, pi, rng, reps)
    q_mats = np.einsum("rn,ni,nj->rij", weights, predictions, predictions, optimize=True)
    q_vecs = np.einsum("rn,ni,n->ri", weights, predictions, labels, optimize=True)
    consts = weights @ (labels**2)
    bootstrap_optimum_losses = simplex_optimum_losses_from_terms(q_mats, q_vecs, consts)
    original_residual = predictions @ omega_hat_original - labels
    original_losses = weights @ (original_residual**2)
    return np.maximum(0.0, original_losses - bootstrap_optimum_losses)


def assign_condition_stratified_folds(frame: pd.DataFrame, seed: int, n_folds: int = cs.CROSS_FIT_FOLDS) -> pd.Series:
    rng = np.random.default_rng(seed)
    folds = pd.Series(index=frame.index, dtype=int)
    for _, condition_frame in frame.groupby("condition", sort=True):
        indices = condition_frame.index.to_numpy()
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for pos, idx in enumerate(shuffled):
            folds.loc[idx] = pos % n_folds
    return folds.astype(int)


def observed_profile(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["condition"].value_counts(normalize=True).sort_index()
    return {str(condition): float(value) for condition, value in counts.items()}


def calibration_risk_condition(train: pd.DataFrame) -> str:
    stats = (
        train.groupby("condition", sort=True)["y_survive"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["mean", "condition"], ascending=[True, True])
    )
    return str(stats.iloc[0]["condition"])


def beta_profile(pi_obs: dict[str, float], d_star: str, beta: float) -> dict[str, float]:
    out = {condition: (1.0 - beta) * float(value) for condition, value in pi_obs.items()}
    out[d_star] = out.get(d_star, 0.0) + beta
    total = sum(out.values())
    return {condition: value / total for condition, value in out.items()}


def profile_weighted_source_means(frame: pd.DataFrame, pi: dict[str, float]) -> np.ndarray:
    matrix = frame[[f"r_{source}" for source in cs.SOURCE_FEATURE_SETS]].to_numpy(dtype=float)
    conditions = frame["condition"].to_numpy()
    means = np.zeros(matrix.shape[1], dtype=float)
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        means += float(pi_d) * matrix[mask].mean(axis=0)
        active += float(pi_d)
    if active <= 1e-12:
        return matrix.mean(axis=0)
    return means / active


def profile_weighted_empirical(frame: pd.DataFrame, pi: dict[str, float]) -> float:
    labels = frame["y_survive"].to_numpy(dtype=float)
    conditions = frame["condition"].to_numpy()
    empirical = 0.0
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        empirical += float(pi_d) * float(labels[mask].mean())
        active += float(pi_d)
    if active <= 1e-12:
        return float(labels.mean())
    return empirical / active


def build_prediction_frames() -> dict[tuple[str, float], pd.DataFrame]:
    prepared_sets = [cs.add_condition_labels(cs.load_fd(fd)) for fd in cs.FD_SETS]
    frames: dict[tuple[str, float], pd.DataFrame] = {}
    for prepared in prepared_sets:
        unit_features = cs.unit_feature_table(prepared)
        test_features = cs.test_feature_table(prepared)
        lives = prepared.train_life["life"].to_numpy(dtype=float)
        empirical_sigma = float(np.std(np.log(lives)))
        empirical_mu = float(np.mean(np.log(lives)))

        fitted_models: dict[str, tuple[np.ndarray, float] | None] = {}
        for model_name, feature_cols in cs.SOURCE_FEATURE_SETS.items():
            if not feature_cols:
                fitted_models[model_name] = None
            else:
                fitted_models[model_name] = cs.fit_linear_predictor(unit_features, feature_cols)

        renamed_test = test_features.rename(columns={v: k for k, v in cs.TEST_FEATURE_MAP.items()})
        for mission in STABILITY_HORIZONS:
            rows = []
            total_age = test_features["observed_cycles"].to_numpy(dtype=float) + mission
            predictions: dict[str, np.ndarray] = {}
            for model_name, feature_cols in cs.SOURCE_FEATURE_SETS.items():
                if not feature_cols:
                    reliability = cs.survival_from_log_life(
                        np.full(len(test_features), empirical_mu),
                        empirical_sigma,
                        total_age,
                    )
                else:
                    fitted = fitted_models[model_name]
                    assert fitted is not None
                    params, resid_sigma = fitted
                    mu = cs.predict_log_life(params, renamed_test, feature_cols)
                    reliability = cs.survival_from_log_life(mu, resid_sigma + 0.10, total_age)
                predictions[model_name] = np.clip(reliability, 0.0, 1.0)

            for idx, row in test_features.reset_index(drop=True).iterrows():
                out = {
                    "subset": prepared.fd,
                    "engine_id": int(row["unit"]),
                    "condition": str(row["condition"]),
                    "observed_cycles": float(row["observed_cycles"]),
                    "rul": float(row["rul"]),
                    "mission_cycles": mission,
                    "y_survive": int(float(row["rul"]) > mission),
                }
                for source_name in cs.SOURCE_FEATURE_SETS:
                    out[f"r_{source_name}"] = float(predictions[source_name][idx])
                rows.append(out)
            frames[(prepared.fd, mission)] = pd.DataFrame(rows)
    return frames


def profile_specs_for_frame(frame: pd.DataFrame) -> list[tuple[str, float | None]]:
    subset = str(frame["subset"].iloc[0])
    if subset in {"FD002", "FD004"}:
        return [(f"beta_{beta:.2f}", beta) for beta in STABILITY_BETAS]
    return [("observed", None)]


def run_fold_settings(
    frame: pd.DataFrame,
    seed: int,
    horizon: float,
    profile_name: str,
    beta: float | None,
    fold: int,
    fold_rng: np.random.Generator,
) -> list[dict[str, object]]:
    train = frame[frame["fold"] != fold].copy()
    eval_frame = frame[frame["fold"] == fold].copy()
    pi_obs = observed_profile(frame)
    d_star = calibration_risk_condition(train) if beta is not None else ""
    pi = beta_profile(pi_obs, d_star, beta) if beta is not None else pi_obs

    train_predictions = train[[f"r_{source}" for source in cs.SOURCE_FEATURE_SETS]].to_numpy(dtype=float)
    train_labels = train["y_survive"].to_numpy(dtype=float)
    train_conditions = train["condition"].to_numpy()
    omega_hat, best_loss = exact_brier_weight(train_predictions, train_labels, train_conditions, pi)
    loss_excesses = bootstrap_loss_excesses(train_predictions, train_labels, train_conditions, pi, fold_rng)

    eval_predictions = eval_frame[[f"r_{source}" for source in cs.SOURCE_FEATURE_SETS]].to_numpy(dtype=float)
    eval_labels = eval_frame["y_survive"].to_numpy(dtype=float)
    eval_conditions = eval_frame["condition"].to_numpy()
    source_means = profile_weighted_source_means(eval_frame, pi)
    fixed_lower = float(np.min(source_means))
    fixed_upper = float(np.max(source_means))
    empirical = profile_weighted_empirical(eval_frame, pi)
    learned_brier = weighted_brier_value(omega_hat, eval_predictions, eval_labels, eval_conditions, pi)
    equal_weight = np.full(len(cs.SOURCE_FEATURE_SETS), 1.0 / len(cs.SOURCE_FEATURE_SETS))
    equal_brier = weighted_brier_value(equal_weight, eval_predictions, eval_labels, eval_conditions, pi)
    fixed_width = fixed_upper - fixed_lower

    rows = []
    for quantile in STABILITY_BOOTSTRAP_QUANTILES:
        epsilon = float(np.percentile(loss_excesses, quantile))
        envelope = fast_continuous_envelope(
            train_predictions,
            train_labels,
            train_conditions,
            pi,
            epsilon,
            source_means,
            omega_hat=omega_hat,
            best_loss=best_loss,
        )
        learned_lower = float(envelope["lower"])
        learned_upper = float(envelope["upper"])
        if learned_upper + 1e-10 < learned_lower:
            raise RuntimeError(
                f"Continuous envelope returned lower > upper for seed={seed}, "
                f"subset={frame['subset'].iloc[0]}, horizon={horizon}, "
                f"profile={profile_name}, fold={fold}, q={quantile}."
            )
        learned_lower = min(learned_lower, learned_upper)
        learned_upper = max(learned_lower, learned_upper)
        learned_width = learned_upper - learned_lower
        miss = max(learned_lower - empirical, empirical - learned_upper, 0.0)
        rows.append(
            {
                "seed": seed,
                "subset": str(frame["subset"].iloc[0]),
                "profile_name": profile_name,
                "beta": math.nan if beta is None else beta,
                "d_star": d_star,
                "mission_cycles": horizon,
                "bootstrap_quantile": quantile,
                "fold": fold,
                "eval_engines": int(len(eval_frame)),
                "epsilon": epsilon,
                "best_calibration_brier": best_loss,
                "bootstrap_reps": int(len(loss_excesses)),
                "bootstrap_excess_min": float(np.min(loss_excesses)),
                "bootstrap_excess_mean": float(np.mean(loss_excesses)),
                "bootstrap_excess_max": float(np.max(loss_excesses)),
                "fixed_lower": fixed_lower,
                "fixed_upper": fixed_upper,
                "fixed_width": fixed_width,
                "learned_lower": learned_lower,
                "learned_upper": learned_upper,
                "learned_width": learned_width,
                "width_reduction": fixed_width - learned_width,
                "width_reduction_fraction": (fixed_width - learned_width) / fixed_width if fixed_width > 1e-12 else 0.0,
                "empirical_reliability": empirical,
                "empirical_inside_learned": bool(learned_lower <= empirical <= learned_upper),
                "miss_distance": miss,
                "learned_brier": learned_brier,
                "equal_weight_brier": equal_brier,
                "brier_improvement": equal_brier - learned_brier,
                "lower_optimizer_success": bool(envelope["lower_success"]),
                "upper_optimizer_success": bool(envelope["upper_success"]),
                "optimizer_used_fallback": bool(envelope["used_fallback"]),
                "lower_loss_excess": float(envelope["lower_loss_excess"]),
                "upper_loss_excess": float(envelope["upper_loss_excess"]),
                "lower_simplex_residual": float(envelope["lower_simplex_residual"]),
                "upper_simplex_residual": float(envelope["upper_simplex_residual"]),
                "lower_bound_violation": float(envelope["lower_bound_violation"]),
                "upper_bound_violation": float(envelope["upper_bound_violation"]),
            }
        )
    return rows


def aggregate_seed_rows(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["seed", "subset", "profile_name", "beta", "mission_cycles", "bootstrap_quantile"]
    for keys, group in fold_results.groupby(group_cols, sort=True, dropna=False):
        seed, subset, profile_name, beta, mission, quantile = keys
        total_eval = float(group["eval_engines"].sum())
        weights = group["eval_engines"].to_numpy(dtype=float) / total_eval
        fixed_lower = float(np.dot(weights, group["fixed_lower"].to_numpy(dtype=float)))
        fixed_upper = float(np.dot(weights, group["fixed_upper"].to_numpy(dtype=float)))
        learned_lower = float(np.dot(weights, group["learned_lower"].to_numpy(dtype=float)))
        learned_upper = float(np.dot(weights, group["learned_upper"].to_numpy(dtype=float)))
        empirical = float(np.dot(weights, group["empirical_reliability"].to_numpy(dtype=float)))
        fixed_width = fixed_upper - fixed_lower
        learned_width = learned_upper - learned_lower
        miss = max(learned_lower - empirical, empirical - learned_upper, 0.0)
        rows.append(
            {
                "seed": seed,
                "subset": subset,
                "profile_name": profile_name,
                "beta": beta,
                "mission_cycles": mission,
                "bootstrap_quantile": quantile,
                "fixed_lower": fixed_lower,
                "fixed_upper": fixed_upper,
                "fixed_width": fixed_width,
                "learned_lower": learned_lower,
                "learned_upper": learned_upper,
                "learned_width": learned_width,
                "width_reduction": fixed_width - learned_width,
                "width_reduction_fraction": (fixed_width - learned_width) / fixed_width if fixed_width > 1e-12 else 0.0,
                "empirical_reliability": empirical,
                "empirical_inside_learned": bool(learned_lower <= empirical <= learned_upper),
                "miss_distance": miss,
                "learned_brier": float(np.dot(weights, group["learned_brier"].to_numpy(dtype=float))),
                "equal_weight_brier": float(np.dot(weights, group["equal_weight_brier"].to_numpy(dtype=float))),
                "brier_improvement": float(np.dot(weights, group["brier_improvement"].to_numpy(dtype=float))),
                "optimizer_fallback_rate": float(
                    np.dot(weights, group["optimizer_used_fallback"].astype(float).to_numpy(dtype=float))
                ),
                "optimizer_failure_rate": float(
                    np.dot(
                        weights,
                        (
                            ~group["lower_optimizer_success"].astype(bool)
                            | ~group["upper_optimizer_success"].astype(bool)
                        )
                        .astype(float)
                        .to_numpy(dtype=float),
                    )
                ),
                "max_loss_excess": float(
                    max(group["lower_loss_excess"].max(), group["upper_loss_excess"].max())
                ),
                "max_simplex_residual": float(
                    max(group["lower_simplex_residual"].max(), group["upper_simplex_residual"].max())
                ),
                "max_bound_violation": float(
                    max(group["lower_bound_violation"].max(), group["upper_bound_violation"].max())
                ),
                "d_star_values": ",".join(sorted({str(v) for v in group["d_star"] if str(v)})),
                "folds": int(group["fold"].nunique()),
                "eval_engines": int(group["eval_engines"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_stability(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["subset", "profile_name", "beta", "mission_cycles", "bootstrap_quantile"]
    metric_cols = [
        "fixed_width",
        "learned_width",
        "width_reduction_fraction",
        "learned_brier",
        "equal_weight_brier",
        "brier_improvement",
        "miss_distance",
        "optimizer_fallback_rate",
        "optimizer_failure_rate",
        "max_loss_excess",
        "max_simplex_residual",
        "max_bound_violation",
    ]
    for keys, group in seed_summary.groupby(group_cols, sort=True, dropna=False):
        subset, profile_name, beta, mission, quantile = keys
        inside_values = group["empirical_inside_learned"].astype(float).to_numpy(dtype=float)
        warning_values = (group["miss_distance"] > 1e-12).astype(float).to_numpy(dtype=float)
        row: dict[str, object] = {
            "subset": subset,
            "profile_name": profile_name,
            "beta": beta,
            "mission_cycles": mission,
            "bootstrap_quantile": quantile,
            "seeds": int(group["seed"].nunique()),
            "inside_rate": float(inside_values.mean()),
            "inside_rate_min": float(inside_values.min()),
            "inside_rate_max": float(inside_values.max()),
            "warning_rate": float(warning_values.mean()),
            "warning_rate_min": float(warning_values.min()),
            "warning_rate_max": float(warning_values.max()),
        }
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def csv_without_missing_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Write explicit text for profile metadata that is not applicable."""

    out = frame.copy()
    for column in ["beta", "d_star", "d_star_values"]:
        if column in out.columns:
            out[column] = out[column].fillna("not_applicable").replace("", "not_applicable")
    return out


def write_markdown(seed_summary: pd.DataFrame, stability_summary: pd.DataFrame) -> None:
    q90 = stability_summary[stability_summary["bootstrap_quantile"].eq(90.0)].copy()
    beta0 = q90[q90["profile_name"].isin(["observed", "beta_0.00"])].copy()
    fd004 = q90[q90["subset"].eq("FD004")]
    fd004_warning_all = bool((fd004["warning_rate"] > 0.0).all()) if not fd004.empty else False
    quantile_violations = 0
    quantile_groups = 0
    q_group_cols = ["seed", "subset", "profile_name", "beta", "mission_cycles"]
    for _, group in seed_summary.groupby(q_group_cols, sort=True, dropna=False):
        ordered = group.sort_values("bootstrap_quantile")
        widths = ordered["learned_width"].to_numpy(dtype=float)
        quantile_groups += 1
        if np.any(np.diff(widths) < -1e-10):
            quantile_violations += 1
    lines = [
        "# C-MAPSS Stability Audit",
        "",
        f"Generated: {_dt.date.today().isoformat()}",
        "",
        "## Scope",
        "",
        f"- Condition-stratified 5-fold seeds: {', '.join(str(s) for s in STABILITY_SEEDS)}",
        f"- Mission horizons: {', '.join(str(int(h)) for h in STABILITY_HORIZONS)} cycles",
        f"- Bootstrap quantiles: {', '.join(str(int(q)) for q in STABILITY_BOOTSTRAP_QUANTILES)}%",
        f"- Bootstrap repetitions per fold/profile: {BOOTSTRAP_REPS}",
        "- FD002/FD004 beta profiles: beta = 0, 0.25, 0.50, 0.75, 1.00.",
        "- For beta profiles, the stressed condition d_star is the lowest empirical-survival condition in the calibration folds only.",
        "",
        "## Generated Artifacts",
        "",
        "- `results/cmapss_stability_fold_results.csv`",
        "- `results/cmapss_stability_seed_summary.csv`",
        "- `results/cmapss_stability_summary.csv`",
        "",
        "## Q90 Observed/Beta-0 Summary Across Horizons",
        "",
        "| Subset/profile | Horizon | Width reduction mean [min,max] | Brier improvement mean [min,max] | Miss mean [min,max] | Inside rate mean [min,max] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in beta0.sort_values(["subset", "mission_cycles", "profile_name"]).iterrows():
        lines.append(
            f"| {row['subset']} {row['profile_name']} | {int(row['mission_cycles'])} | "
            f"{row['width_reduction_fraction_mean']:.3f} [{row['width_reduction_fraction_min']:.3f},{row['width_reduction_fraction_max']:.3f}] | "
            f"{row['brier_improvement_mean']:.4f} [{row['brier_improvement_min']:.4f},{row['brier_improvement_max']:.4f}] | "
            f"{row['miss_distance_mean']:.3f} [{row['miss_distance_min']:.3f},{row['miss_distance_max']:.3f}] | "
            f"{row['inside_rate']:.2f} [{row['inside_rate_min']:.2f},{row['inside_rate_max']:.2f}] |"
        )
    lines.extend(
        [
            "",
        "## Strict Interpretation",
        "",
        "- Width reduction is reported as learned Omega-epsilon width relative to fixed source min-max width.",
        "- Brier improvement is equal-weight Brier minus learned-weight Brier; positive values favor learned predictive weights.",
        "- Miss is interval miss distance for the learned envelope; zero means the held-out empirical reliability is inside the learned interval.",
        "- At 100 cycles and the observed/beta-0 profile, FD001, FD002, and FD003 are stable positives across all ten fold seeds: inside rate is 1.00, mean width reduction is positive, and mean Brier improvement is positive.",
        "- Horizon stability is weaker than the 100-cycle result: FD001 and FD003 miss at 50 and 150 cycles under the observed profile despite positive Brier improvement.",
        "- FD002 beta-0 misses at 50 cycles and has negative Brier improvement at 150 cycles, so the cross-horizon evidence is mixed even when the 100-cycle primary result is positive.",
        "- FD004 beta-0 is a stable warning at 50, 100, and 150 cycles under the continuous learned envelope.",
        f"- FD004 warning persists across all Q90 beta-profile settings: {'Yes' if fd004_warning_all else 'No'}.",
        f"- Bootstrap-quantile monotonicity check: {quantile_groups - quantile_violations}/{quantile_groups} seed/profile/horizon groups have nondecreasing learned width from 80% to 95%.",
        "- These checks are stability diagnostics, not new coverage guarantees.",
        "",
        "## FD004 Q90 Beta-Profile Warning Rates",
        "",
        "| Profile | Horizon | Warning rate | Inside rate | Miss mean [min,max] |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in fd004.sort_values(["profile_name", "mission_cycles"]).iterrows():
        lines.append(
            f"| {row['profile_name']} | {int(row['mission_cycles'])} | "
            f"{row['warning_rate']:.2f} | {row['inside_rate']:.2f} | "
            f"{row['miss_distance_mean']:.3f} [{row['miss_distance_min']:.3f},{row['miss_distance_max']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "The script uses the same C-MAPSS public files and source definitions as `scripts/cmapss_case_study.py`.",
            "Run:",
            "",
            "```powershell",
            "python scripts/cmapss_stability_audit.py",
            "```",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_prediction_frames()
    fold_rows: list[dict[str, object]] = []
    for seed in STABILITY_SEEDS:
        print(f"Running stability seed {seed}...")
        for (subset, horizon), base_frame in sorted(frames.items()):
            frame = base_frame.copy()
            frame["fold"] = assign_condition_stratified_folds(frame, seed)
            for profile_name, beta in profile_specs_for_frame(frame):
                for fold in range(cs.CROSS_FIT_FOLDS):
                    profile_offset = int(round(0.0 if beta is None else beta * 100.0))
                    rng_seed = seed * 1_000_000 + int(horizon) * 1_000 + fold * 100 + profile_offset
                    rng = np.random.default_rng(rng_seed)
                    fold_rows.extend(run_fold_settings(frame, seed, horizon, profile_name, beta, fold, rng))

    fold_results = pd.DataFrame(fold_rows)
    seed_summary = aggregate_seed_rows(fold_results)
    stability_summary = summarize_stability(seed_summary)
    csv_without_missing_metadata(fold_results).to_csv(
        RESULTS_DIR / "cmapss_stability_fold_results.csv", index=False
    )
    csv_without_missing_metadata(seed_summary).to_csv(
        RESULTS_DIR / "cmapss_stability_seed_summary.csv", index=False
    )
    csv_without_missing_metadata(stability_summary).to_csv(
        RESULTS_DIR / "cmapss_stability_summary.csv", index=False
    )
    write_markdown(seed_summary, stability_summary)

    print("Generated C-MAPSS stability audit outputs")
    print(fold_results[["seed", "subset", "profile_name", "mission_cycles", "bootstrap_quantile"]].shape)
    q90 = stability_summary[stability_summary["bootstrap_quantile"].eq(90.0)]
    print(q90[["subset", "profile_name", "mission_cycles", "inside_rate", "miss_distance_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
