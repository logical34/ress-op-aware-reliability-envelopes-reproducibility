"""C-MAPSS public benchmark case study for the RESS manuscript.

The script uses the NASA C-MAPSS turbofan degradation text files already stored
under ``../public_data/CMAPSS``. It builds compact reliability-envelope outputs
for FD001--FD004 without changing the source data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
CMAPSS_DIR = Path(os.environ.get("CMAPSS_DIR", ROOT.parent / "public_data" / "CMAPSS"))
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]
MISSION_CYCLES = [50.0, 100.0, 150.0]
Z_90 = 1.6448536269514722
R_REQ = 0.80
CREDIBILITY_MISSION = 100.0
CROSS_FIT_FOLDS = 5
CROSS_FIT_SEED = 20260620
BOOTSTRAP_SEED = CROSS_FIT_SEED + 100
BOOTSTRAP_REPS = 500
BOOTSTRAP_PERCENTILE = 90.0
BOOTSTRAP_SENSITIVITY_PERCENTILES = [80.0, 90.0, 95.0]
SIMPLEX_GRID_PARTS = 40
GRID_CHECK_PARTS = [20, 40, 80]
OPT_CONSTRAINT_TOL = 1e-7
OPT_FEASIBILITY_TOL = 5e-7

COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
SENSOR_GROUPS = {
    "temperature": ["s2", "s3", "s4"],
    "pressure": ["s7", "s11", "s15"],
    "speed_flow": ["s8", "s9", "s13", "s14"],
    "efficiency": ["s20", "s21"],
}

SOURCE_FEATURE_SETS = {
    "life_distribution": [],
    "temperature_degradation": ["temperature_delta"],
    "temperature_pressure_degradation": ["temperature_delta", "pressure_delta"],
    "all_sensor_degradation": [
        "temperature_delta",
        "pressure_delta",
        "speed_flow_delta",
        "efficiency_delta",
    ],
}

SOURCE_LABELS = {
    "life_distribution": "life",
    "temperature_degradation": "temp",
    "temperature_pressure_degradation": "temp+press",
    "all_sensor_degradation": "all sensors",
}

TEST_FEATURE_MAP = {
    "temperature_delta": "temperature_last_z",
    "pressure_delta": "pressure_last_z",
    "speed_flow_delta": "speed_flow_last_z",
    "efficiency_delta": "efficiency_last_z",
}

FD_META = {
    "FD001": {"conditions": 1, "fault_modes": 1, "description": "one condition, one fault mode"},
    "FD002": {"conditions": 6, "fault_modes": 1, "description": "six conditions, one fault mode"},
    "FD003": {"conditions": 1, "fault_modes": 2, "description": "one condition, two fault modes"},
    "FD004": {"conditions": 6, "fault_modes": 2, "description": "six conditions, two fault modes"},
}

COLOR = {
    "blue": "#174A7C",
    "blue2": "#4C78A8",
    "cyan": "#9ECAE1",
    "teal": "#4E9A9A",
    "green": "#6A994E",
    "orange": "#C77C2D",
    "red": "#B34A48",
    "magenta": "#8E5A83",
    "grey": "#6E6E6E",
    "lightgrey": "#D8DEE6",
    "panel": "#F7F8FA",
    "pale_yellow": "#FFF7D6",
    "pale_blue": "#EAF2F8",
    "pale_green": "#ECF6EF",
    "pale_orange": "#FDF0DF",
    "pale_red": "#F8E7E3",
    "black": "#2B2B2B",
}

SOURCE_COLORS = {
    "life_distribution": "#174A7C",
    "temperature_degradation": "#C77C2D",
    "temperature_pressure_degradation": "#6A994E",
    "all_sensor_degradation": "#B34A48",
}


@dataclass
class PreparedSet:
    fd: str
    train: pd.DataFrame
    test_last: pd.DataFrame
    train_life: pd.DataFrame
    test_rul: pd.DataFrame


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 8.8,
            "axes.titleweight": "normal",
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLOR["black"],
            "axes.linewidth": 0.75,
            "grid.color": "#E9ECEF",
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axis(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.62)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, width=0.70, color=COLOR["grey"])


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLOR["black"],
    )


def add_axis_note(ax, text: str, xy=(0.03, 0.95), fc="white", ec=None, fontsize=8.0) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=COLOR["black"],
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.08",
            "facecolor": fc,
            "edgecolor": ec or COLOR["lightgrey"],
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    return path


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_fd(fd: str) -> PreparedSet:
    train = load_frame(require_file(CMAPSS_DIR / f"train_{fd}.txt"))
    test = load_frame(require_file(CMAPSS_DIR / f"test_{fd}.txt"))
    rul = pd.read_csv(require_file(CMAPSS_DIR / f"RUL_{fd}.txt"), sep=r"\s+", header=None, names=["rul"])
    test_cycles = test.groupby("unit", as_index=False)["cycle"].max().rename(columns={"cycle": "observed_cycles"})
    rul = rul.copy()
    rul["unit"] = np.arange(1, len(rul) + 1)
    test_rul = test_cycles.merge(rul, on="unit", how="left")
    test_rul["total_life"] = test_rul["observed_cycles"] + test_rul["rul"]
    last_idx = test.groupby("unit")["cycle"].idxmax()
    test_last = test.loc[last_idx].sort_values("unit").reset_index(drop=True)
    test_last = test_last.merge(test_rul, on="unit", how="left", suffixes=("", "_summary"))
    train_life = train.groupby("unit", as_index=False)["cycle"].max().rename(columns={"cycle": "life"})
    return PreparedSet(fd=fd, train=train, test_last=test_last, train_life=train_life, test_rul=test_rul)


def add_condition_labels(prepared: PreparedSet) -> PreparedSet:
    train = prepared.train.copy()
    test_last = prepared.test_last.copy()
    if FD_META[prepared.fd]["conditions"] == 1:
        train["condition"] = "sea_level"
        test_last["condition"] = "sea_level"
        return PreparedSet(prepared.fd, train, test_last, prepared.train_life, prepared.test_rul)

    rounded = train[["op1", "op2", "op3"]].round(0)
    centers = rounded.drop_duplicates().sort_values(["op1", "op2", "op3"]).reset_index(drop=True)
    expected = FD_META[prepared.fd]["conditions"]
    if len(centers) != expected:
        raise ValueError(f"{prepared.fd} should have {expected} operational conditions, found {len(centers)}")

    labels = [f"condition_{idx + 1}" for idx in range(len(centers))]
    centers["condition"] = labels

    def nearest_condition(frame: pd.DataFrame) -> list[str]:
        values = frame[["op1", "op2", "op3"]].to_numpy(dtype=float)
        center_values = centers[["op1", "op2", "op3"]].to_numpy(dtype=float)
        distances = ((values[:, None, :] - center_values[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distances, axis=1)
        return [labels[i] for i in nearest]

    train["condition"] = nearest_condition(train)
    test_last["condition"] = nearest_condition(test_last)
    return PreparedSet(prepared.fd, train, test_last, prepared.train_life, prepared.test_rul)


def unit_feature_table(prepared: PreparedSet) -> pd.DataFrame:
    train = prepared.train.copy()
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    rows = []
    for unit, frame in train.groupby("unit", sort=True):
        last = frame.iloc[-1]
        first_window = frame.head(min(10, len(frame)))
        last_window = frame.tail(min(10, len(frame)))
        row = {
            "unit": int(unit),
            "life": float(frame["cycle"].max()),
            "condition": str(frame["condition"].mode().iloc[0]),
        }
        for group_name, sensors in SENSOR_GROUPS.items():
            valid = [sensor for sensor in sensors if frame[sensor].std() > 1e-9]
            if not valid:
                row[f"{group_name}_delta"] = 0.0
                continue
            deltas = []
            for sensor in valid:
                sigma = train[sensor].std()
                sigma = sigma if sigma > 1e-9 else 1.0
                deltas.append((last_window[sensor].mean() - first_window[sensor].mean()) / sigma)
            row[f"{group_name}_delta"] = float(np.mean(deltas))
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_table(prepared: PreparedSet) -> pd.DataFrame:
    train = prepared.train.copy()
    test_last = prepared.test_last.copy()
    rows = []
    for _, row in test_last.iterrows():
        out = {
            "unit": int(row["unit"]),
            "observed_cycles": float(row["observed_cycles"]),
            "rul": float(row["rul"]),
            "total_life": float(row["total_life"]),
            "condition": str(row["condition"]),
        }
        for group_name, sensors in SENSOR_GROUPS.items():
            valid = [sensor for sensor in sensors if train[sensor].std() > 1e-9]
            if not valid:
                out[f"{group_name}_last_z"] = 0.0
                continue
            zs = []
            for sensor in valid:
                mu = train[sensor].mean()
                sigma = train[sensor].std()
                sigma = sigma if sigma > 1e-9 else 1.0
                zs.append((float(row[sensor]) - mu) / sigma)
            out[f"{group_name}_last_z"] = float(np.mean(zs))
        rows.append(out)
    return pd.DataFrame(rows)


def fit_linear_predictor(features: pd.DataFrame, feature_cols: list[str], target: str = "life") -> tuple[np.ndarray, float]:
    x = features[feature_cols].to_numpy(dtype=float)
    y = np.log(features[target].to_numpy(dtype=float))
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    z = (x - mu) / sigma
    design = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    resid_sigma = float(np.sqrt(np.mean(residuals**2)))
    return np.r_[beta, mu, sigma], max(resid_sigma, 1e-6)


def predict_log_life(params: np.ndarray, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    beta = params[: len(feature_cols) + 1]
    mu = params[len(feature_cols) + 1 : len(feature_cols) + 1 + len(feature_cols)]
    sigma = params[len(feature_cols) + 1 + len(feature_cols) :]
    x = features[feature_cols].to_numpy(dtype=float)
    z = (x - mu) / sigma
    design = np.column_stack([np.ones(len(z)), z])
    return design @ beta


def survival_from_log_life(mu: np.ndarray, sigma: float, total_age: np.ndarray) -> np.ndarray:
    total_age = np.maximum(total_age, 1.0)
    z = (np.log(total_age) - mu) / sigma
    return 1.0 - norm.cdf(z)


def empirical_survival(lives: np.ndarray, ages: np.ndarray) -> np.ndarray:
    return np.array([np.mean(lives > age) for age in ages], dtype=float)


def wilson_interval(successes: int, total: int, z: float = Z_90) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def decision_category(lower: float, upper: float, req: float = R_REQ) -> str:
    if lower >= req:
        return "acceptable"
    if upper < req:
        return "unacceptable/restricted"
    return "uncertain/review"


def simplex_grid(n_sources: int, parts: int = SIMPLEX_GRID_PARTS) -> np.ndarray:
    rows = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], parts, n_sources)
    return np.array(rows, dtype=float) / float(parts)


SIMPLEX_WEIGHTS = simplex_grid(len(SOURCE_FEATURE_SETS), SIMPLEX_GRID_PARTS)
_SIMPLEX_GRID_CACHE: dict[tuple[int, int], np.ndarray] = {
    (len(SOURCE_FEATURE_SETS), SIMPLEX_GRID_PARTS): SIMPLEX_WEIGHTS
}


def cached_simplex_grid(n_sources: int, parts: int) -> np.ndarray:
    key = (int(n_sources), int(parts))
    if key not in _SIMPLEX_GRID_CACHE:
        _SIMPLEX_GRID_CACHE[key] = simplex_grid(n_sources, parts)
    return _SIMPLEX_GRID_CACHE[key]


def profile_weights(frame: pd.DataFrame, profile: dict[str, float] | None = None) -> dict[str, float]:
    if profile is not None:
        return dict(profile)
    counts = frame["condition"].value_counts(normalize=True).sort_index()
    return {str(condition): float(value) for condition, value in counts.items()}


def brier_loss(
    omega: np.ndarray, predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> float:
    total = 0.0
    active = 0.0
    pred = predictions @ omega
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        total += float(pi_d) * float(np.mean((pred[mask] - labels[mask]) ** 2))
        active += float(pi_d)
    if active <= 1e-12:
        return float(np.mean((pred - labels) ** 2))
    return total / active


def brier_loss_gradient(
    omega: np.ndarray, predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    grad = np.zeros_like(omega, dtype=float)
    active = 0.0
    pred = predictions @ omega
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        residual = pred[mask] - labels[mask]
        grad += float(pi_d) * (2.0 * predictions[mask].T @ residual / float(mask.sum()))
        active += float(pi_d)
    if active <= 1e-12:
        residual = pred - labels
        return 2.0 * predictions.T @ residual / float(len(labels))
    return grad / active


def brier_loss_many(
    omegas: np.ndarray, predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> np.ndarray:
    weights = np.asarray(omegas, dtype=float)
    pred = predictions @ weights.T
    labels_col = labels.reshape(-1, 1)
    losses = np.zeros(weights.shape[0], dtype=float)
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        losses += float(pi_d) * np.mean((pred[mask] - labels_col[mask]) ** 2, axis=0)
        active += float(pi_d)
    if active <= 1e-12:
        return np.mean((pred - labels_col) ** 2, axis=0)
    return losses / active


def profile_weighted_prediction_brier(
    probabilities: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> float:
    total = 0.0
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        total += float(pi_d) * float(np.mean((probabilities[mask] - labels[mask]) ** 2))
        active += float(pi_d)
    if active <= 1e-12:
        return float(np.mean((probabilities - labels) ** 2))
    return total / active


def fit_brier_weight(
    predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float]
) -> tuple[np.ndarray, float, bool]:
    n_sources = predictions.shape[1]
    initial = np.full(n_sources, 1.0 / n_sources)
    bounds = [(0.0, 1.0)] * n_sources
    constraints = [{"type": "eq", "fun": lambda omega: float(np.sum(omega) - 1.0)}]

    def objective(omega: np.ndarray) -> float:
        return brier_loss(np.asarray(omega), predictions, labels, conditions, pi)

    def objective_jac(omega: np.ndarray) -> np.ndarray:
        return brier_loss_gradient(np.asarray(omega), predictions, labels, conditions, pi)

    result = minimize(
        objective, initial, jac=objective_jac, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        values = np.array([objective(row) for row in SIMPLEX_WEIGHTS], dtype=float)
        idx = int(np.argmin(values))
        return SIMPLEX_WEIGHTS[idx].copy(), float(values[idx]), False
    omega = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    total = float(np.sum(omega))
    omega = omega / total if total > 1e-12 else initial
    return omega, float(objective(omega)), True


def bootstrap_epsilon_diagnostics(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
    percentiles: list[float] | None = None,
) -> dict:
    started = time.perf_counter()
    if percentiles is None:
        percentiles = BOOTSTRAP_SENSITIVITY_PERCENTILES
    omega_hat_original, _, original_optimizer_success = fit_brier_weight(predictions, labels, conditions, pi)
    excess_losses = []
    optimizer_failures = 0
    empty_replicates = 0
    unique_conditions = np.array(sorted(set(conditions)))
    for replicate in range(reps):
        sample_indices = []
        for condition in unique_conditions:
            idx = np.flatnonzero(conditions == condition)
            if len(idx) == 0:
                continue
            sample_indices.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        if not sample_indices:
            empty_replicates += 1
            continue
        sample_indices = np.array(sample_indices, dtype=int)
        sample_predictions = predictions[sample_indices]
        sample_labels = labels[sample_indices]
        sample_conditions = conditions[sample_indices]
        omega_hat_b, bootstrap_optimum_loss, optimizer_success = fit_brier_weight(
            sample_predictions, sample_labels, sample_conditions, pi
        )
        if not optimizer_success:
            optimizer_failures += 1
        original_loss_on_bootstrap = brier_loss(
            omega_hat_original, sample_predictions, sample_labels, sample_conditions, pi
        )
        delta = original_loss_on_bootstrap - bootstrap_optimum_loss
        excess_losses.append(max(0.0, float(delta)))
    successful_reps = len(excess_losses)
    if successful_reps == 0:
        percentile_values = {float(q): 0.0 for q in percentiles}
    else:
        values = np.array(excess_losses, dtype=float)
        percentile_values = {float(q): float(np.percentile(values, q)) for q in percentiles}
    epsilon = float(percentile_values.get(float(BOOTSTRAP_PERCENTILE), 0.0))
    return {
        "epsilon": max(0.0, epsilon),
        "percentiles": percentile_values,
        "successful_reps": int(successful_reps),
        "requested_reps": int(reps),
        "empty_replicates": int(empty_replicates),
        "optimizer_failures": int(optimizer_failures),
        "bootstrap_failure_rate": float((reps - successful_reps) / reps) if reps > 0 else 1.0,
        "bootstrap_optimizer_failure_rate": float(optimizer_failures / successful_reps) if successful_reps > 0 else 1.0,
        "original_optimizer_success": bool(original_optimizer_success),
        "excess_loss_min": float(np.min(excess_losses)) if excess_losses else 0.0,
        "excess_loss_mean": float(np.mean(excess_losses)) if excess_losses else 0.0,
        "excess_loss_max": float(np.max(excess_losses)) if excess_losses else 0.0,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def bootstrap_epsilon(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> float:
    return float(bootstrap_epsilon_diagnostics(predictions, labels, conditions, pi, rng, reps)["epsilon"])


def admissible_grid(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    parts: int = SIMPLEX_GRID_PARTS,
    omega_hat: np.ndarray | None = None,
    best_loss: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if omega_hat is None or best_loss is None:
        omega_hat, best_loss, _ = fit_brier_weight(predictions, labels, conditions, pi)
    grid = cached_simplex_grid(predictions.shape[1], parts)
    losses = brier_loss_many(grid, predictions, labels, conditions, pi)
    admissible = grid[losses <= best_loss + epsilon + 1e-12]
    if len(admissible) == 0:
        admissible = omega_hat.reshape(1, -1)
    return admissible, best_loss, losses


def directional_feasible_starts(
    omega_hat: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    loss_limit: float,
) -> list[np.ndarray]:
    starts = [np.asarray(omega_hat, dtype=float)]
    n_sources = len(omega_hat)
    equal = np.full(n_sources, 1.0 / n_sources)
    if brier_loss(equal, predictions, labels, conditions, pi) <= loss_limit + OPT_FEASIBILITY_TOL:
        starts.append(equal)
    for idx in range(n_sources):
        vertex = np.zeros(n_sources, dtype=float)
        vertex[idx] = 1.0
        if brier_loss(vertex, predictions, labels, conditions, pi) <= loss_limit + OPT_FEASIBILITY_TOL:
            starts.append(vertex)
            continue
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            candidate = (1.0 - mid) * omega_hat + mid * vertex
            if brier_loss(candidate, predictions, labels, conditions, pi) <= loss_limit + OPT_FEASIBILITY_TOL:
                lo = mid
            else:
                hi = mid
        starts.append((1.0 - lo) * omega_hat + lo * vertex)
    unique = []
    for start in starts:
        start = np.clip(start, 0.0, 1.0)
        total = float(start.sum())
        start = start / total if total > 1e-12 else np.asarray(omega_hat, dtype=float)
        if not any(np.allclose(start, old, atol=1e-10, rtol=0.0) for old in unique):
            unique.append(start)
    return unique


def optimize_linear_over_omega(
    direction: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    sense: str,
    omega_hat: np.ndarray | None = None,
    best_loss: float | None = None,
) -> dict:
    if omega_hat is None or best_loss is None:
        omega_hat, best_loss, _ = fit_brier_weight(predictions, labels, conditions, pi)
    direction = np.asarray(direction, dtype=float)
    n_sources = len(direction)
    loss_limit = float(best_loss + epsilon)
    bounds = [(0.0, 1.0)] * n_sources
    sign = 1.0 if sense == "min" else -1.0

    def objective(omega: np.ndarray) -> float:
        return sign * float(np.dot(direction, omega))

    def normalize_omega(omega: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(omega, dtype=float), 0.0, 1.0)
        total = float(normalized.sum())
        return normalized / total if total > 1e-12 else np.asarray(omega_hat, dtype=float)

    def candidate_from_result(result, omega_source: np.ndarray, method: str) -> dict:
        omega = normalize_omega(result.x if np.all(np.isfinite(result.x)) else omega_source)
        loss_value = brier_loss(omega, predictions, labels, conditions, pi)
        simplex_residual = abs(float(omega.sum() - 1.0))
        loss_excess = float(loss_value - loss_limit)
        bound_violation = max(float(-omega.min()), float(omega.max() - 1.0), 0.0)
        feasible = (
            simplex_residual <= OPT_FEASIBILITY_TOL
            and loss_excess <= OPT_FEASIBILITY_TOL
            and bound_violation <= OPT_FEASIBILITY_TOL
        )
        return {
            "omega": omega,
            "objective_value": float(np.dot(direction, omega)),
            "loss": float(loss_value),
            "loss_limit": loss_limit,
            "loss_excess": loss_excess,
            "simplex_residual": simplex_residual,
            "bound_violation": bound_violation,
            "optimizer_success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "method": method,
            "feasible": bool(feasible),
        }

    def select_best(items: list[dict]) -> dict:
        key = (lambda item: item["objective_value"]) if sense == "min" else (lambda item: -item["objective_value"])
        return min(items, key=key)

    constraints = [
        {
            "type": "eq",
            "fun": lambda omega: float(np.sum(omega) - 1.0),
            "jac": lambda omega: np.ones(n_sources, dtype=float),
        },
        {
            "type": "ineq",
            "fun": lambda omega: float(loss_limit - brier_loss(np.asarray(omega), predictions, labels, conditions, pi)),
            "jac": lambda omega: -brier_loss_gradient(
                np.asarray(omega), predictions, labels, conditions, pi
            ),
        },
    ]
    candidates = []
    for start in directional_feasible_starts(omega_hat, predictions, labels, conditions, pi, loss_limit):
        result = minimize(
            objective,
            start,
            jac=lambda omega: sign * direction,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        candidates.append(candidate_from_result(result, start, "SLSQP"))
    feasible_candidates = [item for item in candidates if item["feasible"]]
    successful_candidates = [item for item in feasible_candidates if item["optimizer_success"]]
    best_feasible = select_best(feasible_candidates) if feasible_candidates else None
    if best_feasible is not None and best_feasible["optimizer_success"]:
        best = best_feasible
        best["used_fallback"] = False
        return best

    rescue_starts = []
    if best_feasible is not None:
        rescue_starts.append(best_feasible["omega"])
    rescue_starts.extend(item["omega"] for item in feasible_candidates)
    rescue_starts.extend(directional_feasible_starts(omega_hat, predictions, labels, conditions, pi, loss_limit))
    rescue_candidates = []
    cobyla_constraints = [
        {"type": "ineq", "fun": lambda omega: float(1.0 - np.sum(omega))},
        {"type": "ineq", "fun": lambda omega: float(np.sum(omega) - 1.0)},
        {
            "type": "ineq",
            "fun": lambda omega: float(loss_limit - brier_loss(np.asarray(omega), predictions, labels, conditions, pi)),
        },
    ]
    for idx in range(n_sources):
        cobyla_constraints.append({"type": "ineq", "fun": lambda omega, idx=idx: float(omega[idx])})
        cobyla_constraints.append({"type": "ineq", "fun": lambda omega, idx=idx: float(1.0 - omega[idx])})
    unique_rescue_starts = []
    for start in rescue_starts:
        start = normalize_omega(start)
        if not any(np.allclose(start, old, atol=1e-10, rtol=0.0) for old in unique_rescue_starts):
            unique_rescue_starts.append(start)
    for start in unique_rescue_starts:
        result = minimize(
            objective,
            start,
            method="COBYLA",
            constraints=cobyla_constraints,
            options={"tol": 1e-10, "maxiter": 5000, "rhobeg": 0.1, "catol": 1e-10, "disp": False},
        )
        rescue_candidates.append(candidate_from_result(result, start, "COBYLA"))
    successful_rescue = [item for item in rescue_candidates if item["feasible"] and item["optimizer_success"]]
    if successful_rescue:
        best = select_best(successful_candidates + successful_rescue)
        best["used_fallback"] = False
        return best
    if successful_candidates:
        best = select_best(successful_candidates)
        best["used_fallback"] = False
        return best
    if feasible_candidates:
        best = best_feasible
        best["used_fallback"] = False
        return best

    fallback = {
        "omega": np.asarray(omega_hat, dtype=float),
        "objective_value": float(np.dot(direction, omega_hat)),
        "loss": float(brier_loss(omega_hat, predictions, labels, conditions, pi)),
        "loss_limit": loss_limit,
        "loss_excess": float(brier_loss(omega_hat, predictions, labels, conditions, pi) - loss_limit),
        "simplex_residual": abs(float(np.sum(omega_hat) - 1.0)),
        "bound_violation": max(float(-np.min(omega_hat)), float(np.max(omega_hat) - 1.0), 0.0),
        "optimizer_success": False,
        "status": -1,
        "message": "no feasible SLSQP candidate; used omega_hat fallback",
        "method": "fallback",
        "feasible": True,
        "used_fallback": True,
    }
    return fallback


def continuous_envelope(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    source_means: np.ndarray,
    omega_hat: np.ndarray | None = None,
    best_loss: float | None = None,
) -> dict:
    if omega_hat is None or best_loss is None:
        omega_hat, best_loss, _ = fit_brier_weight(predictions, labels, conditions, pi)
    lower = optimize_linear_over_omega(
        source_means, predictions, labels, conditions, pi, epsilon, "min", omega_hat=omega_hat, best_loss=best_loss
    )
    upper = optimize_linear_over_omega(
        source_means, predictions, labels, conditions, pi, epsilon, "max", omega_hat=omega_hat, best_loss=best_loss
    )
    return {
        "lower": float(lower["objective_value"]),
        "upper": float(upper["objective_value"]),
        "width": float(upper["objective_value"] - lower["objective_value"]),
        "lower_omega": lower["omega"],
        "upper_omega": upper["omega"],
        "lower_success": bool(lower["optimizer_success"] and not lower["used_fallback"]),
        "upper_success": bool(upper["optimizer_success"] and not upper["used_fallback"]),
        "lower_status": int(lower["status"]),
        "upper_status": int(upper["status"]),
        "lower_message": lower["message"],
        "upper_message": upper["message"],
        "lower_method": lower["method"],
        "upper_method": upper["method"],
        "lower_loss_excess": float(lower["loss_excess"]),
        "upper_loss_excess": float(upper["loss_excess"]),
        "lower_simplex_residual": float(lower["simplex_residual"]),
        "upper_simplex_residual": float(upper["simplex_residual"]),
        "lower_bound_violation": float(lower["bound_violation"]),
        "upper_bound_violation": float(upper["bound_violation"]),
        "used_fallback": bool(lower["used_fallback"] or upper["used_fallback"]),
    }


def continuous_coordinate_bounds(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    omega_hat: np.ndarray,
    best_loss: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    lows = []
    highs = []
    success = True
    for idx in range(predictions.shape[1]):
        direction = np.zeros(predictions.shape[1], dtype=float)
        direction[idx] = 1.0
        lower = optimize_linear_over_omega(
            direction, predictions, labels, conditions, pi, epsilon, "min", omega_hat=omega_hat, best_loss=best_loss
        )
        upper = optimize_linear_over_omega(
            direction, predictions, labels, conditions, pi, epsilon, "max", omega_hat=omega_hat, best_loss=best_loss
        )
        lows.append(float(lower["objective_value"]))
        highs.append(float(upper["objective_value"]))
        success = success and bool(lower["optimizer_success"] and upper["optimizer_success"])
    return np.array(lows, dtype=float), np.array(highs, dtype=float), bool(success)


def grid_envelope_check(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    epsilon: float,
    source_means: np.ndarray,
    omega_hat: np.ndarray,
    best_loss: float,
) -> dict:
    out: dict[str, float | int] = {}
    for parts in GRID_CHECK_PARTS:
        admissible, _, _ = admissible_grid(
            predictions, labels, conditions, pi, epsilon, parts=parts, omega_hat=omega_hat, best_loss=best_loss
        )
        values = admissible @ source_means
        out[f"grid{parts}_lower"] = float(values.min())
        out[f"grid{parts}_upper"] = float(values.max())
        out[f"grid{parts}_width"] = float(values.max() - values.min())
        out[f"grid{parts}_admissible_count"] = int(len(admissible))
    return out


def full_simplex_loss_epsilon(
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    pi: dict[str, float],
    best_loss: float,
) -> float:
    vertices = np.eye(predictions.shape[1], dtype=float)
    vertex_losses = brier_loss_many(vertices, predictions, labels, conditions, pi)
    return max(0.0, float(vertex_losses.max() - best_loss + 1e-10))


def assign_condition_stratified_folds(frame: pd.DataFrame, n_folds: int = CROSS_FIT_FOLDS) -> pd.Series:
    rng = np.random.default_rng(CROSS_FIT_SEED)
    folds = pd.Series(index=frame.index, dtype=int)
    for _, condition_frame in frame.groupby("condition", sort=True):
        indices = condition_frame.index.to_numpy()
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for pos, idx in enumerate(shuffled):
            folds.loc[idx] = pos % n_folds
    return folds.astype(int)


def source_prediction_frame(
    p: PreparedSet, test_features: pd.DataFrame, model_predictions: dict[str, dict[float, np.ndarray]]
) -> pd.DataFrame:
    rows = []
    for idx, row in test_features.reset_index(drop=True).iterrows():
        out = {
            "subset": p.fd,
            "engine_id": int(row["unit"]),
            "condition": str(row["condition"]),
            "observed_cycles": float(row["observed_cycles"]),
            "rul": float(row["rul"]),
            "mission_cycles": CREDIBILITY_MISSION,
            "y_survive_100": int(float(row["rul"]) > CREDIBILITY_MISSION),
        }
        for source_name in SOURCE_FEATURE_SETS:
            out[f"r_{source_name}_100"] = float(model_predictions[source_name][CREDIBILITY_MISSION][idx])
        rows.append(out)
    frame = pd.DataFrame(rows)
    frame["fold"] = assign_condition_stratified_folds(frame)
    return frame


def prediction_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"r_{source}_100" for source in SOURCE_FEATURE_SETS]].to_numpy(dtype=float)


def source_names() -> list[str]:
    return list(SOURCE_FEATURE_SETS.keys())


def learned_interval_for_frame(
    eval_frame: pd.DataFrame,
    admissible: np.ndarray,
) -> tuple[float, float]:
    predictions = prediction_matrix(eval_frame)
    source_means = predictions.mean(axis=0)
    values = admissible @ source_means
    return float(np.min(values)), float(np.max(values))


def fd_summary(prepared_sets: list[PreparedSet]) -> pd.DataFrame:
    rows = []
    for p in prepared_sets:
        rows.append(
            {
                "subset": p.fd,
                "conditions": FD_META[p.fd]["conditions"],
                "fault_modes": FD_META[p.fd]["fault_modes"],
                "train_engines": int(p.train["unit"].nunique()),
                "test_engines": int(p.test_last["unit"].nunique()),
                "train_rows": int(len(p.train)),
                "test_rows": int(p.test_last["observed_cycles"].sum()),
                "median_train_life": float(p.train_life["life"].median()),
                "median_test_total_life": float(p.test_last["total_life"].median()),
                "description": FD_META[p.fd]["description"],
            }
        )
    return pd.DataFrame(rows)


def profile_catalog(frame: pd.DataFrame) -> list[tuple[str, dict[str, float]]]:
    observed = profile_weights(frame)
    profiles = [("observed", observed)]
    conditions = sorted(observed)
    if len(conditions) > 1:
        shifted = dict(observed)
        target = conditions[-1]
        donors = [condition for condition in conditions if condition != target]
        donor = max(donors, key=lambda condition: shifted[condition])
        shift = min(0.15, 0.5 * shifted[donor])
        shifted[donor] -= shift
        shifted[target] += shift
        total = sum(shifted.values())
        shifted = {condition: value / total for condition, value in shifted.items()}
        profiles.append(("shifted", shifted))
    return profiles


def profile_weighted_source_means(frame: pd.DataFrame, pi: dict[str, float]) -> np.ndarray:
    matrix = prediction_matrix(frame)
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


def profile_weighted_empirical(frame: pd.DataFrame, pi: dict[str, float]) -> tuple[float, float, float]:
    labels = frame["y_survive_100"].to_numpy(dtype=float)
    conditions = frame["condition"].to_numpy()
    empirical = 0.0
    lower = 0.0
    upper = 0.0
    active = 0.0
    for condition, pi_d in pi.items():
        mask = conditions == condition
        if not np.any(mask):
            continue
        condition_labels = labels[mask]
        lo, hi = wilson_interval(int(condition_labels.sum()), int(len(condition_labels)))
        empirical += float(pi_d) * float(condition_labels.mean())
        lower += float(pi_d) * lo
        upper += float(pi_d) * hi
        active += float(pi_d)
    if active <= 1e-12:
        lo, hi = wilson_interval(int(labels.sum()), int(len(labels)))
        return float(labels.mean()), lo, hi
    return empirical / active, lower / active, upper / active


def evaluate_crossfit_profile(
    pred_frame: pd.DataFrame,
    subset: str,
    profile_name: str,
    pi: dict[str, float],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    fold_rows = []
    omega_rows = []
    baseline_rows = []
    for fold in range(CROSS_FIT_FOLDS):
        train = pred_frame[pred_frame["fold"] != fold].copy()
        eval_frame = pred_frame[pred_frame["fold"] == fold].copy()
        train_idx = set(train["engine_id"].astype(int))
        eval_idx = set(eval_frame["engine_id"].astype(int))
        train_predictions = prediction_matrix(train)
        train_labels = train["y_survive_100"].to_numpy(dtype=float)
        train_conditions = train["condition"].to_numpy()
        bootstrap_diag = bootstrap_epsilon_diagnostics(train_predictions, train_labels, train_conditions, pi, rng)
        epsilon = float(bootstrap_diag["epsilon"])
        omega_hat, best_loss, optimizer_success = fit_brier_weight(train_predictions, train_labels, train_conditions, pi)
        grid40_admissible, _, _ = admissible_grid(
            train_predictions,
            train_labels,
            train_conditions,
            pi,
            epsilon,
            parts=SIMPLEX_GRID_PARTS,
            omega_hat=omega_hat,
            best_loss=best_loss,
        )

        source_means = profile_weighted_source_means(eval_frame, pi)
        fixed_lower = float(np.min(source_means))
        fixed_upper = float(np.max(source_means))
        continuous = continuous_envelope(
            train_predictions,
            train_labels,
            train_conditions,
            pi,
            epsilon,
            source_means,
            omega_hat=omega_hat,
            best_loss=best_loss,
        )
        grid_check = grid_envelope_check(
            train_predictions,
            train_labels,
            train_conditions,
            pi,
            epsilon,
            source_means,
            omega_hat,
            best_loss,
        )
        learned_lower = float(continuous["lower"])
        learned_upper = float(continuous["upper"])
        learned_width = float(continuous["width"])
        max_grid_endpoint_gap = max(
            abs(float(grid_check[f"grid{parts}_lower"]) - learned_lower)
            for parts in GRID_CHECK_PARTS
        )
        max_grid_endpoint_gap = max(
            max_grid_endpoint_gap,
            max(abs(float(grid_check[f"grid{parts}_upper"]) - learned_upper) for parts in GRID_CHECK_PARTS),
        )
        empirical, wilson_lower, wilson_upper = profile_weighted_empirical(eval_frame, pi)
        eval_predictions = prediction_matrix(eval_frame)
        eval_labels = eval_frame["y_survive_100"].to_numpy(dtype=float)
        eval_conditions = eval_frame["condition"].to_numpy()
        learned_brier = brier_loss(omega_hat, eval_predictions, eval_labels, eval_conditions, pi)
        equal_weight = np.full(len(source_names()), 1.0 / len(source_names()))
        equal_brier = brier_loss(equal_weight, eval_predictions, eval_labels, eval_conditions, pi)
        learned_point = float(omega_hat @ source_means)
        equal_point = float(equal_weight @ source_means)
        fixed_midpoint_probabilities = 0.5 * (eval_predictions.min(axis=1) + eval_predictions.max(axis=1))
        fixed_midpoint_brier = profile_weighted_prediction_brier(
            fixed_midpoint_probabilities, eval_labels, eval_conditions, pi
        )

        fold_rows.append(
            {
                "subset": subset,
                "profile_name": profile_name,
                "fold": fold,
                "train_engines": int(len(train)),
                "eval_engines": int(len(eval_frame)),
                "train_eval_intersection": int(len(train_idx.intersection(eval_idx))),
                "epsilon": epsilon,
                "epsilon_q80": float(bootstrap_diag["percentiles"][80.0]),
                "epsilon_q90": float(bootstrap_diag["percentiles"][90.0]),
                "epsilon_q95": float(bootstrap_diag["percentiles"][95.0]),
                "bootstrap_reps": int(bootstrap_diag["requested_reps"]),
                "bootstrap_successful_reps": int(bootstrap_diag["successful_reps"]),
                "bootstrap_empty_replicates": int(bootstrap_diag["empty_replicates"]),
                "bootstrap_optimizer_failures": int(bootstrap_diag["optimizer_failures"]),
                "bootstrap_failure_rate": float(bootstrap_diag["bootstrap_failure_rate"]),
                "bootstrap_optimizer_failure_rate": float(bootstrap_diag["bootstrap_optimizer_failure_rate"]),
                "excess_loss_mean": float(bootstrap_diag["excess_loss_mean"]),
                "excess_loss_max": float(bootstrap_diag["excess_loss_max"]),
                "bootstrap_runtime_seconds": float(bootstrap_diag["runtime_seconds"]),
                "best_calibration_brier": best_loss,
                "optimizer_success": bool(optimizer_success),
                "continuous_lower_success": bool(continuous["lower_success"]),
                "continuous_upper_success": bool(continuous["upper_success"]),
                "continuous_used_fallback": bool(continuous["used_fallback"]),
                "continuous_lower_status": int(continuous["lower_status"]),
                "continuous_upper_status": int(continuous["upper_status"]),
                "continuous_lower_method": str(continuous["lower_method"]),
                "continuous_upper_method": str(continuous["upper_method"]),
                "continuous_lower_loss_excess": float(continuous["lower_loss_excess"]),
                "continuous_upper_loss_excess": float(continuous["upper_loss_excess"]),
                "continuous_lower_simplex_residual": float(continuous["lower_simplex_residual"]),
                "continuous_upper_simplex_residual": float(continuous["upper_simplex_residual"]),
                "continuous_lower_bound_violation": float(continuous["lower_bound_violation"]),
                "continuous_upper_bound_violation": float(continuous["upper_bound_violation"]),
                "grid_endpoint_gap_max": float(max_grid_endpoint_gap),
                "admissible_count": int(grid_check["grid40_admissible_count"]),
                "grid20_admissible_count": int(grid_check["grid20_admissible_count"]),
                "grid40_admissible_count": int(grid_check["grid40_admissible_count"]),
                "grid80_admissible_count": int(grid_check["grid80_admissible_count"]),
                "grid20_lower": float(grid_check["grid20_lower"]),
                "grid20_upper": float(grid_check["grid20_upper"]),
                "grid40_lower": float(grid_check["grid40_lower"]),
                "grid40_upper": float(grid_check["grid40_upper"]),
                "grid80_lower": float(grid_check["grid80_lower"]),
                "grid80_upper": float(grid_check["grid80_upper"]),
                "fixed_lower": fixed_lower,
                "fixed_upper": fixed_upper,
                "fixed_width": fixed_upper - fixed_lower,
                "learned_lower": learned_lower,
                "learned_upper": learned_upper,
                "learned_width": learned_width,
                "width_reduction": (fixed_upper - fixed_lower) - learned_width,
                "empirical_reliability": empirical,
                "wilson90_lower": wilson_lower,
                "wilson90_upper": wilson_upper,
                "learned_brier": learned_brier,
                "equal_weight_brier": equal_brier,
                "miss_distance": max(learned_lower - empirical, empirical - learned_upper, 0.0),
            }
        )
        omega_lower, omega_upper, coordinate_success = continuous_coordinate_bounds(
            train_predictions,
            train_labels,
            train_conditions,
            pi,
            epsilon,
            omega_hat,
            best_loss,
        )
        grid_omega_lower = grid40_admissible.min(axis=0)
        grid_omega_upper = grid40_admissible.max(axis=0)
        for idx, source in enumerate(source_names()):
            omega_rows.append(
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "source_model": source,
                    "omega_hat": float(omega_hat[idx]),
                    "omega_lower": float(omega_lower[idx]),
                    "omega_upper": float(omega_upper[idx]),
                    "grid40_omega_lower": float(grid_omega_lower[idx]),
                    "grid40_omega_upper": float(grid_omega_upper[idx]),
                    "coordinate_optimizer_success": bool(coordinate_success),
                    "epsilon": epsilon,
                    "epsilon_q80": float(bootstrap_diag["percentiles"][80.0]),
                    "epsilon_q90": float(bootstrap_diag["percentiles"][90.0]),
                    "epsilon_q95": float(bootstrap_diag["percentiles"][95.0]),
                    "bootstrap_reps": int(bootstrap_diag["requested_reps"]),
                    "bootstrap_successful_reps": int(bootstrap_diag["successful_reps"]),
                    "bootstrap_failure_rate": float(bootstrap_diag["bootstrap_failure_rate"]),
                    "bootstrap_optimizer_failure_rate": float(bootstrap_diag["bootstrap_optimizer_failure_rate"]),
                    "bootstrap_runtime_seconds": float(bootstrap_diag["runtime_seconds"]),
                    "best_calibration_brier": best_loss,
                    "admissible_count": int(grid_check["grid40_admissible_count"]),
                    "train_engines": int(len(train)),
                    "eval_engines": int(len(eval_frame)),
                }
            )
        for idx, source in enumerate(source_names()):
            point = float(source_means[idx])
            baseline_rows.append(
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "method_id": source,
                    "method_label": SOURCE_LABELS[source],
                    "method_class": "point",
                    "brier_definition": "point_prediction",
                    "lower": point,
                    "upper": point,
                    "point_estimate": point,
                    "width": 0.0,
                    "empirical_reliability": empirical,
                    "miss_distance": abs(point - empirical),
                    "empirical_inside_interval": bool(abs(point - empirical) <= 1e-12),
                    "brier_score": profile_weighted_prediction_brier(
                        eval_predictions[:, idx], eval_labels, eval_conditions, pi
                    ),
                    "eval_engines": int(len(eval_frame)),
                }
            )
        baseline_rows.extend(
            [
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "method_id": "equal_weight_pool",
                    "method_label": "equal-weight pool",
                    "method_class": "point",
                    "brier_definition": "point_prediction",
                    "lower": equal_point,
                    "upper": equal_point,
                    "point_estimate": equal_point,
                    "width": 0.0,
                    "empirical_reliability": empirical,
                    "miss_distance": abs(equal_point - empirical),
                    "empirical_inside_interval": bool(abs(equal_point - empirical) <= 1e-12),
                    "brier_score": equal_brier,
                    "eval_engines": int(len(eval_frame)),
                },
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "method_id": "brier_optimal_point_pool",
                    "method_label": r"Brier-optimal point pool",
                    "method_class": "point",
                    "brier_definition": "point_prediction",
                    "lower": learned_point,
                    "upper": learned_point,
                    "point_estimate": learned_point,
                    "width": 0.0,
                    "empirical_reliability": empirical,
                    "miss_distance": abs(learned_point - empirical),
                    "empirical_inside_interval": bool(abs(learned_point - empirical) <= 1e-12),
                    "brier_score": learned_brier,
                    "eval_engines": int(len(eval_frame)),
                },
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "method_id": "fixed_minmax_envelope",
                    "method_label": "fixed min-max envelope",
                    "method_class": "interval",
                    "brier_definition": "interval_midpoint_surrogate",
                    "lower": fixed_lower,
                    "upper": fixed_upper,
                    "point_estimate": 0.5 * (fixed_lower + fixed_upper),
                    "width": fixed_upper - fixed_lower,
                    "empirical_reliability": empirical,
                    "miss_distance": max(fixed_lower - empirical, empirical - fixed_upper, 0.0),
                    "empirical_inside_interval": bool(fixed_lower <= empirical <= fixed_upper),
                    "brier_score": fixed_midpoint_brier,
                    "eval_engines": int(len(eval_frame)),
                },
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "fold": fold,
                    "method_id": "omega_epsilon_envelope",
                    "method_label": r"$\Omega_\epsilon$ envelope",
                    "method_class": "interval",
                    "brier_definition": "omega_hat_companion_point",
                    "lower": learned_lower,
                    "upper": learned_upper,
                    "point_estimate": learned_point,
                    "width": learned_upper - learned_lower,
                    "empirical_reliability": empirical,
                    "miss_distance": max(learned_lower - empirical, empirical - learned_upper, 0.0),
                    "empirical_inside_interval": bool(learned_lower <= empirical <= learned_upper),
                    "brier_score": learned_brier,
                    "eval_engines": int(len(eval_frame)),
                },
            ]
        )
    return fold_rows, omega_rows, baseline_rows


def aggregate_oof_rows(fold_rows: pd.DataFrame, prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (subset, profile_name), group in fold_rows.groupby(["subset", "profile_name"], sort=True):
        frame = prediction_frames[subset]
        profiles = dict(profile_catalog(frame))
        pi = profiles[profile_name]
        total_eval = float(group["eval_engines"].sum())
        weights = group["eval_engines"].to_numpy(dtype=float) / total_eval
        fixed_lower = float(np.dot(weights, group["fixed_lower"].to_numpy(dtype=float)))
        fixed_upper = float(np.dot(weights, group["fixed_upper"].to_numpy(dtype=float)))
        learned_lower = float(np.dot(weights, group["learned_lower"].to_numpy(dtype=float)))
        learned_upper = float(np.dot(weights, group["learned_upper"].to_numpy(dtype=float)))
        empirical, wilson_lower, wilson_upper = profile_weighted_empirical(frame, pi)
        learned_brier = float(np.dot(weights, group["learned_brier"].to_numpy(dtype=float)))
        equal_brier = float(np.dot(weights, group["equal_weight_brier"].to_numpy(dtype=float)))
        fixed_width = fixed_upper - fixed_lower
        learned_width = learned_upper - learned_lower
        width_reduction = fixed_width - learned_width
        rows.append(
            {
                "subset": subset,
                "profile_name": profile_name,
                "mission_cycles": CREDIBILITY_MISSION,
                "fixed_lower": fixed_lower,
                "fixed_upper": fixed_upper,
                "fixed_width": fixed_width,
                "learned_lower": learned_lower,
                "learned_upper": learned_upper,
                "learned_width": learned_width,
                "width_reduction": width_reduction,
                "width_reduction_fraction": width_reduction / fixed_width if fixed_width > 1e-12 else 0.0,
                "empirical_reliability": empirical,
                "wilson90_lower": wilson_lower,
                "wilson90_upper": wilson_upper,
                "learned_brier": learned_brier,
                "equal_weight_brier": equal_brier,
                "brier_improvement": equal_brier - learned_brier,
                "miss_distance": max(learned_lower - empirical, empirical - learned_upper, 0.0),
                "empirical_inside_learned": bool(learned_lower <= empirical <= learned_upper),
                "decision_category": decision_category(learned_lower, learned_upper),
                "folds": int(group["fold"].nunique()),
                "eval_engines": int(group["eval_engines"].sum()),
            }
        )
    return pd.DataFrame(rows)


def full_data_refit_envelopes(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(CROSS_FIT_SEED + 700)
    for subset, frame in prediction_frames.items():
        predictions = prediction_matrix(frame)
        labels = frame["y_survive_100"].to_numpy(dtype=float)
        conditions = frame["condition"].to_numpy()
        for profile_name, pi in profile_catalog(frame):
            epsilon_diag = bootstrap_epsilon_diagnostics(predictions, labels, conditions, pi, rng)
            epsilon = float(epsilon_diag["epsilon"])
            omega_hat, best_loss, optimizer_success = fit_brier_weight(predictions, labels, conditions, pi)
            source_means = profile_weighted_source_means(frame, pi)
            fixed_lower = float(source_means.min())
            fixed_upper = float(source_means.max())
            continuous = continuous_envelope(
                predictions,
                labels,
                conditions,
                pi,
                epsilon,
                source_means,
                omega_hat=omega_hat,
                best_loss=best_loss,
            )
            grid_check = grid_envelope_check(predictions, labels, conditions, pi, epsilon, source_means, omega_hat, best_loss)
            empirical, wilson_lower, wilson_upper = profile_weighted_empirical(frame, pi)
            learned_lower = float(continuous["lower"])
            learned_upper = float(continuous["upper"])
            fixed_width = fixed_upper - fixed_lower
            learned_width = learned_upper - learned_lower
            rows.append(
                {
                    "subset": subset,
                    "profile_name": profile_name,
                    "mission_cycles": CREDIBILITY_MISSION,
                    "construction": "full_data_refitted_operational_envelope",
                    "calibration_engines": int(len(frame)),
                    "evaluation_engines": int(len(frame)),
                    "epsilon": epsilon,
                    "epsilon_q80": float(epsilon_diag["percentiles"][80.0]),
                    "epsilon_q90": float(epsilon_diag["percentiles"][90.0]),
                    "epsilon_q95": float(epsilon_diag["percentiles"][95.0]),
                    "bootstrap_reps": int(epsilon_diag["requested_reps"]),
                    "bootstrap_successful_reps": int(epsilon_diag["successful_reps"]),
                    "bootstrap_failure_rate": float(epsilon_diag["bootstrap_failure_rate"]),
                    "bootstrap_optimizer_failure_rate": float(epsilon_diag["bootstrap_optimizer_failure_rate"]),
                    "bootstrap_runtime_seconds": float(epsilon_diag["runtime_seconds"]),
                    "optimizer_success": bool(optimizer_success),
                    "best_calibration_brier": float(best_loss),
                    "fixed_lower": fixed_lower,
                    "fixed_upper": fixed_upper,
                    "fixed_width": fixed_width,
                    "refit_lower": learned_lower,
                    "refit_upper": learned_upper,
                    "refit_width": learned_width,
                    "width_reduction": fixed_width - learned_width,
                    "width_reduction_fraction": (fixed_width - learned_width) / fixed_width
                    if fixed_width > 1e-12
                    else 0.0,
                    "empirical_reliability": empirical,
                    "wilson90_lower": wilson_lower,
                    "wilson90_upper": wilson_upper,
                    "miss_distance": max(learned_lower - empirical, empirical - learned_upper, 0.0),
                    "empirical_inside_refit": bool(learned_lower <= empirical <= learned_upper),
                    "decision_category": decision_category(learned_lower, learned_upper),
                    "continuous_lower_success": bool(continuous["lower_success"]),
                    "continuous_upper_success": bool(continuous["upper_success"]),
                    "continuous_used_fallback": bool(continuous["used_fallback"]),
                    "continuous_lower_method": str(continuous["lower_method"]),
                    "continuous_upper_method": str(continuous["upper_method"]),
                    "continuous_lower_status": int(continuous["lower_status"]),
                    "continuous_upper_status": int(continuous["upper_status"]),
                    "continuous_lower_loss_excess": float(continuous["lower_loss_excess"]),
                    "continuous_upper_loss_excess": float(continuous["upper_loss_excess"]),
                    "continuous_lower_simplex_residual": float(continuous["lower_simplex_residual"]),
                    "continuous_upper_simplex_residual": float(continuous["upper_simplex_residual"]),
                    "continuous_lower_bound_violation": float(continuous["lower_bound_violation"]),
                    "continuous_upper_bound_violation": float(continuous["upper_bound_violation"]),
                    "grid20_lower": float(grid_check["grid20_lower"]),
                    "grid20_upper": float(grid_check["grid20_upper"]),
                    "grid40_lower": float(grid_check["grid40_lower"]),
                    "grid40_upper": float(grid_check["grid40_upper"]),
                    "grid80_lower": float(grid_check["grid80_lower"]),
                    "grid80_upper": float(grid_check["grid80_upper"]),
                    "grid_endpoint_gap_max": max(
                        abs(float(grid_check[f"grid{parts}_lower"]) - learned_lower)
                        for parts in GRID_CHECK_PARTS
                    ),
                    "omega_hat": ";".join(f"{value:.10f}" for value in omega_hat),
                }
            )
    return pd.DataFrame(rows)


def compare_oof_and_refit(oof: pd.DataFrame, refit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in oof.iterrows():
        match = refit[
            refit["subset"].eq(row["subset"]) & refit["profile_name"].eq(row["profile_name"])
        ].iloc[0]
        rows.append(
            {
                "subset": row["subset"],
                "profile_name": row["profile_name"],
                "mission_cycles": CREDIBILITY_MISSION,
                "oof_lower": float(row["learned_lower"]),
                "oof_upper": float(row["learned_upper"]),
                "oof_width": float(row["learned_width"]),
                "oof_width_reduction_fraction": float(row["width_reduction_fraction"]),
                "oof_miss_distance": float(row["miss_distance"]),
                "oof_inside": bool(row["empirical_inside_learned"]),
                "refit_lower": float(match["refit_lower"]),
                "refit_upper": float(match["refit_upper"]),
                "refit_width": float(match["refit_width"]),
                "refit_width_reduction_fraction": float(match["width_reduction_fraction"]),
                "refit_miss_distance": float(match["miss_distance"]),
                "refit_inside": bool(match["empirical_inside_refit"]),
                "lower_delta_refit_minus_oof": float(match["refit_lower"] - row["learned_lower"]),
                "upper_delta_refit_minus_oof": float(match["refit_upper"] - row["learned_upper"]),
                "width_delta_refit_minus_oof": float(match["refit_width"] - row["learned_width"]),
                "epsilon_refit": float(match["epsilon"]),
                "folds_oof": int(row["folds"]),
                "calibration_engines_refit": int(match["calibration_engines"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_baseline_rows(baseline_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["subset", "profile_name", "method_id", "method_label", "method_class", "brier_definition"]
    for keys, group in baseline_rows.groupby(group_cols, sort=True):
        subset, profile_name, method_id, method_label, method_class, brier_definition = keys
        total_eval = float(group["eval_engines"].sum())
        weights = group["eval_engines"].to_numpy(dtype=float) / total_eval
        lower = float(np.dot(weights, group["lower"].to_numpy(dtype=float)))
        upper = float(np.dot(weights, group["upper"].to_numpy(dtype=float)))
        empirical = float(np.dot(weights, group["empirical_reliability"].to_numpy(dtype=float)))
        brier = float(np.dot(weights, group["brier_score"].to_numpy(dtype=float)))
        width = upper - lower
        if method_class == "interval":
            miss = max(lower - empirical, empirical - upper, 0.0)
            inside = bool(lower <= empirical <= upper)
        else:
            point = float(np.dot(weights, group["point_estimate"].to_numpy(dtype=float)))
            miss = abs(point - empirical)
            inside = bool(abs(point - empirical) <= 1e-12)
        rows.append(
            {
                "subset": subset,
                "profile_name": profile_name,
                "method_id": method_id,
                "method_label": method_label,
                "method_class": method_class,
                "brier_definition": brier_definition,
                "mission_cycles": CREDIBILITY_MISSION,
                "lower": lower,
                "upper": upper,
                "width": width,
                "empirical_reliability": empirical,
                "miss_distance": miss,
                "empirical_inside_interval": inside,
                "brier_score": brier,
                "folds": int(group["fold"].nunique()),
                "eval_engines": int(group["eval_engines"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["brier_rank_within_profile"] = (
        out.groupby(["subset", "profile_name"])["brier_score"].rank(method="min", ascending=True).astype(int)
    )
    out["miss_rank_within_profile"] = (
        out.groupby(["subset", "profile_name"])["miss_distance"].rank(method="min", ascending=True).astype(int)
    )
    out["width_rank_within_profile"] = (
        out.groupby(["subset", "profile_name"])["width"].rank(method="min", ascending=True).astype(int)
    )
    return out.sort_values(["subset", "profile_name", "brier_rank_within_profile", "method_id"]).reset_index(drop=True)


def split_point_and_interval_evaluations(
    baseline_comparison: pd.DataFrame, conditioned_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = baseline_comparison[baseline_comparison["method_class"].eq("point")].copy()
    point["point_estimate"] = point["lower"].astype(float)
    point["calibration_error"] = (point["point_estimate"] - point["empirical_reliability"]).abs()
    point = point.rename(columns={"brier_score": "oof_brier_score"})
    point["oof_brier_rank_within_profile"] = (
        point.groupby(["subset", "profile_name"])["oof_brier_score"].rank(method="min", ascending=True).astype(int)
    )
    point["calibration_error_rank_within_profile"] = (
        point.groupby(["subset", "profile_name"])["calibration_error"].rank(method="min", ascending=True).astype(int)
    )
    point = point[
        [
            "subset",
            "profile_name",
            "method_id",
            "method_label",
            "mission_cycles",
            "point_estimate",
            "empirical_reliability",
            "oof_brier_score",
            "calibration_error",
            "oof_brier_rank_within_profile",
            "calibration_error_rank_within_profile",
            "folds",
            "eval_engines",
        ]
    ]

    interval = baseline_comparison[baseline_comparison["method_class"].eq("interval")].copy()
    interval = interval.drop(columns=["brier_score", "brier_rank_within_profile"], errors="ignore")
    interval["wilson90_lower"] = np.nan
    interval["wilson90_upper"] = np.nan
    for idx, row in interval.iterrows():
        match = conditioned_rows.loc[
            conditioned_rows["subset"].eq(row["subset"])
            & conditioned_rows["profile_name"].eq(row["profile_name"])
        ].iloc[0]
        interval.at[idx, "wilson90_lower"] = float(match["wilson90_lower"])
        interval.at[idx, "wilson90_upper"] = float(match["wilson90_upper"])
    interval["wilson_overlap"] = (
        interval["upper"].astype(float) >= interval["wilson90_lower"].astype(float)
    ) & (interval["lower"].astype(float) <= interval["wilson90_upper"].astype(float))
    interval["inside_indicator"] = interval["empirical_inside_interval"].astype(bool)
    interval["width_rank_within_profile"] = (
        interval.groupby(["subset", "profile_name"])["width"].rank(method="min", ascending=True).astype(int)
    )
    interval["miss_rank_within_profile"] = (
        interval.groupby(["subset", "profile_name"])["miss_distance"].rank(method="min", ascending=True).astype(int)
    )
    interval = interval[
        [
            "subset",
            "profile_name",
            "method_id",
            "method_label",
            "mission_cycles",
            "lower",
            "upper",
            "width",
            "empirical_reliability",
            "miss_distance",
            "inside_indicator",
            "wilson90_lower",
            "wilson90_upper",
            "wilson_overlap",
            "width_rank_within_profile",
            "miss_rank_within_profile",
            "folds",
            "eval_engines",
        ]
    ]
    point = point.sort_values(["subset", "profile_name", "oof_brier_rank_within_profile", "method_id"]).reset_index(
        drop=True
    )
    interval = interval.sort_values(["subset", "profile_name", "method_id"]).reset_index(drop=True)
    return point, interval


def crossfit_audit(fold_rows: pd.DataFrame, prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for subset, frame in prediction_frames.items():
        observed = fold_rows[(fold_rows["subset"] == subset) & (fold_rows["profile_name"] == "observed")]
        fold_counts = frame.groupby("engine_id")["fold"].nunique()
        rows.append(
            {
                "subset": subset,
                "n_engines": int(frame["engine_id"].nunique()),
                "folds": int(frame["fold"].nunique()),
                "min_eval_engines": int(observed["eval_engines"].min()),
                "max_eval_engines": int(observed["eval_engines"].max()),
                "max_train_eval_intersection": int(observed["train_eval_intersection"].max()),
                "each_engine_one_fold": bool((fold_counts == 1).all()),
                "reproducibility_seed": CROSS_FIT_SEED,
            }
        )
    return pd.DataFrame(rows)


def epsilon_audit(fold_rows: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "subset",
        "profile_name",
        "fold",
        "epsilon_q80",
        "epsilon_q90",
        "epsilon_q95",
        "epsilon",
        "bootstrap_reps",
        "bootstrap_successful_reps",
        "bootstrap_empty_replicates",
        "bootstrap_optimizer_failures",
        "bootstrap_failure_rate",
        "bootstrap_optimizer_failure_rate",
        "excess_loss_mean",
        "excess_loss_max",
        "bootstrap_runtime_seconds",
        "best_calibration_brier",
        "admissible_count",
        "train_engines",
        "eval_engines",
    ]
    available = [col for col in cols if col in fold_rows.columns]
    return fold_rows[available].sort_values(["subset", "profile_name", "fold"]).reset_index(drop=True)


def epsilon_monotonicity(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(CROSS_FIT_SEED + 900)
    for subset, frame in prediction_frames.items():
        pi = dict(profile_catalog(frame))["observed"]
        predictions = prediction_matrix(frame)
        labels = frame["y_survive_100"].to_numpy(dtype=float)
        conditions = frame["condition"].to_numpy()
        epsilon_diag = bootstrap_epsilon_diagnostics(predictions, labels, conditions, pi, rng)
        epsilon0 = float(epsilon_diag["epsilon"])
        source_means = profile_weighted_source_means(frame, pi)
        omega_hat, best_loss_all, _ = fit_brier_weight(predictions, labels, conditions, pi)
        full_simplex_epsilon = full_simplex_loss_epsilon(predictions, labels, conditions, pi, best_loss_all)
        for label, epsilon in [
            ("0", 0.0),
            ("0.5epsilon0", 0.5 * epsilon0),
            ("epsilon0", epsilon0),
            ("2epsilon0", 2.0 * epsilon0),
            ("full_simplex", full_simplex_epsilon),
        ]:
            continuous = continuous_envelope(
                predictions,
                labels,
                conditions,
                pi,
                epsilon,
                source_means,
                omega_hat=omega_hat,
                best_loss=best_loss_all,
            )
            grid_check = grid_envelope_check(predictions, labels, conditions, pi, epsilon, source_means, omega_hat, best_loss_all)
            rows.append(
                {
                    "subset": subset,
                    "profile_name": "observed",
                    "epsilon_label": label,
                    "epsilon": epsilon,
                    "epsilon_is_infinite": bool(label == "full_simplex"),
                    "epsilon0": epsilon0,
                    "bootstrap_reps": int(epsilon_diag["requested_reps"]),
                    "bootstrap_successful_reps": int(epsilon_diag["successful_reps"]),
                    "admissible_count": int(grid_check["grid40_admissible_count"]),
                    "best_calibration_brier": best_loss_all,
                    "lower": float(continuous["lower"]),
                    "upper": float(continuous["upper"]),
                    "width": float(continuous["width"]),
                    "fixed_lower_full": float(source_means.min()),
                    "fixed_upper_full": float(source_means.max()),
                    "grid20_lower": float(grid_check["grid20_lower"]),
                    "grid20_upper": float(grid_check["grid20_upper"]),
                    "grid20_width": float(grid_check["grid20_width"]),
                    "grid40_lower": float(grid_check["grid40_lower"]),
                    "grid40_upper": float(grid_check["grid40_upper"]),
                    "grid40_width": float(grid_check["grid40_width"]),
                    "grid80_lower": float(grid_check["grid80_lower"]),
                    "grid80_upper": float(grid_check["grid80_upper"]),
                    "grid80_width": float(grid_check["grid80_width"]),
                    "continuous_lower_success": bool(continuous["lower_success"]),
                    "continuous_upper_success": bool(continuous["upper_success"]),
                    "continuous_used_fallback": bool(continuous["used_fallback"]),
                    "max_loss_excess": max(
                        float(continuous["lower_loss_excess"]), float(continuous["upper_loss_excess"])
                    ),
                    "max_simplex_residual": max(
                        float(continuous["lower_simplex_residual"]), float(continuous["upper_simplex_residual"])
                    ),
                    "max_bound_violation": max(
                        float(continuous["lower_bound_violation"]), float(continuous["upper_bound_violation"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def continuous_optimizer_audit(fold_rows: pd.DataFrame, monotonicity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fold_rows.iterrows():
        grid_diffs = []
        for parts in GRID_CHECK_PARTS:
            grid_diffs.append(abs(float(row[f"grid{parts}_lower"]) - float(row["learned_lower"])))
            grid_diffs.append(abs(float(row[f"grid{parts}_upper"]) - float(row["learned_upper"])))
        rows.append(
            {
                "check_type": "fold_envelope",
                "subset": row["subset"],
                "profile_name": row["profile_name"],
                "fold": int(row["fold"]),
                "epsilon_label": "epsilon0",
                "continuous_lower_success": bool(row["continuous_lower_success"]),
                "continuous_upper_success": bool(row["continuous_upper_success"]),
                "continuous_used_fallback": bool(row["continuous_used_fallback"]),
                "continuous_lower_method": str(row.get("continuous_lower_method", "")),
                "continuous_upper_method": str(row.get("continuous_upper_method", "")),
                "max_loss_excess": max(
                    float(row["continuous_lower_loss_excess"]), float(row["continuous_upper_loss_excess"])
                ),
                "max_simplex_residual": max(
                    float(row["continuous_lower_simplex_residual"]),
                    float(row["continuous_upper_simplex_residual"]),
                ),
                "max_bound_violation": max(
                    float(row["continuous_lower_bound_violation"]),
                    float(row["continuous_upper_bound_violation"]),
                ),
                "continuous_lower": float(row["learned_lower"]),
                "continuous_upper": float(row["learned_upper"]),
                "continuous_width": float(row["learned_width"]),
                "grid20_lower": float(row["grid20_lower"]),
                "grid20_upper": float(row["grid20_upper"]),
                "grid40_lower": float(row["grid40_lower"]),
                "grid40_upper": float(row["grid40_upper"]),
                "grid80_lower": float(row["grid80_lower"]),
                "grid80_upper": float(row["grid80_upper"]),
                "max_grid_endpoint_gap": float(max(grid_diffs)),
                "full_simplex_limit_pass": "not_applicable",
                "monotonicity_pass": "not_applicable",
            }
        )
    order = ["0", "0.5epsilon0", "epsilon0", "2epsilon0", "full_simplex"]
    for subset, group in monotonicity.groupby("subset", sort=True):
        ordered = group.set_index("epsilon_label").loc[order].reset_index()
        widths = ordered["width"].to_numpy(dtype=float)
        monotonic_pass = bool(np.all(np.diff(widths) >= -1e-7))
        full_row = ordered[ordered["epsilon_label"].eq("full_simplex")].iloc[0]
        fixed_lower = float(full_row["fixed_lower_full"])
        fixed_upper = float(full_row["fixed_upper_full"])
        full_pass = bool(
            abs(float(full_row["lower"]) - fixed_lower) <= 1e-6
            and abs(float(full_row["upper"]) - fixed_upper) <= 1e-6
        )
        rows.append(
            {
                "check_type": "monotonicity_full_simplex",
                "subset": subset,
                "profile_name": "observed",
                "fold": -1,
                "epsilon_label": "all",
                "continuous_lower_success": bool(ordered["continuous_lower_success"].all()),
                "continuous_upper_success": bool(ordered["continuous_upper_success"].all()),
                "continuous_used_fallback": bool(ordered["continuous_used_fallback"].any()),
                "continuous_lower_method": "mixed",
                "continuous_upper_method": "mixed",
                "max_loss_excess": float(ordered["max_loss_excess"].max()),
                "max_simplex_residual": float(ordered["max_simplex_residual"].max()),
                "max_bound_violation": float(ordered["max_bound_violation"].max()),
                "continuous_lower": float(full_row["lower"]),
                "continuous_upper": float(full_row["upper"]),
                "continuous_width": float(full_row["width"]),
                "grid20_lower": float(full_row["grid20_lower"]),
                "grid20_upper": float(full_row["grid20_upper"]),
                "grid40_lower": float(full_row["grid40_lower"]),
                "grid40_upper": float(full_row["grid40_upper"]),
                "grid80_lower": float(full_row["grid80_lower"]),
                "grid80_upper": float(full_row["grid80_upper"]),
                "max_grid_endpoint_gap": max(
                    abs(float(full_row["grid20_lower"]) - float(full_row["lower"])),
                    abs(float(full_row["grid20_upper"]) - float(full_row["upper"])),
                    abs(float(full_row["grid40_lower"]) - float(full_row["lower"])),
                    abs(float(full_row["grid40_upper"]) - float(full_row["upper"])),
                    abs(float(full_row["grid80_lower"]) - float(full_row["lower"])),
                    abs(float(full_row["grid80_upper"]) - float(full_row["upper"])),
                ),
                "full_simplex_limit_pass": bool(full_pass),
                "monotonicity_pass": bool(monotonic_pass),
            }
        )
    return pd.DataFrame(rows)


def build_case_outputs(
    prepared_sets: list[PreparedSet],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    envelope_rows = []
    point_rows = []
    source_rows = []
    profile_rows = []
    engine_prediction_frames = {}
    fold_rows = []
    omega_rows = []
    baseline_rows = []
    rng = np.random.default_rng(CROSS_FIT_SEED + 100)

    for p in prepared_sets:
        unit_features = unit_feature_table(p)
        test_features = test_feature_table(p)
        lives = p.train_life["life"].to_numpy(dtype=float)
        empirical_sigma = float(np.std(np.log(lives)))
        empirical_mu = float(np.mean(np.log(lives)))

        model_predictions: dict[str, dict[float, np.ndarray]] = {}
        for model_name, feature_cols in SOURCE_FEATURE_SETS.items():
            model_predictions[model_name] = {}
            for mission in MISSION_CYCLES:
                total_age = test_features["observed_cycles"].to_numpy(dtype=float) + mission
                if not feature_cols:
                    reliability = survival_from_log_life(
                        np.full(len(test_features), empirical_mu),
                        empirical_sigma,
                        total_age,
                    )
                else:
                    params, resid_sigma = fit_linear_predictor(unit_features, feature_cols)
                    renamed = test_features.rename(columns={v: k for k, v in TEST_FEATURE_MAP.items()})
                    mu = predict_log_life(params, renamed, feature_cols)
                    reliability = survival_from_log_life(mu, resid_sigma + 0.10, total_age)
                model_predictions[model_name][mission] = np.clip(reliability, 0.0, 1.0)

        for mission in MISSION_CYCLES:
            true_survive = (test_features["rul"].to_numpy(dtype=float) > mission).astype(float)
            empirical_rel = true_survive.mean()
            successes = int(true_survive.sum())
            wilson_lower, wilson_upper = wilson_interval(successes, len(true_survive))

            values = np.vstack([model_predictions[name][mission] for name in SOURCE_FEATURE_SETS])
            source_means = values.mean(axis=1)
            avg_lower = float(np.min(source_means))
            avg_upper = float(np.max(source_means))
            empirical_inside = bool(avg_lower <= empirical_rel <= avg_upper)
            envelope_row = {
                "subset": p.fd,
                "mission_cycles": mission,
                "lower": avg_lower,
                "upper": avg_upper,
                "width": avg_upper - avg_lower,
                "empirical_reliability": float(empirical_rel),
                "wilson90_lower": float(wilson_lower),
                "wilson90_upper": float(wilson_upper),
                "empirical_inside_envelope": empirical_inside,
                "decision_category": decision_category(avg_lower, avg_upper),
                "test_engines": int(len(true_survive)),
                "construction": "source_minmax_no_wilson",
            }
            envelope_rows.append(envelope_row)

            for model_name in SOURCE_FEATURE_SETS:
                rel = model_predictions[model_name][mission]
                mae = float(np.mean(np.abs(rel - true_survive)))
                point_rows.append(
                    {
                        "subset": p.fd,
                        "mission_cycles": mission,
                        "model": model_name,
                        "mean_reliability": float(np.mean(rel)),
                        "mae_vs_survival_indicator": mae,
                        "empirical_reliability": float(empirical_rel),
                    }
                )

            # Source-inclusion rows focus on 100-cycle mission.
            if mission == 100.0:
                for model_name in SOURCE_FEATURE_SETS:
                    rel = model_predictions[model_name][mission]
                    source_rows.append(
                        {
                            "subset": p.fd,
                            "source_set": model_name,
                            "mean_reliability_100": float(np.mean(rel)),
                            "mae_vs_survival_indicator_100": float(np.mean(np.abs(rel - true_survive))),
                        }
                    )

        # Profile/condition shift rows for six-condition subsets; single-condition subsets are retained as sea_level.
        for condition, condition_frame in test_features.groupby("condition", sort=True):
            observed = condition_frame["observed_cycles"].to_numpy(dtype=float)
            rul = condition_frame["rul"].to_numpy(dtype=float)
            true_survive = (rul > 100.0).astype(float)
            if len(true_survive) == 0:
                continue
            successes = int(true_survive.sum())
            wilson_lower, wilson_upper = wilson_interval(successes, len(true_survive))
            profile_rows.append(
                {
                    "subset": p.fd,
                    "condition": condition,
                    "n_test_engines": int(len(condition_frame)),
                    "median_observed_cycles": float(np.median(observed)),
                    "empirical_reliability_100": float(np.mean(true_survive)),
                    "wilson90_lower_100": float(wilson_lower),
                    "wilson90_upper_100": float(wilson_upper),
                    "width_100": float(wilson_upper - wilson_lower),
                }
            )

        pred_frame = source_prediction_frame(p, test_features, model_predictions)
        engine_prediction_frames[p.fd] = pred_frame
        for profile_name, pi in profile_catalog(pred_frame):
            fold_part, omega_part, baseline_part = evaluate_crossfit_profile(pred_frame, p.fd, profile_name, pi, rng)
            fold_rows.extend(fold_part)
            omega_rows.extend(omega_part)
            baseline_rows.extend(baseline_part)

    engine_predictions = pd.concat(engine_prediction_frames.values(), ignore_index=True)
    fold_summary = pd.DataFrame(fold_rows)
    omega_projection = pd.DataFrame(omega_rows)
    baseline_comparison = aggregate_baseline_rows(pd.DataFrame(baseline_rows))
    conditioned_rows = aggregate_oof_rows(fold_summary, engine_prediction_frames)
    point_evaluation, interval_evaluation = split_point_and_interval_evaluations(
        baseline_comparison, conditioned_rows
    )
    full_refit = full_data_refit_envelopes(engine_prediction_frames)
    oof_refit_comparison = compare_oof_and_refit(conditioned_rows, full_refit)
    audit = crossfit_audit(fold_summary, engine_prediction_frames)
    epsilon_checks = epsilon_audit(fold_summary)
    monotonicity = epsilon_monotonicity(engine_prediction_frames)
    continuous_checks = continuous_optimizer_audit(fold_summary, monotonicity)

    return (
        pd.DataFrame(envelope_rows),
        pd.DataFrame(point_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(profile_rows),
        engine_predictions,
        omega_projection,
        conditioned_rows,
        fold_summary,
        full_refit,
        oof_refit_comparison,
        baseline_comparison,
        point_evaluation,
        interval_evaluation,
        audit,
        epsilon_checks,
        monotonicity,
        continuous_checks,
    )


def width_decomposition(
    envelope: pd.DataFrame, point_models: pd.DataFrame, profile_shift: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for fd in FD_SETS:
        env = envelope[(envelope["subset"] == fd) & (envelope["mission_cycles"] == 100.0)].iloc[0]
        points = point_models[(point_models["subset"] == fd) & (point_models["mission_cycles"] == 100.0)]
        prof = profile_shift[profile_shift["subset"] == fd]
        source_model_width = float(points["mean_reliability"].max() - points["mean_reliability"].min())
        empirical_width = float(env["wilson90_upper"] - env["wilson90_lower"])
        if prof["condition"].nunique() > 1:
            profile_width = float(prof["empirical_reliability_100"].max() - prof["empirical_reliability_100"].min())
        else:
            profile_width = 0.0
        total_diagnostic = source_model_width + empirical_width + profile_width
        total_diagnostic = total_diagnostic if total_diagnostic > 1e-12 else 1.0
        rows.append(
            {
                "subset": fd,
                "mission_cycles": 100.0,
                "envelope_width": float(env["width"]),
                "source_model_width": source_model_width,
                "empirical_interval_width": empirical_width,
                "profile_condition_width": profile_width,
                "source_model_share": source_model_width / total_diagnostic,
                "empirical_interval_share": empirical_width / total_diagnostic,
                "profile_condition_share": profile_width / total_diagnostic,
                "dominant_width_driver": max(
                    [
                        ("source/model", source_model_width),
                        ("empirical finite-sample", empirical_width),
                        ("profile condition", profile_width),
                    ],
                    key=lambda item: item[1],
                )[0],
            }
        )
    return pd.DataFrame(rows)


def validation_planning(envelope: pd.DataFrame, profile_shift: pd.DataFrame) -> pd.DataFrame:
    fd004 = envelope[(envelope["subset"] == "FD004") & (envelope["mission_cycles"] == 100.0)].iloc[0]
    initial_width = float(fd004["width"])
    profile_width = float(profile_shift[profile_shift["subset"] == "FD004"]["width_100"].mean())
    budgets = np.arange(0, 13)
    rows = []
    for budget in budgets:
        uniform = max(0.0, initial_width * math.exp(-0.045 * budget))
        profile_targeted = max(0.0, initial_width * math.exp(-0.065 * budget))
        risk_targeted = max(0.0, initial_width * math.exp(-0.095 * budget) - 0.15 * profile_width * (budget / 12.0))
        rows.append(
            {
                "budget": int(budget),
                "uniform": float(uniform),
                "profile_targeted": float(profile_targeted),
                "risk_targeted": float(max(0.0, risk_targeted)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    summary = summary.set_index("subset").loc[FD_SETS].reset_index()
    rows = [
        ("Training engines", "train_engines"),
        ("Test engines", "test_engines"),
        ("Operating conditions", "conditions"),
        ("Fault modes", "fault_modes"),
    ]
    values = np.array([[float(row[col]) for _, row in summary.iterrows()] for _, col in rows])
    ax.set_xlim(-0.5, len(FD_SETS) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in range(len(rows)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COLOR["panel"], alpha=0.65, linewidth=0)
    for i, (_, col) in enumerate(rows):
        for j, row in summary.iterrows():
            val = int(row[col])
            ax.text(j, i, f"{val}", ha="center", va="center", fontsize=8.2, color=COLOR["black"])
    ax.set_xticks(np.arange(len(FD_SETS)))
    ax.set_xticklabels(FD_SETS)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    for x in np.arange(-0.5, len(FD_SETS), 1):
        ax.axvline(x, color="white", linewidth=0.9)
    for y in np.arange(-0.5, len(rows), 1):
        ax.axhline(y, color="white", linewidth=0.9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.75)
    save_figure(fig, "fig10_cmapss_dataset_overview")


def plot_envelope(envelope: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.35), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for ax, fd in zip(flat_axes, FD_SETS):
        sub = envelope[envelope["subset"] == fd].sort_values("mission_cycles")
        x = sub["mission_cycles"].to_numpy(dtype=float)
        lower = sub["lower"].to_numpy(dtype=float)
        upper = sub["upper"].to_numpy(dtype=float)
        empirical = sub["empirical_reliability"].to_numpy(dtype=float)
        ax.axhspan(-0.03, R_REQ, color=COLOR["pale_red"], alpha=0.18, linewidth=0)
        ax.fill_between(x, lower, upper, color=COLOR["cyan"], alpha=0.28, linewidth=0, label="finite envelope")
        ax.plot(x, lower, color=COLOR["blue"], linewidth=1.55)
        ax.plot(x, upper, color=COLOR["blue"], linewidth=1.55)
        ax.plot(x, empirical, color=COLOR["orange"], marker="o", markersize=3.6, linewidth=1.35, label="empirical")
        ax.axhline(R_REQ, color=COLOR["red"], linestyle=(0, (4, 2)), linewidth=0.95, label=r"$R_{\mathrm{req}}$")
        row100 = sub[sub["mission_cycles"] == CREDIBILITY_MISSION].iloc[0]
        ax.plot([CREDIBILITY_MISSION, CREDIBILITY_MISSION], [row100["lower"], row100["upper"]], color=COLOR["black"], lw=0.75)
        ax.scatter([CREDIBILITY_MISSION], [row100["empirical_reliability"]], color=COLOR["orange"], s=18, zorder=4, edgecolor="white", linewidth=0.45)
        ax.set_title(fd)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(MISSION_CYCLES)
        polish_axis(ax)
        if ax is flat_axes[-1]:
            ax.text(0.95, R_REQ + 0.015, r"$R_{\mathrm{req}}$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color=COLOR["red"])
    axes[0, 0].set_ylabel("Fleet reliability")
    axes[1, 0].set_ylabel("Fleet reliability")
    axes[1, 0].set_xlabel("Mission cycles")
    axes[1, 1].set_xlabel("Mission cycles")
    handles, labels = flat_axes[-1].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), handlelength=2.0)
    fig.tight_layout(rect=(0, 0.070, 1, 1), pad=0.85)
    save_figure(fig, "fig11_cmapss_reliability_envelopes")


def plot_source_inclusion(source: pd.DataFrame) -> None:
    order = ["life_distribution", "temperature_degradation", "temperature_pressure_degradation", "all_sensor_degradation"]
    labels = ["life", "temp", "temp+press", "all sensors"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 4.75), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for ax, fd in zip(flat_axes, FD_SETS):
        sub = source[source["subset"] == fd].set_index("source_set").loc[order]
        y = np.arange(len(order))
        values = sub["mean_reliability_100"].to_numpy(dtype=float)
        ax.hlines(y, 0.0, values, color=COLOR["lightgrey"], linewidth=1.3)
        ax.scatter(values, y, s=30, color=COLOR["blue"], edgecolor="white", linewidth=0.55, zorder=3)
        ax.set_title(fd)
        ax.set_xlim(0, 0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", labelsize=8.8)
        polish_axis(ax, grid_axis="x")
    flat_axes[0].set_ylim(len(order) - 0.5, -0.5)
    axes[1, 0].set_xlabel("Mean reliability at 100 cycles")
    axes[1, 1].set_xlabel("Mean reliability at 100 cycles")
    fig.tight_layout(pad=0.85)
    save_figure(fig, "fig12_cmapss_source_inclusion")


def plot_profile_shift(profile: pd.DataFrame) -> None:
    multi = profile[profile["subset"].isin(["FD002", "FD004"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.35), sharey=True)
    for ax, fd in zip(axes, ["FD002", "FD004"]):
        sub = (
            multi[multi["subset"] == fd]
            .sort_values(["empirical_reliability_100", "condition"])
            .reset_index(drop=True)
        )
        x = np.arange(len(sub))
        lower = sub["wilson90_lower_100"].to_numpy(dtype=float)
        upper = sub["wilson90_upper_100"].to_numpy(dtype=float)
        empirical = sub["empirical_reliability_100"].to_numpy(dtype=float)
        ax.vlines(x, lower, upper, color=COLOR["cyan"], linewidth=8.0, alpha=0.34)
        ax.scatter(x, empirical, color=COLOR["orange"], edgecolor=COLOR["black"], linewidth=0.45, zorder=3, s=28)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_title(fd)
        ax.set_xlabel("Condition rank by reliability")
        ax.set_ylim(-0.03, 1.03)
        polish_axis(ax, grid_axis="y")
    axes[0].set_ylabel("100-cycle empirical reliability")
    fig.tight_layout(pad=1.2)
    save_figure(fig, "fig13_cmapss_profile_shift")


def plot_validation(validation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    ax.plot(
        validation["budget"],
        validation["uniform"],
        marker="o",
        linestyle="-",
        markersize=4.0,
        linewidth=2.0,
        color=COLOR["grey"],
        label="uniform",
    )
    ax.plot(
        validation["budget"],
        validation["profile_targeted"],
        marker="s",
        linestyle=(0, (4, 2)),
        markersize=4.0,
        linewidth=2.0,
        color=COLOR["teal"],
        label="profile-targeted",
    )
    ax.plot(
        validation["budget"],
        validation["risk_targeted"],
        marker="^",
        linestyle=(0, (5, 2, 1.4, 2)),
        markersize=4.0,
        linewidth=2.0,
        color=COLOR["red"],
        label="risk-targeted",
    )
    for col, label, color, dy in [
        ("uniform", "uniform", COLOR["grey"], 0.004),
        ("profile_targeted", "profile-targeted", COLOR["teal"], 0.000),
        ("risk_targeted", "risk-targeted", COLOR["red"], 0.006),
    ]:
        ax.text(validation["budget"].max() + 0.15, validation[col].iloc[-1] + dy, label, ha="left", va="center", fontsize=7.8, color=color)
    ax.set_xlabel("Additional validation budget units")
    ax.set_ylabel("Expected FD004 100-cycle width")
    ax.set_xlim(validation["budget"].min(), validation["budget"].max() + 1.8)
    ax.set_ylim(0, validation[["uniform", "profile_targeted", "risk_targeted"]].to_numpy().max() * 1.12)
    polish_axis(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, "fig14_cmapss_validation_planning")


def plot_width_decomposition(decomposition: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    x = np.arange(len(decomposition))
    source = decomposition["source_model_share"].to_numpy(dtype=float)
    empirical = decomposition["empirical_interval_share"].to_numpy(dtype=float)
    profile = decomposition["profile_condition_share"].to_numpy(dtype=float)
    width = 0.62
    ax.bar(
        x,
        source,
        width=width,
        color=COLOR["blue"],
        edgecolor="white",
        linewidth=0.8,
        hatch="///",
        label="source/model",
    )
    ax.bar(
        x,
        empirical,
        bottom=source,
        width=width,
        color=COLOR["teal"],
        edgecolor="white",
        linewidth=0.8,
        hatch="\\\\\\",
        label="empirical interval",
    )
    ax.bar(
        x,
        profile,
        bottom=source + empirical,
        width=width,
        color=COLOR["orange"],
        edgecolor="white",
        linewidth=0.8,
        hatch="..",
        label="profile condition",
    )
    for xi, s, e, p in zip(x, source, empirical, profile):
        vals = [s, e, p]
        mids = [s / 2, s + e / 2, s + e + p / 2]
        for value, mid in zip(vals, mids):
            if value >= 0.17:
                ax.text(xi, mid, f"{value:.0%}", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(decomposition["subset"])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Share of diagnostic width")
    polish_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), handlelength=1.7)
    fig.tight_layout(rect=(0, 0.12, 1, 1), pad=0.85)
    save_figure(fig, "fig15_cmapss_width_decomposition")


def plot_profile_conditioned_envelope(conditioned: pd.DataFrame, full_refit: pd.DataFrame) -> None:
    sub = conditioned[conditioned["profile_name"] == "observed"].sort_values("subset").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 3.45))
    x = np.arange(len(sub))
    offset = 0.18
    for idx, row in sub.iterrows():
        refit = full_refit[
            full_refit["subset"].eq(row["subset"]) & full_refit["profile_name"].eq(row["profile_name"])
        ].iloc[0]
        ax.hlines(idx - offset, row["fixed_lower"], row["fixed_upper"], color=COLOR["lightgrey"], linewidth=4.8, alpha=0.86)
        ax.hlines(idx - offset, row["fixed_lower"], row["fixed_upper"], color=COLOR["grey"], linewidth=1.0, alpha=0.88)
        learned_color = COLOR["red"] if row["miss_distance"] > 0 else COLOR["blue"]
        ax.hlines(idx, row["learned_lower"], row["learned_upper"], color=learned_color, linewidth=3.2, alpha=0.94)
        refit_color = COLOR["magenta"] if refit["miss_distance"] > 0 else COLOR["green"]
        ax.hlines(idx + offset, refit["refit_lower"], refit["refit_upper"], color=refit_color, linewidth=2.6, alpha=0.94)
        ax.vlines(
            row["empirical_reliability"],
            idx - 0.32,
            idx + 0.32,
            color=COLOR["orange"],
            linewidth=1.4,
        )
        ax.hlines(
            idx,
            row["wilson90_lower"],
            row["wilson90_upper"],
            color=COLOR["orange"],
            linewidth=0.95,
            linestyle="-",
        )
        if row["miss_distance"] > 0:
            ax.scatter(row["empirical_reliability"], idx, marker="x", s=30, color=COLOR["red"], linewidth=1.05, zorder=5)
    ax.set_yticks(x)
    ax.set_yticklabels(sub["subset"])
    ax.set_xlabel("100-cycle reliability", labelpad=4)
    ax.set_xlim(0, 0.62)
    polish_axis(ax, grid_axis="x")
    legend_handles = [
        Line2D([0], [0], color=COLOR["grey"], lw=2.4, label="fixed"),
        Line2D([0], [0], color=COLOR["blue"], lw=2.4, label="OOF"),
        Line2D([0], [0], color=COLOR["green"], lw=2.4, label="refit"),
        Line2D([0], [0], color=COLOR["orange"], lw=1.1, label="Wilson"),
        Line2D([0], [0], marker="x", color=COLOR["red"], lw=0, markersize=5.0, label="miss"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        handlelength=1.25,
        columnspacing=0.85,
        labelspacing=0.30,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), pad=0.85)
    save_figure(fig, "fig16_cmapss_profile_conditioned_envelope")


def plot_omega_projection(omega_projection: pd.DataFrame) -> None:
    sub = omega_projection[omega_projection["profile_name"] == "observed"].copy()
    grouped = (
        sub.groupby(["subset", "source_model"], sort=True)
        .agg(omega_hat=("omega_hat", "mean"), omega_lower=("omega_lower", "min"), omega_upper=("omega_upper", "max"))
        .reset_index()
    )
    sources = source_names()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.15), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    y = np.arange(len(sources))
    for ax, fd in zip(axes_flat, FD_SETS):
        fd_rows = grouped[grouped["subset"] == fd].set_index("source_model").loc[sources]
        lower = fd_rows["omega_lower"].to_numpy(dtype=float)
        upper = fd_rows["omega_upper"].to_numpy(dtype=float)
        point = fd_rows["omega_hat"].to_numpy(dtype=float)
        for yi, lo, hi, p in zip(y, lower, upper, point):
            ax.hlines(yi, lo, hi, color=COLOR["lightgrey"], linewidth=3.0, zorder=1)
            ax.hlines(yi, lo, hi, color=COLOR["blue"], linewidth=1.2, zorder=2)
            ax.scatter(p, yi, s=20, facecolor="white", edgecolor=COLOR["blue"], linewidth=0.9, zorder=3)
        ax.set_title(fd)
        ax.set_xlim(0, 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([SOURCE_LABELS[source] for source in sources])
        polish_axis(ax, grid_axis="x")
    for ax in axes[:, 0]:
        ax.set_ylabel("Source/model")
    axes_flat[0].set_ylim(len(sources) - 0.5, -0.5)
    for ax in axes_flat:
        ax.set_xlabel(r"$\omega_s$")
    legend_handles = [
        Line2D([0], [0], color=COLOR["blue"], lw=1.2, label="admissible interval"),
        Line2D([0], [0], marker="o", color=COLOR["blue"], markerfacecolor="white", lw=0, markersize=4.2, label=r"$\hat{\omega}$"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.75, h_pad=1.0, w_pad=0.9)
    save_figure(fig, "fig17_cmapss_omega_projection")


def plot_epsilon_monotonicity(monotonicity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.35))
    order = ["0", "0.5epsilon0", "epsilon0", "2epsilon0", "full_simplex"]
    x = np.arange(len(order))
    colors = {
        "FD001": COLOR["blue"],
        "FD002": COLOR["orange"],
        "FD003": COLOR["green"],
        "FD004": COLOR["red"],
    }
    markers = {"FD001": "o", "FD002": "s", "FD003": "^", "FD004": "D"}
    for fd in FD_SETS:
        sub = monotonicity[monotonicity["subset"] == fd].set_index("epsilon_label").loc[order].reset_index()
        ax.plot(x, sub["width"], marker=markers[fd], color=colors[fd], linewidth=2.0, markersize=4.2, label=fd)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$0$", r"$0.5\epsilon_0$", r"$\epsilon_0$", r"$2\epsilon_0$", r"$\infty$"])
    ax.set_ylabel("Envelope width")
    for fd in FD_SETS:
        sub = monotonicity[monotonicity["subset"] == fd].set_index("epsilon_label").loc[order].reset_index()
        ax.text(x[-1] + 0.10, sub["width"].iloc[-1], fd, ha="left", va="center", fontsize=7.8, color=colors[fd])
    ax.set_ylim(0, monotonicity["width"].max() * 1.14)
    ax.set_xlim(x[0], x[-1] + 0.65)
    polish_axis(ax)
    fig.tight_layout(pad=0.95)
    save_figure(fig, "fig18_cmapss_epsilon_monotonicity")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    prepared = [add_condition_labels(load_fd(fd)) for fd in FD_SETS]
    summary = fd_summary(prepared)
    (
        envelope,
        point_models,
        source,
        profile,
        engine_predictions,
        omega_projection,
        conditioned,
        fold_summary,
        full_refit,
        oof_refit_comparison,
        baseline_comparison,
        point_evaluation,
        interval_evaluation,
        crossfit_checks,
        epsilon_checks,
        monotonicity,
        continuous_checks,
    ) = build_case_outputs(prepared)
    decomposition = width_decomposition(envelope, point_models, profile)
    validation = validation_planning(envelope, profile)

    summary.to_csv(RESULTS_DIR / "cmapss_dataset_summary.csv", index=False)
    envelope.to_csv(RESULTS_DIR / "cmapss_envelope_summary.csv", index=False)
    point_models.to_csv(RESULTS_DIR / "cmapss_point_model_summary.csv", index=False)
    source.to_csv(RESULTS_DIR / "cmapss_source_inclusion_summary.csv", index=False)
    profile.to_csv(RESULTS_DIR / "cmapss_profile_shift_summary.csv", index=False)
    engine_predictions.to_csv(RESULTS_DIR / "cmapss_engine_level_predictions.csv", index=False)
    omega_projection.to_csv(RESULTS_DIR / "cmapss_source_credibility_intervals.csv", index=False)
    conditioned.to_csv(RESULTS_DIR / "cmapss_profile_conditioned_envelope_summary.csv", index=False)
    fold_summary.to_csv(RESULTS_DIR / "cmapss_profile_conditioned_fold_summary.csv", index=False)
    full_refit.to_csv(RESULTS_DIR / "cmapss_full_refit_operational_envelope_summary.csv", index=False)
    oof_refit_comparison.to_csv(RESULTS_DIR / "cmapss_oof_vs_full_refit_comparison.csv", index=False)
    baseline_comparison.to_csv(RESULTS_DIR / "cmapss_oof_baseline_comparison.csv", index=False)
    point_evaluation.to_csv(RESULTS_DIR / "cmapss_oof_point_predictor_evaluation.csv", index=False)
    interval_evaluation.to_csv(RESULTS_DIR / "cmapss_oof_set_valued_evaluation.csv", index=False)
    crossfit_checks.to_csv(RESULTS_DIR / "cmapss_crossfit_leakage_audit.csv", index=False)
    epsilon_checks.to_csv(RESULTS_DIR / "cmapss_epsilon_bootstrap_audit.csv", index=False)
    monotonicity.to_csv(RESULTS_DIR / "cmapss_epsilon_monotonicity_summary.csv", index=False)
    continuous_checks.to_csv(RESULTS_DIR / "cmapss_continuous_envelope_audit.csv", index=False)
    decomposition.to_csv(RESULTS_DIR / "cmapss_width_decomposition_summary.csv", index=False)
    validation.to_csv(RESULTS_DIR / "cmapss_validation_planning_summary.csv", index=False)

    plot_dataset_summary(summary)
    plot_envelope(envelope)
    plot_source_inclusion(source)
    plot_profile_shift(profile)
    plot_width_decomposition(decomposition)
    plot_profile_conditioned_envelope(conditioned, full_refit)
    plot_omega_projection(omega_projection)
    plot_epsilon_monotonicity(monotonicity)
    plot_validation(validation)

    print("Generated C-MAPSS Case C outputs")
    print(summary[["subset", "train_engines", "test_engines", "conditions", "fault_modes"]].to_string(index=False))
    for fd in FD_SETS:
        row = envelope[(envelope["subset"] == fd) & (envelope["mission_cycles"] == 100.0)].iloc[0]
        print(
            f"{fd} 100-cycle: lower={row['lower']:.3f}, upper={row['upper']:.3f}, "
            f"width={row['width']:.3f}, empirical={row['empirical_reliability']:.3f}, "
            f"decision={row['decision_category']}"
        )
    observed_conditioned = conditioned[conditioned["profile_name"] == "observed"].sort_values("subset")
    for _, row in observed_conditioned.iterrows():
        refit_row = full_refit[
            full_refit["subset"].eq(row["subset"]) & full_refit["profile_name"].eq("observed")
        ].iloc[0]
        print(
            f"{row['subset']} OOF learned 100-cycle: "
            f"lower={row['learned_lower']:.3f}, "
            f"upper={row['learned_upper']:.3f}, "
            f"width={row['learned_width']:.3f}, "
            f"reduction={100.0 * row['width_reduction_fraction']:.1f}%, "
            f"empirical_inside={row['empirical_inside_learned']}, "
            f"miss_distance={row['miss_distance']:.3f}; "
            f"full-refit=[{refit_row['refit_lower']:.3f},{refit_row['refit_upper']:.3f}], "
            f"width={refit_row['refit_width']:.3f}"
        )


if __name__ == "__main__":
    main()
