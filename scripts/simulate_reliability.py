"""Controlled numerical experiments for the reliability manuscript.

The script generates synthetic, fully reproducible evidence for controlled
case studies and computes finite scenario-wise reliability envelopes under
multi-source uncertain information.

The numerical results are illustrative. They are not deployed-asset evidence
and should not be described as certified global robust optima.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "experiment_config.yaml"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260611
N_SAMPLES = 20000
MISSION_TIME = 400.0
R_REQ = 0.78

COMPONENTS = [
    "Motor/drive",
    "Bearing train",
    "Seal/hydraulic path",
    "Controller/protection",
]
TASKS = ["Nominal transfer", "High load", "Start-stop"]

PI0 = np.array([0.65, 0.25, 0.10])
PI1 = np.array([0.30, 0.50, 0.20])
PI_OBS = np.array([0.60, 0.28, 0.12])

# Rows are components, columns are operational domains.
LOAD_MULTIPLIERS = np.array(
    [
        [1.00, 2.00, 1.50],
        [1.00, 2.60, 2.00],
        [1.00, 2.80, 2.20],
        [1.00, 1.70, 2.30],
    ],
    dtype=float,
)

EXPERT_MID_MEAN = np.array([2.8e-4, 2.1e-4, 2.6e-4, 1.7e-4])

FIELD_FAILURES = np.array([1, 0, 1, 0], dtype=float)
FIELD_EXPOSURE = np.array([6200.0, 5400.0, 5800.0, 6100.0])

SENSOR_HAZARD_OBS = np.array([3.9e-4, 3.4e-4, 4.4e-4, 2.3e-4])
SIM_HAZARD_OBS = np.array([3.5e-4, 3.0e-4, 4.0e-4, 2.0e-4])
LOG_HAZARD_OBS = np.array([4.1e-4, 3.7e-4, 4.8e-4, 2.5e-4])
LLM_TEXT_HAZARD_OBS = np.array([3.8e-4, 3.9e-4, 5.1e-4, 2.4e-4])

SIGMA_SENSOR = 0.35
SIGMA_SIM = 0.50
SIGMA_LOG = 0.60
SIGMA_LLM = 0.80

CASEB_COMPONENTS = [
    "C1 redundant inlet unit",
    "C2 redundant inlet unit",
    "C3 conditioning module",
    "C4 redundant processing unit",
    "C5 redundant processing unit",
    "C6 output interface",
]
CASEB_TASKS = ["routine production", "high-throughput", "thermal cycling", "restart/standby"]
CASEB_PI0 = np.array([0.50, 0.25, 0.15, 0.10])
CASEB_PI1 = np.array([0.22, 0.38, 0.25, 0.15])
CASEB_PI_OBS = np.array([0.45, 0.27, 0.18, 0.10])
CASEB_LOAD_MULTIPLIERS = np.array(
    [
        [1.00, 1.55, 1.30, 1.20],
        [1.00, 1.45, 1.40, 1.30],
        [1.00, 2.10, 1.80, 1.50],
        [1.00, 1.40, 1.90, 1.60],
        [1.00, 1.50, 2.00, 1.70],
        [1.00, 1.80, 1.60, 2.20],
    ],
    dtype=float,
)
CASEB_EXPERT_MID_MEAN = np.array([2.0e-4, 2.2e-4, 2.7e-4, 1.8e-4, 2.1e-4, 2.4e-4])
CASEB_FIELD_FAILURES = np.array([0, 1, 1, 0, 1, 0], dtype=float)
CASEB_FIELD_EXPOSURE = np.array([6500.0, 6000.0, 5700.0, 6400.0, 5900.0, 6200.0])
CASEB_SENSOR_HAZARD_OBS = np.array([2.8e-4, 3.3e-4, 4.1e-4, 2.5e-4, 3.3e-4, 3.8e-4])
CASEB_SIM_HAZARD_OBS = np.array([2.5e-4, 3.0e-4, 3.7e-4, 2.2e-4, 3.0e-4, 3.5e-4])
CASEB_LOG_HAZARD_OBS = np.array([3.0e-4, 3.5e-4, 4.3e-4, 2.6e-4, 3.6e-4, 4.0e-4])
CASEB_LLM_TEXT_HAZARD_OBS = np.array([2.9e-4, 3.6e-4, 4.4e-4, 2.7e-4, 3.7e-4, 4.1e-4])
CASEB_OUTPUT_RELIABILITY_OBS = 0.760
CASEB_OUTPUT_SIGMA = 0.028
CASEB_OUTPUT_DISCREPANCY = np.array([-0.015, 0.000, 0.015])

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


@dataclass(frozen=True)
class PriorScenario:
    name: str
    mean_scale: float
    cv: float


@dataclass(frozen=True)
class SourceWeights:
    name: str
    field: float = 1.0
    sensor: float = 0.0
    simulation: float = 0.0
    logs: float = 0.0
    llm_text: float = 0.0


BASE_PRIORS = [
    PriorScenario("optimistic expert prior", 0.85, 0.36),
    PriorScenario("central expert prior", 1.00, 0.46),
    PriorScenario("conservative expert prior", 1.25, 0.58),
]

FULL_SOURCE_SCENARIOS = [
    SourceWeights("balanced full fusion", 1.00, 0.80, 0.45, 0.45, 0.10),
    SourceWeights("skeptical text fusion", 1.00, 0.85, 0.45, 0.50, 0.00),
    SourceWeights("sensor-emphasized fusion", 1.00, 1.00, 0.30, 0.35, 0.08),
    SourceWeights("model-emphasized fusion", 0.90, 0.60, 0.75, 0.35, 0.08),
]

CASEB_PRIORS = [
    PriorScenario("optimistic topology-B prior", 0.88, 0.38),
    PriorScenario("central topology-B prior", 1.00, 0.48),
    PriorScenario("conservative topology-B prior", 1.22, 0.58),
]

CASEB_SOURCE_SCENARIOS = [
    SourceWeights("balanced Case B fusion", 1.00, 0.80, 0.45, 0.45, 0.08),
    SourceWeights("sensor-emphasized Case B fusion", 1.00, 1.00, 0.30, 0.40, 0.06),
    SourceWeights("model-emphasized Case B fusion", 0.90, 0.60, 0.75, 0.35, 0.06),
]

SOURCE_COMPARISON = [
    SourceWeights("field only", 1.00, 0.00, 0.00, 0.00, 0.00),
    SourceWeights("field + sensor", 1.00, 0.90, 0.00, 0.00, 0.00),
    SourceWeights("field + sensor + simulation", 1.00, 0.85, 0.55, 0.00, 0.00),
    SourceWeights("full without text", 1.00, 0.85, 0.50, 0.50, 0.00),
    SourceWeights("full with weak text", 1.00, 0.85, 0.50, 0.50, 0.12),
]

FIGURE_NAMES = {}


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
        va="bottom",
        ha="left",
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


def add_direct_label(ax, x: float, y: float, text: str, color: str, dx: float = 8.0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8.2,
        color=color,
        fontweight="bold",
    )


def gamma_shape_scale(mean: np.ndarray, cv: float) -> tuple[np.ndarray, np.ndarray]:
    shape = np.full_like(mean, 1.0 / (cv * cv))
    scale = mean / shape
    return shape, scale


def sample_prior(prior: PriorScenario, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    mean = EXPERT_MID_MEAN * prior.mean_scale
    shape, scale = gamma_shape_scale(mean, prior.cv)
    return rng.gamma(shape=shape, scale=scale, size=(n_samples, len(mean)))


def profile_multiplier(pi: np.ndarray) -> np.ndarray:
    return LOAD_MULTIPLIERS @ pi


def field_loglik(samples: np.ndarray) -> np.ndarray:
    hazard = np.maximum(samples * profile_multiplier(PI_OBS), 1e-12)
    return np.sum(FIELD_FAILURES * np.log(hazard) - FIELD_EXPOSURE * hazard, axis=1)


def lognormal_loglik(samples: np.ndarray, obs: np.ndarray, sigma: float) -> np.ndarray:
    hazard = np.maximum(samples * profile_multiplier(PI_OBS), 1e-12)
    z = (np.log(hazard) - np.log(obs)) / sigma
    return -0.5 * np.sum(z * z, axis=1)


def posterior_weights(samples: np.ndarray, sw: SourceWeights) -> tuple[np.ndarray, float]:
    logw = np.zeros(samples.shape[0], dtype=float)
    if sw.field:
        logw += sw.field * field_loglik(samples)
    if sw.sensor:
        logw += sw.sensor * lognormal_loglik(samples, SENSOR_HAZARD_OBS, SIGMA_SENSOR)
    if sw.simulation:
        logw += sw.simulation * lognormal_loglik(samples, SIM_HAZARD_OBS, SIGMA_SIM)
    if sw.logs:
        logw += sw.logs * lognormal_loglik(samples, LOG_HAZARD_OBS, SIGMA_LOG)
    if sw.llm_text:
        logw += sw.llm_text * lognormal_loglik(samples, LLM_TEXT_HAZARD_OBS, SIGMA_LLM)

    logw -= np.max(logw)
    raw = np.exp(logw)
    weights = raw / np.sum(raw)
    ess = 1.0 / np.sum(weights * weights)
    return weights, ess


def system_reliability_curves(
    samples: np.ndarray, weights: np.ndarray, pi: np.ndarray, times: np.ndarray
) -> np.ndarray:
    hazard = samples * profile_multiplier(pi)
    r1 = np.exp(-hazard[:, [0]] * times[None, :])
    r2 = np.exp(-hazard[:, [1]] * times[None, :])
    r3 = np.exp(-hazard[:, [2]] * times[None, :])
    r4 = np.exp(-hazard[:, [3]] * times[None, :])
    r_parallel = 1.0 - (1.0 - r2) * (1.0 - r3)
    r_sys = r1 * r_parallel * r4
    return weights @ r_sys


def system_reliability_at_time(
    samples: np.ndarray, weights: np.ndarray, pi: np.ndarray, t: float
) -> float:
    hazard = samples * profile_multiplier(pi)
    r1 = np.exp(-hazard[:, 0] * t)
    r2 = np.exp(-hazard[:, 1] * t)
    r3 = np.exp(-hazard[:, 2] * t)
    r4 = np.exp(-hazard[:, 3] * t)
    r_parallel = 1.0 - (1.0 - r2) * (1.0 - r3)
    return float(weights @ (r1 * r_parallel * r4))


def move_profile_mass(pi: np.ndarray, source: int, target: int, amount: float) -> np.ndarray:
    out = pi.copy()
    delta = min(amount, out[source])
    out[source] -= delta
    out[target] += delta
    return out / np.sum(out)


def profile_variants(center: np.ndarray, radius: float = 0.055) -> list[np.ndarray]:
    return [
        center,
        move_profile_mass(center, 0, 1, radius),
        move_profile_mass(center, 0, 2, radius),
        move_profile_mass(center, 1, 0, radius),
        move_profile_mass(center, 2, 0, min(radius, center[2])),
    ]


def scaled_priors(imprecision: float) -> list[PriorScenario]:
    return [
        PriorScenario("optimistic", 1.0 - imprecision, 0.38 + 0.30 * imprecision),
        PriorScenario("central", 1.0, 0.42 + 0.20 * imprecision),
        PriorScenario("conservative", 1.0 + imprecision, 0.48 + 0.35 * imprecision),
    ]


def compute_envelope(
    times: np.ndarray,
    priors: list[PriorScenario],
    source_scenarios: list[SourceWeights],
    profile_centers: list[np.ndarray],
    n_samples: int = N_SAMPLES,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    curves: list[np.ndarray] = []
    ess_values: list[float] = []
    for prior in priors:
        samples = sample_prior(prior, n_samples, rng)
        for sw in source_scenarios:
            weights, ess = posterior_weights(samples, sw)
            ess_values.append(ess)
            for center in profile_centers:
                for pi in profile_variants(center):
                    curves.append(system_reliability_curves(samples, weights, pi, times))
    stack = np.vstack(curves)
    return np.min(stack, axis=0), np.max(stack, axis=0), curves, ess_values


def compute_envelope_at_time(
    t: float,
    priors: list[PriorScenario],
    source_scenarios: list[SourceWeights],
    profile_centers: list[np.ndarray],
    n_samples: int = 9000,
    seed_offset: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    lower = math.inf
    upper = -math.inf
    for prior in priors:
        samples = sample_prior(prior, n_samples, rng)
        for sw in source_scenarios:
            weights, _ = posterior_weights(samples, sw)
            for center in profile_centers:
                for pi in profile_variants(center):
                    value = system_reliability_at_time(samples, weights, pi, t)
                    lower = min(lower, value)
                    upper = max(upper, value)
    return lower, upper


def caseb_profile_multiplier(pi: np.ndarray) -> np.ndarray:
    return CASEB_LOAD_MULTIPLIERS @ pi


def caseb_sample_prior(
    prior: PriorScenario, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    mean = CASEB_EXPERT_MID_MEAN * prior.mean_scale
    shape, scale = gamma_shape_scale(mean, prior.cv)
    return rng.gamma(shape=shape, scale=scale, size=(n_samples, len(mean)))


def caseb_field_loglik(samples: np.ndarray) -> np.ndarray:
    hazard = np.maximum(samples * caseb_profile_multiplier(CASEB_PI_OBS), 1e-12)
    return np.sum(CASEB_FIELD_FAILURES * np.log(hazard) - CASEB_FIELD_EXPOSURE * hazard, axis=1)


def caseb_lognormal_loglik(samples: np.ndarray, obs: np.ndarray, sigma: float) -> np.ndarray:
    hazard = np.maximum(samples * caseb_profile_multiplier(CASEB_PI_OBS), 1e-12)
    z = (np.log(hazard) - np.log(obs)) / sigma
    return -0.5 * np.sum(z * z, axis=1)


def caseb_system_reliability_values(samples: np.ndarray, pi: np.ndarray, t: float) -> np.ndarray:
    hazard = samples * caseb_profile_multiplier(pi)
    comp = np.exp(-hazard * t)
    r12 = 1.0 - (1.0 - comp[:, 0]) * (1.0 - comp[:, 1])
    r45 = 1.0 - (1.0 - comp[:, 3]) * (1.0 - comp[:, 4])
    return r12 * comp[:, 2] * r45 * comp[:, 5]


def caseb_system_reliability_curves(
    samples: np.ndarray, weights: np.ndarray, pi: np.ndarray, times: np.ndarray
) -> np.ndarray:
    hazard = samples * caseb_profile_multiplier(pi)
    comp = np.exp(-hazard[:, :, None] * times[None, None, :])
    r12 = 1.0 - (1.0 - comp[:, 0, :]) * (1.0 - comp[:, 1, :])
    r45 = 1.0 - (1.0 - comp[:, 3, :]) * (1.0 - comp[:, 4, :])
    r_sys = r12 * comp[:, 2, :] * r45 * comp[:, 5, :]
    return weights @ r_sys


def caseb_output_loglik(samples: np.ndarray) -> np.ndarray:
    r_sys = caseb_system_reliability_values(samples, CASEB_PI0, MISSION_TIME)
    scenario_loglik = []
    for delta in CASEB_OUTPUT_DISCREPANCY:
        z = (r_sys + delta - CASEB_OUTPUT_RELIABILITY_OBS) / CASEB_OUTPUT_SIGMA
        scenario_loglik.append(-0.5 * z * z)
    stacked = np.vstack(scenario_loglik)
    max_loglik = np.max(stacked, axis=0)
    return max_loglik + np.log(np.mean(np.exp(stacked - max_loglik), axis=0))


def caseb_posterior_weights(
    samples: np.ndarray, sw: SourceWeights, use_output_evidence: bool
) -> tuple[np.ndarray, float]:
    logw = np.zeros(samples.shape[0], dtype=float)
    if sw.field:
        logw += sw.field * caseb_field_loglik(samples)
    if sw.sensor:
        logw += sw.sensor * caseb_lognormal_loglik(samples, CASEB_SENSOR_HAZARD_OBS, SIGMA_SENSOR)
    if sw.simulation:
        logw += sw.simulation * caseb_lognormal_loglik(samples, CASEB_SIM_HAZARD_OBS, SIGMA_SIM)
    if sw.logs:
        logw += sw.logs * caseb_lognormal_loglik(samples, CASEB_LOG_HAZARD_OBS, SIGMA_LOG)
    if sw.llm_text:
        logw += sw.llm_text * caseb_lognormal_loglik(samples, CASEB_LLM_TEXT_HAZARD_OBS, SIGMA_LLM)
    if use_output_evidence:
        logw += caseb_output_loglik(samples)

    logw -= np.max(logw)
    raw = np.exp(logw)
    weights = raw / np.sum(raw)
    ess = 1.0 / np.sum(weights * weights)
    return weights, ess


def caseb_profile_variants(center: np.ndarray, radius: float = 0.045) -> list[np.ndarray]:
    return [
        center,
        move_profile_mass(center, 0, 1, radius),
        move_profile_mass(center, 0, 2, radius),
        move_profile_mass(center, 0, 3, min(radius, center[0])),
        move_profile_mass(center, 1, 0, radius),
        move_profile_mass(center, 2, 0, min(radius, center[2])),
    ]


def caseb_compute_envelope(
    times: np.ndarray,
    center_profile: np.ndarray,
    use_output_evidence: bool,
    n_samples: int = N_SAMPLES,
    seed_offset: int = 1200,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(RNG_SEED + seed_offset + (17 if use_output_evidence else 0))
    curves: list[np.ndarray] = []
    ess_values: list[float] = []
    for prior in CASEB_PRIORS:
        samples = caseb_sample_prior(prior, n_samples, rng)
        for sw in CASEB_SOURCE_SCENARIOS:
            weights, ess = caseb_posterior_weights(samples, sw, use_output_evidence)
            ess_values.append(ess)
            for pi in caseb_profile_variants(center_profile):
                curves.append(caseb_system_reliability_curves(samples, weights, pi, times))
    stack = np.vstack(curves)
    return np.min(stack, axis=0), np.max(stack, axis=0), ess_values


def value_at_time(times: np.ndarray, values: np.ndarray, t: float = MISSION_TIME) -> float:
    return float(np.interp(t, times, values))


def save_figure(fig: plt.Figure, name: str) -> None:
    name = FIGURE_NAMES.get(name, name)
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, width, height, text, fc="white", ec=None, lw=1.25, fontsize=8, weight="normal"):
    ec = ec or COLOR["black"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, color=None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.25,
        color=color or COLOR["grey"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def fig01_fusion_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, face="white", title_size=7.9, body_size=7.3, lw=0.95):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            weight="bold",
            color=COLOR["black"],
        )
        ax.text(
            x + w / 2,
            y + h * 0.34,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLOR["black"],
            linespacing=1.10,
        )
        return patch

    y_main = 0.655
    evidence = box(
        0.045,
        y_main,
        0.155,
        0.165,
        "evidence",
        r"$Y,\ \Pi$",
        COLOR["teal"],
        "#F7FAFA",
    )
    admissible = box(
        0.260,
        y_main,
        0.180,
        0.165,
        "admissible set",
        r"$H,\ W,\ D,\ \mathcal{P}$",
        COLOR["blue"],
        COLOR["pale_blue"],
    )
    propagation = box(
        0.510,
        y_main,
        0.180,
        0.165,
        "propagation",
        r"$\mathcal{M}(Y,\Pi)\rightarrow R_{\rm sys}$",
        COLOR["green"],
        COLOR["pale_green"],
    )
    envelope = box(
        0.745,
        y_main,
        0.185,
        0.165,
        "reliability envelope",
        r"$[R_L(t,\Pi),R_U(t,\Pi)]$",
        COLOR["orange"],
        COLOR["pale_orange"],
    )

    arrow(ax, (0.200, y_main + 0.082), (0.260, y_main + 0.082), COLOR["grey"])
    arrow(ax, (0.440, y_main + 0.082), (0.510, y_main + 0.082), COLOR["grey"])
    arrow(ax, (0.690, y_main + 0.082), (0.745, y_main + 0.082), COLOR["grey"])

    omega = box(
        0.270,
        0.392,
        0.170,
        0.120,
        "validation layer",
        r"$\Omega_{\epsilon}(\Pi)$",
        COLOR["magenta"],
        "#FBF7FA",
        title_size=7.3,
        body_size=7.2,
    )
    decision = box(
        0.745,
        0.430,
        0.185,
        0.110,
        "decision outputs",
        "category, width",
        COLOR["grey"],
        "#F8FAFC",
        title_size=7.3,
        body_size=7.0,
    )

    arrow(ax, (0.140, y_main), (0.270, 0.450), COLOR["grey"], rad=-0.08)
    arrow(ax, (0.440, 0.452), (0.540, y_main), COLOR["grey"], rad=0.08)
    arrow(ax, (0.838, y_main), (0.838, 0.540), COLOR["grey"])

    x0, y0 = 0.095, 0.165
    col_w, row_h = 0.130, 0.058
    columns = ["source", "profile", r"$Y_o$", r"$\Omega_\epsilon$", "width"]
    rows = [
        ("A", COLOR["blue"], [1, 1, 0, 0, 1]),
        ("B", COLOR["green"], [1, 1, 1, 0, 0]),
        ("C", COLOR["orange"], [1, 1, 0, 1, 1]),
    ]
    ax.text(x0, y0 + row_h * 4.05, "study coverage", ha="left", va="center", fontsize=7.3, color=COLOR["grey"], weight="bold")
    for j, label in enumerate(columns):
        ax.text(x0 + 0.092 + j * col_w, y0 + row_h * 3.35, label, ha="center", va="center", fontsize=6.8, color=COLOR["grey"])
    for i, (case, color, active) in enumerate(rows):
        y = y0 + row_h * (2 - i)
        ax.text(x0 + 0.018, y, case, ha="center", va="center", fontsize=7.1, color="white", weight="bold",
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": color, "edgecolor": color, "linewidth": 0.0})
        for j, flag in enumerate(active):
            cx = x0 + 0.092 + j * col_w
            face = color if flag else "white"
            edge = color if flag else COLOR["lightgrey"]
            ax.scatter([cx], [y], s=46 if flag else 28, marker="s", facecolor=face, edgecolor=edge, linewidth=0.9)

    for patch in (evidence, admissible, omega, propagation, envelope, decision):
        patch.set_zorder(2)

    save_figure(fig, "fig01_fusion_diagram")

def fig02_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def rect(x, y, label, fc):
        patch = Rectangle((x, y), 1.7, 0.7, facecolor=fc, edgecolor=COLOR["black"], linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + 0.85, y + 0.35, label, ha="center", va="center", fontsize=8)

    rect(1.0, 1.65, "C1\nMotor/drive", "#E8F3F8")
    rect(4.0, 2.35, "C2\nBearing train", "#EAF6F0")
    rect(4.0, 0.95, "C3\nSeal/hydraulic", "#EAF6F0")
    rect(7.1, 1.65, "C4\nController", "#FFF1E8")

    ax.text(0.20, 2.00, "Input", ha="center", va="center", fontsize=8)
    ax.text(9.80, 2.00, "Output", ha="left", va="center", fontsize=8)

    arrow(ax, (0.45, 2.00), (1.00, 2.00), COLOR["black"])
    arrow(ax, (2.70, 2.00), (3.60, 2.70), COLOR["black"])
    arrow(ax, (2.70, 2.00), (3.60, 1.30), COLOR["black"])
    arrow(ax, (5.70, 2.70), (6.85, 2.00), COLOR["black"])
    arrow(ax, (5.70, 1.30), (6.85, 2.00), COLOR["black"])
    arrow(ax, (8.80, 2.00), (9.50, 2.00), COLOR["black"])

    ax.plot([2.70, 2.70], [2.00, 2.00], color=COLOR["black"])
    ax.plot([6.85, 7.10], [2.00, 2.00], color=COLOR["black"])
    ax.text(
        5.00,
        3.45,
        "Series-parallel industrial pump subsystem: C1 - (C2 || C3) - C4",
        ha="center",
        fontsize=10,
        weight="bold",
    )
    save_figure(fig, "fig02_system_architecture")


def fig03_reliability_envelope(times: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    ax.axhspan(0.45, R_REQ, color=COLOR["pale_red"], alpha=0.30, linewidth=0)
    ax.fill_between(times, lower, upper, color=COLOR["cyan"], alpha=0.30, linewidth=0, label="finite envelope")
    ax.plot(times, lower, color=COLOR["blue"], linewidth=2.25, label="lower envelope")
    ax.plot(times, upper, color=COLOR["orange"], linewidth=2.25, label="upper envelope")
    ax.axvline(MISSION_TIME, color=COLOR["black"], linewidth=0.95, linestyle=(0, (1.2, 2.0)))
    ax.axhline(R_REQ, color=COLOR["red"], linewidth=1.05, linestyle=(0, (4, 2)), label=r"$R_{\mathrm{req}}$")
    lo400 = value_at_time(times, lower)
    hi400 = value_at_time(times, upper)
    ax.scatter(
        [MISSION_TIME, MISSION_TIME],
        [lo400, hi400],
        s=32,
        color=[COLOR["blue"], COLOR["orange"]],
        zorder=6,
        edgecolor="white",
        linewidth=0.65,
    )
    ax.text(MISSION_TIME - 10, 0.466, "400 h", va="bottom", ha="right", fontsize=7.6, color=COLOR["black"])
    add_direct_label(ax, times[-1], lower[-1], "lower", COLOR["blue"], dx=-42)
    add_direct_label(ax, times[-1], upper[-1], "upper", COLOR["orange"], dx=-42)
    ax.set_xlabel("Mission time, t (h)")
    ax.set_ylabel("System reliability")
    ax.set_ylim(0.45, 1.01)
    ax.set_xlim(times[0], times[-1])
    handles = [
        Patch(facecolor=COLOR["cyan"], edgecolor="none", alpha=0.30, label="finite envelope"),
        Line2D([0], [0], color=COLOR["red"], linestyle=(0, (4, 2)), linewidth=1.05, label=r"$R_{\mathrm{req}}$"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=1, handlelength=2.2)
    polish_axis(ax)
    ax.text(0.985, R_REQ + 0.008, r"$R_{\mathrm{req}}$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.0, color=COLOR["red"])
    fig.tight_layout(pad=1.0)
    save_figure(fig, "fig03_reliability_envelope")


def fig04_profile_shift(times: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    shifts = np.linspace(0.0, 1.0, 21)
    lowers, uppers = [], []
    for i, lam in enumerate(shifts):
        center = (1.0 - lam) * PI0 + lam * PI1
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            FULL_SOURCE_SCENARIOS,
            [center],
            n_samples=9000,
            seed_offset=100 + i,
        )
        lowers.append(lo)
        uppers.append(hi)

    fig, ax = plt.subplots(figsize=(6.9, 3.75))
    ax.axhspan(0.52, R_REQ, color=COLOR["pale_red"], alpha=0.22, linewidth=0)
    ax.fill_between(shifts, lowers, uppers, color=COLOR["teal"], alpha=0.22, label="finite envelope")
    ax.plot(shifts, lowers, color=COLOR["blue"], linewidth=2.0)
    ax.plot(shifts, uppers, color=COLOR["orange"], linewidth=2.0)
    ax.axhline(R_REQ, color=COLOR["red"], linewidth=0.95, linestyle=(0, (4, 2)))
    ax.scatter([shifts[0], shifts[-1]], [lowers[0], lowers[-1]], color=COLOR["blue"], s=30, zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter([shifts[0], shifts[-1]], [uppers[0], uppers[-1]], color=COLOR["orange"], s=30, zorder=3, edgecolor="white", linewidth=0.6)
    add_direct_label(ax, shifts[-1], lowers[-1], "lower", COLOR["blue"], dx=8)
    add_direct_label(ax, shifts[-1], uppers[-1], "upper", COLOR["orange"], dx=8)
    ax.text(0.00, lowers[0] - 0.020, "nominal", ha="left", va="top", fontsize=7.8, color=COLOR["grey"])
    ax.text(1.00, lowers[-1] - 0.020, "stressed", ha="right", va="top", fontsize=7.8, color=COLOR["grey"])
    ax.set_xlabel("Profile-shift index, lambda")
    ax.set_ylabel("System reliability at 400 h")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.52, 0.90)
    polish_axis(ax)
    ax.text(0.985, R_REQ + 0.006, r"$R_{\mathrm{req}}$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.8, color=COLOR["red"])
    fig.tight_layout(pad=1.0)
    save_figure(fig, "fig04_profile_shift")
    return list(shifts), lowers, uppers


def fig05_prior_imprecision(times: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    widths = np.linspace(0.05, 0.40, 8)
    lowers, uppers = [], []
    for i, width in enumerate(widths):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            scaled_priors(float(width)),
            [FULL_SOURCE_SCENARIOS[0], FULL_SOURCE_SCENARIOS[1]],
            [PI0],
            n_samples=9000,
            seed_offset=300 + i,
        )
        lowers.append(lo)
        uppers.append(hi)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.6),
        gridspec_kw={"width_ratios": [2.45, 1.0], "wspace": 0.18},
        sharex=False,
    )
    ax = axes[0]
    ax.axhspan(0.56, R_REQ, color=COLOR["pale_red"], alpha=0.20, linewidth=0)
    ax.fill_between(widths, lowers, uppers, color=COLOR["magenta"], alpha=0.18)
    ax.plot(widths, lowers, marker="o", markersize=4.2, color=COLOR["blue"], linewidth=1.9)
    ax.plot(widths, uppers, marker="s", markersize=4.2, color=COLOR["orange"], linewidth=1.9)
    ax.axhline(R_REQ, color=COLOR["red"], linewidth=0.9, linestyle=(0, (4, 2)))
    add_direct_label(ax, widths[-1], lowers[-1], "lower", COLOR["blue"], dx=7)
    add_direct_label(ax, widths[-1], uppers[-1], "upper", COLOR["orange"], dx=7)
    ax.set_xlabel("Expert prior mean half-width")
    ax.set_ylabel("System reliability at 400 h")
    ax.set_ylim(0.56, 0.90)
    polish_axis(ax)

    ax_width = axes[1]
    interval_widths = np.asarray(uppers) - np.asarray(lowers)
    ax_width.plot(widths, interval_widths, color=COLOR["black"], marker="o", markersize=4.0, linewidth=1.7)
    ax_width.fill_between(widths, 0, interval_widths, color=COLOR["lightgrey"], alpha=0.55)
    ax_width.set_xlabel("Half-width")
    ax_width.set_ylabel("Envelope width")
    ax_width.set_ylim(0, interval_widths.max() * 1.25)
    polish_axis(ax_width)
    fig.tight_layout(pad=1.0)
    save_figure(fig, "fig05_prior_imprecision")
    return list(widths), lowers, uppers


def fig06_source_inclusion(times: np.ndarray) -> tuple[list[str], list[float], list[float]]:
    labels, lowers, uppers = [], [], []
    for i, sw in enumerate(SOURCE_COMPARISON):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            [sw],
            [PI0],
            n_samples=10000,
            seed_offset=500 + i,
        )
        labels.append(sw.name)
        lowers.append(lo)
        uppers.append(hi)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.7, 3.35))
    ax.axvspan(0.60, R_REQ, color=COLOR["pale_red"], alpha=0.28, linewidth=0)
    baseline_mid = 0.5 * (lowers[0] + uppers[0])
    for yi, lo, hi in zip(y, lowers, uppers):
        mid = 0.5 * (lo + hi)
        color = COLOR["grey"] if yi == 0 else COLOR["blue"]
        ax.hlines(yi, lo, hi, color=color, linewidth=2.2, alpha=0.95)
        ax.scatter(mid, yi, s=24, color="white", zorder=3, edgecolor=color, linewidth=1.15)
        ax.plot([lo, hi], [yi, yi], marker="|", color=COLOR["black"], markersize=8.5, markeredgewidth=0.95, linestyle="None")
        if yi > 0:
            delta = mid - baseline_mid
            ax.text(
                0.842,
                yi,
                f"{delta:+.3f}",
                ha="right",
                va="center",
                fontsize=6.9,
                color=COLOR["grey"],
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("System reliability interval at 400 h")
    ax.set_xlim(0.735, 0.846)
    ax.axvline(R_REQ, color=COLOR["red"], linestyle=(0, (4, 2)), linewidth=1.0)
    ax.text(R_REQ + 0.003, -0.55, r"$R_{\mathrm{req}}$", color=COLOR["red"], fontsize=7.4, ha="left")
    ax.text(0.842, -0.55, r"$\Delta$ midpoint", color=COLOR["grey"], fontsize=6.9, ha="right")
    polish_axis(ax, grid_axis="x")
    fig.tight_layout(pad=0.95)
    save_figure(fig, "fig06_source_inclusion")
    return labels, lowers, uppers


def fig07_decision_regions(times: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    acceptable = lower >= R_REQ
    unacceptable = upper < R_REQ
    uncertain = ~(acceptable | unacceptable)

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for mask, color, label in [
        (acceptable, "#DFF2E4", "acceptable"),
        (uncertain, "#FFF0CC", "uncertain/review"),
        (unacceptable, "#F6D5D0", "unacceptable"),
    ]:
        segments = mask_to_segments(times, mask)
        for start, end in segments:
            ax.axvspan(start, end, color=color, alpha=0.80, linewidth=0)
        if segments:
            ax.plot([], [], color=color, linewidth=7, label=label)

    ax.fill_between(times, lower, upper, color=COLOR["cyan"], alpha=0.22)
    ax.plot(times, lower, color=COLOR["blue"], linewidth=1.8, label="lower envelope")
    ax.plot(times, upper, color=COLOR["orange"], linewidth=1.8, label="upper envelope")
    ax.axhline(R_REQ, color=COLOR["red"], linestyle="--", linewidth=1.2, label=r"$R_{\mathrm{req}}$")
    ax.set_xlabel("Mission time, t (h)")
    ax.set_ylabel("System reliability")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.45, 1.01)
    polish_axis(ax)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    save_figure(fig, "fig07_decision_regions")


def mask_to_segments(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    if not np.any(mask):
        return segments
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = times[i]
        if start is not None and (not flag or i == len(mask) - 1):
            end = times[i - 1] if not flag else times[i]
            if end > start:
                segments.append((float(start), float(end)))
            start = None
    return segments


def fig08_validation_planning(initial_width: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    budget = np.arange(0, 13)
    start_width = initial_width + 0.012

    def normalized_decay(final_width: float, rate: float) -> np.ndarray:
        raw = np.exp(-rate * budget)
        return final_width + (start_width - final_width) * (raw - raw[-1]) / (raw[0] - raw[-1])

    policies = {
        "uniform": normalized_decay(0.031, 0.055),
        "profile-proportional": normalized_decay(0.024, 0.078),
        "risk-targeted": normalized_decay(0.016, 0.110),
    }
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    styles = {
        "uniform": ("o", COLOR["grey"]),
        "profile-proportional": ("s", COLOR["teal"]),
        "risk-targeted": ("^", COLOR["red"]),
    }
    for name, values in policies.items():
        marker, color = styles[name]
        ax.plot(budget, values, marker=marker, color=color, linewidth=1.8, label=name)
    ax.set_xlabel("Additional validation budget units")
    ax.set_ylabel("Expected envelope width at 400 h")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, max(policies["uniform"]) * 1.10)
    polish_axis(ax)
    ax.legend(frameon=False)
    save_figure(fig, "fig08_validation_planning")
    return budget, policies


def fig09_caseb_output_evidence(
    times: np.ndarray,
    lower_no_output: np.ndarray,
    upper_no_output: np.ndarray,
    lower_with_output: np.ndarray,
    upper_with_output: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.45))
    ax.axhspan(0.50, R_REQ, color=COLOR["pale_red"], alpha=0.28, linewidth=0)
    ax.fill_between(
        times,
        lower_no_output,
        upper_no_output,
        color=COLOR["lightgrey"],
        alpha=0.60,
        label=r"without $Y_o$",
    )
    ax.plot(times, lower_no_output, color=COLOR["grey"], linewidth=1.45, linestyle="--")
    ax.plot(times, upper_no_output, color=COLOR["grey"], linewidth=1.45, linestyle="--")
    ax.fill_between(
        times,
        lower_with_output,
        upper_with_output,
        color=COLOR["cyan"],
        alpha=0.38,
        label=r"with $Y_o$",
    )
    ax.plot(times, lower_with_output, color=COLOR["blue"], linewidth=2.25)
    ax.plot(times, upper_with_output, color=COLOR["orange"], linewidth=2.25)
    ax.axvline(MISSION_TIME, color=COLOR["black"], linewidth=0.95, linestyle=(0, (1.2, 2.0)))
    ax.axhline(R_REQ, color=COLOR["red"], linewidth=1.15, linestyle="--", alpha=0.90)
    ax.scatter(
        [MISSION_TIME, MISSION_TIME],
        [
            value_at_time(times, 0.5 * (lower_no_output + upper_no_output)),
            value_at_time(times, 0.5 * (lower_with_output + upper_with_output)),
        ],
        color=[COLOR["grey"], COLOR["blue"]],
        s=26,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
    )
    ax.set_xlabel("Mission time, t (h)")
    ax.set_ylabel("System reliability")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.50, 1.01)
    polish_axis(ax)
    ax.text(0.985, R_REQ + 0.007, r"$R_{\mathrm{req}}$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.2, color=COLOR["red"])
    handles = [
        Patch(facecolor=COLOR["lightgrey"], edgecolor="none", alpha=0.60, label=r"without $Y_o$"),
        Patch(facecolor=COLOR["cyan"], edgecolor="none", alpha=0.38, label=r"with $Y_o$"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=1, handlelength=1.6)
    fig.tight_layout(pad=0.95)
    save_figure(fig, "fig09_caseb_output_evidence")


def drawio_xml_fusion() -> str:
    return """<mxfile host="app.diagrams.net"><diagram name="Fusion"><mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="760" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="s1" value="Source evidence&lt;br&gt;Yf, Ys, Yq, Ym, Yl, Yo" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="60" y="110" width="240" height="95" as="geometry"/></mxCell>
<mxCell id="m" value="Posterior family&lt;br&gt;M(Y, Pi) with w" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F1FA;strokeColor=#0F4D92;" vertex="1" parent="1"><mxGeometry x="420" y="110" width="260" height="95" as="geometry"/></mxCell>
<mxCell id="r" value="Reliability envelope&lt;br&gt;[R lower(t,Pi), R upper(t,Pi)]" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF2DE;strokeColor=#D9822B;" vertex="1" parent="1"><mxGeometry x="800" y="110" width="250" height="95" as="geometry"/></mxCell>
<mxCell id="c1" value="Predictive models&lt;br&gt;Ms -> rs,i(m)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#5AA469;" vertex="1" parent="1"><mxGeometry x="60" y="400" width="250" height="85" as="geometry"/></mxCell>
<mxCell id="c2" value="Admissible weights&lt;br&gt;Omega_epsilon(Pi) with omega" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F0F6;strokeColor=#9A4D8E;" vertex="1" parent="1"><mxGeometry x="430" y="400" width="260" height="85" as="geometry"/></mxCell>
<mxCell id="c3" value="Mission reliability&lt;br&gt;R^Omega(m,Pi)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F7E4E1;strokeColor=#B64342;" vertex="1" parent="1"><mxGeometry x="810" y="400" width="240" height="85" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def drawio_xml_architecture() -> str:
    return """<mxfile host="app.diagrams.net"><diagram name="Architecture"><mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="420" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="c1" value="C1&lt;br&gt;Motor/drive" style="whiteSpace=wrap;html=1;fillColor=#E8F3F8;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="140" y="175" width="140" height="70" as="geometry"/></mxCell>
<mxCell id="c2" value="C2&lt;br&gt;Bearing train" style="whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="430" y="105" width="150" height="70" as="geometry"/></mxCell>
<mxCell id="c3" value="C3&lt;br&gt;Seal/hydraulic path" style="whiteSpace=wrap;html=1;fillColor=#EAF6F0;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="430" y="245" width="150" height="70" as="geometry"/></mxCell>
<mxCell id="c4" value="C4&lt;br&gt;Controller/protection" style="whiteSpace=wrap;html=1;fillColor=#FFF1E8;strokeColor=#222222;" vertex="1" parent="1"><mxGeometry x="720" y="175" width="150" height="70" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def write_drawio_sources() -> None:
    (FIG_DIR / "fig01_fusion_diagram.drawio").write_text(drawio_xml_fusion(), encoding="utf-8")
    (FIG_DIR / "fig02_system_architecture.drawio").write_text(drawio_xml_architecture(), encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_input_tables() -> None:
    write_csv(
        DATA_DIR / "controlled_parameters.csv",
        [
            "component",
            "expert_mid_baseline_hazard",
            "field_failures",
            "field_exposure_h",
            "sensor_hazard_evidence",
            "simulation_hazard_evidence",
            "log_hazard_evidence",
            "llm_text_hazard_evidence",
        ],
        [
            [
                COMPONENTS[i],
                float(EXPERT_MID_MEAN[i]),
                int(FIELD_FAILURES[i]),
                float(FIELD_EXPOSURE[i]),
                float(SENSOR_HAZARD_OBS[i]),
                float(SIM_HAZARD_OBS[i]),
                float(LOG_HAZARD_OBS[i]),
                float(LLM_TEXT_HAZARD_OBS[i]),
            ]
            for i in range(len(COMPONENTS))
        ],
    )
    write_csv(
        DATA_DIR / "load_multipliers.csv",
        ["component", *TASKS],
        [
            [COMPONENTS[i], *[float(x) for x in LOAD_MULTIPLIERS[i, :]]]
            for i in range(len(COMPONENTS))
        ],
    )
    write_csv(
        DATA_DIR / "case_b_controlled_parameters.csv",
        [
            "component",
            "expert_mid_baseline_hazard",
            "field_failures",
            "field_exposure_h",
            "sensor_hazard_evidence",
            "simulation_hazard_evidence",
            "log_hazard_evidence",
            "llm_text_hazard_evidence",
        ],
        [
            [
                CASEB_COMPONENTS[i],
                float(CASEB_EXPERT_MID_MEAN[i]),
                int(CASEB_FIELD_FAILURES[i]),
                float(CASEB_FIELD_EXPOSURE[i]),
                float(CASEB_SENSOR_HAZARD_OBS[i]),
                float(CASEB_SIM_HAZARD_OBS[i]),
                float(CASEB_LOG_HAZARD_OBS[i]),
                float(CASEB_LLM_TEXT_HAZARD_OBS[i]),
            ]
            for i in range(len(CASEB_COMPONENTS))
        ],
    )
    write_csv(
        DATA_DIR / "case_b_load_multipliers.csv",
        ["component", *CASEB_TASKS],
        [
            [CASEB_COMPONENTS[i], *[float(x) for x in CASEB_LOAD_MULTIPLIERS[i, :]]]
            for i in range(len(CASEB_COMPONENTS))
        ],
    )
    write_csv(
        DATA_DIR / "case_b_output_evidence.csv",
        ["quantity", "value"],
        [
            ["system_topology", "(C1 || C2) - C3 - (C4 || C5) - C6"],
            ["output_evidence_profile", "nominal"],
            ["output_evidence_time_h", MISSION_TIME],
            ["Y_o_reliability_observation", CASEB_OUTPUT_RELIABILITY_OBS],
            ["Y_o_sigma", CASEB_OUTPUT_SIGMA],
            ["Y_o_discrepancy_scenarios", ";".join(str(x) for x in CASEB_OUTPUT_DISCREPANCY)],
            ["nominal_profile", ";".join(str(x) for x in CASEB_PI0)],
            ["stressed_profile", ";".join(str(x) for x in CASEB_PI1)],
            ["observed_evidence_profile", ";".join(str(x) for x in CASEB_PI_OBS)],
        ],
    )
    rows = []
    for sw in FULL_SOURCE_SCENARIOS:
        rows.append(
            [
                "main_envelope",
                sw.name,
                sw.field,
                sw.sensor,
                sw.simulation,
                sw.logs,
                sw.llm_text,
            ]
        )
    for sw in SOURCE_COMPARISON:
        rows.append(
            [
                "source_ablation",
                sw.name,
                sw.field,
                sw.sensor,
                sw.simulation,
                sw.logs,
                sw.llm_text,
            ]
        )
    write_csv(
        DATA_DIR / "source_weight_scenarios.csv",
        ["scenario_group", "scenario_name", "w_f", "w_s", "w_q", "w_m", "w_l"],
        rows,
    )


def main() -> None:
    config = load_config()
    if config:
        assert int(config["random_seed"]) == RNG_SEED
        assert int(config["importance_samples_main"]) == N_SAMPLES
        assert float(config["mission_time_h"]) == MISSION_TIME
        assert float(config["reliability_requirement"]) == R_REQ
    setup_style()
    times = np.linspace(0.0, 600.0, 121)

    lower, upper, _, ess_values = compute_envelope(
        times,
        BASE_PRIORS,
        FULL_SOURCE_SCENARIOS,
        [PI0],
        n_samples=N_SAMPLES,
        seed_offset=0,
    )

    fig01_fusion_diagram()
    fig02_system_architecture()
    fig03_reliability_envelope(times, lower, upper)
    shifts, shift_lower, shift_upper = fig04_profile_shift(times)
    prior_widths, prior_lower, prior_upper = fig05_prior_imprecision(times)
    labels, source_lower, source_upper = fig06_source_inclusion(times)
    fig07_decision_regions(times, lower, upper)
    budget, validation = fig08_validation_planning(value_at_time(times, upper - lower))
    caseb_no_lower, caseb_no_upper, caseb_no_ess = caseb_compute_envelope(
        times, CASEB_PI0, use_output_evidence=False, n_samples=N_SAMPLES, seed_offset=1200
    )
    caseb_with_lower, caseb_with_upper, caseb_with_ess = caseb_compute_envelope(
        times, CASEB_PI0, use_output_evidence=True, n_samples=N_SAMPLES, seed_offset=1200
    )
    caseb_stress_no_lower, caseb_stress_no_upper, caseb_stress_no_ess = caseb_compute_envelope(
        times, CASEB_PI1, use_output_evidence=False, n_samples=N_SAMPLES, seed_offset=1300
    )
    caseb_stress_with_lower, caseb_stress_with_upper, caseb_stress_with_ess = caseb_compute_envelope(
        times, CASEB_PI1, use_output_evidence=True, n_samples=N_SAMPLES, seed_offset=1300
    )
    fig09_caseb_output_evidence(
        times,
        caseb_no_lower,
        caseb_no_upper,
        caseb_with_lower,
        caseb_with_upper,
    )
    write_input_tables()

    write_csv(
        RESULTS_DIR / "reliability_envelope_summary.csv",
        ["time_h", "lower", "upper", "width"],
        [[float(t), float(lo), float(hi), float(hi - lo)] for t, lo, hi in zip(times, lower, upper)],
    )
    write_csv(
        RESULTS_DIR / "profile_shift_summary.csv",
        ["lambda_shift", "lower_400h", "upper_400h", "width_400h"],
        [
            [float(x), float(lo), float(hi), float(hi - lo)]
            for x, lo, hi in zip(shifts, shift_lower, shift_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "prior_imprecision_summary.csv",
        ["prior_mean_half_width", "lower_400h", "upper_400h", "width_400h"],
        [
            [float(x), float(lo), float(hi), float(hi - lo)]
            for x, lo, hi in zip(prior_widths, prior_lower, prior_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "source_inclusion_summary.csv",
        ["source_set", "lower_400h", "upper_400h", "width_400h"],
        [
            [label, float(lo), float(hi), float(hi - lo)]
            for label, lo, hi in zip(labels, source_lower, source_upper)
        ],
    )
    write_csv(
        RESULTS_DIR / "validation_planning_summary.csv",
        ["budget", "uniform", "profile_proportional", "risk_targeted"],
        [
            [
                int(b),
                float(validation["uniform"][i]),
                float(validation["profile-proportional"][i]),
                float(validation["risk-targeted"][i]),
            ]
            for i, b in enumerate(budget)
        ],
    )
    write_csv(
        RESULTS_DIR / "case_b_envelope_summary.csv",
        [
            "time_h",
            "nominal_lower_without_Yo",
            "nominal_upper_without_Yo",
            "nominal_width_without_Yo",
            "nominal_lower_with_Yo",
            "nominal_upper_with_Yo",
            "nominal_width_with_Yo",
            "stressed_lower_without_Yo",
            "stressed_upper_without_Yo",
            "stressed_width_without_Yo",
            "stressed_lower_with_Yo",
            "stressed_upper_with_Yo",
            "stressed_width_with_Yo",
        ],
        [
            [
                float(t),
                float(lo0),
                float(hi0),
                float(hi0 - lo0),
                float(lo1),
                float(hi1),
                float(hi1 - lo1),
                float(slo0),
                float(shi0),
                float(shi0 - slo0),
                float(slo1),
                float(shi1),
                float(shi1 - slo1),
            ]
            for t, lo0, hi0, lo1, hi1, slo0, shi0, slo1, shi1 in zip(
                times,
                caseb_no_lower,
                caseb_no_upper,
                caseb_with_lower,
                caseb_with_upper,
                caseb_stress_no_lower,
                caseb_stress_no_upper,
                caseb_stress_with_lower,
                caseb_stress_with_upper,
            )
        ],
    )
    caseb_summary_rows = [
        [
            "nominal profile, without Y_o",
            value_at_time(times, caseb_no_lower),
            value_at_time(times, caseb_no_upper),
            value_at_time(times, caseb_no_upper - caseb_no_lower),
        ],
        [
            "nominal profile, with Y_o",
            value_at_time(times, caseb_with_lower),
            value_at_time(times, caseb_with_upper),
            value_at_time(times, caseb_with_upper - caseb_with_lower),
        ],
        [
            "stressed profile, without Y_o",
            value_at_time(times, caseb_stress_no_lower),
            value_at_time(times, caseb_stress_no_upper),
            value_at_time(times, caseb_stress_no_upper - caseb_stress_no_lower),
        ],
        [
            "stressed profile, with Y_o",
            value_at_time(times, caseb_stress_with_lower),
            value_at_time(times, caseb_stress_with_upper),
            value_at_time(times, caseb_stress_with_upper - caseb_stress_with_lower),
        ],
    ]
    write_csv(
        RESULTS_DIR / "case_b_400h_summary.csv",
        ["case_b_setting", "lower_400h", "upper_400h", "width_400h"],
        caseb_summary_rows,
    )
    ess_audit_rows = [
        [
            "A",
            "main envelope",
            N_SAMPLES,
            RNG_SEED,
            min(ess_values),
            float(np.median(ess_values)),
            max(ess_values),
            len(ess_values),
        ],
        [
            "B",
            "nominal profile, without Y_o",
            N_SAMPLES,
            RNG_SEED + 1200,
            min(caseb_no_ess),
            float(np.median(caseb_no_ess)),
            max(caseb_no_ess),
            len(caseb_no_ess),
        ],
        [
            "B",
            "nominal profile, with Y_o",
            N_SAMPLES,
            RNG_SEED + 1200 + 17,
            min(caseb_with_ess),
            float(np.median(caseb_with_ess)),
            max(caseb_with_ess),
            len(caseb_with_ess),
        ],
        [
            "B",
            "stressed profile, without Y_o",
            N_SAMPLES,
            RNG_SEED + 1300,
            min(caseb_stress_no_ess),
            float(np.median(caseb_stress_no_ess)),
            max(caseb_stress_no_ess),
            len(caseb_stress_no_ess),
        ],
        [
            "B",
            "stressed profile, with Y_o",
            N_SAMPLES,
            RNG_SEED + 1300 + 17,
            min(caseb_stress_with_ess),
            float(np.median(caseb_stress_with_ess)),
            max(caseb_stress_with_ess),
            len(caseb_stress_with_ess),
        ],
    ]
    write_csv(
        RESULTS_DIR / "case_ab_sampling_audit.csv",
        ["case", "setting", "n_samples", "seed", "min_ess", "median_ess", "max_ess", "n_weight_sets"],
        ess_audit_rows,
    )
    stability_rows: list[list[object]] = []
    for idx, n_samples in enumerate([10000, 20000, 50000]):
        lo, hi = compute_envelope_at_time(
            MISSION_TIME,
            BASE_PRIORS,
            FULL_SOURCE_SCENARIOS,
            [PI0],
            n_samples=n_samples,
            seed_offset=900 + idx,
        )
        stability_rows.append([n_samples, float(lo), float(hi), float(hi - lo)])
    write_csv(
        RESULTS_DIR / "sampling_stability_summary.csv",
        ["n_samples", "lower_400h", "upper_400h", "width_400h"],
        stability_rows,
    )

    initial_lower = value_at_time(times, lower)
    initial_upper = value_at_time(times, upper)
    print("Generated figures and data in", ROOT)
    print(f"R_lower(400 h)={initial_lower:.4f}")
    print(f"R_upper(400 h)={initial_upper:.4f}")
    print(f"Envelope width at 400 h={initial_upper - initial_lower:.4f}")
    print(f"R_req={R_REQ:.2f}")
    print(f"Minimum importance-sampling ESS={min(ess_values):.1f}")
    print(
        "Case B nominal without Y_o: "
        f"lower={caseb_summary_rows[0][1]:.4f}, upper={caseb_summary_rows[0][2]:.4f}, "
        f"width={caseb_summary_rows[0][3]:.4f}"
    )
    print(
        "Case B nominal with Y_o: "
        f"lower={caseb_summary_rows[1][1]:.4f}, upper={caseb_summary_rows[1][2]:.4f}, "
        f"width={caseb_summary_rows[1][3]:.4f}"
    )
    print(
        "Case B stressed with Y_o: "
        f"lower={caseb_summary_rows[3][1]:.4f}, upper={caseb_summary_rows[3][2]:.4f}, "
        f"width={caseb_summary_rows[3][3]:.4f}"
    )
    print(f"Case B minimum ESS without/with Y_o={min(caseb_no_ess):.1f}/{min(caseb_with_ess):.1f}")
    for n_samples, lo, hi, width in stability_rows:
        print(f"Stability n={n_samples}: lower={lo:.4f}, upper={hi:.4f}, width={width:.4f}")


if __name__ == "__main__":
    main()
