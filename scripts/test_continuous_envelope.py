"""Invariant checks for continuous C-MAPSS envelope optimization.

This is a lightweight numerical test for the optimizer used by
``scripts/cmapss_case_study.py``. It does not touch C-MAPSS data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cmapss_case_study import (  # noqa: E402
    GRID_CHECK_PARTS,
    OPT_FEASIBILITY_TOL,
    brier_loss,
    continuous_envelope,
    fit_brier_weight,
    full_simplex_loss_epsilon,
    grid_envelope_check,
)


def synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    predictions = np.array(
        [
            [0.88, 0.78, 0.62, 0.71],
            [0.81, 0.72, 0.58, 0.69],
            [0.76, 0.70, 0.54, 0.67],
            [0.25, 0.41, 0.47, 0.32],
            [0.30, 0.44, 0.51, 0.35],
            [0.68, 0.61, 0.55, 0.59],
            [0.37, 0.48, 0.46, 0.42],
            [0.91, 0.80, 0.65, 0.73],
        ],
        dtype=float,
    )
    labels = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=float)
    conditions = np.array(["A", "A", "B", "B", "B", "A", "B", "A"], dtype=object)
    pi = {"A": 0.55, "B": 0.45}
    source_means = np.array([0.64, 0.59, 0.52, 0.57], dtype=float)
    return predictions, labels, conditions, pi, source_means


def assert_feasible(result: dict, predictions: np.ndarray, labels: np.ndarray, conditions: np.ndarray, pi: dict[str, float], loss_limit: float) -> None:
    for key in ["lower_omega", "upper_omega"]:
        omega = np.asarray(result[key], dtype=float)
        assert np.all(omega >= -OPT_FEASIBILITY_TOL), f"{key} has negative coordinate: {omega}"
        assert np.all(omega <= 1.0 + OPT_FEASIBILITY_TOL), f"{key} exceeds one: {omega}"
        assert abs(float(omega.sum()) - 1.0) <= OPT_FEASIBILITY_TOL, f"{key} is off simplex: {omega.sum()}"
        loss = brier_loss(omega, predictions, labels, conditions, pi)
        assert loss <= loss_limit + OPT_FEASIBILITY_TOL, f"{key} violates loss limit: {loss} > {loss_limit}"


def main() -> None:
    predictions, labels, conditions, pi, source_means = synthetic_problem()
    omega_hat, best_loss, fit_success = fit_brier_weight(predictions, labels, conditions, pi)
    assert fit_success, "Brier-weight optimizer failed on the synthetic problem"

    epsilons = [0.0, 0.001, 0.01]
    widths = []
    for epsilon in epsilons:
        result = continuous_envelope(
            predictions,
            labels,
            conditions,
            pi,
            epsilon,
            source_means,
            omega_hat=omega_hat,
            best_loss=best_loss,
        )
        assert result["lower_success"], f"lower optimizer failed for epsilon={epsilon}: {result['lower_message']}"
        assert result["upper_success"], f"upper optimizer failed for epsilon={epsilon}: {result['upper_message']}"
        assert result["lower"] <= result["upper"] + 1e-10, "continuous endpoint order failed"
        assert_feasible(result, predictions, labels, conditions, pi, best_loss + epsilon)

        grid = grid_envelope_check(predictions, labels, conditions, pi, epsilon, source_means, omega_hat, best_loss)
        for parts in GRID_CHECK_PARTS:
            assert result["lower"] <= float(grid[f"grid{parts}_lower"]) + 5e-6, f"continuous lower excludes grid {parts}"
            assert result["upper"] >= float(grid[f"grid{parts}_upper"]) - 5e-6, f"continuous upper excludes grid {parts}"
        widths.append(float(result["width"]))

    assert np.all(np.diff(widths) >= -1e-7), f"widths are not monotone: {widths}"

    full_epsilon = full_simplex_loss_epsilon(predictions, labels, conditions, pi, best_loss)
    full = continuous_envelope(
        predictions,
        labels,
        conditions,
        pi,
        full_epsilon,
        source_means,
        omega_hat=omega_hat,
        best_loss=best_loss,
    )
    assert abs(float(full["lower"]) - float(source_means.min())) <= 1e-6, "full-simplex lower limit failed"
    assert abs(float(full["upper"]) - float(source_means.max())) <= 1e-6, "full-simplex upper limit failed"
    print("continuous envelope invariant checks passed")


if __name__ == "__main__":
    main()
