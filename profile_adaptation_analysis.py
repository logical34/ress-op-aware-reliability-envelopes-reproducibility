"""Isolate calibration-profile adaptation using the verified cross-fitted predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from complete_revision_analysis import engine_weights, face_extrema, load_module, save_table


INPUT_NAMES = (
    "cmapss_engine_level_predictions.csv",
    "cmapss_profile_conditioned_fold_summary.csv",
    "cmapss_source_credibility_intervals.csv",
)
ARM_LABELS = {
    "transported": "Original-profile set",
    "adapted_common_tolerance": "Target-profile set, common tolerance",
    "adapted_target_tolerance": "Target-profile set, target tolerance",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, inputs, output = args.repo_root.resolve(), args.results_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    c = load_module(repo / "scripts/cmapss_case_study.py", "profile_adaptation_cmapss")
    predictions = pd.read_csv(inputs / INPUT_NAMES[0])
    existing_folds = pd.read_csv(inputs / INPUT_NAMES[1])
    coordinates = pd.read_csv(inputs / INPUT_NAMES[2])
    rows, profile_rows, changes = [], [], []
    for fd in ("FD002", "FD004"):
        frame = predictions[predictions.subset == fd].reset_index(drop=True)
        assert frame.engine_id.is_unique
        np.testing.assert_array_equal(c.assign_condition_stratified_folds(frame), frame.fold)
        profiles = dict(c.profile_catalog(frame))
        original, target = profiles["observed"], profiles["shifted"]
        for condition in sorted(original):
            profile_rows.append({"subset": fd, "condition": condition,
                "original_mass": original[condition], "target_mass": target[condition]})
        for fold in range(c.CROSS_FIT_FOLDS):
            calibration = frame[frame.fold != fold]
            evaluation = frame[frame.fold == fold]
            assert set(calibration.engine_id).isdisjoint(evaluation.engine_id)
            matrix = c.prediction_matrix(calibration)
            labels = calibration.y_survive_100.to_numpy()
            conditions = calibration.condition.to_numpy()
            eval_matrix = c.prediction_matrix(evaluation)
            eval_labels = evaluation.y_survive_100.to_numpy()
            eval_rho = engine_weights(evaluation.condition.to_numpy(), target)
            means = eval_rho @ eval_matrix
            assert all((calibration.condition == d).any() and (evaluation.condition == d).any()
                       for d in target), "Profile control requires represented conditions"
            fitted, archived = {}, {}
            for name, pi in (("observed", original), ("shifted", target)):
                fitted[name] = c.fit_brier_weight(matrix, labels, conditions, pi)
                omega, loss, success = fitted[name]
                assert success
                selected = existing_folds[(existing_folds.subset == fd)
                    & (existing_folds.profile_name == name) & (existing_folds.fold == fold)]
                assert len(selected) == 1
                archived[name] = selected.iloc[0]
                assert abs(loss - archived[name].best_calibration_brier) < 1e-10
                saved = coordinates[(coordinates.subset == fd) & (coordinates.profile_name == name)
                    & (coordinates.fold == fold)].set_index("source_model").reindex(c.source_names())
                np.testing.assert_allclose(omega, saved.omega_hat, atol=2e-7, rtol=0)
            epsilon_original = float(archived["observed"].epsilon)
            epsilon_target = float(archived["shifted"].epsilon)
            for arm, loss_profile, epsilon in (
                ("transported", "observed", epsilon_original),
                ("adapted_common_tolerance", "shifted", epsilon_original),
                ("adapted_target_tolerance", "shifted", epsilon_target),
            ):
                pi = profiles[loss_profile]
                omega, best_loss, _ = fitted[loss_profile]
                envelope = c.continuous_envelope(matrix, labels, conditions, pi, epsilon, means,
                    omega_hat=omega, best_loss=best_loss)
                rho = engine_weights(conditions, pi)
                optimum, lower, upper = face_extrema(matrix, labels, rho, means, epsilon)
                objective_error = abs(best_loss - optimum)
                endpoint_error = max(abs(lower - envelope["lower"]), abs(upper - envelope["upper"]))
                assert objective_error < 1e-9 and endpoint_error < 2e-7
                assert envelope["lower_success"] and envelope["upper_success"]
                assert not envelope["used_fallback"]
                assert -1e-9 <= lower <= upper <= 1 + 1e-9
                archived_error = 0.0
                if arm == "adapted_target_tolerance":
                    saved = archived["shifted"]
                    archived_error = max(abs(saved.learned_lower - lower), abs(saved.learned_upper - upper))
                    assert archived_error < 2e-7
                forecast = eval_matrix @ omega
                empirical = float(eval_rho @ eval_labels)
                rows.append({"subset": fd, "fold": fold, "arm": arm,
                    "calibration_profile": loss_profile, "evaluation_profile": "shifted",
                    "calibration_engines": len(calibration), "evaluation_engines": len(evaluation),
                    "train_eval_intersection": 0, "epsilon": epsilon,
                    "lower": lower, "upper": upper, "width": upper - lower,
                    "empirical_reliability": empirical, "point_prediction": float(means @ omega),
                    "held_out_brier": float(eval_rho @ ((forecast - eval_labels) ** 2)),
                    "miss_distance": max(0, lower - empirical, empirical - upper),
                    "decision_at_080": c.decision_category(lower, upper, 0.8),
                    "objective_crosscheck_error": objective_error,
                    "endpoint_crosscheck_error": endpoint_error,
                    "archived_endpoint_error": archived_error,
                    "lower_feasibility_excess": envelope["lower_loss_excess"],
                    "upper_feasibility_excess": envelope["upper_loss_excess"]})
            observed_omega, _, _ = fitted["observed"]
            target_omega, _, _ = fitted["shifted"]
            changes.append({"subset": fd, "fold": fold,
                "optimizer_l1_change": float(np.abs(target_omega - observed_omega).sum()),
                "original_epsilon": epsilon_original, "target_epsilon": epsilon_target,
                "calibration_profile_mass": sum(original.values()),
                "evaluation_profile_mass": sum(target.values())})

    fold_table = pd.DataFrame(rows)
    summary_rows = []
    for (fd, arm), group in fold_table.groupby(["subset", "arm"], sort=False):
        weights = group.evaluation_engines.to_numpy() / group.evaluation_engines.sum()
        summary = {"subset": fd, "arm": arm, "arm_label": ARM_LABELS[arm],
                   "folds": len(group), "evaluation_engines": int(group.evaluation_engines.sum())}
        for name in ("lower", "upper", "width", "point_prediction", "held_out_brier",
                     "empirical_reliability", "miss_distance"):
            summary[name] = float(weights @ group[name].to_numpy())
        summary["decision_at_080"] = c.decision_category(summary["lower"], summary["upper"], 0.8)
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)
    for fd in ("FD002", "FD004"):
        group = summary[summary.subset == fd].set_index("arm")
        for arm in ARM_LABELS:
            selection = (summary.subset == fd) & (summary.arm == arm)
            for name in ("lower", "upper", "width", "held_out_brier"):
                summary.loc[selection, f"{name}_change_from_transported"] = (
                    group.loc[arm, name] - group.loc["transported", name])
    for name, table in (
        ("profile_adaptation_folds", fold_table),
        ("profile_adaptation_summary", summary),
        ("profile_adaptation_profiles", pd.DataFrame(profile_rows)),
        ("profile_adaptation_weight_changes", pd.DataFrame(changes)),
    ):
        save_table(output, name, table)
    latex_rows = []
    for fd in ("FD002", "FD004"):
        for index, arm in enumerate(ARM_LABELS):
            row = summary[(summary.subset == fd) & (summary.arm == arm)].iloc[0]
            latex_rows.append(f"{fd if index == 0 else ''} & {ARM_LABELS[arm]} & "
                f"[{row.lower:.4f},{row.upper:.4f}] & {row.held_out_brier:.6f} " + r"\\")
    (output / "profile_adaptation_numbers.tex").write_text(
        "% Generated by profile_adaptation_analysis.py.\n"
        + "\\newcommand{\\ProfileAdaptationRows}{\n" + "\n".join(latex_rows) + "\n}\n",
        encoding="ascii")
    evidence = {"python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "input_sha256": {name: digest(inputs / name) for name in INPUT_NAMES},
        "source_sha256": {"scripts/cmapss_case_study.py": digest(repo / "scripts/cmapss_case_study.py"),
            "complete_revision_analysis.py": digest(Path(__file__).with_name("complete_revision_analysis.py")),
            "profile_adaptation_analysis.py": digest(Path(__file__))},
        "reused_bootstrap_tolerances": True, "bootstrap_resamples_per_original_fit": c.BOOTSTRAP_REPS,
        "fold_seed": c.CROSS_FIT_SEED, "folds": c.CROSS_FIT_FOLDS,
        "mission_cycles": c.CREDIBILITY_MISSION, "arms": ARM_LABELS,
        "independent_face_checks": len(fold_table),
        "max_objective_crosscheck_error": float(fold_table.objective_crosscheck_error.max()),
        "max_endpoint_crosscheck_error": float(fold_table.endpoint_crosscheck_error.max()),
        "max_archived_endpoint_error": float(fold_table.archived_endpoint_error.max()),
        "no_held_out_labels_used_for_fitting": True,
        "common_tolerance_control": "observed-profile tolerance in both transported and adapted sets",
        "aggregation": "fold-engine-count weighted endpoints and scores",
        "evaluation_profile": "same declared shifted profile in every arm"}
    (output / "profile_adaptation_checks.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
