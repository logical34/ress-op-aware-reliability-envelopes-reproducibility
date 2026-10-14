"""Recompute revision evidence without rendering or changing manuscript figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import null_space
from scipy.special import logsumexp

OUTPUT_NAMES = [
    "cmapss_envelope_summary", "cmapss_point_model_summary",
    "cmapss_source_inclusion_summary", "cmapss_profile_shift_summary",
    "cmapss_engine_level_predictions", "cmapss_source_credibility_intervals",
    "cmapss_profile_conditioned_envelope_summary", "cmapss_profile_conditioned_fold_summary",
    "cmapss_full_refit_operational_envelope_summary", "cmapss_oof_vs_full_refit_comparison",
    "cmapss_oof_baseline_comparison", "cmapss_oof_point_predictor_evaluation",
    "cmapss_oof_set_valued_evaluation", "cmapss_crossfit_leakage_audit",
    "cmapss_epsilon_bootstrap_audit", "cmapss_epsilon_monotonicity_summary",
    "cmapss_continuous_envelope_audit",
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def save_table(output, name, frame):
    numeric = frame.select_dtypes(include=np.number)
    if len(numeric.columns) and not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"Nonfinite numerical output in {name}")
    frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")


def source_predictions(c, train_features, test_features):
    age = test_features["observed_cycles"].to_numpy() + c.CREDIBILITY_MISSION
    test_renamed = test_features.rename(columns={v: k for k, v in c.TEST_FEATURE_MAP.items()})
    predictions = []
    for columns in c.SOURCE_FEATURE_SETS.values():
        if columns:
            params, sigma = c.fit_linear_predictor(train_features, columns)
            mu = c.predict_log_life(params, test_renamed, columns)
            probabilities = c.survival_from_log_life(mu, sigma + 0.10, age)
        else:
            log_life = np.log(train_features["life"].to_numpy())
            probabilities = c.survival_from_log_life(
                np.full(len(age), log_life.mean()), float(log_life.std()), age,
            )
        predictions.append(probabilities)
    return np.clip(np.column_stack(predictions), 0, 1)


def engine_weights(conditions, pi):
    rho = np.zeros(len(conditions))
    for name, mass in pi.items():
        mask = conditions == name
        if mask.any():
            rho[mask] = mass / mask.sum()
    return rho / rho.sum() if rho.sum() > 0 else np.full(len(conditions), 1 / len(conditions))


def face_extrema(matrix, labels, rho, means, epsilon):
    """Solve the small quadratic problem on every simplex face, independently of SLSQP."""
    size = matrix.shape[1]
    gram = (matrix.T * rho) @ matrix
    linear = matrix.T @ (rho * labels)
    constant = float(rho @ (labels ** 2))
    faces, minima = [], []
    for count in range(1, size + 1):
        for indices in combinations(range(size), count):
            indices = np.array(indices)
            center = np.full(count, 1 / count)
            basis = null_space(np.ones((1, count)))
            subgram = gram[np.ix_(indices, indices)]
            if count > 1:
                reduced = basis.T @ subgram @ basis
                if np.linalg.eigvalsh(reduced).min() <= 1e-14:
                    raise ValueError("Independent face solver requires nondegenerate source contrasts")
                center -= basis @ np.linalg.solve(reduced, basis.T @ (subgram @ center - linear[indices]))
            else:
                reduced = np.empty((0, 0))
            loss = float(center @ subgram @ center - 2 * linear[indices] @ center + constant)
            faces.append((indices, center, basis, reduced, loss))
            if center.min() >= -1e-10:
                minima.append(loss)
    optimum = min(minima)
    cap = optimum + epsilon
    predictions = []
    for indices, center, basis, reduced, loss in faces:
        if loss > cap + 1e-12:
            continue
        candidates = [center]
        if len(indices) > 1:
            direction = basis.T @ means[indices]
            inverse_direction = np.linalg.solve(reduced, direction)
            denominator = float(direction @ inverse_direction)
            if denominator > 1e-20:
                offset = basis @ inverse_direction * np.sqrt(max(0, cap - loss) / denominator)
                candidates.extend([center - offset, center + offset])
        for candidate in candidates:
            if candidate.min() >= -1e-9:
                predictions.append(float(means[indices] @ candidate))
    assert predictions, "No feasible independent endpoint"
    return optimum, min(predictions), max(predictions)


def predictive_audit(c, tables, prepared, output):
    predictions = tables["cmapss_engine_level_predictions"]
    folds = tables["cmapss_profile_conditioned_fold_summary"]
    coordinates = tables["cmapss_source_credibility_intervals"]
    fold_checks, geometry, interval_rows, training_rows, training_draws = [], [], [], [], []
    optimizer_checks, point_checks = [], []
    bootstrap_rng = np.random.default_rng(20261013)
    for data in prepared:
        fd = data.fd
        frame = predictions[predictions.subset == fd].reset_index(drop=True)
        assert frame.engine_id.is_unique
        independent = np.empty(len(frame), dtype=int)
        generator = np.random.default_rng(c.CROSS_FIT_SEED)
        for condition in sorted(frame.condition.unique()):
            indices = np.flatnonzero(frame.condition.to_numpy() == condition)
            generator.shuffle(indices)
            independent[indices] = np.arange(len(indices)) % c.CROSS_FIT_FOLDS
        assert np.array_equal(independent, frame.fold.to_numpy())
        train_features = c.unit_feature_table(data)
        test_features = c.test_feature_table(data)
        original_predictions = source_predictions(c, train_features, test_features)
        source_error = float(np.max(np.abs(original_predictions - c.prediction_matrix(frame))))
        assert source_error < 1e-12, (fd, source_error)
        profiles = dict(c.profile_catalog(frame))
        frozen_weights = {}
        for name, pi in profiles.items():
            independent_scores = np.zeros(len(c.source_names()) + 2)
            independent_means = np.zeros(len(c.source_names()) + 2)
            for fold in range(c.CROSS_FIT_FOLDS):
                calibration = frame[frame.fold != fold]
                evaluation = frame[frame.fold == fold]
                assert set(calibration.engine_id).isdisjoint(set(evaluation.engine_id))
                fold_checks.append({
                    "subset": fd, "profile_name": name, "fold": fold,
                    "calibration_engines": len(calibration), "evaluation_engines": len(evaluation),
                    "intersection": 0, "source_probability_max_error": source_error,
                })
                omega_frame = coordinates[
                    (coordinates.subset == fd) & (coordinates.profile_name == name) & (coordinates.fold == fold)
                ].set_index("source_model").reindex(c.source_names())
                omega = omega_frame.omega_hat.to_numpy()
                assert np.min(omega) >= -1e-10 and abs(omega.sum() - 1) < 1e-9
                frozen_weights[name, fold] = omega
                row = folds[(folds.subset == fd) & (folds.profile_name == name) & (folds.fold == fold)].iloc[0]
                rho = engine_weights(calibration.condition.to_numpy(), pi)
                matrix = c.prediction_matrix(calibration)
                labels = calibration.y_survive_100.to_numpy()
                eval_rho = engine_weights(evaluation.condition.to_numpy(), pi)
                eval_matrix = c.prediction_matrix(evaluation)
                eval_labels = evaluation.y_survive_100.to_numpy()
                a = eval_rho @ eval_matrix
                independent_optimum, lower, upper = face_extrema(matrix, labels, rho, a, row.epsilon)
                objective_gap = abs(independent_optimum - row.best_calibration_brier)
                endpoint_gap = max(abs(lower - row.learned_lower), abs(upper - row.learned_upper))
                assert objective_gap < 1e-9 and endpoint_gap < 2e-7, (fd, name, fold, objective_gap, endpoint_gap)
                assert max(abs(a.min() - row.fixed_lower), abs(a.max() - row.fixed_upper)) < 1e-12
                assert abs(eval_rho @ eval_labels - row.empirical_reliability) < 1e-12
                optimizer_checks.append({"subset": fd, "profile_name": name, "fold": fold,
                    "construction": "oof", "objective_gap": objective_gap, "endpoint_gap": endpoint_gap})
                candidate_predictions = np.column_stack([eval_matrix, eval_matrix.mean(axis=1), eval_matrix @ omega])
                independent_scores += len(evaluation) / len(frame) * (eval_rho @ ((candidate_predictions - eval_labels[:, None]) ** 2))
                independent_means += len(evaluation) / len(frame) * (eval_rho @ candidate_predictions)
                curvature = (matrix.T * rho) @ matrix
                basis = null_space(np.ones((1, matrix.shape[1])))
                mu = float(np.linalg.eigvalsh(basis.T @ curvature @ basis).min())
                a = c.profile_weighted_source_means(evaluation, pi)
                radius = float(np.linalg.norm(basis.T @ a) * np.sqrt(row.epsilon / mu)) if mu > 0 else 0.0
                geometry.append({
                    "subset": fd, "profile_name": name, "fold": fold, "tangent_min_eigenvalue": mu,
                    "curvature_available": bool(mu > 0), "epsilon": row.epsilon,
                    "conditional_radius": radius, "learned_width": row.learned_width,
                })
            for index, method in enumerate(c.source_names() + ["equal_weight_pool", "brier_optimal_point_pool"]):
                reported = tables["cmapss_oof_point_predictor_evaluation"]
                reported = reported[(reported.subset == fd) & (reported.profile_name == name) & (reported.method_id == method)].iloc[0]
                score_gap = abs(independent_scores[index] - reported.oof_brier_score)
                mean_gap = abs(independent_means[index] - reported.point_estimate)
                assert max(score_gap, mean_gap) < 1e-12
                point_checks.append({"subset": fd, "profile_name": name, "method": method,
                    "independent_brier": independent_scores[index], "independent_mean": independent_means[index],
                    "score_gap": score_gap, "mean_gap": mean_gap})
            refits = tables["cmapss_full_refit_operational_envelope_summary"]
            refit = refits[(refits.subset == fd) & (refits.profile_name == name)].iloc[0]
            full_rho = engine_weights(frame.condition.to_numpy(), pi)
            full_matrix = c.prediction_matrix(frame)
            optimum, lower, upper = face_extrema(full_matrix, frame.y_survive_100.to_numpy(), full_rho,
                full_rho @ full_matrix, refit.epsilon)
            objective_gap = abs(optimum - refit.best_calibration_brier)
            endpoint_gap = max(abs(lower - refit.refit_lower), abs(upper - refit.refit_upper))
            assert objective_gap < 1e-9 and endpoint_gap < 2e-7
            optimizer_checks.append({"subset": fd, "profile_name": name, "fold": -1,
                "construction": "full_refit", "objective_gap": objective_gap, "endpoint_gap": endpoint_gap})
            selected = folds[(folds.subset == fd) & (folds.profile_name == name)]
            count_weights = selected.eval_engines.to_numpy() / selected.eval_engines.sum()
            empirical = selected.empirical_reliability.to_numpy()
            fixed_gap = np.maximum.reduce([
                selected.fixed_lower.to_numpy() - empirical, empirical - selected.fixed_upper.to_numpy(),
                np.zeros(len(selected)),
            ])
            learned_gap = np.maximum.reduce([
                selected.learned_lower.to_numpy() - empirical, empirical - selected.learned_upper.to_numpy(),
                np.zeros(len(selected)),
            ])
            full_empirical = c.profile_weighted_empirical(frame, pi)[0]
            interval_rows.append({
                "subset": fd, "profile_name": name,
                "fixed_inside_folds": int(np.sum(fixed_gap <= 1e-12)),
                "learned_inside_folds": int(np.sum(learned_gap <= 1e-12)),
                "folds": len(selected), "fixed_mean_fold_gap": float(count_weights @ fixed_gap),
                "learned_mean_fold_gap": float(count_weights @ learned_gap),
                "fold_count_weighted_empirical": float(count_weights @ empirical),
                "all_engine_profile_empirical": full_empirical,
                "aggregation_difference": float(count_weights @ empirical - full_empirical),
            })
        # Hold feature extraction and OOF mixture weights fixed in this sensitivity study.
        pi = profiles["observed"]
        pooling_coefficients = np.zeros_like(original_predictions)
        for fold in range(c.CROSS_FIT_FOLDS):
            mask = frame.fold.to_numpy() == fold
            pooling_coefficients[mask] = engine_weights(frame.loc[mask, "condition"].to_numpy(), pi)[:, None]
            pooling_coefficients[mask] *= frozen_weights["observed", fold][None, :] * mask.sum() / len(frame)
        original = float(np.sum(pooling_coefficients * original_predictions))
        draws = []
        for replicate in range(200):
            indices = bootstrap_rng.integers(0, len(train_features), len(train_features))
            probabilities = source_predictions(c, train_features.iloc[indices], test_features)
            value = float(np.sum(pooling_coefficients * probabilities))
            draws.append(value)
            training_draws.append({"subset": fd, "replicate": replicate, "oof_companion_point_pool": value})
        q05, q95 = np.quantile(draws, [0.05, 0.95])
        training_rows.append({
            "subset": fd, "profile_name": "observed", "original_point_pool": original,
            "q05": q05, "q95": q95, "spread": q95 - q05,
            "sd": float(np.std(draws, ddof=1)), "resamples": 200, "seed": 20261013,
        })
        print(f"Audited folds and training resamples: {fd}", flush=True)
    save_table(output, "independent_fold_checks", pd.DataFrame(fold_checks))
    save_table(output, "independent_optimizer_checks", pd.DataFrame(optimizer_checks))
    save_table(output, "independent_point_checks", pd.DataFrame(point_checks))
    save_table(output, "conditional_curvature_diagnostics", pd.DataFrame(geometry))
    intervals = pd.DataFrame(interval_rows)
    training = pd.DataFrame(training_rows)
    save_table(output, "fold_interval_diagnostics", intervals)
    save_table(output, "training_resample_summary", training)
    save_table(output, "training_resample_draws", pd.DataFrame(training_draws))
    return intervals, training


def nonlinear_analysis(s, output):
    rng = np.random.default_rng(20261014)
    shapes = np.array([1.4, 2.0, 2.6, 1.8, 2.2, 3.0])
    failures = np.array([1, 2, 2, 1, 2, 2])
    profiles = {"nominal": s.CASEB_PI0, "stressed": s.CASEB_PI1}
    times = np.arange(0, 1201, 20, dtype=float)
    curves, diagnostics = [], []
    strengths = s.FULL_SOURCE_SCENARIOS
    for prior in s.CASEB_PRIORS:
        theta = 400 * s.caseb_sample_prior(prior, 20000, rng)
        failure_probability = -np.expm1(-theta * s.caseb_profile_multiplier(s.CASEB_PI_OBS))
        field = np.sum(
            failures * np.log(failure_probability) + (12 - failures) * np.log1p(-failure_probability), axis=1,
        )
        blocks = [("sensor", s.CASEB_SENSOR_HAZARD_OBS, s.SIGMA_SENSOR),
                  ("simulation", s.CASEB_SIM_HAZARD_OBS, s.SIGMA_SIM),
                  ("logs", s.CASEB_LOG_HAZARD_OBS, s.SIGMA_LOG),
                  ("llm_text", s.CASEB_LLM_TEXT_HAZARD_OBS, s.SIGMA_LLM)]
        other = {name: s.caseb_lognormal_loglik(theta / 400, observations, sigma)
                 for name, observations, sigma in blocks}
        output_likelihood = s.caseb_output_loglik(theta / 400)
        for strength in strengths:
            logw = strength.field * field + output_likelihood
            for name, values in other.items():
                logw = logw + getattr(strength, name) * values
            weights = np.exp(logw - logsumexp(logw))
            ess = float(1 / np.sum(weights ** 2))
            diagnostics.append({"prior": prior.name, "strength": strength.name, "samples": 20000, "ess": ess})
            for name, profile in profiles.items():
                component = np.exp(-theta[:, :, None] * s.caseb_profile_multiplier(profile)[None, :, None]
                                   * (times[None, None, :] / 400) ** shapes[None, :, None])
                values = (1 - (1 - component[:, 0]) * (1 - component[:, 1])) * component[:, 2]
                values *= (1 - (1 - component[:, 3]) * (1 - component[:, 4])) * component[:, 5]
                for time, probability in zip(times, weights @ values):
                    curves.append({"profile": name, "prior": prior.name, "strength": strength.name,
                                   "time_h": time, "posterior_mean_reliability": probability})
    curves = pd.DataFrame(curves)
    envelope = curves.groupby(["profile", "time_h"], as_index=False).posterior_mean_reliability.agg(
        lower="min", upper="max",
    )
    diagnostics = pd.DataFrame(diagnostics)
    save_table(output, "nonlinear_scenario_curves", curves)
    save_table(output, "nonlinear_envelopes", envelope)
    save_table(output, "nonlinear_sampling_diagnostics", diagnostics)
    inputs = {
        "seed": 20261014, "samples_per_prior": 20000, "time_h": times.tolist(),
        "reference_time_h": 400, "weibull_shapes": shapes.tolist(),
        "component_test_trials": 12, "component_failures": failures.tolist(),
        "input_evidence_profile": s.CASEB_PI_OBS.tolist(), "output_evidence_profile": s.CASEB_PI0.tolist(),
        "query_profiles": {k: v.tolist() for k, v in profiles.items()},
        "prior_hazard_means_per_hour": s.CASEB_EXPERT_MID_MEAN.tolist(),
        "prior_scenarios": [asdict(x) for x in s.CASEB_PRIORS],
        "strength_scenarios": [asdict(x) for x in strengths],
        "source_lognormal_observations_per_hour": {name: observations.tolist() for name, observations, sigma in blocks},
        "source_log_standard_deviations": {name: sigma for name, observations, sigma in blocks},
        "load_multiplier_matrix": s.CASEB_LOAD_MULTIPLIERS.tolist(),
        "output_observation": s.CASEB_OUTPUT_RELIABILITY_OBS, "output_sigma": s.CASEB_OUTPUT_SIGMA,
        "output_discrepancy_shifts": s.CASEB_OUTPUT_DISCREPANCY.tolist(),
        "output_discrepancy_masses": [1 / 3] * 3,
        "theta_definition": "dimensionless cumulative-hazard coefficient at reference time; theta=400*baseline hazard",
    }
    (output / "nonlinear_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return envelope, diagnostics


def budget_and_thresholds(c, tables, output):
    planning = c.validation_planning(tables["cmapss_envelope_summary"], tables["cmapss_profile_shift_summary"])
    envelope = tables["cmapss_envelope_summary"].query("subset == 'FD004' and mission_cycles == 100").iloc[0]
    wp = tables["cmapss_profile_shift_summary"].query("subset == 'FD004'").width_100.mean()
    budgets = planning.budget.to_numpy()
    direct = np.column_stack([
        envelope.width * np.exp(-0.045 * budgets), envelope.width * np.exp(-0.065 * budgets),
        np.maximum(0, envelope.width * np.exp(-0.095 * budgets) - 0.15 * wp * budgets / 12),
    ])
    assert np.max(np.abs(planning[["uniform", "profile_targeted", "risk_targeted"]].to_numpy() - direct)) < 1e-12
    save_table(output, "hypothetical_budget_curves", planning)
    threshold_rows = []
    for row in tables["cmapss_profile_conditioned_envelope_summary"].itertuples():
        for requirement in (0.25, 0.35, 0.45, 0.65, 0.80):
            threshold_rows.append({"subset": row.subset, "profile_name": row.profile_name,
                "lower": row.learned_lower, "upper": row.learned_upper, "requirement": requirement,
                "decision": c.decision_category(row.learned_lower, row.learned_upper, requirement)})
    save_table(output, "threshold_sensitivity", pd.DataFrame(threshold_rows))
    return {"initial_width": float(envelope.width), "mean_condition_wilson_width": float(wp),
            "observations_simulated": 0, "posterior_updates": 0, "assumed_equal_policy_costs": True}


def proof_checks():
    p = np.linspace(0, 1, 10001)
    empirical = 0.16 + (p - 0.8) ** 2
    population = 0.21 + (p - 0.7) ** 2
    delta = float(np.max(np.abs(empirical - population)))
    epsilon = 0.0025
    admissible = empirical <= 0.16 + epsilon + 1e-12
    assert np.max(population[admissible] - 0.21) <= epsilon + 2 * delta + 1e-12
    assert 0.16 + (0.7 - 0.8) ** 2 <= 0.16 + 2 * delta + 1e-12
    radius = np.sqrt(0.02) * np.sqrt(epsilon / 0.5)
    assert np.max(np.abs(0.7 + 0.2 * p[admissible] - 0.86)) <= radius + 1e-12
    boundary_predictions = np.array([0.7, 0.9])
    for weight in (0, 0.25, 0.5, 1):
        v = np.array([-weight, weight])
        gradient = 2 * 0.7 * boundary_predictions
        assert gradient @ v >= -1e-12
    mean_a, mean_b = 0.8, (0.6 + 1.0) / 2
    compliance_a, compliance_b = float(0.8 >= 0.75), (float(0.6 >= 0.75) + float(1 >= 0.75)) / 2
    assert mean_a == mean_b and compliance_a == 1 and compliance_b == 0.5
    return {"uniform_loss_delta_example": delta, "excess_loss_bound_checked": True,
            "population_minimizer_inclusion_checked": True, "curvature_radius_example": radius,
            "boundary_first_order_condition_checked": True,
            "zero_tolerance_degenerate_example": {"calibration_sources": [0.5, 0.5],
                "evaluation_sources": [0.2, 0.9], "epsilon": 0, "width": 0.7, "mu": 0},
            "single_source_simplex_width": 0, "compliance_counterexample": [compliance_a, compliance_b]}


def generate_numbers(tables, intervals, training, nonlinear, diagnostics, output):
    macros, endrow = [], chr(92) * 2
    def macro(name, value, new=False):
        command = "newcommand" if new else "renewcommand"
        macros.append("\\" + command + "{\\" + name + "}{" + value + "}")
    oof = tables["cmapss_profile_conditioned_envelope_summary"]
    refit = tables["cmapss_full_refit_operational_envelope_summary"]
    rows = []
    for row in oof.itertuples():
        ref = refit[(refit.subset == row.subset) & (refit.profile_name == row.profile_name)].iloc[0]
        name = row.subset + (" obs." if row.profile_name == "observed" else " shift")
        rows.append(f"{name} & [{row.fixed_lower:.4f},{row.fixed_upper:.4f}] & "
            f"[{row.learned_lower:.4f},{row.learned_upper:.4f}] & [{ref.refit_lower:.4f},{ref.refit_upper:.4f}] & "
            f"{row.learned_width:.4f} & {ref.refit_width:.4f} & {ref.refit_width-row.learned_width:.4f} " + endrow)
    macro("CmapssProfileConditionedRows", "\n".join(rows))
    points = tables["cmapss_oof_point_predictor_evaluation"]
    rows, better, best = [], 0, 0
    for (fd, name), group in points.groupby(["subset", "profile_name"], sort=True):
        singles = group[group.method_id.isin(["life_distribution", "temperature_degradation",
            "temperature_pressure_degradation", "all_sensor_degradation"])]
        oracle = singles.sort_values("oof_brier_score").iloc[0]
        equal = group[group.method_id == "equal_weight_pool"].iloc[0]
        learned = group[group.method_id == "brier_optimal_point_pool"].iloc[0]
        better += int(learned.oof_brier_score < equal.oof_brier_score)
        best += int(learned.oof_brier_score <= group.oof_brier_score.min() + 1e-12)
        label = fd + (" shifted" if name == "shifted" else "")
        for position, (item, title) in enumerate([(oracle, f"Best single oracle ({oracle.method_label})"),
            (equal, "Equal-weight pool"), (learned, "Brier-optimal pool")]):
            rows.append(f"{label if position == 0 else ''} & {title} & {item.oof_brier_score:.6f} & {item.point_estimate:.4f} " + endrow)
    macro("CmapssPointPredictorRows", "\n".join(rows))
    macro("CmapssLearnedPointBetterCount", str(better))
    macro("CmapssLearnedPointBestOrTiedCount", str(best))
    rows = []
    for row in intervals.itertuples():
        label = row.subset + (" shifted" if row.profile_name == "shifted" else "")
        fixed = "0 (exact)" if row.fixed_mean_fold_gap == 0 else f"{row.fixed_mean_fold_gap:.4f}"
        learned = "0 (exact)" if row.learned_mean_fold_gap == 0 else f"{row.learned_mean_fold_gap:.4f}"
        rows.append(f"{label} & {row.fixed_inside_folds}/{row.folds} & {row.learned_inside_folds}/{row.folds} & {fixed} & {learned} " + endrow)
    macro("CmapssSetValuedRows", "\n".join(rows))
    rows = [f"{r.subset} & {r.original_point_pool:.3f} & {r.q05:.3f}--{r.q95:.3f} & {r.sd:.3f} " + endrow
            for r in training.itertuples()]
    macro("RevisionTrainingRows", "\n".join(rows))
    fd001 = oof[(oof.subset == "FD001") & (oof.profile_name == "observed")].iloc[0]
    macro("RevisionFDOneThresholdInterval", f"[{fd001.learned_lower:.4f},{fd001.learned_upper:.4f}]", True)
    for name, prefix in [("nominal", "Nominal"), ("stressed", "Stressed")]:
        row = nonlinear[(nonlinear.profile == name) & (nonlinear.time_h == 400)].iloc[0]
        macro(f"RevisionNonlinear{prefix}Interval", f"[{row.lower:.3f},{row.upper:.3f}]", True)
    macro("RevisionNonlinearMinESS", f"{diagnostics.ess.min():.1f}", True)
    text = "% Generated by complete_revision_analysis.py; overrides verified revision tables.\n" + "\n".join(macros) + "\n"
    (output / "revision_numbers.tex").write_text(text, encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cmapss-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-predictive", action="store_true", help="Reuse an already completed revision output directory.")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = (args.output_dir or repo / "revision_results").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cmapss_dir:
        os.environ["CMAPSS_DIR"] = str(args.cmapss_dir.resolve())
    c = load_module(repo / "scripts/cmapss_case_study.py", "revision_cmapss")
    s = load_module(repo / "scripts/simulate_reliability.py", "revision_controlled")
    prepared = [c.add_condition_labels(c.load_fd(fd)) for fd in c.FD_SETS]
    figure_hashes = {p.name: sha256(p) for p in (repo / "figures").glob("*") if p.is_file()}
    if args.reuse_predictive:
        tables = {name: pd.read_csv(output / f"{name}.csv") for name in OUTPUT_NAMES}
    else:
        print("Recomputing four C-MAPSS subsets, 30 held-out folds, 500 calibration bootstraps per fit.", flush=True)
        values = c.build_case_outputs(prepared)
        assert len(values) == len(OUTPUT_NAMES)
        tables = dict(zip(OUTPUT_NAMES, values))
        for name, value in tables.items():
            save_table(output, name, value)
    save_table(output, "cmapss_dataset_summary", c.fd_summary(prepared))
    intervals, training = predictive_audit(c, tables, prepared, output)
    nonlinear, diagnostics = nonlinear_analysis(s, output)
    budget = budget_and_thresholds(c, tables, output)
    generate_numbers(tables, intervals, training, nonlinear, diagnostics, output)
    assert all(sha256(repo / "figures" / name) == digest for name, digest in figure_hashes.items())
    provenance = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "source_files": {path.relative_to(repo).as_posix(): source_sha256(path) for path in [
            repo / "scripts/cmapss_case_study.py", repo / "scripts/simulate_reliability.py"]},
        "revision_script_sha256": source_sha256(Path(__file__)),
        "raw_data_sha256": {path.name: sha256(path) for path in sorted(c.CMAPSS_DIR.glob("*.txt"))
                             if path.name.startswith(("train_", "test_", "RUL_"))},
        "seed_folds": c.CROSS_FIT_SEED, "seed_calibration_bootstrap": c.CROSS_FIT_SEED + 100,
        "seed_full_data_refit_bootstrap": c.CROSS_FIT_SEED + 700, "calibration_resamples": c.BOOTSTRAP_REPS,
        "seed_training_resamples": 20261013, "training_resamples": 200, "seed_nonlinear": 20261014,
        "proof_and_decision_checks": proof_checks(), "budget_curve_check": budget, "figures_unchanged": True,
    }
    (output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(training.to_string(index=False), flush=True)
    print(nonlinear[nonlinear.time_h == 400].to_string(index=False), flush=True)
    print(f"Minimum nonlinear ESS: {diagnostics.ess.min():.6f}", flush=True)
    print(f"Completed data-only revision reproduction: {output}", flush=True)


if __name__ == "__main__":
    main()
