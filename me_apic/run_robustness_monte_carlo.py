"""Paired Monte Carlo robustness test for intermittent pantograph control."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_baseline import Parameters
from run_pantograph_experiments import (
    ControllerParameters,
    PantographParameters,
    RobustnessParameters,
    metrics,
    simulate,
)


METHODS = ("Passive", "Continuous", "T-APIC", "E-APIC", "ME-APIC")
COLORS = {
    "Passive": "#7A7A7A",
    "Continuous": "#4C78A8",
    "T-APIC": "#F58518",
    "E-APIC": "#E7B34B",
    "ME-APIC": "#7B5AA6",
}


def perturbed_pantograph(
    nominal: PantographParameters, rng: np.random.Generator
) -> tuple[PantographParameters, dict[str, float]]:
    mass_scale = rng.uniform(0.9, 1.1, 3)
    stiffness_scale = rng.uniform(0.9, 1.1, 3)
    damping_scale = rng.uniform(0.9, 1.1, 3)
    contact_scale = float(rng.uniform(0.9, 1.1))
    model = PantographParameters(
        masses_kg=tuple(np.asarray(nominal.masses_kg)*mass_scale),
        damping_Ns_m=tuple(np.asarray(nominal.damping_Ns_m)*damping_scale),
        stiffness_N_m=tuple(np.asarray(nominal.stiffness_N_m)*stiffness_scale),
        contact_stiffness_N_m=nominal.contact_stiffness_N_m*contact_scale,
    )
    record = {
        "mass_scale_1": float(mass_scale[0]),
        "mass_scale_2": float(mass_scale[1]),
        "mass_scale_3": float(mass_scale[2]),
        "stiffness_scale_1": float(stiffness_scale[0]),
        "stiffness_scale_2": float(stiffness_scale[1]),
        "stiffness_scale_3": float(stiffness_scale[2]),
        "damping_scale_3": float(damping_scale[2]),
        "contact_stiffness_scale": contact_scale,
    }
    return model, record


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    from scipy.stats import t as student_t, wilcoxon

    output = []
    for speed in sorted({float(row["speed_kmh"]) for row in rows}):
        speed_rows = [row for row in rows if float(row["speed_kmh"]) == speed]
        passive = {
            int(row["trial"]): float(row["std_force_N"])
            for row in speed_rows if row["controller"] == "Passive"
        }
        continuous_effort = {
            int(row["trial"]): float(row["effort_N2s"])
            for row in speed_rows if row["controller"] == "Continuous"
        }
        continuous_work = {
            int(row["trial"]): float(row["mechanical_work_J"])
            for row in speed_rows if row["controller"] == "Continuous"
        }
        for method in METHODS:
            selected = [row for row in speed_rows if row["controller"] == method]
            std = np.asarray([float(row["std_force_N"]) for row in selected])
            improvements = np.asarray([
                100*(passive[int(row["trial"])]-float(row["std_force_N"]))
                / passive[int(row["trial"])] for row in selected
            ])
            effort_reduction = np.asarray([
                100*(continuous_effort[int(row["trial"])]-float(row["effort_N2s"]))
                / max(continuous_effort[int(row["trial"])], 1e-12)
                for row in selected
            ])
            work_reduction = np.asarray([
                100*(continuous_work[int(row["trial"])]-float(row["mechanical_work_J"]))
                / max(continuous_work[int(row["trial"])], 1e-12)
                for row in selected
            ])
            mean_improvement = float(np.mean(improvements))
            sem = float(np.std(improvements, ddof=1)/np.sqrt(len(improvements))) if len(improvements) > 1 else 0.0
            critical = float(student_t.ppf(0.975, len(improvements)-1)) if len(improvements) > 1 else 0.0
            if method == "Passive":
                wilcoxon_p, effect_dz = 1.0, 0.0
            else:
                wilcoxon_p = float(wilcoxon(improvements, alternative="two-sided").pvalue)
                effect_dz = float(mean_improvement/max(np.std(improvements, ddof=1), 1e-12))
            output.append({
                "controller": method,
                "speed_kmh": speed,
                "n_trials": len(selected),
                "mean_std_force_N": float(np.mean(std)),
                "sd_std_force_N": float(np.std(std, ddof=1)),
                "median_paired_improvement_percent": float(np.median(improvements)),
                "mean_paired_improvement_percent": mean_improvement,
                "mean_improvement_ci95_low_percent": mean_improvement-critical*sem,
                "mean_improvement_ci95_high_percent": mean_improvement+critical*sem,
                "paired_effect_size_dz": effect_dz,
                "wilcoxon_p_two_sided": wilcoxon_p,
                "q25_paired_improvement_percent": float(np.percentile(improvements, 25)),
                "q75_paired_improvement_percent": float(np.percentile(improvements, 75)),
                "win_rate_vs_passive_percent": float(100*np.mean(improvements > 0)),
                "mean_ATR_percent": float(np.mean([
                    float(row["ATR_percent"]) for row in selected
                ])),
                "mean_effort_reduction_vs_continuous_percent": float(
                    np.mean(effort_reduction)
                ),
                "mean_mechanical_work_J": float(np.mean([
                    float(row["mechanical_work_J"]) for row in selected
                ])),
                "mean_mechanical_work_reduction_vs_continuous_percent": float(
                    np.mean(work_reduction)
                ),
                "mean_contact_loss_percent": float(np.mean([
                    float(row["contact_loss_percent"]) for row in selected
                ])),
            })
    return output


def publication_figure(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    speeds = (300.0, 350.0)
    compared = ("Continuous", "T-APIC", "E-APIC", "ME-APIC")
    n_trials = len({int(row["trial"]) for row in rows})
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0), hspace=0.38, wspace=0.32)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    rng = np.random.default_rng(404)
    positions, labels = [], []
    for speed_index, speed in enumerate(speeds):
        passive = {
            int(row["trial"]): float(row["std_force_N"])
            for row in rows if row["controller"] == "Passive"
            and float(row["speed_kmh"]) == speed
        }
        for method_index, method in enumerate(compared):
            selected = [row for row in rows if row["controller"] == method
                        and float(row["speed_kmh"]) == speed]
            values = np.asarray([
                100*(passive[int(row["trial"])]-float(row["std_force_N"]))
                / passive[int(row["trial"])] for row in selected
            ])
            position = speed_index*5.2+method_index
            positions.append(position)
            labels.append(method)
            box = ax_a.boxplot(
                values, positions=[position], widths=0.55, patch_artist=True,
                showfliers=False, medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            )
            box["boxes"][0].set(facecolor=COLORS[method], alpha=0.42,
                                edgecolor=COLORS[method])
            jitter = rng.uniform(-0.16, 0.16, len(values))
            ax_a.scatter(position+jitter, values, s=14, color=COLORS[method],
                         edgecolor="white", linewidth=0.35, zorder=3)
    ax_a.axhline(0, color="0.25", ls="--", lw=0.8)
    ax_a.set_xticks([1.5, 6.7], ["300 km/h", "350 km/h"])
    ax_a.set_ylabel("Paired reduction in force s.d. vs Passive (%)")
    ax_a.grid(axis="y", alpha=0.22)
    handles = [mpl.patches.Patch(facecolor=COLORS[m], alpha=0.55, label=m) for m in compared]
    ax_a.legend(handles=handles, ncol=3, loc="upper right")

    width = 0.19
    for method_index, method in enumerate(compared):
        atr_means = []
        win_rates = []
        for speed in speeds:
            selected = [row for row in rows if row["controller"] == method
                        and float(row["speed_kmh"]) == speed]
            passive = {
                int(row["trial"]): float(row["std_force_N"])
                for row in rows if row["controller"] == "Passive"
                and float(row["speed_kmh"]) == speed
            }
            atr_means.append(np.mean([float(row["ATR_percent"]) for row in selected]))
            win_rates.append(100*np.mean([
                float(row["std_force_N"]) < passive[int(row["trial"])]
                for row in selected
            ]))
        x = np.arange(len(speeds))+(method_index-1.5)*width
        atr_bars = ax_b.bar(x, atr_means, width=width, color=COLORS[method], alpha=0.82)
        win_bars = ax_c.bar(x, win_rates, width=width, color=COLORS[method], alpha=0.82)
        ax_b.bar_label(atr_bars, fmt="%.0f", padding=2, fontsize=5.5)
        ax_c.bar_label(win_bars, fmt="%.0f", padding=2, fontsize=5.5)
    for axis in (ax_b, ax_c):
        axis.set_xticks(np.arange(2), ["300", "350"])
        axis.set_xlabel("Speed (km/h)")
        axis.grid(axis="y", alpha=0.22)
    ax_b.set_ylabel("Mean activation-time rate (%)")
    ax_c.set_ylabel("Win rate vs Passive (%)")
    ax_c.set_ylim(0, 105)
    for label, axis in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
        axis.text(-0.08, 1.04, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="bottom")
    fig.suptitle(
        f"Paired robustness test with uncertainty, noise and actuator lag "
        f"(n = {n_trials} per speed)", y=0.995,
    )
    fig.subplots_adjust(top=0.93, bottom=0.11, left=0.09, right=0.98)
    stem = output/"robustness_figure"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def run_trial_job(
    trial: int,
    scenario_seed: int,
    speeds: tuple[float, ...],
    beam: Parameters,
    nominal_panto: PantographParameters,
    control: ControllerParameters,
) -> list[dict[str, float | int | str]]:
    """Run one paired scenario; kept top-level for Windows multiprocessing."""
    scenario_rng = np.random.default_rng(scenario_seed)
    plant_panto, perturbation = perturbed_pantograph(nominal_panto, scenario_rng)
    phases = tuple(scenario_rng.uniform(-np.pi, np.pi, 3))
    robust = RobustnessParameters(
        measurement_noise_std_N=2.0,
        actuator_delay_s=0.005,
        actuator_time_constant_s=0.015,
        irregularity_scale=float(scenario_rng.uniform(0.8, 1.2)),
        irregularity_phase_offsets_rad=phases,
    )
    trial_rows: list[dict[str, float | int | str]] = []
    for speed in speeds:
        for method in METHODS:
            run = simulate(
                method, speed, beam, plant_panto, control,
                feedback_kind="modal_lqr", robustness=robust,
                random_seed=scenario_seed,
                controller_model_pantograph=nominal_panto,
            )
            row = metrics(run)
            row.update({
                "trial": trial,
                "scenario_seed": scenario_seed,
                "measurement_noise_std_N": robust.measurement_noise_std_N,
                "actuator_delay_s": robust.actuator_delay_s,
                "actuator_time_constant_s": robust.actuator_time_constant_s,
                "irregularity_scale": robust.irregularity_scale,
                **perturbation,
            })
            trial_rows.append(row)
    return trial_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("robustness_results"))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--speeds", type=float, nargs="+", default=(300, 350))
    parser.add_argument("--seed", type=int, default=9103)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    beam = Parameters()
    nominal_panto = PantographParameters()
    control = ControllerParameters()
    master = np.random.default_rng(args.seed)
    scenario_seeds = [
        int(master.integers(0, 2**31-1)) for _ in range(args.trials)
    ]
    speeds = tuple(float(speed) for speed in args.speeds)
    rows: list[dict[str, float | int | str]] = []
    if args.workers <= 1:
        for trial, scenario_seed in enumerate(scenario_seeds):
            rows.extend(run_trial_job(
                trial, scenario_seed, speeds, beam, nominal_panto, control
            ))
            print(f"completed trial {trial+1}/{args.trials}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_trial_job, trial, scenario_seed, speeds,
                    beam, nominal_panto, control,
                ): trial
                for trial, scenario_seed in enumerate(scenario_seeds)
            }
            completed = 0
            for future in as_completed(futures):
                rows.extend(future.result())
                completed += 1
                print(f"completed trial {completed}/{args.trials}", flush=True)
    method_order = {method: index for index, method in enumerate(METHODS)}
    rows.sort(key=lambda row: (
        int(row["trial"]), float(row["speed_kmh"]), method_order[str(row["controller"])]
    ))

    with (args.output/"monte_carlo_trials.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (args.output/"monte_carlo_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output/"monte_carlo_metadata.json").write_text(
        json.dumps({
            "n_trials": args.trials,
            "speeds_kmh": args.speeds,
            "master_seed": args.seed,
            "workers": args.workers,
            "nominal_pantograph": asdict(nominal_panto),
            "controller": asdict(control),
            "uncertainty": {
                "pantograph_parameters": "independent uniform +/-10%",
                "irregularity_scale": "uniform 0.8--1.2 with random phases",
                "measurement_noise_std_N": 2.0,
                "actuator_delay_s": 0.005,
                "actuator_time_constant_s": 0.015,
                "pairing": "same plant, irregularity and noise seed across methods",
                "controller_model": "fixed nominal DSA380",
            },
        }, indent=2), encoding="utf-8",
    )
    publication_figure(rows, args.output)


if __name__ == "__main__":
    main()

