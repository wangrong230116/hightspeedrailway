"""Reviewer-driven convergence, equal-budget, and spectral experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_baseline import Parameters
from run_pantograph_experiments import (
    ControllerParameters,
    PantographParameters,
    metrics,
    simulate,
)


OUT = Path("revision_results")
OUT.mkdir(parents=True, exist_ok=True)
SPEEDS = (300.0, 350.0)


def write_csv(name: str, records: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_metrics(controller, speed, beam, panto, control):
    return metrics(simulate(
        controller, speed, beam, panto, control, feedback_kind="modal_lqr"
    ))


def sensitivity_study() -> list[dict]:
    beam0, panto0, control0 = Parameters(), PantographParameters(), ControllerParameters()
    cases = []
    for value in (30, 60, 120):
        cases.append(("elements", value, replace(beam0, elements=value), panto0, control0))
    for value in (8, 12, 16, 24):
        cases.append(("modes", value, beam0, panto0, replace(control0, modal_modes=value)))
    for multiplier in (2.0, 1.0, 0.5):
        cases.append(("dt_multiplier", multiplier, replace(beam0, dt=beam0.dt*multiplier), panto0, control0))
    for value in (2.5e4, 5.0e4, 1.0e5):
        cases.append(("contact_stiffness_N_m", value, beam0, replace(panto0, contact_stiffness_N_m=value), control0))
    for value in (1.10, 1.25, 1.50):
        cases.append(("energy_sigma", value, beam0, panto0, replace(control0, modal_energy_sigma=value)))
    for value in (1.5, 2.0, 2.5):
        cases.append(("burst_distance_m", value, beam0, panto0, replace(control0, adaptive_burst_distance_m=value)))

    output = []
    for index, (parameter, value, beam, panto, control) in enumerate(cases, 1):
        passive = run_metrics("Passive", 350.0, beam, panto, control)
        me = run_metrics("ME-APIC", 350.0, beam, panto, control)
        improvement = 100.0*(passive["std_force_N"]-me["std_force_N"])/passive["std_force_N"]
        output.append({
            "parameter": parameter,
            "value": value,
            "passive_std_force_N": passive["std_force_N"],
            "me_apic_std_force_N": me["std_force_N"],
            "paired_improvement_percent": improvement,
            "ATR_percent": me["ATR_percent"],
            "effort_N2s": me["effort_N2s"],
            "mechanical_work_J": me["mechanical_work_J"],
        })
        print(f"sensitivity {index}/{len(cases)}: {parameter}={value}", flush=True)
    write_csv("numerical_sensitivity.csv", output)
    return output


def pareto_study() -> tuple[list[dict], list[dict]]:
    beam, panto, base = Parameters(), PantographParameters(), ControllerParameters()
    deadlines = (0.055, 0.070, 0.090, 0.115, 0.140, 0.180, 0.230, 0.300)
    output = []
    for speed in SPEEDS:
        passive = run_metrics("Passive", speed, beam, panto, base)
        for method in ("E-APIC", "ME-APIC"):
            sigmas = (1.5, 3.0, 6.0) if method == "E-APIC" else (1.10, 1.25, 1.50)
            for sigma in sigmas:
                for deadline in deadlines:
                    if method == "E-APIC":
                        control = replace(base, event_deadline_s=deadline, event_sigma=sigma)
                    else:
                        control = replace(base, modal_energy_deadline_s=deadline, modal_energy_sigma=sigma)
                    row = run_metrics(method, speed, beam, panto, control)
                    row.update({
                        "trigger_sigma": sigma,
                        "deadline_s": deadline,
                        "paired_improvement_percent": 100.0*(passive["std_force_N"]-row["std_force_N"])/passive["std_force_N"],
                    })
                    output.append(row)
                    print(f"pareto {speed:.0f} {method} sigma={sigma:.2f} deadline={deadline:.3f}", flush=True)
    write_csv("equal_budget_sweep.csv", output)

    matched = []
    for speed in SPEEDS:
        e_rows = [r for r in output if r["controller"] == "E-APIC" and r["speed_kmh"] == speed]
        m_rows = [r for r in output if r["controller"] == "ME-APIC" and r["speed_kmh"] == speed]
        budget_sets = {
            "ATR": ("ATR_percent", (15, 20, 25, 30, 35, 40)),
            "effort": ("effort_N2s", (30, 45, 60, 80, 100)),
            "mechanical_work": ("mechanical_work_J", (0.4, 0.6, 0.8, 1.1, 1.4)),
        }
        for budget_kind, (field, targets) in budget_sets.items():
            for target in targets:
                e = min(e_rows, key=lambda x: abs(float(x[field])-target))
                m = min(m_rows, key=lambda x: abs(float(x[field])-target))
                gap = 100.0*abs(float(e[field])-float(m[field]))/max(target, 1e-12)
                if gap > 10.0:
                    continue
                matched.append({
                    "speed_kmh": speed,
                    "budget_kind": budget_kind,
                    "target_budget": target,
                    "e_budget": e[field],
                    "me_budget": m[field],
                    "relative_budget_gap_percent": gap,
                    "e_improvement_percent": e["paired_improvement_percent"],
                    "me_improvement_percent": m["paired_improvement_percent"],
                    "me_minus_e_improvement_points": float(m["paired_improvement_percent"])-float(e["paired_improvement_percent"]),
                    "e_trigger_sigma": e["trigger_sigma"],
                    "me_trigger_sigma": m["trigger_sigma"],
                    "e_deadline_s": e["deadline_s"],
                    "me_deadline_s": m["deadline_s"],
                })
    write_csv("matched_budget_comparison.csv", matched)
    return output, matched


def spectral_study() -> list[dict]:
    from scipy.signal import find_peaks, welch

    beam, panto, control = Parameters(), PantographParameters(), ControllerParameters()
    records = []
    spectra = {}
    for speed in SPEEDS:
        run = simulate("Passive", speed, beam, panto, control, feedback_kind="modal_lqr")
        mask = (run.time >= 0.05*run.time[-1]) & (run.time <= 0.95*run.time[-1])
        fs = 1.0/(run.time[1]-run.time[0])
        frequency, density = welch(run.force[mask]-np.mean(run.force[mask]), fs=fs, nperseg=min(512, np.sum(mask)))
        spectra[speed] = (frequency, density)
        peaks, _ = find_peaks(density)
        order = peaks[np.argsort(density[peaks])[-5:]][::-1]
        for rank, idx in enumerate(order, 1):
            records.append({
                "speed_kmh": speed,
                "rank": rank,
                "frequency_Hz": frequency[idx],
                "PSD_N2_Hz": density[idx],
            })
    write_csv("passive_spectral_peaks.csv", records)
    return records, spectra


def figures(sensitivity, pareto, spectra) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8,
        "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    parameters = ["elements", "modes", "dt_multiplier", "contact_stiffness_N_m", "energy_sigma", "burst_distance_m"]
    titles = [r"$N_e$", r"$N_m$", r"$\Delta t/\Delta t_0$", r"$k_c$", r"$\sigma_V$", r"$d_a$"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    for ax, parameter, title in zip(axes.ravel(), parameters, titles):
        rr = [r for r in sensitivity if r["parameter"] == parameter]
        x = np.arange(len(rr))
        ax.plot(x, [r["paired_improvement_percent"] for r in rr], marker="o", color="#2E7D32")
        ax.axhline(0, color="0.2", lw=0.7)
        ax.set_xticks(x, [f"{float(r['value']):g}" for r in rr])
        ax.set_title(title)
        ax.set_ylabel("ME-APIC improvement (%)")
    fig.tight_layout()
    fig.savefig(OUT/"numerical_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT/"numerical_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    fields = [("ATR_percent", "ATR (%)"), ("effort_N2s", r"$J_F$ (N$^2$s)"), ("mechanical_work_J", r"$J_P$ (J)")]
    colors = {"E-APIC": "#D95F02", "ME-APIC": "#1B9E77"}
    for ax, (field, label) in zip(axes, fields):
        for speed, marker in zip(SPEEDS, ("o", "s")):
            for method in ("E-APIC", "ME-APIC"):
                rr = sorted([r for r in pareto if r["speed_kmh"] == speed and r["controller"] == method], key=lambda r: r[field])
                ax.plot([r[field] for r in rr], [r["paired_improvement_percent"] for r in rr], marker=marker, color=colors[method], label=f"{method}, {speed:.0f}")
        ax.axhline(0, color="0.2", lw=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel(r"Reduction in $\sigma_F$ (%)")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/"equal_budget_pareto.pdf", bbox_inches="tight")
    fig.savefig(OUT/"equal_budget_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for speed, color in zip(SPEEDS, ("#377EB8", "#E41A1C")):
        f, p = spectra[speed]
        ax.semilogy(f, p, color=color, label=f"{speed:.0f} km/h")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (N$^2$/Hz)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/"high_speed_spectrum.pdf", bbox_inches="tight")
    fig.savefig(OUT/"high_speed_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sensitivity = sensitivity_study()
    pareto, matched = pareto_study()
    peaks, spectra = spectral_study()
    figures(sensitivity, pareto, spectra)
    (OUT/"revision_metadata.json").write_text(json.dumps({
        "sensitivity_speed_kmh": 350,
        "pareto_speeds_kmh": SPEEDS,
        "sensitivity_cases": len(sensitivity),
        "pareto_cases": len(pareto),
        "matched_pairs": len(matched),
        "spectral_peaks": len(peaks),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

