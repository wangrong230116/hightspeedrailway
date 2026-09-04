"""Compare passive, continuous, T-APIC and E-APIC on the Cardelli baseline.

The active actuator is applied to the moving-mass DOF.  A bounded contact-force
feedback law is shared by all active strategies; only its activation schedule
changes.  This is an algorithm-integration benchmark, not yet a production
three-DOF pantograph model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from run_baseline import Parameters, assemble_beam, contact_vectors


@dataclass(frozen=True)
class ControlParameters:
    force_reference_N: float = 50.0
    proportional_gain: float = 1.50
    actuator_limit_N: float = 35.0
    # Time-triggered APIC: fixed width, aperiodic start-to-start intervals.
    activation_width_s: float = 0.020
    tapic_interval_min_s: float = 0.055
    tapic_interval_jitter_s: float = 0.025
    tapic_seed: int = 2026
    # Event-triggered APIC: paper-style growth threshold plus a check deadline.
    event_sigma: float = 5.0
    event_floor_N2: float = 20.0**2
    event_check_period_s: float = 0.160
    evaluation_end_s: float = 0.95


@dataclass
class ExperimentResult:
    controller: str
    time: np.ndarray
    contact_force: np.ndarray
    actuator_force: np.ndarray
    active: np.ndarray
    starts: np.ndarray


def feedback_force(measured_force: float, c: ControlParameters) -> float:
    error = measured_force - c.force_reference_N
    return float(np.clip(-c.proportional_gain*error, -c.actuator_limit_N, c.actuator_limit_N))


def tapic_schedule(end_time: float, c: ControlParameters) -> np.ndarray:
    rng = np.random.default_rng(c.tapic_seed)
    starts = [0.0]
    while starts[-1] < end_time:
        starts.append(
            starts[-1] + c.tapic_interval_min_s
            + c.tapic_interval_jitter_s*rng.random()
        )
    return np.asarray(starts)


def simulate_controller(name: str, p: Parameters, c: ControlParameters) -> ExperimentResult:
    k_beam, m_beam, free = assemble_beam(p)
    n_beam = len(free)
    n_total = n_beam + 1
    k_total = np.zeros((n_total, n_total))
    m_total = np.zeros_like(k_total)
    k_total[:n_beam, :n_beam] = k_beam
    m_total[:n_beam, :n_beam] = m_beam
    m_total[-1, -1] = p.moving_mass
    base_force = np.zeros(n_total)
    base_force[-1] = p.preload

    b0, _ = contact_vectors(0.0, p, free)
    q = np.linalg.solve(k_total + p.contact_stiffness*np.outer(b0, b0), base_force)
    qd = np.zeros(n_total)
    qdd = np.zeros(n_total)

    end_time = p.span_length/p.speed
    steps = int(np.floor(end_time/p.dt))
    times = np.arange(steps + 1)*p.dt
    force_hist = np.empty(steps + 1)
    actuator_hist = np.zeros(steps + 1)
    active_hist = np.zeros(steps + 1, dtype=bool)
    force_hist[0] = p.contact_stiffness*(b0 @ q)

    gamma, beta, dt = p.newmark_gamma, p.newmark_beta, p.dt
    scheduled_starts = tapic_schedule(end_time, c) if name == "T-APIC" else np.empty(0)
    starts: list[float] = [0.0] if name == "Continuous" else []
    active_until = -np.inf
    reference_energy = (force_hist[0]-c.force_reference_N)**2
    last_activation_start = 0.0

    if name in {"T-APIC", "E-APIC"}:
        starts.append(0.0)
        active_until = c.activation_width_s

    for step in range(steps):
        t = times[step]
        measured_force = force_hist[step]

        if name == "Passive":
            is_active = False
        elif name == "Continuous":
            is_active = True
        elif name == "T-APIC":
            idx = max(0, int(np.searchsorted(scheduled_starts, t, side="right")-1))
            is_active = t < scheduled_starts[idx] + c.activation_width_s
            if idx + 1 > len(starts) and scheduled_starts[idx] < end_time:
                starts.append(float(scheduled_starts[idx]))
        elif name == "E-APIC":
            if t < active_until:
                is_active = True
            else:
                if active_hist[step-1] if step else True:
                    reference_energy = (measured_force-c.force_reference_N)**2
                energy = (measured_force-c.force_reference_N)**2
                threshold = c.event_sigma*reference_energy + c.event_floor_N2
                deadline = t-last_activation_start >= c.event_check_period_s
                if energy >= threshold or deadline:
                    starts.append(float(t))
                    last_activation_start = float(t)
                    active_until = t+c.activation_width_s
                    is_active = True
                else:
                    is_active = False
        else:
            raise ValueError(f"Unknown controller: {name}")

        u = feedback_force(measured_force, c) if is_active else 0.0
        active_hist[step] = is_active
        actuator_hist[step] = u
        external_force = base_force.copy()
        external_force[-1] += u

        x = p.speed*times[step + 1]
        b, convective = contact_vectors(x, p, free)
        k_contact = (
            p.contact_stiffness*np.outer(b, b)
            + p.contact_damping*np.outer(b, convective)
        )
        c_contact = p.contact_damping*np.outer(b, b)
        k_eff = k_total + k_contact
        c_eff = c_contact
        k_dyn = k_eff + (gamma/(beta*dt))*c_eff + (1/(beta*dt**2))*m_total
        rhs = external_force.copy()
        rhs += m_total @ (
            (1/(beta*dt**2))*q + (1/(beta*dt))*qd + (1/(2*beta)-1)*qdd
        )
        rhs += c_eff @ (
            (gamma/(beta*dt))*q + (gamma/beta-1)*qd
            + dt*(gamma/(2*beta)-1)*qdd
        )
        q_new = np.linalg.solve(k_dyn, rhs)
        qdd_new = (q_new-q-dt*qd)/(beta*dt**2) - (1/(2*beta)-1)*qdd
        qd_new = qd + dt*((1-gamma)*qdd + gamma*qdd_new)
        q, qd, qdd = q_new, qd_new, qdd_new
        force_hist[step + 1] = (
            p.contact_stiffness*(b @ q)
            + p.contact_damping*(b @ qd)
            + p.contact_damping*(convective @ q)
        )

    active_hist[-1] = active_hist[-2]
    actuator_hist[-1] = actuator_hist[-2]
    return ExperimentResult(
        name, times, force_hist, actuator_hist, active_hist, np.asarray(starts)
    )


def metrics(result: ExperimentResult, c: ControlParameters) -> dict[str, float | int | str]:
    evaluation = result.time <= c.evaluation_end_s
    force = result.contact_force[evaluation]
    time = result.time[evaluation]
    actuator = result.actuator_force[evaluation]
    active = result.active[evaluation]
    dt = time[1]-time[0]
    active_seconds = float(np.sum(active[:-1])*dt)
    error = force-c.force_reference_N
    return {
        "controller": result.controller,
        "mean_force_N": float(np.mean(force)),
        "std_force_N": float(np.std(force)),
        "cv_percent": float(100*np.std(force)/np.mean(force)),
        "rmse_to_reference_N": float(np.sqrt(np.mean(error**2))),
        "min_force_N": float(np.min(force)),
        "max_force_N": float(np.max(force)),
        "contact_loss_percent": float(100*np.mean(force <= 0.0)),
        "NoC": int(np.sum(result.starts <= c.evaluation_end_s)),
        "ATR_percent": float(100*active_seconds/time[-1]),
        "actuator_rms_N": float(np.sqrt(np.mean(actuator**2))),
        "control_energy_proxy_N2s": float(np.sum(actuator[:-1]**2)*dt),
    }


def save(results: list[ExperimentResult], p: Parameters, c: ControlParameters, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    rows = [metrics(result, c) for result in results]
    with (output/"control_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evaluation = results[0].time <= c.evaluation_end_s
    time_series_fields = ["time_s"]
    for result in results:
        time_series_fields.extend([
            f"{result.controller}_contact_force_N",
            f"{result.controller}_actuator_force_N",
            f"{result.controller}_active",
        ])
    with (output/"control_timeseries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=time_series_fields)
        writer.writeheader()
        for index in np.flatnonzero(evaluation):
            row: dict[str, float | int] = {"time_s": float(results[0].time[index])}
            for result in results:
                row[f"{result.controller}_contact_force_N"] = float(result.contact_force[index])
                row[f"{result.controller}_actuator_force_N"] = float(result.actuator_force[index])
                row[f"{result.controller}_active"] = int(result.active[index])
            writer.writerow(row)
    (output/"control_metadata.json").write_text(
        json.dumps({"model": asdict(p), "control": asdict(c), "metrics": rows}, indent=2),
        encoding="utf-8",
    )

    colors = {
        "Passive": "#7f7f7f", "Continuous": "#0072BD",
        "T-APIC": "#D95319", "E-APIC": "#EDB120",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for result in results:
        evaluation = result.time <= c.evaluation_end_s
        axes[0].plot(
            result.time[evaluation], result.contact_force[evaluation], lw=1.0,
            label=result.controller, color=colors[result.controller],
        )
        if result.controller != "Passive":
            axes[1].plot(
                result.time[evaluation], result.actuator_force[evaluation], lw=0.9,
                label=result.controller, color=colors[result.controller],
            )
    axes[0].axhline(c.force_reference_N, color="0.25", ls="--", lw=1.0, label="50 N reference")
    axes[0].set_ylabel("Contact force [N]")
    axes[1].set_ylabel("Active force [N]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        ax.set_xlim(0, c.evaluation_end_s)
    fig.suptitle("Cardelli baseline with intermittent active contact-force control")
    fig.tight_layout()
    fig.savefig(output/"contact_force_control_comparison.png", dpi=220)
    plt.close(fig)

    names = [row["controller"] for row in rows]
    std = [row["std_force_N"] for row in rows]
    atr = [row["ATR_percent"] for row in rows]
    energy = [row["control_energy_proxy_N2s"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, values, title, ylabel in zip(
        axes,
        (std, atr, energy),
        ("Contact-force fluctuation", "Activation time rate", "Control effort"),
        ("Standard deviation [N]", "ATR [%]", r"$\int u^2dt$ [N$^2$s]"),
    ):
        ax.bar(names, values, color=["#7f7f7f", "#0072BD", "#D95319", "#EDB120"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output/"performance_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("control_results"))
    args = parser.parse_args()
    p, c = Parameters(), ControlParameters()
    controllers = ["Passive", "Continuous", "T-APIC", "E-APIC"]
    results = [simulate_controller(name, p, c) for name in controllers]
    save(results, p, c, args.output)
    for result in results:
        row = metrics(result, c)
        print(
            f"{result.controller:10s} std={row['std_force_N']:7.3f} N  "
            f"NoC={row['NoC']:3d}  ATR={row['ATR_percent']:6.2f}%  "
            f"energy={row['control_energy_proxy_N2s']:8.2f} N^2s"
        )


if __name__ == "__main__":
    main()

