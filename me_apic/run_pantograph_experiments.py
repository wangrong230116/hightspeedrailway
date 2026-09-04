"""Hybrid Cardelli catenary + DSA380 three-DOF pantograph experiments.

The contact wire is Cardelli's Euler--Bernoulli FE span.  The pantograph uses
the DSA380 lumped parameters reported in the FENet appendix.  Contact is a
unilateral penalty law, so tensile contact force is clipped by an active-set
step rather than being interpreted as physical adhesion.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from run_baseline import Parameters, assemble_beam, hermite


@dataclass(frozen=True)
class PantographParameters:
    masses_kg: tuple[float, float, float] = (7.12, 6.0, 5.8)
    damping_Ns_m: tuple[float, float, float] = (0.0, 0.0, 70.0)
    stiffness_N_m: tuple[float, float, float] = (9430.0, 14100.0, 0.1)
    contact_stiffness_N_m: float = 5.0e4


@dataclass(frozen=True)
class ControllerParameters:
    gain: float = 1.15
    limit_N: float = 80.0
    burst_width_s: float = 0.030
    tapic_interval_min_s: float = 0.080
    tapic_interval_jitter_s: float = 0.040
    tapic_seed: int = 2026
    event_sigma: float = 3.0
    event_floor_N2: float = 15.0**2
    event_deadline_s: float = 0.180
    # Modal-energy E-APIC: V=x'Px growth and a speed-scaled spatial burst.
    modal_energy_sigma: float = 1.25
    modal_energy_floor_force_factor: float = 3.0
    modal_energy_deadline_s: float = 0.140
    adaptive_burst_distance_m: float = 2.0
    adaptive_burst_min_s: float = 0.015
    adaptive_burst_max_s: float = 0.040
    modal_modes: int = 16
    modal_force_weight: float = 0.20


@dataclass(frozen=True)
class RobustnessParameters:
    measurement_noise_std_N: float = 0.0
    actuator_delay_s: float = 0.0
    actuator_time_constant_s: float = 0.0
    irregularity_scale: float = 1.0
    irregularity_phase_offsets_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Run:
    controller: str
    speed_kmh: float
    target_force_N: float
    time: np.ndarray
    force: np.ndarray
    actuator: np.ndarray
    actuator_velocity: np.ndarray
    active: np.ndarray
    starts: np.ndarray
    feedback_kind: str = "proportional"
    lyapunov: np.ndarray | None = None


def target_force(speed_kmh: float) -> float:
    """DSA380 target force from FENet Table A.1; speed is in km/h."""
    return 0.00097*speed_kmh**2 + 70.0


def chain_matrix(values: tuple[float, float, float]) -> np.ndarray:
    """Three masses: elements 1/2 connect neighbours; element 3 to ground."""
    v1, v2, v3 = values
    return np.array(
        [[v1, -v1, 0.0], [-v1, v1+v2, -v2], [0.0, -v2, v2+v3]],
        dtype=float,
    )


def moving_contact_vector(
    position: float, beam: Parameters, free: np.ndarray
) -> np.ndarray:
    length = beam.span_length/beam.elements
    element = min(int(np.floor(position/length)), beam.elements-1)
    local_x = position-element*length
    n_local, _ = hermite(local_x, length)
    full = np.zeros(2*(beam.elements+1))
    dofs = np.array([2*element, 2*element+1, 2*element+2, 2*element+3])
    full[dofs] = n_local
    # penetration = collector-head displacement - wire displacement
    return np.concatenate((-full[free], [1.0, 0.0, 0.0]))


def wire_irregularity(
    position: float,
    scale: float = 1.0,
    phase_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> float:
    """Deterministic 0.26-mm multi-wavelength height irregularity."""
    phase_1, phase_2, phase_3 = phase_offsets
    return scale*(
        0.15e-3*np.sin(2*np.pi*position/12.0 + phase_1)
        + 0.08e-3*np.sin(2*np.pi*position/5.0 + 0.7 + phase_2)
        + 0.03e-3*np.sin(2*np.pi*position/1.5 + 1.3 + phase_3)
    )


def feedback(force: float, reference: float, control: ControllerParameters) -> float:
    return float(np.clip(-control.gain*(force-reference), -control.limit_N, control.limit_N))


def design_lqi_gain(pantograph: PantographParameters) -> np.ndarray:
    """LQI gain for the grounded-wire linearization and integrated force error."""
    from scipy.linalg import solve_continuous_are

    panto_mass = np.diag(pantograph.masses_kg)
    panto_stiffness = chain_matrix(pantograph.stiffness_N_m)
    panto_stiffness[0, 0] += pantograph.contact_stiffness_N_m
    panto_damping = chain_matrix(pantograph.damping_Ns_m)
    state_a = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [-np.linalg.solve(panto_mass, panto_stiffness),
         -np.linalg.solve(panto_mass, panto_damping)],
    ])
    state_b = np.concatenate(
        (np.zeros(3), np.linalg.solve(panto_mass, np.array([0.0, 0.0, 1.0])))
    )[:, None]
    force_output = np.array(
        [[pantograph.contact_stiffness_N_m, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    augmented_a = np.block([
        [state_a, np.zeros((6, 1))],
        [force_output, np.zeros((1, 1))],
    ])
    augmented_b = np.vstack((state_b, [[0.0]]))
    # State scaling prioritizes collector displacement and accumulated force error.
    state_cost = np.diag((1e7, 1e5, 1e4, 10.0, 5.0, 2.0, 1e-2))
    riccati = solve_continuous_are(augmented_a, augmented_b, state_cost, [[1.0]])
    return np.asarray(augmented_b.T@riccati).ravel()


def design_modal_lqr(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    contact_vector: np.ndarray,
    contact_stiffness: float,
    modes: int = 16,
    force_weight: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced full-system LQR around a closed-contact mid-span equilibrium."""
    from scipy.linalg import eigh, solve_continuous_are

    closed_stiffness = stiffness + contact_stiffness*np.outer(
        contact_vector, contact_vector
    )
    _, shapes = eigh(
        closed_stiffness, mass, subset_by_index=(0, min(modes, mass.shape[0])-1)
    )
    modal_stiffness = shapes.T@closed_stiffness@shapes
    modal_damping = shapes.T@damping@shapes
    count = shapes.shape[1]
    state_a = np.block([
        [np.zeros((count, count)), np.eye(count)],
        [-modal_stiffness, -modal_damping],
    ])
    physical_input = np.zeros(mass.shape[0])
    physical_input[-1] = 1.0
    modal_input = shapes.T@physical_input
    state_b = np.concatenate((np.zeros(count), modal_input))[:, None]
    force_output = contact_stiffness*(contact_vector@shapes)
    output_state = np.concatenate((force_output, np.zeros(count)))
    state_cost = force_weight*np.outer(output_state, output_state)
    state_cost += np.diag(np.concatenate((np.full(count, 1e-5), np.full(count, 1e-4))))
    riccati = solve_continuous_are(state_a, state_b, state_cost, [[1.0]])
    return shapes, np.asarray(state_b.T@riccati).ravel(), riccati


def tapic_starts(horizon: float, control: ControllerParameters) -> np.ndarray:
    rng = np.random.default_rng(control.tapic_seed)
    starts = [0.0]
    while starts[-1] < horizon:
        starts.append(
            starts[-1]+control.tapic_interval_min_s
            + control.tapic_interval_jitter_s*rng.random()
        )
    return np.asarray(starts)


def adaptive_burst_width(speed_kmh: float, control: ControllerParameters) -> float:
    speed_ms = speed_kmh/3.6
    return float(np.clip(
        control.adaptive_burst_distance_m/speed_ms,
        control.adaptive_burst_min_s,
        control.adaptive_burst_max_s,
    ))


def newmark_solve(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    load: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    dt: float,
    gamma: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dynamic = stiffness + gamma/(beta*dt)*damping + mass/(beta*dt**2)
    rhs = load.copy()
    rhs += mass @ ((q/(beta*dt**2)) + qd/(beta*dt) + (1/(2*beta)-1)*qdd)
    rhs += damping @ (
        gamma*q/(beta*dt) + (gamma/beta-1)*qd
        + dt*(gamma/(2*beta)-1)*qdd
    )
    q_new = np.linalg.solve(dynamic, rhs)
    qdd_new = (q_new-q-dt*qd)/(beta*dt**2) - (1/(2*beta)-1)*qdd
    qd_new = qd + dt*((1-gamma)*qdd + gamma*qdd_new)
    return q_new, qd_new, qdd_new


def simulate(
    controller: str,
    speed_kmh: float,
    beam_template: Parameters,
    pantograph: PantographParameters,
    control: ControllerParameters,
    feedback_kind: str = "proportional",
    robustness: RobustnessParameters | None = None,
    random_seed: int = 0,
    controller_model_pantograph: PantographParameters | None = None,
) -> Run:
    robustness = robustness or RobustnessParameters()
    rng = np.random.default_rng(random_seed)
    irregularity = lambda position: wire_irregularity(
        position,
        robustness.irregularity_scale,
        robustness.irregularity_phase_offsets_rad,
    )
    speed_ms = speed_kmh/3.6
    beam = replace(beam_template, speed=speed_ms)
    k_beam, m_beam, free = assemble_beam(beam)
    n_beam = len(free)
    size = n_beam+3
    stiffness = np.zeros((size, size))
    damping = np.zeros_like(stiffness)
    mass = np.zeros_like(stiffness)
    stiffness[:n_beam, :n_beam] = k_beam
    mass[:n_beam, :n_beam] = m_beam
    stiffness[n_beam:, n_beam:] = chain_matrix(pantograph.stiffness_N_m)
    damping[n_beam:, n_beam:] = chain_matrix(pantograph.damping_Ns_m)
    mass[n_beam:, n_beam:] = np.diag(pantograph.masses_kg)

    reference = target_force(speed_kmh)
    base_load = np.zeros(size)
    base_load[-1] = reference
    b0 = moving_contact_vector(0.0, beam, free)
    r0 = irregularity(0.0)
    q = np.linalg.solve(
        stiffness+pantograph.contact_stiffness_N_m*np.outer(b0, b0),
        base_load+pantograph.contact_stiffness_N_m*b0*r0,
    )
    qd = np.zeros(size)
    qdd = np.zeros(size)
    # LQI states are expressed relative to the moving wire surface, preventing
    # rigid wire-following motion from being mistaken for pantograph error.
    panto_reference = q[n_beam:].copy()-r0
    previous_wire_surface = r0
    previous_equilibrium = q.copy()
    integral_force_error = 0.0
    lqi_gain = design_lqi_gain(pantograph) if feedback_kind == "lqi" else None
    if feedback_kind == "modal_lqr":
        controller_panto = controller_model_pantograph or pantograph
        controller_stiffness = np.zeros_like(stiffness)
        controller_damping = np.zeros_like(damping)
        controller_mass = np.zeros_like(mass)
        controller_stiffness[:n_beam, :n_beam] = k_beam
        controller_damping[:n_beam, :n_beam] = damping[:n_beam, :n_beam]
        controller_mass[:n_beam, :n_beam] = m_beam
        controller_stiffness[n_beam:, n_beam:] = chain_matrix(
            controller_panto.stiffness_N_m
        )
        controller_damping[n_beam:, n_beam:] = chain_matrix(
            controller_panto.damping_Ns_m
        )
        controller_mass[n_beam:, n_beam:] = np.diag(controller_panto.masses_kg)
        midpoint_b = moving_contact_vector(0.5*beam.span_length, beam, free)
        modal_shapes, modal_gain, modal_riccati = design_modal_lqr(
            controller_mass, controller_damping, controller_stiffness, midpoint_b,
            controller_panto.contact_stiffness_N_m,
            modes=control.modal_modes,
            force_weight=control.modal_force_weight,
        )
    else:
        modal_shapes, modal_gain, modal_riccati = None, None, None
    horizon = beam.span_length/speed_ms
    steps = int(np.floor(horizon/beam.dt))
    time = np.arange(steps+1)*beam.dt
    lyapunov_hist = np.full(steps+1, np.nan)
    force_hist = np.zeros(steps+1)
    force_hist[0] = max(
        0.0, pantograph.contact_stiffness_N_m*(b0@q-r0)
    )
    actuator = np.zeros(steps+1)
    actuator_velocity = np.zeros(steps+1)
    applied_actuator = 0.0
    delay_steps = max(0, int(round(robustness.actuator_delay_s/beam.dt)))
    delay_buffer = [0.0]*delay_steps
    active = np.zeros(steps+1, dtype=bool)
    scheduled = tapic_starts(horizon, control) if controller == "T-APIC" else np.empty(0)
    starts: list[float] = [0.0] if controller != "Passive" else []
    if controller == "E-APIC":
        active_until = control.burst_width_s
    elif controller == "ME-APIC":
        active_until = adaptive_burst_width(speed_kmh, control)
    else:
        active_until = -np.inf
    last_start = 0.0
    reference_energy = (force_hist[0]-reference)**2
    reference_modal_energy = 0.0

    for step, t in enumerate(time[:-1]):
        measured = force_hist[step] + rng.normal(
            0.0, robustness.measurement_noise_std_N
        )
        current_b = moving_contact_vector(speed_ms*t, beam, free)
        current_irregularity = irregularity(speed_ms*t)
        wire_surface = -current_b[:n_beam]@q[:n_beam]+current_irregularity
        wire_velocity = (wire_surface-previous_wire_surface)/beam.dt if step else 0.0
        if feedback_kind == "modal_lqr":
            controller_kc = controller_panto.contact_stiffness_N_m
            equilibrium = np.linalg.solve(
                controller_stiffness+controller_kc*np.outer(current_b, current_b),
                base_load+controller_kc*current_b*current_irregularity,
            )
            equilibrium_velocity = (
                (equilibrium-previous_equilibrium)/beam.dt if step else np.zeros(size)
            )
            modal_displacement = modal_shapes.T@controller_mass@(q-equilibrium)
            modal_velocity = modal_shapes.T@controller_mass@(qd-equilibrium_velocity)
            modal_state = np.concatenate((modal_displacement, modal_velocity))
            lyapunov_hist[step] = float(modal_state@modal_riccati@modal_state)
        if controller == "Passive":
            is_active = False
        elif controller == "Continuous":
            is_active = True
        elif controller == "T-APIC":
            index = max(0, int(np.searchsorted(scheduled, t, side="right")-1))
            is_active = t < scheduled[index]+control.burst_width_s
            if index+1 > len(starts):
                starts.append(float(scheduled[index]))
        elif controller == "E-APIC":
            if t < active_until:
                is_active = True
            else:
                if active[step-1] if step else True:
                    reference_energy = (measured-reference)**2
                energy = (measured-reference)**2
                threshold = control.event_sigma*reference_energy+control.event_floor_N2
                deadline = t-last_start >= control.event_deadline_s
                if energy >= threshold or deadline:
                    starts.append(float(t))
                    last_start = float(t)
                    active_until = t+control.burst_width_s
                    is_active = True
                else:
                    is_active = False
        elif controller == "ME-APIC":
            if feedback_kind != "modal_lqr":
                raise ValueError("ME-APIC requires modal_lqr feedback")
            if t < active_until:
                is_active = True
            else:
                if active[step-1] if step else True:
                    reference_modal_energy = lyapunov_hist[step]
                threshold = (
                    control.modal_energy_sigma*reference_modal_energy
                    + control.modal_energy_floor_force_factor*reference**2
                )
                deadline = t-last_start >= control.modal_energy_deadline_s
                if lyapunov_hist[step] >= threshold or deadline:
                    starts.append(float(t))
                    last_start = float(t)
                    active_until = t+adaptive_burst_width(speed_kmh, control)
                    is_active = True
                else:
                    is_active = False
        else:
            raise ValueError(f"Unknown controller {controller}")

        if is_active and feedback_kind == "lqi":
            augmented_state = np.concatenate((
                q[n_beam:]-wire_surface-panto_reference,
                qd[n_beam:]-wire_velocity,
                [integral_force_error],
            ))
            command = float(np.clip(-lqi_gain@augmented_state, -control.limit_N, control.limit_N))
        elif is_active and feedback_kind == "modal_lqr":
            command = float(np.clip(
                -modal_gain@modal_state, -control.limit_N, control.limit_N
            ))
        elif is_active and feedback_kind == "proportional":
            command = feedback(measured, reference, control)
        elif is_active:
            raise ValueError(f"Unknown feedback kind {feedback_kind}")
        else:
            command = 0.0
        if delay_steps:
            delay_buffer.append(command)
            delayed_command = delay_buffer.pop(0)
        else:
            delayed_command = command
        if robustness.actuator_time_constant_s > 0.0:
            alpha = 1.0-np.exp(-beam.dt/robustness.actuator_time_constant_s)
            applied_actuator += alpha*(delayed_command-applied_actuator)
        else:
            applied_actuator = delayed_command
        u = float(np.clip(applied_actuator, -control.limit_N, control.limit_N))
        active[step] = is_active
        actuator[step] = u
        load = base_load.copy()
        # The actuator is mounted between the base and lower frame (mass 3).
        load[-1] += u

        position = speed_ms*time[step+1]
        b = moving_contact_vector(position, beam, free)
        irregularity_value = irregularity(position)
        kc = pantograph.contact_stiffness_N_m
        # Active-set unilateral contact: first try the closed-contact branch.
        q_contact, qd_contact, qdd_contact = newmark_solve(
            mass, damping, stiffness+kc*np.outer(b, b),
            load+kc*b*irregularity_value, q, qd, qdd, beam.dt,
            beam.newmark_gamma, beam.newmark_beta,
        )
        candidate_force = kc*(b@q_contact-irregularity_value)
        if candidate_force >= 0.0:
            q, qd, qdd = q_contact, qd_contact, qdd_contact
            force_hist[step+1] = candidate_force
        else:
            q, qd, qdd = newmark_solve(
                mass, damping, stiffness, load, q, qd, qdd, beam.dt,
                beam.newmark_gamma, beam.newmark_beta,
            )
            force_hist[step+1] = 0.0
        actuator_velocity[step+1] = qd[-1]
        integral_force_error += 0.5*beam.dt*(
            (measured-reference)+(force_hist[step+1]-reference)
        )
        previous_wire_surface = wire_surface
        if feedback_kind == "modal_lqr":
            previous_equilibrium = equilibrium

    active[-1] = active[-2]
    actuator[-1] = actuator[-2]
    lyapunov_hist[-1] = lyapunov_hist[-2]
    return Run(
        controller, speed_kmh, reference, time, force_hist, actuator,
        actuator_velocity, active,
        np.asarray(starts), feedback_kind, lyapunov_hist,
    )


def metrics(run: Run) -> dict[str, float | int | str]:
    # Exclude support-entry/exit transients and evaluate the central 90% of span.
    lo, hi = 0.05*run.time[-1], 0.95*run.time[-1]
    mask = (run.time >= lo) & (run.time <= hi)
    force = run.force[mask]
    active = run.active[mask]
    actuator = run.actuator[mask]
    actuator_velocity = run.actuator_velocity[mask]
    dt = run.time[1]-run.time[0]
    duration = float(np.sum(mask)*dt)
    starts = run.starts[run.starts <= run.time[-1]]
    return {
        "controller": run.controller,
        "speed_kmh": run.speed_kmh,
        "target_force_N": run.target_force_N,
        "mean_force_N": float(np.mean(force)),
        "std_force_N": float(np.std(force)),
        "cv_percent": float(100*np.std(force)/max(np.mean(force), 1e-12)),
        "min_force_N": float(np.min(force)),
        "max_force_N": float(np.max(force)),
        "contact_loss_percent": float(100*np.mean(force <= 0.0)),
        "NoC": int(starts.size),
        "ATR_percent": float(100*np.sum(active)*dt/duration),
        "effort_N2s": float(np.sum(actuator**2)*dt),
        "mechanical_work_J": float(np.sum(np.abs(actuator*actuator_velocity))*dt),
        "signed_mechanical_work_J": float(np.sum(actuator*actuator_velocity)*dt),
    }


def save(runs: list[Run], rows: list[dict[str, float | int | str]], output: Path,
         beam: Parameters, panto: PantographParameters, control: ControllerParameters) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with (output/"speed_sweep_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output/"speed_sweep_timeseries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "controller", "feedback_kind", "speed_kmh", "target_force_N", "time_s",
            "contact_force_N", "actuator_force_N", "actuator_velocity_m_s",
            "active", "lyapunov_V",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for index, time_value in enumerate(run.time):
                writer.writerow({
                    "controller": run.controller,
                    "feedback_kind": run.feedback_kind,
                    "speed_kmh": run.speed_kmh,
                    "target_force_N": run.target_force_N,
                    "time_s": float(time_value),
                    "contact_force_N": float(run.force[index]),
                    "actuator_force_N": float(run.actuator[index]),
                    "actuator_velocity_m_s": float(run.actuator_velocity[index]),
                    "active": int(run.active[index]),
                    "lyapunov_V": (
                        float(run.lyapunov[index]) if run.lyapunov is not None else ""
                    ),
                })
    (output/"speed_sweep_metadata.json").write_text(
        json.dumps(
            {"beam": asdict(beam), "pantograph": asdict(panto),
             "feedback_kinds": sorted({run.feedback_kind for run in runs}),
             "controller": asdict(control), "metrics": rows},
            indent=2,
        ), encoding="utf-8",
    )

    colors = {"Passive": "#7f7f7f", "Continuous": "#0072BD",
              "T-APIC": "#D95319", "E-APIC": "#EDB120",
              "ME-APIC": "#7E2F8E"}
    speeds = sorted({run.speed_kmh for run in runs})
    names = list(dict.fromkeys(run.controller for run in runs))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for name in names:
        selected = [row for row in rows if row["controller"] == name]
        axes[0].plot(speeds, [r["std_force_N"] for r in selected], "o-", label=name,
                     color=colors[name])
        axes[1].plot(speeds, [r["max_force_N"] for r in selected], "o-",
                     label=name, color=colors[name])
        axes[2].plot(speeds, [r["ATR_percent"] for r in selected], "o-", label=name,
                     color=colors[name])
    for ax, ylabel in zip(
        axes, ("Force standard deviation [N]", "Peak contact force [N]", "ATR [%]")
    ):
        ax.set_xlabel("Speed [km/h]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    feedback_label = runs[0].feedback_kind.replace("_", " ").upper()
    fig.suptitle(f"DSA380 unilateral-contact speed sweep ({feedback_label})")
    fig.tight_layout()
    fig.savefig(output/"speed_sweep_summary.png", dpi=220)
    plt.close(fig)

    chosen_speed = 300.0
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for run in runs:
        if run.speed_kmh == chosen_speed:
            central = (run.time >= 0.05*run.time[-1]) & (run.time <= 0.95*run.time[-1])
            ax.plot(run.time[central], run.force[central], lw=1.0, label=run.controller,
                    color=colors[run.controller])
    ax.axhline(target_force(chosen_speed), color="0.2", ls="--", lw=1,
               label="Target force")
    ax.set(xlabel="Time [s]", ylabel="Contact force [N]",
           title=f"Contact-force histories at 300 km/h ({feedback_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output/"contact_force_300kmh.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("pantograph_results"))
    parser.add_argument("--speeds", type=float, nargs="+", default=(200, 250, 300, 350))
    args = parser.parse_args()
    beam, panto, control = Parameters(), PantographParameters(), ControllerParameters()
    names = ("Passive", "Continuous", "T-APIC", "E-APIC")
    runs = [simulate(name, speed, beam, panto, control)
            for speed in args.speeds for name in names]
    # Controller-major ordering is convenient for plotting by speed.
    runs.sort(key=lambda run: (names.index(run.controller), run.speed_kmh))
    rows = [metrics(run) for run in runs]
    save(runs, rows, args.output, beam, panto, control)
    for row in rows:
        print(
            f"{row['controller']:10s} {row['speed_kmh']:3.0f} km/h  "
            f"std={row['std_force_N']:7.2f} N  loss={row['contact_loss_percent']:6.2f}%  "
            f"ATR={row['ATR_percent']:6.2f}%"
        )


if __name__ == "__main__":
    main()

