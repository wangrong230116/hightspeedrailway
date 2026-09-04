"""Run the Cardelli 2026 k=0, 60-element Collina--Bruni benchmark.

This is a NumPy port of ``Confront_collina_RMSE_RMSRE.m``.  It preserves the
published mesh, moving-mass, contact and Newmark parameters while replacing
MATLAB symbolic construction by the closed-form cubic Hermite matrices for
the k=0 element.
"""

from __future__ import annotations

import argparse
import csv
import json
import time as walltime
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parameters:
    span_length: float = 60.0
    elements: int = 60
    axial_tension: float = 20_000.0
    linear_density: float = 1.35
    bending_stiffness: float = 136.0
    moving_mass: float = 3.0
    preload: float = 50.0
    speed: float = 60.0
    contact_stiffness: float = 5.0e4
    contact_damping: float = 1.0e4
    newmark_gamma: float = 0.55
    newmark_beta: float = 0.275
    dt: float = 8.2082e-4


def element_matrices(length: float, p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Euler--Bernoulli mass/stiffness plus tensile geometric stiffness."""
    l = length
    elastic = (p.bending_stiffness / l**3) * np.array(
        [
            [12, 6*l, -12, 6*l],
            [6*l, 4*l*l, -6*l, 2*l*l],
            [-12, -6*l, 12, -6*l],
            [6*l, 2*l*l, -6*l, 4*l*l],
        ], dtype=float,
    )
    geometric = (p.axial_tension / (30*l)) * np.array(
        [
            [36, 3*l, -36, 3*l],
            [3*l, 4*l*l, -3*l, -l*l],
            [-36, -3*l, 36, -3*l],
            [3*l, -l*l, -3*l, 4*l*l],
        ], dtype=float,
    )
    mass = (p.linear_density*l / 420) * np.array(
        [
            [156, 22*l, 54, -13*l],
            [22*l, 4*l*l, 13*l, -3*l*l],
            [54, 13*l, 156, -22*l],
            [-13*l, -3*l*l, -22*l, 4*l*l],
        ], dtype=float,
    )
    return elastic + geometric, mass


def assemble_beam(p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = p.elements + 1
    full_dofs = 2 * nodes
    length = p.span_length / p.elements
    ke, me = element_matrices(length, p)
    stiffness = np.zeros((full_dofs, full_dofs))
    mass = np.zeros_like(stiffness)
    for element in range(p.elements):
        dofs = np.array([2*element, 2*element+1, 2*element+2, 2*element+3])
        stiffness[np.ix_(dofs, dofs)] += ke
        mass[np.ix_(dofs, dofs)] += me
    # Simply supported: constrain vertical displacement at both ends only.
    constrained = {0, 2*(nodes-1)}
    free = np.array([i for i in range(full_dofs) if i not in constrained])
    return stiffness[np.ix_(free, free)], mass[np.ix_(free, free)], free


def hermite(position: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Displacement shape vector and physical spatial derivative."""
    xi = position / length
    n = np.array(
        [
            1 - 3*xi**2 + 2*xi**3,
            length*(xi - 2*xi**2 + xi**3),
            3*xi**2 - 2*xi**3,
            length*(-xi**2 + xi**3),
        ]
    )
    dn_dxi = np.array(
        [
            -6*xi + 6*xi**2,
            length*(1 - 4*xi + 3*xi**2),
            6*xi - 6*xi**2,
            length*(-2*xi + 3*xi**2),
        ]
    )
    return n, dn_dxi / length


def contact_vectors(x: float, p: Parameters, free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = p.span_length / p.elements
    element = min(int(np.floor(x / length)), p.elements - 1)
    local_x = x - element*length
    n_local, nx_local = hermite(local_x, length)
    full_size = 2*(p.elements+1)
    n_full = np.zeros(full_size)
    nx_full = np.zeros(full_size)
    dofs = np.array([2*element, 2*element+1, 2*element+2, 2*element+3])
    n_full[dofs] = n_local
    nx_full[dofs] = nx_local
    n_red = n_full[free]
    nx_red = nx_full[free]
    b = np.concatenate((-n_red, [1.0]))
    c = p.speed * np.concatenate((-nx_red, [0.0]))
    return b, c


def simulate(p: Parameters) -> dict[str, np.ndarray | float]:
    start_clock = walltime.perf_counter()
    k_beam, m_beam, free = assemble_beam(p)
    n_beam = len(free)
    n_total = n_beam + 1
    k_total = np.zeros((n_total, n_total))
    m_total = np.zeros_like(k_total)
    k_total[:n_beam, :n_beam] = k_beam
    m_total[:n_beam, :n_beam] = m_beam
    m_total[-1, -1] = p.moving_mass
    c_total = np.zeros_like(k_total)
    force = np.zeros(n_total)
    force[-1] = p.preload

    b0, _ = contact_vectors(0.0, p, free)
    q = np.linalg.solve(k_total + p.contact_stiffness*np.outer(b0, b0), force)
    qd = np.zeros(n_total)
    qdd = np.zeros(n_total)

    end_time = p.span_length / p.speed
    steps = int(np.floor(end_time / p.dt))
    times = np.arange(steps + 1, dtype=float)*p.dt
    contact_force = np.empty(steps + 1)
    mass_displacement = np.empty(steps + 1)
    contact_force[0] = p.contact_stiffness*(b0 @ q)
    mass_displacement[0] = q[-1]
    gamma, beta, dt = p.newmark_gamma, p.newmark_beta, p.dt

    setup_seconds = walltime.perf_counter() - start_clock
    integration_clock = walltime.perf_counter()
    for step in range(steps):
        x = p.speed*times[step + 1]
        b, c = contact_vectors(x, p, free)
        k_contact = p.contact_stiffness*np.outer(b, b) + p.contact_damping*np.outer(b, c)
        c_contact = p.contact_damping*np.outer(b, b)
        k_eff = k_total + k_contact
        c_eff = c_total + c_contact

        k_dyn = k_eff + (gamma/(beta*dt))*c_eff + (1/(beta*dt**2))*m_total
        rhs = force.copy()
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

        contact_force[step + 1] = (
            p.contact_stiffness*(b @ q)
            + p.contact_damping*(b @ qd)
            + p.contact_damping*(c @ q)
        )
        mass_displacement[step + 1] = q[-1]

    integration_seconds = walltime.perf_counter() - integration_clock
    return {
        "time": times,
        "contact_force": contact_force,
        "mass_displacement": mass_displacement,
        "setup_seconds": setup_seconds,
        "integration_seconds": integration_seconds,
    }


def compare_reference(
    time: np.ndarray, force: np.ndarray, reference_csv: Path
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    from scipy.interpolate import PchipInterpolator

    reference = np.genfromtxt(reference_csv, delimiter=",", names=True)
    t_ref = np.asarray(reference["t_s"])
    f_ref = np.asarray(reference["F_N"])
    mask = (time >= t_ref.min()) & (time <= min(0.95, t_ref.max()))
    t_use = time[mask]
    f_use = force[mask]
    # Match MATLAB's interp1(..., 'pchip') used by the source project.
    f_ref_i = PchipInterpolator(t_ref, f_ref)(t_use)
    residual = f_use - f_ref_i
    nonzero = np.abs(f_ref_i) > 1e-12*max(1.0, float(np.max(np.abs(f_ref_i))))
    metrics = {
        "rmse_N": float(np.sqrt(np.mean(residual**2))),
        "rmsre": float(np.sqrt(np.mean((residual[nonzero]/f_ref_i[nonzero])**2))),
        "correlation": float(np.corrcoef(f_use, f_ref_i)[0, 1]),
        "mean_force_N": float(np.mean(f_use)),
        "std_force_N": float(np.std(f_use)),
        "min_force_N": float(np.min(f_use)),
        "max_force_N": float(np.max(f_use)),
    }
    return metrics, t_use, f_ref_i


def save_results(result: dict, metrics: dict, p: Parameters, output: Path, reference: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    time = np.asarray(result["time"])
    force = np.asarray(result["contact_force"])
    _, t_compare, f_reference = compare_reference(time, force, reference)
    f_compare = np.interp(t_compare, time, force)

    with (output/"baseline_contact_force.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "contact_force_N", "moving_mass_displacement_m"])
        writer.writerows(zip(time, force, np.asarray(result["mass_displacement"])))

    payload = {
        "parameters": asdict(p),
        "metrics_vs_digitized_reference_0_to_0.95s": metrics,
        "runtime": {
            "setup_seconds": result["setup_seconds"],
            "integration_seconds": result["integration_seconds"],
        },
        "reference_csv": str(reference),
    }
    (output/"metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(t_compare, f_compare, lw=1.2, label="Python port: N=60, k=0")
    ax.plot(t_compare, f_reference, "--", lw=1.2, label="Cardelli digitized reference")
    ax.set(xlabel="Time [s]", ylabel="Contact force [N]", xlim=(0, 0.95))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Cardelli baseline validation: RMSE={metrics['rmse_N']:.3f} N")
    fig.tight_layout()
    fig.savefig(output/"baseline_vs_reference.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Cardelli 2026" / "Code MATLAB Cardelli Alberto"
        / "Functions Cardelli Alberto" / "F_img_digitized_refined.csv",
    )
    args = parser.parse_args()
    p = Parameters()
    result = simulate(p)
    metrics, _, _ = compare_reference(
        np.asarray(result["time"]), np.asarray(result["contact_force"]), args.reference
    )
    save_results(result, metrics, p, args.output, args.reference)
    print(json.dumps({**metrics, "integration_seconds": result["integration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

