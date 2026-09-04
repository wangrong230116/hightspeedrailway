"""Compare force-error E-APIC with modal-energy E-APIC (ME-APIC)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_baseline import Parameters
from run_pantograph_experiments import (
    ControllerParameters,
    PantographParameters,
    metrics,
    save,
    simulate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("energy_eapic_results"))
    parser.add_argument("--speeds", type=float, nargs="+", default=(200, 250, 300, 350))
    args = parser.parse_args()
    beam, panto, control = Parameters(), PantographParameters(), ControllerParameters()
    names = ("Passive", "Continuous", "T-APIC", "E-APIC", "ME-APIC")
    runs = [
        simulate(name, speed, beam, panto, control, feedback_kind="modal_lqr")
        for speed in args.speeds for name in names
    ]
    runs.sort(key=lambda run: (names.index(run.controller), run.speed_kmh))
    rows = [metrics(run) for run in runs]
    save(runs, rows, args.output, beam, panto, control)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diagnostic = next(
        run for run in runs if run.controller == "ME-APIC" and run.speed_kmh == 350
    )
    central = (
        (diagnostic.time >= 0.05*diagnostic.time[-1])
        & (diagnostic.time <= 0.95*diagnostic.time[-1])
    )
    diagnostic_time = diagnostic.time[central]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.4), sharex=True)
    axes[0].plot(diagnostic_time, diagnostic.force[central], color="#7E2F8E", lw=1.0)
    axes[0].axhline(diagnostic.target_force_N, color="0.25", ls="--", lw=1.0)
    axes[0].set_ylabel("Contact force [N]")
    axes[1].semilogy(
        diagnostic_time, np.maximum(diagnostic.lyapunov[central], 1e-12),
        color="#0072BD", lw=1.0,
    )
    axes[1].set_ylabel(r"Modal energy $V$")
    axes[2].plot(diagnostic_time, diagnostic.actuator[central], color="#D95319", lw=1.0)
    axes[2].fill_between(
        diagnostic_time, -control.limit_N, control.limit_N,
        where=diagnostic.active[central], color="#EDB120", alpha=0.18,
        label="ME-APIC active",
    )
    axes[2].set_ylabel("Actuator [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper left")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("ME-APIC diagnostic at 350 km/h")
    fig.tight_layout()
    fig.savefig(args.output/"me_apic_diagnostic_350kmh.png", dpi=220)
    plt.close(fig)
    for row in rows:
        print(
            f"{row['controller']:10s} {row['speed_kmh']:3.0f} km/h  "
            f"std={row['std_force_N']:7.2f} N  NoC={row['NoC']:2d}  "
            f"ATR={row['ATR_percent']:6.2f}%  effort={row['effort_N2s']:8.2f}"
        )


if __name__ == "__main__":
    main()

