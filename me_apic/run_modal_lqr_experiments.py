"""Run PIC/T-APIC/E-APIC with one shared coupled-system modal LQR law."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--output", type=Path, default=Path("modal_lqr_results"))
    parser.add_argument("--speeds", type=float, nargs="+", default=(200, 250, 300, 350))
    args = parser.parse_args()
    beam, panto, control = Parameters(), PantographParameters(), ControllerParameters()
    names = ("Passive", "Continuous", "T-APIC", "E-APIC")
    runs = [
        simulate(name, speed, beam, panto, control, feedback_kind="modal_lqr")
        for speed in args.speeds for name in names
    ]
    runs.sort(key=lambda run: (names.index(run.controller), run.speed_kmh))
    rows = [metrics(run) for run in runs]
    save(runs, rows, args.output, beam, panto, control)
    for row in rows:
        print(
            f"{row['controller']:10s} {row['speed_kmh']:3.0f} km/h  "
            f"std={row['std_force_N']:7.2f} N  "
            f"ATR={row['ATR_percent']:6.2f}%  effort={row['effort_N2s']:8.2f}"
        )


if __name__ == "__main__":
    main()

