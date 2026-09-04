# ME-APIC Pantograph-Catenary Control Code

Reproducible Python implementation of the Cardelli finite-element contact-wire baseline and the modal-energy-triggered aperiodic intermittent control (ME-APIC) extensions used for active pantograph-catenary studies.

## Repository scope

This directory contains source code and unit tests only. Experimental outputs, time histories, CSV files, figures, generated datasets, manuscript files, and caches are intentionally excluded.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib

Install dependencies:

    python -m pip install -r requirements.txt

Run tests:

    python -m unittest discover -v

Available entry points include the baseline reproduction, controller comparisons, modal LQR, ME-APIC, paired robustness study, modal/theory diagnostics, sensitivity analysis, and the five-span two-wire/dropper transfer stress test. Each script creates outputs only in a user-selected or script-default local result directory; those directories are ignored by Git.

## Data policy

No experimental result or generated dataset is tracked in this repository. Re-run the scripts locally to generate outputs.