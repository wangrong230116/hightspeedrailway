# ME-APIC Reviewer Dataset

This repository contains one dataset and this short description only.

## Dataset

`ME_APIC_monte_carlo_dataset.csv` contains the complete paired Monte Carlo simulation records used to evaluate pantograph-catenary control robustness.

- Observations: 2,000 rows
- Speeds: 300 and 350 km/h
- Trials: 200 paired trials at each speed
- Strategies per trial: Passive, Continuous LQR, T-APIC, force-error E-APIC, and ME-APIC
- Pairing key: `speed_kmh` + `trial`
- Randomization key: `scenario_seed`

Within each `(speed_kmh, trial)` pair, all five strategies use the same pantograph/catenary parameter perturbations, irregularity scale, measurement-noise setting, actuator delay, and actuator time constant. This paired design permits direct within-trial comparisons against Passive.

## Main response columns

- `mean_force_N`, `std_force_N`, `cv_percent`: contact-force statistics
- `min_force_N`, `max_force_N`: force extrema
- `contact_loss_percent`: percentage of samples with zero contact force
- `NoC`: number of controller activations
- `ATR_percent`: activation-time ratio
- `effort_N2s`: integral of squared actuator force
- `mechanical_work_J`: absolute actuator-interface mechanical work
- `signed_mechanical_work_J`: signed mechanical work

## Scenario columns

`measurement_noise_std_N`, `actuator_delay_s`, `actuator_time_constant_s`, `irregularity_scale`, `mass_scale_1` to `mass_scale_3`, `stiffness_scale_1` to `stiffness_scale_3`, `damping_scale_3`, and `contact_stiffness_scale` define the shared uncertainty realization for each paired trial.

## Suggested reviewer check

For each speed and trial, compare a controller's `std_force_N` with the Passive row having the same pairing key. A positive passive-relative improvement is:

`100 * (std_force_N_passive - std_force_N_controller) / std_force_N_passive`

No source code, figures, manuscript files, or additional experiment outputs are included.