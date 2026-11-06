# Calibrated ME-APIC reviewer dataset

This release contains ME_APIC_monte_carlo_dataset.csv (2,000 controller records), summary.json (paired statistics and selected deterministic sweep summaries), and this description.

The controller uses nine controllable modes, an 18-dimensional state, and rho = 0.012907490610006619. Closed-loop simulations use the full finite-element plant.

## Paired design

Five methods (Passive, Continuous, T-APIC, E-APIC, ME-APIC) share 200 scenarios at each of 300 and 350 km/h. Pair records by speed_kmh and trial; scenario_seed identifies the shared realization. There are 400 speed-trial scenarios, not 2,000 independent paired scenarios.

For each controller and trial, passive-relative improvement is 100 * (sigma_passive - sigma_controller) / sigma_passive. Summary improvement percentages average these within-trial percentages. ME-APIC means are 2.379327% at 300 km/h and 0.710968% at 350 km/h.

Mechanical-work and squared-force reductions average trial-wise ratios against the paired Continuous controller. Energy-start shares exclude initialization; simultaneous energy/deadline crossings are recorded separately. At 350 km/h, energy crossings account for 986/1400 non-initial ME-APIC starts.

The primary CSV provides force statistics, actuator utilization and costs, and shared parameter/noise/delay settings. It contains controller-level trial summaries, not raw force/state histories or executable FE simulation code. Deterministic sweep comparisons in the JSON are distinct from the 200-trial uncertainty statistics.
