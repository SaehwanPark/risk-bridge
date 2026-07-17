# Replication Cases

Cases are versioned applied-analysis harnesses built on the public `risk_bridge` package.
They contain configuration, orchestration, synthetic tests, and replication documentation.
Generated outputs must remain untracked under `data/`.

The public reproduction bundle includes only privacy-safe cases:

- [Independent numerical validation](numerical_validation/README.md): regenerable derivative,
  optimizer-comparison, recovery, and invariance artifacts.
- [External calibration validation](external_calibration_validation/README.md): regenerable
  synthetic fixed-summary recovery/calibration study with pass/fail gates.
- [Synthetic transport example](synthetic_transport_example/README.md): privacy-safe
  Scenario-2-shaped user-data second example with smoke and mid-size profiles.
- [Runtime and support-scaling protocol](runtime_support_scaling/README.md): regenerable
  environment-captured runtime timings and Cartesian-support scaling tables.

Private patient-level replication cases are not distributed in this repository.
See [REPRODUCTION.md](../REPRODUCTION.md) for the end-to-end regeneration runbook.
