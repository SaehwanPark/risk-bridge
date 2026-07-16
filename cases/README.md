# Replication Cases

Cases are versioned applied-analysis harnesses built on the public `risk_bridge` package.
They contain configuration, orchestration, synthetic tests, and replication documentation.
Generated outputs must remain untracked under `data/`.

The public reproduction bundle includes only privacy-safe cases:

- [Independent numerical validation](numerical_validation/README.md): regenerable derivative,
  optimizer-comparison, recovery, and invariance artifacts.
- [Synthetic transport example](synthetic_transport_example/README.md): privacy-safe
  Scenario-2-shaped user-data second example with smoke and mid-size profiles.

Private patient-level replication cases are not distributed in this repository.
See [REPRODUCTION.md](../REPRODUCTION.md) for the end-to-end regeneration runbook.
