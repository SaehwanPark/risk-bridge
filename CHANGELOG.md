# Changelog

All notable user-facing changes to Risk Bridge are recorded here.

## 1.0.0 - Public repository release

### Added

- Public Python package layout under `src/risk_bridge`.
- Command-line workflows for simulated and user-data constrained MLE runs.
- Built-in Scenario 1-3 simulation configuration builders.
- User-data schema mapping for target, source, and reference cohorts.
- Propensity-score matched and random-sampled source evaluation paths.
- Constrained MLE solver ladder with fit diagnostics and calibration-violation reporting.
- CSV output contract for estimates, metrics, thresholds, prevalence, and run metadata.
- Unit test suite for core numerical and pipeline modules.
- Public documentation: README, quickstart, user guide, API reference, and architecture notes.
- Apache 2.0 license.

### Changed

- Reframed repository documentation for first-time external users.
- Removed internal agent harness files, scratch instructions, and bundled research PDFs from the public distribution.
- Updated package metadata for a stable public `v1.0.0` release.
