# Changelog

All notable user-facing changes to Risk Bridge are recorded here.

## Unreleased

### Added

- CI workflow, `CITATION.cff`, and `REPRODUCTION.md` for the public reproduction
  bundle.
- Privacy-safe `cases/numerical_validation` and `cases/synthetic_transport_example`
  harnesses with regenerable artifacts under `data/`.
- Schema-versioned calibration exports (`calibration_metrics.csv`,
  `calibration_residuals.csv`, `schema_version=1.1.0`).
- Public GitHub repository and package-index publication for `risk-bridge==1.0.1`
  (GitHub release `v1.0.1`). Archival DOI remains pending Zenodo deposit.

## 1.0.1 - Dependency and tooling compatibility

### Added

- Added `comp-builders` as a package-index dependency for explicit `Result` composition in recoverable orchestration validation.
- Added focused scenario runtime validation coverage for non-positive `n_jobs`, `path_jobs`, and `intermediate_flush_every`.

### Changed

- Lowered the package Python floor to `>=3.11` while keeping Python 3.13 compatibility.
- Replaced the local `mypy` development check with `basedpyright` over the public typed surface.
- Updated public documentation to describe the 1.0.1 dependency and type-checking workflow.

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
