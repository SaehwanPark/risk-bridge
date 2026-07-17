# Contributing to Risk Bridge

Thanks for improving Risk Bridge. This document covers the expected development
workflow for the public package.

## Support expectations

- Bug reports, feature requests, and questions belong in
  [GitHub Issues](https://github.com/SaehwanPark/risk-bridge/issues).
- Maintainers triage issues as capacity allows; there is no guaranteed SLA.
- Security-sensitive reports should use a private maintainer contact when
  available, otherwise open a minimal public issue without sensitive detail.

## Development setup

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uvx ruff check .
```

Use `uv` for dependency and command execution. Do not introduce a parallel
`requirements.txt` or unmanaged virtualenv workflow for this repository.

## Spec-driven changes

Meaningful package changes should keep project state docs aligned:

- Update `CHANGELOG.md` for user-visible completed work.
- Keep `ARCHITECTURE.md` accurate for structural or data-flow changes.
- Prefer focused tests next to the touched modules.

## Pull requests

- Keep each PR scoped to one coherent slice.
- Include or update tests for behavior changes.
- Do not commit generated artifacts under `data/`.
- Do not include patient-level or other private datasets in public PRs.
- Ensure `uv run pytest` and `uv run basedpyright` are clean before review.

## Privacy

The public repository ships only privacy-safe synthetic and numerical cases.
Private patient-data replication harnesses are intentionally excluded from this
tree. Do not reopen them in public contributions.

## Release ownership

Tagged releases, PyPI uploads, and Zenodo deposits are maintainer-operated steps.
Contributors should not publish versions unless a maintainer explicitly asks them
to run the release checklist.
