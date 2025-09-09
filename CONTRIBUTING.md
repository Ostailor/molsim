# Contributing

Thanks for your interest in contributing! This project aims to deliver a research‑grade, safety‑aware molecule discovery pipeline.

## Development Setup

1. Create a virtual environment and install dev deps:
   - `make setup`
2. Run checks locally:
   - `make lint && make type && make test`
3. Enable pre‑commit hooks:
   - Installed by `make setup`; run `pre-commit run --all-files` to check.

## Code Style and Quality

- Python 3.11+ with type hints.
- Formatting: black; Linting: ruff; Types: mypy.
- Keep functions small and pure where practical; add docstrings.

## Tests

- Add unit tests for new logic; keep them fast and deterministic.
- Mark slow/integration tests distinctly.

## Safety & Ethics

- Never add actionable synthesis instructions or enable harmful outputs.
- Follow the Safety & Compliance policies.

## Commit and PR Process

- Small, focused PRs with clear titles (imperative mood).
- Link to an issue or TASKS.md phase/deliverable.
- CI must pass: lint, type, test. Include docs updates when needed.
