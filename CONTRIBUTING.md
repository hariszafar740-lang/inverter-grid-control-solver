# Contributing to Grid-Forming Inverter Control Solver

Thank you for your interest in contributing!
We welcome bug fixes, numerical validation improvements, and control-law extensions.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Quality Gates Before Opening a Pull Request

1. Linting: `python -m ruff check .` must pass with zero warnings or errors.
2. Reproducible Figure & Benchmark: `python main.py` and `python -m src.benchmark` must succeed cleanly.
3. Unit & Cross-Validation Suite: `python -m pytest -v` must pass across all tests, including SciPy ODE cross-checks.
4. Scientific Honesty: Document exact physical assumptions, coordinate conventions (amplitude-invariant peak Clarke αβ), and model scope boundaries without claiming hardware certification.
