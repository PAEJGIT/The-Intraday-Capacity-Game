# The Intraday Capacity Game

A formal model of competitive intraday cross-border electricity trading, cast as a Simple Stochastic Game (SSG) and solved as a fixed-point problem. This repository accompanies the bachelor thesis *The Intraday Capacity Game* (Aalborg University, Department of Computer Science, 2026).

## Overview

The model treats intraday trading for scarce cross-border interconnector capacity (DK1–Germany, SIDC/XBID framework) as a stopping SSG between three actors:

- **MAX** — the trader, who chooses trade size,
- **MIN** — a budget-constrained rival firm, who may block trades,
- **AVG** — the market matching process, resolving outcomes stochastically.

## Repository structure
Figures are produced by `figures.py` (see below for its location in your layout).

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for figures only)

```bash
pip install numpy matplotlib
```

## Usage

All modules import from the `src` package, so run them as modules from the repository root.

Run the sanity tests:

```bash
python -m src.tests
```

Reproduce the experimental tables and JSON data (`results/all.json`):

```bash
python -m src.experiments
```

Run the random-initial-policy convergence study (`results/convergence_random.json`):

```bash
python -m src.convergence_random
```

Generate the figures (`figures/*.pdf`):

```bash
python -m src.figures
```

