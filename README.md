# Algorithmic Optimization for Intelligent Decision Support Systems

A comparative study of three metaheuristic optimization algorithms — **Genetic Algorithm (GA)**, **Simulated Annealing (SA)**, and **Particle Swarm Optimization (PSO)** — applied to constrained project portfolio selection, with graph-based dependency modeling and an interactive web interface.

## Overview

This project solves a real-world decision problem: given a set of candidate projects (each with a cost, expected benefit, and risk score), which subset should be selected to **maximize total benefit** while respecting:

- A **budget constraint**
- A **maximum average risk** constraint
- **Project dependencies** — some projects require another project to be selected first, modeled as a directed graph

All three algorithms are benchmarked against a greedy baseline, and results are validated statistically across multiple runs and scenarios rather than relying on a single lucky result.

## Key Finding

No single algorithm wins universally. **GA** achieves the highest average benefit under moderate constraints, but becomes unstable (high variance) under tightly constrained scenarios. **PSO** and **SA** are more consistent when the feasible search space shrinks. See the full report for details.

## Project Structure

```
├── projects.csv                  # Dataset: 20 candidate projects with cost, benefit, risk, dependencies
├── ga_optimizer.py                # Core algorithms: GA, SA, PSO, Greedy baseline, dependency graph
├── run_experiments.py             # Statistical rigor: multi-run experiments across 3 scenarios
├── app.py                         # Interactive Streamlit web application
├── comparison_chart.png           # Algorithm benefit comparison (generated)
├── convergence_chart.png          # GA convergence over generations (generated)
├── statistical_comparison.png     # Mean ± std dev across scenarios (generated)
└── README.md
```

## How to Run

**1. Install dependencies:**
```bash
pip install numpy pandas matplotlib streamlit
```

**2. Run the core optimizer (terminal output + charts):**
```bash
python ga_optimizer.py
```

**3. Run the statistical rigor experiments (15 runs × 3 scenarios):**
```bash
python run_experiments.py
```

**4. Launch the interactive web app:**
```bash
streamlit run app.py
```

## Algorithms Implemented

| Algorithm | Strategy | Notes |
|---|---|---|
| Genetic Algorithm | Population-based, evolves solutions via selection/crossover/mutation | Strong average performance, higher variance under tight constraints |
| Simulated Annealing | Single-solution, local search with temperature-based acceptance | Most stable under strict constraints |
| Particle Swarm Optimization | Population-based, particles guided by personal/global best | Consistent performance across all tested scenarios |
| Greedy (baseline) | Ranks by benefit/cost ratio | Fast but ignores dependency constraints entirely |

## Author

Paulika Rath — B.Tech, Electrical Engineering, National Institute of Technology Rourkela
Summer Internship Project, Department of Computer Science & Engineering
