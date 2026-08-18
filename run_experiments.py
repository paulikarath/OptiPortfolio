"""
Statistical Rigor Experiments
--------------------------------
Since GA, SA, and PSO all involve randomness, one run isn't enough to trust.
This script runs each algorithm MANY times per scenario and reports the
average (mean) and how much results vary (standard deviation) - this is
what makes a result trustworthy instead of a lucky fluke.

Run with:  python run_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class SuppressPrints:
    """Temporarily silences print() output - used so 15 runs x 3 algorithms
    x 3 scenarios doesn't flood the terminal with hundreds of 'Generation N...' lines."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

N_RUNS = 15  # how many times to repeat each algorithm per scenario

with SuppressPrints():
    import ga_optimizer as go  # reuses your existing algorithms, no duplicate code

# ---------- DEFINE TEST SCENARIOS ----------
# Each scenario is a different (budget, risk_limit) combination -
# this tests whether the algorithms hold up under different real-world conditions.
scenarios = {
    "Default":      {"BUDGET": 500000, "RISK_LIMIT": 6},
    "Tight Budget": {"BUDGET": 300000, "RISK_LIMIT": 6},
    "Strict Risk":  {"BUDGET": 500000, "RISK_LIMIT": 3},
}

algorithms = {
    "Greedy": lambda: go.greedy_selection(),          # returns (chosen, benefit)
    "GA":     lambda: go.run_ga()[:2],                # returns (solution, score, history) -> trim to 2
    "SA":     lambda: go.simulated_annealing()[:2],
    "PSO":    lambda: go.particle_swarm_optimization()[:2],
}

def get_benefit(result, is_greedy=False):
    """Greedy already returns benefit directly; others return a fitness score,
    so we recompute the real benefit (not the penalized fitness score) from the solution."""
    solution, score_or_benefit = result
    if is_greedy:
        return score_or_benefit
    selected = go.data[solution == 1]
    return selected["benefit"].sum()

# ---------- RUN EVERYTHING ----------
results = {}  # results[scenario][algorithm] = list of benefit values across N_RUNS

for scenario_name, settings in scenarios.items():
    print(f"\n=== Scenario: {scenario_name} (Budget=${settings['BUDGET']:,}, Risk<={settings['RISK_LIMIT']}) ===")
    go.BUDGET = settings["BUDGET"]
    go.RISK_LIMIT = settings["RISK_LIMIT"]

    results[scenario_name] = {}

    for algo_name, algo_func in algorithms.items():
        benefits = []
        for run in range(N_RUNS):
            with SuppressPrints():
                result = algo_func()
            is_greedy = (algo_name == "Greedy")
            benefit = get_benefit(result, is_greedy=is_greedy)
            benefits.append(benefit)

        mean_benefit = np.mean(benefits)
        std_benefit = np.std(benefits)
        results[scenario_name][algo_name] = benefits

        print(f"  {algo_name:8s}: mean={mean_benefit:7.1f}  std={std_benefit:5.1f}  "
              f"(min={min(benefits)}, max={max(benefits)})")

# ---------- CHART: MEAN + VARIABILITY ACROSS SCENARIOS ----------
fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), sharey=False)
if len(scenarios) == 1:
    axes = [axes]

colors = {"Greedy": "#94a3b8", "SA": "#3b82f6", "PSO": "#f59e0b", "GA": "#22c55e"}

for ax, (scenario_name, algo_results) in zip(axes, results.items()):
    names = list(algo_results.keys())
    means = [np.mean(algo_results[n]) for n in names]
    stds = [np.std(algo_results[n]) for n in names]
    bar_colors = [colors[n] for n in names]

    ax.bar(names, means, yerr=stds, capsize=6, color=bar_colors)
    ax.set_title(scenario_name)
    ax.set_ylabel("Mean Total Benefit")
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle(f"Algorithm Performance Across Scenarios (mean of {N_RUNS} runs, error bars = std dev)")
plt.tight_layout()
plt.savefig("statistical_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved chart: statistical_comparison.png")

# ---------- RESTORE DEFAULT SETTINGS ----------
go.BUDGET = 500000
go.RISK_LIMIT = 6

print("\nDone. This chart and the printed mean/std numbers above are your")
print("statistically rigorous results - use them in your report's Results section.")
