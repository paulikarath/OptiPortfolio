"""
Genetic Algorithm - Project Portfolio Optimizer
--------------------------------------------------
Goal: pick the BEST combination of projects that:
  1. Stays under our budget
  2. Keeps average risk under our risk limit
  3. Maximizes total benefit

Think of each "solution" as a row of 0s and 1s, one per project.
1 = we picked this project, 0 = we didn't.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- STEP 0: SETTINGS (change these numbers to test different scenarios) ----------
BUDGET = 500000          # total money we can spend
RISK_LIMIT = 6            # max average risk we're okay with
POPULATION_SIZE = 50       # how many guesses we keep per round
GENERATIONS = 100          # how many rounds of improvement
MUTATION_RATE = 0.05       # 5% chance a single project flips on/off during mutation
PENALTY = 100000           # big number to punish "bad" guesses that break rules

# ---------- STEP 1: LOAD THE PROJECTS ----------
data = pd.read_csv("projects.csv")
costs = data["cost"].values
benefits = data["benefit"].values
risks = data["risk_score"].values
n_projects = len(data)

print(f"Loaded {n_projects} projects.")

# ---------- BUILD THE DEPENDENCY GRAPH ----------
# This is a real graph structure: each project (node) points to the project
# it depends on (a directed edge). We store it as a dictionary:
#   dependency_graph[project_index] = index of the project it requires (or None)
name_to_index = {name: i for i, name in enumerate(data["project_name"])}
dependency_graph = {}

for i, row in data.iterrows():
    dep_name = row["depends_on"]
    if pd.isna(dep_name) or str(dep_name).strip() == "":
        dependency_graph[i] = None
    else:
        dependency_graph[i] = name_to_index[str(dep_name).strip()]

print("Dependency graph built:")
for i, dep in dependency_graph.items():
    if dep is not None:
        print(f"  '{data['project_name'][i]}' requires '{data['project_name'][dep]}'")

def count_dependency_violations(chromosome):
    """
    Walks the graph: for every selected project, check if its required
    prerequisite (if any) is ALSO selected. Returns how many are broken.
    """
    violations = 0
    for i in range(n_projects):
        if chromosome[i] == 1:
            required = dependency_graph[i]
            if required is not None and chromosome[required] == 0:
                violations += 1
    return violations

# ---------- STEP 2: SCORE A GUESS (the "fitness function") ----------
def fitness(chromosome):
    """
    chromosome = array of 0s and 1s, one per project.
    Higher score = better solution.
    """
    total_cost = np.sum(chromosome * costs)
    total_benefit = np.sum(chromosome * benefits)
    selected_count = np.sum(chromosome)

    if selected_count == 0:
        return 0  # picking nothing is useless

    avg_risk = np.sum(chromosome * risks) / selected_count

    score = total_benefit

    # Punish going over budget
    if total_cost > BUDGET:
        score -= PENALTY * (total_cost - BUDGET) / BUDGET

    # Punish going over risk limit
    if avg_risk > RISK_LIMIT:
        score -= PENALTY * (avg_risk - RISK_LIMIT)

    # Punish broken dependencies (picked a project without its prerequisite)
    violations = count_dependency_violations(chromosome)
    if violations > 0:
        score -= PENALTY * violations

    return score

# ---------- STEP 3: MAKE RANDOM STARTING GUESSES ----------
def create_population():
    return [np.random.randint(0, 2, n_projects) for _ in range(POPULATION_SIZE)]

# ---------- STEP 4: PICK PARENTS (better guesses are more likely to be picked) ----------
def select_parent(population, scores):
    adjusted = np.array([max(s, 0) + 1e-6 for s in scores])  # avoid all-zero probs
    probs = adjusted / adjusted.sum()  # numpy division keeps this summing to exactly 1
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

# ---------- STEP 5: MIX TWO PARENTS TOGETHER (crossover) ----------
def crossover(parent1, parent2):
    point = np.random.randint(1, n_projects)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

# ---------- STEP 6: RANDOMLY FLIP A FEW BITS (mutation) ----------
def mutate(chromosome):
    for i in range(n_projects):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]  # flip 0 to 1 or 1 to 0
    return chromosome

# ---------- STEP 7: RUN THE WHOLE THING ----------
def run_ga():
    population = create_population()
    best_solution = None
    best_score = -np.inf
    history = []  # track best score per generation, for a chart later

    for gen in range(GENERATIONS):
        scores = [fitness(ind) for ind in population]

        # track the best one we've ever seen
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        history.append(best_score)

        # build the next generation
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = select_parent(population, scores)
            parent2 = select_parent(population, scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        if gen % 10 == 0:
            print(f"Generation {gen}: best score so far = {best_score:.2f}")

    return best_solution, best_score, history

# ---------- BASELINE: GREEDY SELECTION (for comparison) ----------
def greedy_selection():
    """
    Simple, 'dumb' strategy to compare against: sort projects by
    benefit-per-dollar (best bang for buck), keep adding until
    budget or risk limit would be broken.
    This is the classic non-AI approach most people would default to.
    """
    ratio = benefits / costs
    order = np.argsort(-ratio)  # sort descending: best ratio first

    chosen = np.zeros(n_projects, dtype=int)
    total_cost = 0
    total_benefit = 0
    total_risk = 0
    count = 0

    for idx in order:
        new_cost = total_cost + costs[idx]
        new_count = count + 1
        new_avg_risk = (total_risk + risks[idx]) / new_count

        if new_cost <= BUDGET and new_avg_risk <= RISK_LIMIT:
            chosen[idx] = 1
            total_cost = new_cost
            total_benefit += benefits[idx]
            total_risk += risks[idx]
            count = new_count

    return chosen, total_benefit

# ---------- ALGORITHM 2: SIMULATED ANNEALING ----------
def simulated_annealing(initial_temp=1000, cooling_rate=0.995, iterations=2000):
    """
    Starts with ONE random solution and tweaks it repeatedly.
    - If the tweak is better, always keep it.
    - If the tweak is worse, sometimes keep it anyway (controlled by 'temperature').
    - Temperature starts high (very open to worse moves) and cools down over time
      (gets pickier), which is why it's called "annealing" - like cooling metal.
    This helps it escape getting stuck on a mediocre answer early on.
    """
    current = np.random.randint(0, 2, n_projects)
    current_score = fitness(current)
    best = current.copy()
    best_score = current_score
    temp = initial_temp
    history = [best_score]

    for i in range(iterations):
        # make a small random tweak: flip one random project on/off
        neighbor = current.copy()
        flip_idx = np.random.randint(0, n_projects)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        neighbor_score = fitness(neighbor)

        delta = neighbor_score - current_score

        # always accept better moves; sometimes accept worse ones based on temperature
        if delta > 0 or np.random.rand() < np.exp(delta / max(temp, 1e-6)):
            current = neighbor
            current_score = neighbor_score

        if current_score > best_score:
            best = current.copy()
            best_score = current_score

        history.append(best_score)
        temp *= cooling_rate  # cool down a little each iteration

    return best, best_score, history

# ---------- STEP 8: SHOW THE RESULTS ----------
if __name__ == "__main__":
    best_solution, best_score, history = run_ga()

    selected = data[best_solution == 1]
    total_cost = selected["cost"].sum()
    total_benefit = selected["benefit"].sum()
    avg_risk = selected["risk_score"].mean() if len(selected) > 0 else 0

    print("\n===== BEST PORTFOLIO FOUND =====")
    print(selected[["project_name", "cost", "benefit", "risk_score"]].to_string(index=False))
    print(f"\nTotal Cost:    ${total_cost:,} / ${BUDGET:,} budget")
    print(f"Total Benefit: {total_benefit}")
    print(f"Average Risk:  {avg_risk:.2f} / {RISK_LIMIT} limit")
    dep_violations = count_dependency_violations(best_solution)
    print(f"Dependency Check: {'PASSED - all prerequisites satisfied' if dep_violations == 0 else f'FAILED - {dep_violations} broken dependencies'}")

    # ---------- COMPARE AGAINST GREEDY BASELINE ----------
    greedy_chosen, greedy_benefit = greedy_selection()

    # ---------- RUN SIMULATED ANNEALING ----------
    sa_solution, sa_score, sa_history = simulated_annealing()
    sa_selected = data[sa_solution == 1]
    sa_benefit = sa_selected["benefit"].sum()

    ga_improvement = ((total_benefit - greedy_benefit) / greedy_benefit) * 100
    sa_improvement = ((sa_benefit - greedy_benefit) / greedy_benefit) * 100

    print("\n===== COMPARISON: GA vs SA vs GREEDY BASELINE =====")
    print(f"Greedy baseline benefit: {greedy_benefit}")
    print(f"GA benefit:              {total_benefit}  ({ga_improvement:.1f}% vs greedy)")
    print(f"SA benefit:              {sa_benefit}  ({sa_improvement:.1f}% vs greedy)")

    # ---------- CHART 1: HOW THE GA IMPROVED OVER TIME ----------
    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2)
    plt.title("GA Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score Found So Far")
    plt.grid(True, alpha=0.3)
    plt.savefig("convergence_chart.png", dpi=150, bbox_inches="tight")
    print("\nSaved chart: convergence_chart.png")

    # ---------- CHART 2: GA vs SA vs GREEDY BAR COMPARISON ----------
    plt.figure(figsize=(6, 5))
    plt.bar(["Greedy\nBaseline", "Simulated\nAnnealing", "Genetic\nAlgorithm"],
            [greedy_benefit, sa_benefit, total_benefit],
            color=["#94a3b8", "#3b82f6", "#22c55e"])
    plt.title("Total Benefit: GA vs SA vs Greedy Baseline")
    plt.ylabel("Total Benefit Score")
    plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
    print("Saved chart: comparison_chart.png")
