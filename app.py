"""
Streamlit UI for the GA Project Portfolio Optimizer
------------------------------------------------------
This wraps your ga_optimizer.py logic in a clickable web app.
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="AI Decision Support System", layout="wide")
st.title("🧬 Genetic Algorithm Decision Support System")
st.caption("Optimal R&D Project Portfolio Selection under Budget & Risk Constraints")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("projects.csv")

data = load_data()

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Settings")
budget = st.sidebar.slider("Budget ($)", min_value=100000, max_value=1000000,
                            value=500000, step=10000)
risk_limit = st.sidebar.slider("Max Average Risk", min_value=1, max_value=10,
                                value=6, step=1)
population_size = st.sidebar.slider("Population Size", min_value=10, max_value=200,
                                     value=50, step=10)
generations = st.sidebar.slider("Generations", min_value=20, max_value=300,
                                 value=100, step=10)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.3,
                                   value=0.05, step=0.01)

run_button = st.sidebar.button("🚀 Run Optimization", type="primary")

# ---------- SHOW AVAILABLE PROJECTS ----------
st.subheader("Available Projects")
st.dataframe(data, use_container_width=True)

# ---------- GA LOGIC (same as ga_optimizer.py, wrapped as functions) ----------
def fitness(chromosome, costs, benefits, risks, budget, risk_limit, penalty=100000):
    total_cost = np.sum(chromosome * costs)
    total_benefit = np.sum(chromosome * benefits)
    selected_count = np.sum(chromosome)
    if selected_count == 0:
        return 0
    avg_risk = np.sum(chromosome * risks) / selected_count
    score = total_benefit
    if total_cost > budget:
        score -= penalty * (total_cost - budget) / budget
    if avg_risk > risk_limit:
        score -= penalty * (avg_risk - risk_limit)
    return score

def select_parent(population, scores):
    adjusted = np.array([max(s, 0) + 1e-6 for s in scores])
    probs = adjusted / adjusted.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

def crossover(parent1, parent2, n_projects):
    point = np.random.randint(1, n_projects)
    return np.concatenate([parent1[:point], parent2[point:]])

def mutate(chromosome, n_projects, mutation_rate):
    for i in range(n_projects):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def run_ga(data, budget, risk_limit, population_size, generations, mutation_rate):
    costs = data["cost"].values
    benefits = data["benefit"].values
    risks = data["risk_score"].values
    n_projects = len(data)

    population = [np.random.randint(0, 2, n_projects) for _ in range(population_size)]
    best_solution = None
    best_score = -np.inf
    history = []

    for gen in range(generations):
        scores = [fitness(ind, costs, benefits, risks, budget, risk_limit) for ind in population]
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_solution = population[gen_best_idx].copy()
        history.append(best_score)

        new_population = []
        for _ in range(population_size):
            p1 = select_parent(population, scores)
            p2 = select_parent(population, scores)
            child = crossover(p1, p2, n_projects)
            child = mutate(child, n_projects, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_solution, best_score, history

def greedy_selection(data, budget, risk_limit):
    costs = data["cost"].values
    benefits = data["benefit"].values
    risks = data["risk_score"].values
    n_projects = len(data)
    ratio = benefits / costs
    order = np.argsort(-ratio)

    chosen = np.zeros(n_projects, dtype=int)
    total_cost = 0
    total_benefit = 0
    total_risk = 0
    count = 0

    for idx in order:
        new_cost = total_cost + costs[idx]
        new_count = count + 1
        new_avg_risk = (total_risk + risks[idx]) / new_count
        if new_cost <= budget and new_avg_risk <= risk_limit:
            chosen[idx] = 1
            total_cost = new_cost
            total_benefit += benefits[idx]
            total_risk += risks[idx]
            count = new_count

    return chosen, total_benefit

# ---------- RUN AND DISPLAY RESULTS ----------
if run_button:
    with st.spinner("Running genetic algorithm..."):
        best_solution, best_score, history = run_ga(
            data, budget, risk_limit, population_size, generations, mutation_rate
        )
        greedy_chosen, greedy_benefit = greedy_selection(data, budget, risk_limit)

    selected = data[best_solution == 1]
    total_cost = selected["cost"].sum()
    total_benefit = selected["benefit"].sum()
    avg_risk = selected["risk_score"].mean() if len(selected) > 0 else 0
    improvement = ((total_benefit - greedy_benefit) / greedy_benefit) * 100 if greedy_benefit > 0 else 0

    st.success("Optimization complete!")

    # ---- KEY METRICS ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Benefit", f"{total_benefit}")
    col2.metric("Total Cost", f"${total_cost:,}", f"${budget - total_cost:,} left")
    col3.metric("Avg Risk", f"{avg_risk:.2f}", f"limit {risk_limit}")
    col4.metric("vs Greedy Baseline", f"+{improvement:.1f}%")

    # ---- SELECTED PORTFOLIO TABLE ----
    st.subheader("Selected Portfolio")
    st.dataframe(selected[["project_name", "cost", "benefit", "risk_score"]],
                 use_container_width=True)

    # ---- CHARTS ----
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(history, linewidth=2, color="#22c55e")
        ax1.set_title("GA Optimization Progress")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best Fitness Score")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["Greedy Baseline", "Genetic Algorithm"],
                [greedy_benefit, total_benefit],
                color=["#94a3b8", "#22c55e"])
        ax2.set_title("Total Benefit: GA vs Greedy")
        ax2.set_ylabel("Total Benefit Score")
        st.pyplot(fig2)

else:
    st.info("👈 Adjust settings in the sidebar and click **Run Optimization** to get started.")
