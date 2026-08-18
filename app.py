"""
Streamlit UI for the Multi-Algorithm Decision Support System
------------------------------------------------------------
Imports GA, SA, PSO, and Greedy directly from ga_optimizer.py so this
UI can never drift out of sync with the core algorithms.
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# ---------- IMPORT CORE LOGIC (suppress its console prints on load) ----------
class _Silence:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    def __exit__(self, *a):
        sys.stdout.close()
        sys.stdout = self._stdout

with _Silence():
    import ga_optimizer as go

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="AI Decision Support System", layout="wide")
st.title("🧬 Multi-Algorithm Decision Support System")
st.caption("Genetic Algorithm vs Simulated Annealing vs Particle Swarm Optimization vs Greedy Baseline — "
           "with graph-based project dependency constraints")

data = go.data

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Settings")
budget = st.sidebar.slider("Budget ($)", min_value=100000, max_value=1000000,
                            value=500000, step=10000)
risk_limit = st.sidebar.slider("Max Average Risk", min_value=1, max_value=10, value=6, step=1)
population_size = st.sidebar.slider("GA/PSO Population Size", min_value=10, max_value=200, value=50, step=10)
generations = st.sidebar.slider("GA Generations", min_value=20, max_value=300, value=100, step=10)
mutation_rate = st.sidebar.slider("GA Mutation Rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01)

run_button = st.sidebar.button("🚀 Run All Algorithms", type="primary")

# ---------- SHOW AVAILABLE PROJECTS + DEPENDENCY GRAPH ----------
st.subheader("Available Projects")
st.dataframe(data, use_container_width=True)

with st.expander("📊 View Dependency Graph"):
    dep_lines = []
    for i, dep in go.dependency_graph.items():
        if dep is not None:
            dep_lines.append(f"**{data['project_name'][i]}** requires *{data['project_name'][dep]}*")
    if dep_lines:
        for line in dep_lines:
            st.markdown(f"- {line}")
    else:
        st.write("No dependencies defined.")

# ---------- RUN AND DISPLAY RESULTS ----------
if run_button:
    # push slider values into the shared module settings the algorithms read from
    go.BUDGET = budget
    go.RISK_LIMIT = risk_limit
    go.POPULATION_SIZE = population_size
    go.GENERATIONS = generations
    go.MUTATION_RATE = mutation_rate

    with st.spinner("Running Greedy, GA, SA, and PSO..."):
        with _Silence():
            greedy_chosen, greedy_benefit = go.greedy_selection()
            ga_solution, ga_score, ga_history = go.run_ga()
            sa_solution, sa_score, sa_history = go.simulated_annealing()
            pso_solution, pso_score, pso_history = go.particle_swarm_optimization()

    results = {
        "Greedy":  {"solution": greedy_chosen, "history": None},
        "GA":      {"solution": ga_solution, "history": ga_history},
        "SA":      {"solution": sa_solution, "history": sa_history},
        "PSO":     {"solution": pso_solution, "history": pso_history},
    }

    for name, r in results.items():
        selected = data[r["solution"] == 1]
        r["benefit"] = selected["benefit"].sum()
        r["cost"] = selected["cost"].sum()
        r["risk"] = selected["risk_score"].mean() if len(selected) > 0 else 0
        r["violations"] = go.count_dependency_violations(r["solution"])
        r["selected_df"] = selected

    best_name = max(results, key=lambda n: results[n]["benefit"])

    st.success(f"Optimization complete! Best result: **{best_name}** with benefit {results[best_name]['benefit']}")

    # ---- SUMMARY TABLE ----
    st.subheader("Algorithm Comparison")
    summary_cols = st.columns(4)
    colors = {"Greedy": "#94a3b8", "SA": "#3b82f6", "PSO": "#f59e0b", "GA": "#22c55e"}
    for col, name in zip(summary_cols, results.keys()):
        r = results[name]
        dep_status = "OK" if r["violations"] == 0 else f"{r['violations']} broken"
        with col:
            st.metric(name, f"{r['benefit']}",
                      f"${budget - r['cost']:,} left" if name != "Greedy" else None)
            st.caption(f"Cost: ${r['cost']:,} | Risk: {r['risk']:.2f} | Deps: {dep_status}")

    # ---- BEST PORTFOLIO TABLE ----
    st.subheader(f"Best Portfolio ({best_name})")
    st.dataframe(results[best_name]["selected_df"][["project_name", "cost", "benefit", "risk_score"]],
                 use_container_width=True)

    # ---- CHARTS ----
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for name in ["GA", "SA", "PSO"]:
            ax1.plot(results[name]["history"], linewidth=2, color=colors[name], label=name)
        ax1.set_title("Optimization Progress (all 3 algorithms)")
        ax1.set_xlabel("Iteration / Generation")
        ax1.set_ylabel("Best Fitness Score Found")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        names = list(results.keys())
        values = [results[n]["benefit"] for n in names]
        ax2.bar(names, values, color=[colors[n] for n in names])
        ax2.set_title("Total Benefit by Algorithm")
        ax2.set_ylabel("Total Benefit Score")
        st.pyplot(fig2)

else:
    st.info("👈 Adjust settings in the sidebar and click **Run All Algorithms** to get started.")
