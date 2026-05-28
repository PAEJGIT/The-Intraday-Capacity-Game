"""
Figures for Chapter 7.

This script parses the SSG model and generates the plots used in the report.

Produces:
  fig_convergence.pdf       VI tail vs SI iterations (log scale residual)
  fig_strategy_threshold.pdf MAX action as a function of (b, blocked-prob)
  fig_W_vs_B.pdf            W at Max(H, B) as a function of B
  fig_W_vs_pH.pdf           W at Max(H, B) as a function of P(H)
  fig_phase_blocked.pdf     Phase diagram for the joint (sb, lb) sweep
  fig_scaling.pdf           Graph size and SI cost as B grows
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Use the Agg backend so matplotlib doesn't require a GUI/display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.game import GameGraph, Params
from src.algorithms import strategy_iteration, value_iteration, bellman

# Define output directories
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"


plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Section 7.1: VI residual vs iteration, with SI markers
# ---------------------------------------------------------------------------

def fig_convergence():
    p = Params()
    g = GameGraph.build(p)

    # Calculate the VI residual ||T(v) - v||_inf step by step
    v = np.zeros(g.n())
    residuals = []
    
    for iteration in range(150):
        v_new = bellman(g, v)
        max_diff = float(np.max(np.abs(v_new - v)))
        residuals.append(max_diff)
        v = v_new

    # Get the exact SI solution to mark it on the graph
    si = strategy_iteration(g)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    
    # Plot VI decay curve
    ax.semilogy(range(1, len(residuals) + 1), residuals,
                label="Value iteration  $\\|T(v_k) - v_k\\|_\\infty$",
                color="#1f3a5f", linewidth=1.6)
                
    # Add target tolerance and SI termination markers
    ax.axhline(1e-9, color="#999999", linestyle="--", linewidth=0.8,
               label="$\\varepsilon = 10^{-9}$")
    ax.axvline(si.eval_calls, color="#a23b3b", linestyle=":", linewidth=1.4,
               label=f"SI: {si.eval_calls} linear solves")
               
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Residual")
    ax.set_xlim(0, len(residuals))
    ax.set_title("Convergence of value iteration vs. strategy iteration")
    ax.legend(loc="upper right", frameon=False)
    
    fig.savefig(FIG / "fig_convergence.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_convergence.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.3.1: W vs MIN's Budget Capacity (B)
# ---------------------------------------------------------------------------

def fig_W_vs_B():
    budget_capacities = list(range(0, 16))
    win_probabilities = []
    
    for budget in budget_capacities:
        g = GameGraph.build(Params(B=budget))
        si = strategy_iteration(g)
        win_probabilities.append(float(si.v[0]))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(budget_capacities, win_probabilities, marker="o", color="#1f3a5f", linewidth=1.5)
    
    ax.set_xlabel("MIN's token budget capacity $B$")
    ax.set_ylabel(r"$W(\mathrm{Max}(H, B))$")
    ax.set_title("Trader's optimal winning probability as MIN's budget grows")
    ax.set_ylim(0.45, 0.80)
    
    fig.savefig(FIG / "fig_W_vs_B.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_W_vs_B.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.3.3: W vs P(spread = H)
# ---------------------------------------------------------------------------

def fig_W_vs_pH():
    # Sweep from 0 to 1 with 40 steps
    pH_values = np.linspace(0.0, 1.0, 41)
    win_probabilities = []
    
    for pH in pH_values:
        g = GameGraph.build(Params(p_H=float(pH)))
        si = strategy_iteration(g)
        win_probabilities.append(float(si.v[0]))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(pH_values, win_probabilities, color="#1f3a5f", linewidth=1.6)
    
    ax.set_xlabel(r"P(spread = H) on continue")
    ax.set_ylabel(r"$W(\mathrm{Max}(H, 6))$")
    ax.set_title("Trader's optimal winning probability vs. spread persistence")
    ax.set_xlim(0, 1)
    
    fig.savefig(FIG / "fig_W_vs_pH.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_W_vs_pH.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.2: Strategy Threshold - MAX action at Max(H, b)
# ---------------------------------------------------------------------------

def fig_strategy_threshold():
    p = Params()
    g = GameGraph.build(p)
    si = strategy_iteration(g)

    budgets = list(range(p.B + 1))
    v_small = []
    v_large = []
    
    # Collect the evaluated values of both choices at every budget level
    for b in budgets:
        max_idx = g.idx[("Max", ("H", b))]
        
        # Create a quick lookup for target indices based on action name
        edge_dict = {}
        for action, target_idx in g.out_max[max_idx]:
            edge_dict[action] = target_idx
            
        v_small.append(float(si.v[edge_dict["commit_small"]]))
        v_large.append(float(si.v[edge_dict["commit_large"]]))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    
    # Plot the competing values
    ax.plot(budgets, v_small, marker="o", label=r"$v(\mathrm{commit\_small})$",
            color="#1f3a5f", linewidth=1.5)
    ax.plot(budgets, v_large, marker="s", label=r"$v(\mathrm{commit\_large})$",
            color="#a23b3b", linewidth=1.5)
            
    # Shade the background where commit_small is strategically superior
    for index, b in enumerate(budgets):
        if v_small[index] > v_large[index]:
            ax.axvspan(b - 0.5, b + 0.5, color="#1f3a5f", alpha=0.06)
            
    ax.set_xlabel(r"MIN's token budget $b$ at $\mathrm{Max}(H, b)$")
    ax.set_ylabel("Value of action")
    ax.set_title("Optimal MAX action by MIN's budget (shaded: MAX prefers small)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xticks(budgets)
    
    fig.savefig(FIG / "fig_strategy_threshold.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_strategy_threshold.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.3.2: Phase diagram for the joint blocked-success sweep
# ---------------------------------------------------------------------------

def fig_phase_blocked():
    """
    Creates a contour plot showing which action MAX prefers at Max(H, 6)
    across a grid of small-block vs large-block success probabilities.
    """
    p_absorb = 0.20
    sb_grid = np.linspace(0.0, 0.5, 41)
    lb_grid = np.linspace(0.0, 0.6, 49)
    
    # Matrices to hold the results for the heatmap
    prefer_large = np.zeros((len(sb_grid), len(lb_grid)), dtype=int)
    W_grid = np.zeros((len(sb_grid), len(lb_grid)))

    # Sweep the 2D space
    for a, sb in enumerate(sb_grid):
        for b, lb in enumerate(lb_grid):
            # The remaining probability goes to 'continue'
            p = Params(
                p_avg_SB=(float(sb), float(1 - p_absorb - sb), p_absorb),
                p_avg_LB=(float(lb), float(1 - p_absorb - lb), p_absorb),
            )
            g = GameGraph.build(p)
            si = strategy_iteration(g)
            
            # Start vertex is 0
            edge_dict = {}
            for action, target_idx in g.out_max[0]:
                edge_dict[action] = target_idx
                
            v_small = float(si.v[edge_dict["commit_small"]])
            v_large = float(si.v[edge_dict["commit_large"]])
            
            # Record preference (1 for large, 0 for small) and total value W
            if v_large > v_small:
                prefer_large[a, b] = 1 
            else:
                prefer_large[a, b] = 0
                
            W_grid[a, b] = float(si.v[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Plot 1: The Phase Diagram (Binary boundary)
    ax1.contourf(lb_grid, sb_grid, prefer_large, levels=[-0.5, 0.5, 1.5],
                 colors=["#dde2ec", "#f5d1c8"])
    ax1.contour(lb_grid, sb_grid, prefer_large.astype(float),
                     levels=[0.5], colors="black", linewidths=1.2)
                     
    # Plot the default project parameters (0.05, 0.30)
    ax1.plot(0.30, 0.05, "k*", markersize=10, label="base (0.05, 0.30)")
    
    ax1.set_xlabel(r"$P(\mathrm{succeed} \mid \mathrm{large},\, \mathrm{blocked})$")
    ax1.set_ylabel(r"$P(\mathrm{succeed} \mid \mathrm{small},\, \mathrm{blocked})$")
    ax1.set_title("MAX's optimal action at $\\mathrm{Max}(H, 6)$")
    ax1.text(0.05, 0.42, "MAX: commit_small", fontsize=9, ha="left")
    ax1.text(0.42, 0.05, "MAX: commit_large", fontsize=9, ha="right")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)

    # Plot 2: The Continuous Heatmap of W
    im = ax2.imshow(W_grid, origin="lower", aspect="auto",
                    extent=[lb_grid[0], lb_grid[-1], sb_grid[0], sb_grid[-1]],
                    cmap="viridis")
    ax2.plot(0.30, 0.05, "w*", markersize=10)
    ax2.set_xlabel(r"$P(\mathrm{succeed} \mid \mathrm{large},\, \mathrm{blocked})$")
    ax2.set_ylabel(r"$P(\mathrm{succeed} \mid \mathrm{small},\, \mathrm{blocked})$")
    ax2.set_title("$W(\\mathrm{Max}(H, 6))$")
    fig.colorbar(im, ax=ax2, shrink=0.85)

    fig.savefig(FIG / "fig_phase_blocked.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_phase_blocked.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.4: Algorithm scaling as graph grows
# ---------------------------------------------------------------------------

def fig_scaling():
    budgets = [2, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100]
    
    n_vertices = []
    n_edges = []
    si_outer_iters = []
    time_si = []
    time_vi = []
    
    for B in budgets:
        g = GameGraph.build(Params(B=B))
        
        # Profile SI
        t0 = time.perf_counter()
        si = strategy_iteration(g)
        t_si = time.perf_counter() - t0
        
        # Profile VI
        t0 = time.perf_counter()
        vi = value_iteration(g, eps=1e-9)
        t_vi = time.perf_counter() - t0
        
        n_vertices.append(g.n())
        n_edges.append(g.edge_count())
        si_outer_iters.append(si.outer_iterations)
        time_si.append(t_si * 1000) # Convert to ms
        time_vi.append(t_vi * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    
    # Plot 1: Graph sizes and SI iterations
    ax1.plot(budgets, n_vertices, marker="o", label="|V|", color="#1f3a5f")
    ax1.plot(budgets, n_edges, marker="s", label="|E|", color="#a23b3b")
    ax1.plot(budgets, si_outer_iters, marker="^", label="SI outer iterations", color="#3f8b3a")
    ax1.set_xlabel("$B$")
    ax1.set_ylabel("count")
    ax1.set_title("Graph size and SI outer iterations vs. $B$")
    ax1.legend(loc="upper left", frameon=False)

    # Plot 2: Wall-clock time comparison
    ax2.plot(budgets, time_si, marker="o", label="Strategy iteration", color="#1f3a5f")
    ax2.plot(budgets, time_vi, marker="s", label=r"Value iteration ($\varepsilon = 10^{-9}$)",
             color="#a23b3b")
    ax2.set_xlabel("$B$")
    ax2.set_ylabel("wall time (ms)")
    ax2.set_yscale("log")
    ax2.set_title("Solve time vs. $B$")
    ax2.legend(loc="upper left", frameon=False)
    
    fig.savefig(FIG / "fig_scaling.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_scaling.pdf'}")


# ---------------------------------------------------------------------------
# Section 7.1: Histogram of SI iteration counts from random initial policies
# ---------------------------------------------------------------------------

def fig_convergence_histogram():
    path = RES / "convergence_random.json"
    if not path.exists():
        print(f"  skip (missing {path}, run convergence_random.py first)")
        return
        
    with open(path) as f:
        data = json.load(f)
        
    outers = data["outers_raw"]
    inners = data["inners_raw"]
    evals = data["evals_raw"]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    
    # Map the data to the three subplots
    plot_definitions = [
        (outers, "Outer (MAX) switches"),
        (inners, "Inner (MIN) switches total"),
        (evals, "Linear solves")
    ]
    
    for index in range(3):
        ax = axes[index]
        dataset = plot_definitions[index][0]
        name = plot_definitions[index][1]
        
        ax.hist(dataset, bins=range(min(dataset), max(dataset) + 2),
                color="#1f3a5f", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(name)
        ax.set_title(f"median {int(np.median(dataset))}, max {max(dataset)}")
        
    axes[0].set_ylabel("count")
    fig.suptitle(f"SI iteration counts over {data['n_trials']} random initial policies")
    
    fig.savefig(FIG / "fig_convergence_histogram.pdf")
    plt.close(fig)
    print(f"  wrote {FIG / 'fig_convergence_histogram.pdf'}")


def main():
    print("Generating figures...")
    fig_convergence()
    fig_W_vs_B()
    fig_W_vs_pH()
    fig_strategy_threshold()
    fig_phase_blocked()
    fig_scaling()
    fig_convergence_histogram()
    print("Done.")


if __name__ == "__main__":
    main()