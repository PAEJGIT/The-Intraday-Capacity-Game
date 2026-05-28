"""
Experiments.

runs the SSG sweeps and generates the raw data

Outputs:
  - JSON tables saved in results/
  - Terminal prints suitable for transcribing into LaTeX tables
"""

from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import Dict
import numpy as np

from src.game import GameGraph, Params
from src.algorithms import strategy_iteration, value_iteration, value_at_start

# Setup paths for outputs
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Ensure output directories exist
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def extract_target_value(g: GameGraph, edges, action_name, value_vector):
    """
    Looks through the outgoing edges of a vertex and returns the computed 
    value of the specific action we are looking for.
    """
    for action, target_index in edges:
        if action == action_name:
            return float(value_vector[target_index])
            
    raise ValueError(f"Action '{action_name}' was not found in the provided edges.")


# ---------------------------------------------------------------------------
# Experiments (Matches Chapter 7 Structure)
# ---------------------------------------------------------------------------

def experiment_convergence() -> Dict:
    """Matches Section 7.1: Compares SI and VI convergence rates."""
    print("=" * 70)
    print("7.1  Convergence")
    print("=" * 70)
    
    # Initialize the base instance
    p = Params()
    g = GameGraph.build(p)
    
    # Run Strategy Iteration (Exact rational fixed point)
    si = strategy_iteration(g)
    
    vi_tols = [1e-3, 1e-6, 1e-9, 1e-12]
    vi_runs = []
    
    # Run Value Iteration at progressively tighter tolerances
    for eps in vi_tols:
        vi = value_iteration(g, eps=eps)
        
        # Manually calculate the max absolute difference between SI and VI
        max_diff = 0.0
        for i in range(len(vi.v)):
            diff = abs(vi.v[i] - si.v[i])
            if diff > max_diff:
                max_diff = diff
                
        vi_runs.append({
            "eps": eps,
            "iterations": vi.iterations,
            "residual": vi.final_residual,
            "max_abs_diff_from_W": max_diff,
            "W_at_start": float(vi.v[0]),
        })
        print(f"  VI eps={eps:.0e}: {vi.iterations:5d} iters,  |VI - W|_inf = {max_diff:.2e}")

    print(f"  SI: {si.outer_iterations} outer / {si.inner_iterations_total} inner switches ({si.eval_calls} linear solves)")
    print(f"  W(Max(H,6)) = {value_at_start(g, si.v):.10f}")

    # Build the graph statistics for reporting
    graph_stats = g.vertex_table()
    graph_stats["edges"] = g.edge_count()
    
    return {
        "graph": graph_stats,
        "W_at_start": float(si.v[0]),
        "strategy_iteration": {
            "outer_iterations": si.outer_iterations,
            "inner_iterations_total": si.inner_iterations_total,
            "inner_per_outer": si.inner_iterations_per_outer,
            "eval_calls": si.eval_calls,
        },
        "value_iteration": vi_runs,
    }


def experiment_base_strategy() -> Dict:
    """Matches Section 7.2: Analyzes the threshold behavior on the base instance."""
    print("\n" + "=" * 70)
    print("7.2  Optimal strategies on the base instance")
    print("=" * 70)
    
    p = Params()
    g = GameGraph.build(p)
    si = strategy_iteration(g)

    max_rows = []
    print(f"  {'Vertex':12s} {'Optimal':16s} {'V(small)':>10s} {'V(large)':>10s}  Gap")
    
    # Walk backwards through the budget levels to see where the MAX strategy flips
    for b in range(p.B, -1, -1):
        v_max = ("Max", ("H", b))
        if v_max not in g.idx:
            continue
            
        i = g.idx[v_max]
        edges = g.out_max[i]
        
        v_small = extract_target_value(g, edges, "commit_small", si.v)
        v_large = extract_target_value(g, edges, "commit_large", si.v)
        
        if v_small > v_large:
            chosen = "commit_small"
        else:
            chosen = "commit_large"
            
        max_rows.append({"b": b, "v_small": v_small, "v_large": v_large, "optimal": chosen})
        gap = v_large - v_small
        print(f"  Max(H,{b})       {chosen:16s} {v_small:10.4f} {v_large:10.4f}  {gap:+.4f}")

    # Inspect MIN strategy to confirm MIN always blocks when budget allows
    print(f"\n  {'Vertex':14s} {'Optimal':16s} {'V(block)':>10s} {'V(pass)':>10s}")
    min_rows = []
    
    for tag in ["MinS", "MinL"]:
        for b in range(p.B, -1, -1):
            v = (tag, (b,))
            if v not in g.idx:
                continue
                
            i = g.idx[v]
            edges = g.out_min[i]
            
            # If MIN only has one edge, it's a forced pass due to low budget
            if len(edges) <= 1:
                continue
                
            # Setup keys based on whether MIN is responding to small or large
            if tag == "MinS":
                block_key = "block_small"
                pass_key = "pass_small"
            else:
                block_key = "block_large"
                pass_key = "pass_large"
                
            v_block = extract_target_value(g, edges, block_key, si.v)
            v_pass = extract_target_value(g, edges, pass_key, si.v)
            
            # MIN wants to minimize the value
            if v_block < v_pass:
                chosen = block_key
            else:
                chosen = pass_key
                
            min_rows.append({"tag": tag, "b": b, "v_block": v_block, "v_pass": v_pass, "optimal": chosen})
            print(f"  {tag}({b})        {chosen:16s} {v_block:10.4f} {v_pass:10.4f}")

    # Save all values for reference
    all_vals = {}
    for i in range(g.n()):
        all_vals[g.label(i)] = float(si.v[i])

    return {
        "max_strategy": max_rows,
        "min_strategy": min_rows,
        "all_values": all_vals,
    }


def experiment_sensitivity_B() -> Dict:
    """Matches Section 7.3.1: Sweeps the maximum budget capacity B."""
    print("\n" + "=" * 70)
    print("7.3a  Sensitivity to MIN's budget capacity B")
    print("=" * 70)
    
    rows = []
    print(f"  {'B':>3s} {'|V|':>5s} {'|E|':>5s} {'W(start)':>10s}  {'outer':>5s} {'inner':>5s} {'VI@1e-9':>8s}")
    
    # Test budget capacities from 0 up to 10
    for B in range(0, 11):
        p = Params(B=B)
        g = GameGraph.build(p)
        
        si = strategy_iteration(g)
        vi = value_iteration(g, eps=1e-9)
        
        rows.append({
            "B": B,
            "n_vertices": g.n(),
            "n_edges": g.edge_count(),
            "W_start": float(si.v[0]),
            "outer_iter": si.outer_iterations,
            "inner_iter": si.inner_iterations_total,
            "vi_iter": vi.iterations,
        })
        print(f"  {B:>3d} {g.n():>5d} {g.edge_count():>5d} {float(si.v[0]):>10.4f}  {si.outer_iterations:>5d} {si.inner_iterations_total:>5d} {vi.iterations:>8d}")
        
    return {"sweep_B": rows}


def experiment_sensitivity_blocked() -> Dict:
    """Matches Section 7.3.2: 2D sweep over block success probabilities."""
    print("\n" + "=" * 70)
    print("7.3b  Sensitivity to blocked-trade success probabilities")
    print("=" * 70)

    p_absorb = 0.20
    rows = []
    
    # Create explicit lists for JSON serialization
    sb_grid = np.linspace(0.0, 0.5, 11).tolist()
    lb_grid = np.linspace(0.0, 0.6, 13).tolist()

    print(f"  Joint sweep: P(succeed|small,B) in [{sb_grid[0]}, {sb_grid[-1]}], P(succeed|large,B) in [{lb_grid[0]}, {lb_grid[-1]}]")
    
    n_large = 0
    total_points = 0

    for sb in sb_grid:
        for lb in lb_grid:
            total_points += 1
            
            # The remaining probability goes to 'continue'
            p_avg_SB = (float(sb), float(1.0 - p_absorb - sb), p_absorb)
            p_avg_LB = (float(lb), float(1.0 - p_absorb - lb), p_absorb)
            
            p = Params(p_avg_SB=p_avg_SB, p_avg_LB=p_avg_LB)
            g = GameGraph.build(p)
            si = strategy_iteration(g)

            # Node 0 is always Max(H, B)
            start_edges = g.out_max[0]
            
            v_small = extract_target_value(g, start_edges, "commit_small", si.v)
            v_large = extract_target_value(g, start_edges, "commit_large", si.v)
            
            prefers_large = (v_large > v_small)
            if prefers_large:
                n_large += 1
                
            rows.append({
                "p_succ_SB": float(sb),
                "p_succ_LB": float(lb),
                "W_start": float(si.v[0]),
                "v_small": v_small,
                "v_large": v_large,
                "max_prefers_large": prefers_large,
            })

    print(f"  Of {total_points} parameter points: MAX prefers large at {n_large}, small at {total_points - n_large}.")

    return {
        "sweep_blocked_success": rows,
        "sb_grid": sb_grid,
        "lb_grid": lb_grid
    }


def experiment_sensitivity_pH() -> Dict:
    """Matches Section 7.3.3: Sensitivity to spread persistence."""
    print("\n" + "=" * 70)
    print("7.3c  Sensitivity to P(spread = H) on continue")
    print("=" * 70)
    
    rows = []
    # Test probabilities from 0.0 to 1.0 in 5% increments
    grid = np.linspace(0.0, 1.0, 21)
    
    print(f"  {'p_H':>5s} {'W(start)':>10s}  outer  inner")
    for pH in grid:
        p = Params(p_H=float(pH))
        g = GameGraph.build(p)
        si = strategy_iteration(g)
        
        rows.append({
            "p_H": float(pH),
            "W_start": float(si.v[0]),
            "outer_iter": si.outer_iterations,
            "inner_iter": si.inner_iterations_total,
        })
        print(f"  {pH:>5.2f} {float(si.v[0]):>10.4f}  {si.outer_iterations:>5d}  {si.inner_iterations_total:>5d}")
        
    return {"sweep_pH": rows}


def experiment_scaling() -> Dict:
    """Matches Section 7.4: Tests algorithmic scaling as the graph grows."""
    print("\n" + "=" * 70)
    print("7.5  Scaling: graph size and SI cost as B grows")
    print("=" * 70)
    
    rows = []
    print(f"  {'B':>3s} {'|V|':>5s} {'|E|':>5s} {'W(start)':>10s} {'outer':>5s} {'inner':>5s} {'eval':>5s} {'t_si(ms)':>10s} {'t_vi(ms)':>10s}")
    
    # Push B up to 50 to observe the exponential bound behavior
    for B in [2, 4, 6, 8, 10, 15, 20, 30, 50]:
        p = Params(B=B)
        g = GameGraph.build(p)
        
        # Profile Strategy Iteration
        t_start = time.perf_counter()
        si = strategy_iteration(g)
        t_si = (time.perf_counter() - t_start) * 1000
        
        # Profile Value Iteration
        t_start = time.perf_counter()
        vi = value_iteration(g, eps=1e-9)
        t_vi = (time.perf_counter() - t_start) * 1000
        
        rows.append({
            "B": B,
            "n_vertices": g.n(),
            "n_edges": g.edge_count(),
            "W_start": float(si.v[0]),
            "outer_iter": si.outer_iterations,
            "inner_iter": si.inner_iterations_total,
            "eval_calls": si.eval_calls,
            "t_si_ms": t_si,
            "t_vi_ms": t_vi,
            "vi_iter": vi.iterations,
        })
        print(f"  {B:>3d} {g.n():>5d} {g.edge_count():>5d} {float(si.v[0]):>10.4f} {si.outer_iterations:>5d} {si.inner_iterations_total:>5d} {si.eval_calls:>5d} {t_si:>10.2f} {t_vi:>10.2f}")
        
    return {"scaling": rows}


def main():
    print("Running Intraday Capacity Game Experiments...")
    
    results = {}
    results["7.1_convergence"] = experiment_convergence()
    results["7.2_base_strategy"] = experiment_base_strategy()
    results["7.3a_sweep_B"] = experiment_sensitivity_B()
    results["7.3b_sweep_blocked"] = experiment_sensitivity_blocked()
    results["7.3c_sweep_pH"] = experiment_sensitivity_pH()
    results["7.5_scaling"] = experiment_scaling()

    # Save everything to a master JSON file
    out_path = RESULTS / "all.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll experiments complete. Wrote data to {out_path}")

if __name__ == "__main__":
    main()