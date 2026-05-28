"""
convergence study: SI from many random initial policies.

The default initial policy turned out to be near-optimal for MIN on the base instance,
producing 0 inner switches. 

To get a non-misleading picture of SI behaviour, i
run from random initial policies and report the distribution of iteration counts.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np

from src.game import GameGraph, Params
from src.algorithms import evaluate, SIResult

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def random_initial_max(g: GameGraph, rng: random.Random) -> dict:
    """assigns a completely random available successor to each MAX vertex."""
    policy = {}
    for i in g.max_indices():
        edges = g.out_max[i]
        choices = []
        for action, target in edges:
            choices.append(target)
        policy[i] = rng.choice(choices)
    return policy


def random_initial_min(g: GameGraph, rng: random.Random) -> dict:
    """Assigns a completely random available successor to each MIN vertex."""
    policy = {}
    for i in g.min_indices():
        edges = g.out_min[i]
        choices = []
        for action, target in edges:
            choices.append(target)
        policy[i] = rng.choice(choices)
    return policy


def min_response_random_start(g: GameGraph, A: dict, I_start: dict) -> tuple:
    """
    Computes MIN's best response against policy A, starting explicitly from I_start.
    """
    I = dict(I_start)
    n_inner = 0
    n_eval = 0
    
    while True:
        v = evaluate(g, A, I)
        n_eval += 1
        improved = False
        
        for i in g.min_indices():
            edges = g.out_min[i]
            if len(edges) <= 1:
                continue
                
            current_target = I[i]
            
            for action, potential_target in edges:
                if v[potential_target] < v[current_target] - 1e-12:
                    I[i] = potential_target
                    n_inner += 1
                    improved = True
                    break
                    
            if improved:
                break
                
        if not improved:
            return I, n_inner, n_eval, v


def strategy_iteration_random(g: GameGraph, A_init: dict, I_init: dict) -> SIResult:
    """
    A full SI run initialized with random policies for BOTH players,
    used to test worst-case iteration counts.
    """
    A = dict(A_init)
    
    # First response pass
    I, inner_first, eval_first, v = min_response_random_start(g, A, I_init)
    
    inner_per_outer = [inner_first]
    n_eval = eval_first
    outer = 0

    while True:
        improved = False
        for i in g.max_indices():
            edges = g.out_max[i]
            if len(edges) <= 1:
                continue
                
            current_target = A[i]
            
            for action, potential_target in edges:
                if v[potential_target] > v[current_target] + 1e-12:
                    A[i] = potential_target
                    improved = True
                    outer += 1
                    break
                    
            if improved:
                break
                
        if not improved:
            return SIResult(
                v=v, 
                A=A, 
                I=I,
                outer_iterations=outer,
                inner_iterations_total=sum(inner_per_outer),
                inner_iterations_per_outer=inner_per_outer,
                eval_calls=n_eval,
            )
            
        I, inner_n, eval_n, v = min_response_random_start(g, A, I)
        inner_per_outer.append(inner_n)
        n_eval += eval_n


def run(n_trials: int = 500, seed: int = 42):
    rng = random.Random(seed)
    p = Params()
    g = GameGraph.build(p)

    outers = []
    inners = []
    evals = []
    W_reference = None
    
    for trial in range(n_trials):
        A0 = random_initial_max(g, rng)
        I0 = random_initial_min(g, rng)
        
        si = strategy_iteration_random(g, A0, I0)
        
        outers.append(si.outer_iterations)
        inners.append(si.inner_iterations_total)
        evals.append(si.eval_calls)
        
        # Verify that all random starts converge to the identical game value
        if W_reference is None:
            W_reference = float(si.v[0])
        else:
            diff = abs(W_reference - float(si.v[0]))
            if diff > 1e-9:
                raise ValueError(f"Trial {trial} converged to a different value!")

    stats = {
        "n_trials": n_trials,
        "W_at_start": W_reference,
        "outer_iterations": {
            "min": int(min(outers)),
            "max": int(max(outers)),
            "mean": float(np.mean(outers)),
            "median": float(np.median(outers)),
            "stdev": float(np.std(outers)),
        },
        "inner_iterations_total": {
            "min": int(min(inners)),
            "max": int(max(inners)),
            "mean": float(np.mean(inners)),
            "median": float(np.median(inners)),
            "stdev": float(np.std(inners)),
        },
        "eval_calls": {
            "min": int(min(evals)),
            "max": int(max(evals)),
            "mean": float(np.mean(evals)),
            "median": float(np.median(evals)),
            "stdev": float(np.std(evals)),
        },
    }

    print("=" * 70)
    print(f"SI convergence over {n_trials} random initial-policy pairs")
    print("=" * 70)
    print(f"  W(Max(H,6)) = {W_reference:.10f}  (identical across all trials)")
    print(f"  Outer iterations:        min={stats['outer_iterations']['min']}  median={stats['outer_iterations']['median']:.1f}  mean={stats['outer_iterations']['mean']:.2f}  max={stats['outer_iterations']['max']}")
    print(f"  Inner iterations total:  min={stats['inner_iterations_total']['min']}  median={stats['inner_iterations_total']['median']:.1f}  mean={stats['inner_iterations_total']['mean']:.2f}  max={stats['inner_iterations_total']['max']}")
    print(f"  Linear solves:           min={stats['eval_calls']['min']}  median={stats['eval_calls']['median']:.1f}  mean={stats['eval_calls']['mean']:.2f}  max={stats['eval_calls']['max']}")

    # Combine stats and raw data for plotting
    output_data = {}
    for key in stats:
        output_data[key] = stats[key]
    output_data["outers_raw"] = outers
    output_data["inners_raw"] = inners
    output_data["evals_raw"] = evals

    out_file = RESULTS / "convergence_random.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nWrote {out_file}")
    return stats

if __name__ == "__main__":
    run()