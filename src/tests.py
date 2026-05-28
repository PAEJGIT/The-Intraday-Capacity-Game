"""
Sanity tests on the implementation of the Intraday Capacity Game.
Run as: python -m src.tests

These tests verify the structural properties of the game graph and ensure
the fixed-point algorithms behave
"""

from __future__ import annotations

import random
import numpy as np

from src.game import GameGraph, Params, VertexKind
from src.algorithms import (
    strategy_iteration, value_iteration, evaluate, bellman,
)


def test_si_vi_agree():
    """
    Strategy Iteration and Value Iteration must agree 
    on the exact same value vector for the base instance.
    """
    g = GameGraph.build(Params())
    
    si = strategy_iteration(g)
    vi = value_iteration(g, eps=1e-13)
    
    max_difference = float(np.max(np.abs(si.v - vi.v)))
    
    status = "OK" if max_difference < 1e-8 else "FAIL"
    print(f"  SI vs VI:    max |Δ| = {max_difference:.2e}   {status}")
    
    assert max_difference < 1e-8


def test_si_is_fixed_point():
    """
    At termination of Strategy Iteration, the value vector must be a strict 
    fixed point of the Bellman operator T
    """
    g = GameGraph.build(Params())
    si = strategy_iteration(g)
    
    # Apply the Bellman operator once to the resulting vector
    v_applied = bellman(g, si.v)
    
    max_difference = float(np.max(np.abs(v_applied - si.v)))
    
    status = "OK" if max_difference < 1e-9 else "FAIL"
    print(f"  v = T(v):    max |Δ| = {max_difference:.2e}   {status}")
    
    assert max_difference < 1e-9


def test_si_robust_to_initial_policy():
    """
    Because the game is a stopping SSG, Strategy Iteration should converge to 
    the exact same unique fixed point regardless of the initial policy.
    """
    rng = random.Random(0)
    g = GameGraph.build(Params())
    
    # Get the reference baseline
    si_ref = strategy_iteration(g)
    
    for trial in range(50):
        # Build a completely random initial MAX policy manually
        random_max_policy = {}
        for i in g.max_indices():
            edges = g.out_max[i]
            possible_targets = []
            for action, target in edges:
                possible_targets.append(target)
            random_max_policy[i] = rng.choice(possible_targets)
            
        # Run SI with the random start
        si_trial = strategy_iteration(g, A_init=random_max_policy)
        
        diff = float(np.max(np.abs(si_trial.v - si_ref.v)))
        assert diff < 1e-9, f"trial {trial}: diff = {diff}"
        
    print(f"  initial-policy independence (50 trials):  OK")


def test_stopping_property():
    """
    Verifies Lemma 6 / Theorem 7: 
    Every AVG vertex must route to the absorbing 
    sink S0 with strict positive probability to guarantee the game halts.
    """
    g = GameGraph.build(Params())
    
    for i in g.avg_indices():
        prob_to_s0 = 0.0
        
        # Manually sum the probabilities pointing to the 0-sink
        for action, prob, target_idx in g.out_avg[i]:
            if target_idx == g.s0_idx:
                prob_to_s0 += prob
                
        assert prob_to_s0 > 0, f"Vertex {g.label(i)} does not route to S0 (violates stopping condition)"
        
    print(f"  stopping property (every AVG -> S0):  OK")


def test_W_in_unit_interval():
    """
    Probabilities must logically fall between 0 and 1. 
    Checks the entire value vector for bounds.
    """
    g = GameGraph.build(Params())
    si = strategy_iteration(g)
    
    min_val = float(si.v.min())
    max_val = float(si.v.max())
    
    assert min_val >= -1e-12, f"Value dropped below 0: {min_val}"
    assert max_val <= 1.0 + 1e-12, f"Value exceeded 1: {max_val}"
    
    print(f"  v in [0,1]:  OK   (min={min_val:.4f}, max={max_val:.4f})")


def test_two_player_consistency():
    """
    Sanity check: 
    If we artificially force the rival (MIN) to 'always pass', 
    the trader's (MAX) value at Max(H,B) should be strictly greater than or 
    equal to the optimal value where MIN plays adversarially.
    """
    g = GameGraph.build(Params())
    si = strategy_iteration(g)

    optimal_max_policy = si.A

    # Force MIN to always pass: pick the pass edge at every MIN vertex.
    always_pass_min = {}
    for i in g.min_indices():
        edges = g.out_min[i]
        chosen = edges[0][1]
        for action, target in edges:
            if action.startswith("pass"):
                chosen = target
                break
        always_pass_min[i] = chosen

    # Evaluate MAX's optimal policy against the non-adversarial MIN.
    v_passive = evaluate(g, optimal_max_policy, always_pass_min)

    w_adversarial = float(si.v[0])
    w_passive = float(v_passive[0])

    status = "OK" if w_passive >= w_adversarial - 1e-9 else "FAIL"
    print(f"  passive MIN >= adversarial:  {w_passive:.4f} >= {w_adversarial:.4f}   {status}")

    assert w_passive >= w_adversarial - 1e-9


def run_all():
    print("Running sanity tests on the Intraday Capacity Game...")
    test_si_vi_agree()
    test_si_is_fixed_point()
    test_si_robust_to_initial_policy()
    test_stopping_property()
    test_W_in_unit_interval()
    test_two_player_consistency()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()