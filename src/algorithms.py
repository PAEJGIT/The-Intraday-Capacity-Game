"""
Algorithms on the labelled SSG: strategy iteration and value iteration.

Chapter 6
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.game import GameGraph, VertexKind


@dataclass
class SIResult:
    v: np.ndarray
    A: Dict[int, int]                  # MAX policy
    I: Dict[int, int]                  # MIN policy
    outer_iterations: int              
    inner_iterations_total: int        
    inner_iterations_per_outer: List[int]
    eval_calls: int                    


@dataclass
class VIResult:
    v: np.ndarray
    iterations: int
    final_residual: float


def evaluate(g: GameGraph, A: Dict[int, int], I: Dict[int, int]) -> np.ndarray:
    """
    solves the linear system for v^{A,I}. 
    calculates the value vector for a fixed policy pair.
    """
    n = g.n()
    v = np.zeros(n)
    v[g.s1_idx] = 1.0  # Sink 1 has value 1.0

    # Get transient vertices (everything except sinks)
    transient = g.transient_indices()
    
    # need a mapping from the full graph index to the matrix index
    pos = {}
    for matrix_idx, graph_idx in enumerate(transient):
        pos[graph_idx] = matrix_idx
        
    m = len(transient)

    # Q is the transition matrix, r is the immediate reward (chance to hit S1)
    Q = np.zeros((m, m))
    r = np.zeros(m)

    for i in transient:
        matrix_row = pos[i]
        kind = g.kinds[i]
        
        if kind == VertexKind.MAX:
            target = A[i]
            if target == g.s1_idx:
                r[matrix_row] = 1.0
            elif target != g.s0_idx:
                Q[matrix_row, pos[target]] = 1.0
                
        elif kind == VertexKind.MIN:
            target = I[i]
            if target == g.s1_idx:
                r[matrix_row] = 1.0
            elif target != g.s0_idx:
                Q[matrix_row, pos[target]] = 1.0
                
        elif kind == VertexKind.AVG:
            for action, prob, target in g.out_avg[i]:
                if target == g.s1_idx:
                    r[matrix_row] += prob
                elif target != g.s0_idx:
                    Q[matrix_row, pos[target]] += prob

    # Solve the system: (I - Q) * v_t = r
    # This works because the game is stopping (guaranteed to hit a sink).
    identity_matrix = np.eye(m)
    v_transient = np.linalg.solve(identity_matrix - Q, r)

    # Map the solved values back to the full vector
    for i in transient:
        v[i] = v_transient[pos[i]]
        
    return v


def get_initial_max_policy(g: GameGraph) -> Dict[int, int]:
    """Default MAX policy: Pick the first available move."""
    policy = {}
    for i in g.max_indices():
        edges = g.out_max[i]
        first_target = edges[0][1]
        policy[i] = first_target
    return policy


def get_initial_min_policy(g: GameGraph) -> Dict[int, int]:
    """Default MIN policy: Pick the first available move."""
    policy = {}
    for i in g.min_indices():
        edges = g.out_min[i]
        first_target = edges[0][1]
        policy[i] = first_target
    return policy


def min_response(g: GameGraph, A: Dict[int, int], I_init: Optional[Dict[int, int]] = None, tol: float = 1e-12):
    """
    Inner subroutine: compute the MIN-optimal response to a fixed MAX policy.
    This corresponds to Condon's Lemma 4 (single-switch improvement).
    """
    if I_init is not None:
        I = dict(I_init)
    else:
        I = get_initial_min_policy(g)
        
    n_inner = 0
    n_eval = 0

    while True:
        v = evaluate(g, A, I)
        n_eval += 1
        improved = False
        
        # Scan all MIN vertices to see if there is a move that Lowers the value (MIN wants to minimize)
        for i in g.min_indices():
            edges = g.out_min[i]
            if len(edges) <= 1:
                continue
                
            current_target = I[i]
            current_value = v[current_target]
            
            for action, potential_target in edges:
                potential_value = v[potential_target]
                
                # If we find a strictly better move for MIN, switch and break
                if potential_value < current_value - tol:
                    I[i] = potential_target
                    n_inner += 1
                    improved = True
                    break
                    
            if improved:
                # We only switch one vertex at a time to ensure monotonic progress
                break
                
        # If we scanned the whole graph and made no improvements, we are done
        if not improved:
            return I, n_inner, n_eval


def strategy_iteration(g: GameGraph, A_init: Optional[Dict[int, int]] = None, tol: float = 1e-12) -> SIResult:
    """
    strategy iteration
    outer loop improves MAX, inner loop optimally responds with MIN.
    """
    if A_init is not None:
        A = dict(A_init)
    else:
        A = get_initial_max_policy(g)
        
    #get the first MIN response
    I, inner_first, eval_first = min_response(g, A)

    inner_per_outer = [inner_first]
    n_eval = eval_first
    outer = 0

    while True:
        v = evaluate(g, A, I)
        n_eval += 1
        improved = False
        
        # Scan MAX vertices to see if there is a move that Increases the value
        for i in g.max_indices():
            edges = g.out_max[i]
            if len(edges) <= 1:
                continue
                
            current_target = A[i]
            current_value = v[current_target]
            
            for action, potential_target in edges:
                potential_value = v[potential_target]
                
                # Check for strict improvement for MAX
                if potential_value > current_value + tol:
                    A[i] = potential_target
                    improved = True
                    outer += 1
                    break
                    
            if improved:
                break
                
        if not improved:
            # Reached the fixed point!
            return SIResult(
                v=v, 
                A=A, 
                I=I,
                outer_iterations=outer,
                inner_iterations_total=sum(inner_per_outer),
                inner_iterations_per_outer=inner_per_outer,
                eval_calls=n_eval,
            )
            
        # If MAX improved, MIN gets a chance to respond to the new strategy
        I, inner_n, eval_n = min_response(g, A, I_init=I)
        inner_per_outer.append(inner_n)
        n_eval += eval_n


def bellman(g: GameGraph, v: np.ndarray) -> np.ndarray:
    """Do one application of the Bellman operator T."""
    n = g.n()
    out = np.zeros(n)
    out[g.s1_idx] = 1.0

    for i in range(n):
        kind = g.kinds[i]
        
        if kind == VertexKind.S0 or kind == VertexKind.S1:
            continue
            
        elif kind == VertexKind.MAX:
            max_val = -1.0
            for action, target in g.out_max[i]:
                if v[target] > max_val:
                    max_val = v[target]
            out[i] = max_val
            
        elif kind == VertexKind.MIN:
            min_val = 2.0
            for action, target in g.out_min[i]:
                if v[target] < min_val:
                    min_val = v[target]
            out[i] = min_val
            
        elif kind == VertexKind.AVG:
            expected_value = 0.0
            for action, prob, target in g.out_avg[i]:
                expected_value += prob * v[target]
            out[i] = expected_value
            
    return out


def value_iteration(g: GameGraph, eps: float = 1e-10, max_iter: int = 100_000) -> VIResult:
    """Standard Value Iteration until convergence below epsilon."""
    v = np.zeros(g.n())
    
    for k in range(1, max_iter + 1):
        v_new = bellman(g, v)
        
        # Calculate max absolute difference
        max_diff = 0.0
        for i in range(g.n()):
            diff = abs(v_new[i] - v[i])
            if diff > max_diff:
                max_diff = diff
                
        v = v_new
        
        if max_diff < eps:
            return VIResult(v=v, iterations=k, final_residual=max_diff)
            
    return VIResult(v=v, iterations=max_iter, final_residual=max_diff)


def evaluate_policy(g: GameGraph, A: Dict[int, int], I: Dict[int, int]) -> np.ndarray:
    """Exposed convenience wrapper"""
    return evaluate(g, A, I)


def value_at_start(g: GameGraph, v: np.ndarray) -> float:
    """Returns W at Max(H, B) which is always index 0."""
    return float(v[0])