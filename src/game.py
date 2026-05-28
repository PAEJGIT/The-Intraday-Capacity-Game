"""
The Capacity Game — Game graph construction.

Constructs the labelled SSG G_trade from the process algebra L_SSG by finite
unfolding starting from Max(H, B). Vertices are closed terms; the partition
into V_max, V_min, V_avg, V_0, V_1 is read off the head operator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class VertexKind(Enum):
    MAX = "max"
    MIN = "min"
    AVG = "avg"
    S0 = "S0"
    S1 = "S1"


# A vertex is represented as a tuple: (Name, (Parameters))
Vertex = Tuple[str, Tuple]


@dataclass(frozen=True)
class Params:
    """Tunable parameters of the trading model based on the project scenario."""
    B: int = 6                # Max token budget
    cost_small: int = 3       # Cost to block a small trade
    cost_large: int = 2       # Cost to block a large trade
    regen: int = 1            # Tokens regenerated when passing

    # AVG distributions: (succeed, continue, absorb)
    p_avg_SP: Tuple[float, float, float] = (0.70, 0.10, 0.20)
    p_avg_SB: Tuple[float, float, float] = (0.05, 0.75, 0.20)
    p_avg_LP: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    p_avg_LB: Tuple[float, float, float] = (0.30, 0.50, 0.20)
    p_avg_L:  Tuple[float, float, float] = (0.00, 0.80, 0.20)

    # Spread re-sample probability on continue
    p_H: float = 0.5

    def __post_init__(self):
        # Sanity check to ensure probabilities sum to 1.0
        distribution_names = ["p_avg_SP", "p_avg_SB", "p_avg_LP", "p_avg_LB", "p_avg_L"]
        for name in distribution_names:
            probabilities = getattr(self, name)
            total = sum(probabilities)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"{name} does not sum to 1.0! It sums to {total}")


def kind(v: Vertex) -> VertexKind:
    """Map a vertex to its corresponding SSG partition class."""
    name = v[0]
    
    if name == "S0":
        return VertexKind.S0
    elif name == "S1":
        return VertexKind.S1
    elif name == "Max":
        return VertexKind.MAX
    elif name == "MinS" or name == "MinL":
        return VertexKind.MIN
    elif name.startswith("Avg"):
        return VertexKind.AVG
    else:
        raise ValueError(f"Unknown vertex process name: {name}")


def regen(b: int, p: Params) -> int:
    """Calculate the budget after regeneration, capped at Max B."""
    new_budget = b + p.regen
    if new_budget > p.B:
        return p.B
    return new_budget


def successors(v: Vertex, p: Params):
    """
    Returns outgoing transitions from a vertex based on Section 4.3 rules.
    MAX/MIN returns: list of (action, successor_vertex)
    AVG returns: list of (action, probability, successor_vertex)
    """
    name = v[0]
    args = v[1]

    # Sinks have no outgoing edges
    if name == "S0" or name == "S1":
        return []

    # MAX, the trader
    if name == "Max":
        sigma = args[0]
        b = args[1]
        
        if sigma == "H":
            # High spread: Trader can choose trade size
            return [
                ("commit_small", ("MinS", (b,))),
                ("commit_large", ("MinL", (b,))),
            ]
        else:
            # Low spread: Forced pass
            return [
                ("low_resolve", ("AvgL", (b,)))
            ]

    # MIN responding to a small trade
    if name == "MinS":
        b = args[0]
        if b >= p.cost_small:
            # MIN has enough budget to block
            return [
                ("block_small", ("AvgSB", (b - p.cost_small,))),
                ("pass_small",  ("AvgSP", (regen(b, p),))),
            ]
        else:
            # Forced to pass
            return [
                ("pass_small", ("AvgSP", (regen(b, p),)))
            ]

    # MIN responding to a large trade
    if name == "MinL":
        b = args[0]
        if b >= p.cost_large:
            return [
                ("block_large", ("AvgLB", (b - p.cost_large,))),
                ("pass_large",  ("AvgLP", (regen(b, p),))),
            ]
        else:
            return [
                ("pass_large", ("AvgLP", (regen(b, p),)))
            ]

    # AVG, the market resolution
    if name.startswith("Avg"):
        b = args[0]
        
        # Look up the correct probability distribution
        if name == "AvgSP":
            p_succ, p_cont, p_abs = p.p_avg_SP
        elif name == "AvgSB":
            p_succ, p_cont, p_abs = p.p_avg_SB
        elif name == "AvgLP":
            p_succ, p_cont, p_abs = p.p_avg_LP
        elif name == "AvgLB":
            p_succ, p_cont, p_abs = p.p_avg_LB
        elif name == "AvgL":
            p_succ, p_cont, p_abs = p.p_avg_L
        else:
            raise ValueError(f"Unknown AVG node: {name}")

        transitions = []
        if p_succ > 0:
            transitions.append(("succeed", p_succ, ("S1", ())))
            
        if p_cont > 0:
            # Split the continue probability between High and Low spread rounds
            prob_high = p_cont * p.p_H
            prob_low = p_cont * (1.0 - p.p_H)
            transitions.append(("continue", prob_high, ("Max", ("H", b))))
            transitions.append(("continue", prob_low,  ("Max", ("L", b))))
            
        if p_abs > 0:
            transitions.append(("absorb", p_abs, ("S0", ())))
            
        return transitions

    raise ValueError(f"Unknown vertex name {name}")


@dataclass
class GameGraph:
    """Materialised game graph built from the L-SSG rules"""
    params: Params
    vertices: List[Vertex]                       
    idx: Dict[Vertex, int]                       
    kinds: List[VertexKind]                      

    out_max: Dict[int, List[Tuple[str, int]]] = field(default_factory=dict)
    out_min: Dict[int, List[Tuple[str, int]]] = field(default_factory=dict)
    out_avg: Dict[int, List[Tuple[str, float, int]]] = field(default_factory=dict)

    s0_idx: int = -1
    s1_idx: int = -1

    @classmethod
    def build(cls, p: Params, start: Vertex = None) -> "GameGraph":
        """Unfold the reachable state space using a worklist algorithm"""
        if start is None:
            start = ("Max", ("H", p.B))

        vertices = []
        idx = {}
        kinds = []
        
        out_max = {}
        out_min = {}
        out_avg = {}

        # Helper to track discovered vertices
        def get_or_create_idx(v: Vertex) -> int:
            if v in idx:
                return idx[v]
            
            new_idx = len(vertices)
            vertices.append(v)
            idx[v] = new_idx
            kinds.append(kind(v))
            return new_idx

        # Initialize the worklist
        start_idx = get_or_create_idx(start)
        worklist = [start_idx]

        # Process the graph until all reachable states are found
        while len(worklist) > 0:
            current_idx = worklist.pop()
            v = vertices[current_idx]
            v_kind = kinds[current_idx]

            # Sinks have no successors, so skip them
            if v_kind == VertexKind.S0 or v_kind == VertexKind.S1:
                continue

            transitions = successors(v, p)

            if v_kind == VertexKind.MAX:
                edges = []
                for action, target_vertex in transitions:
                    is_new = target_vertex not in idx
                    target_idx = get_or_create_idx(target_vertex)
                    if is_new:
                        worklist.append(target_idx)
                    edges.append((action, target_idx))
                out_max[current_idx] = edges

            elif v_kind == VertexKind.MIN:
                edges = []
                for action, target_vertex in transitions:
                    is_new = target_vertex not in idx
                    target_idx = get_or_create_idx(target_vertex)
                    if is_new:
                        worklist.append(target_idx)
                    edges.append((action, target_idx))
                out_min[current_idx] = edges

            elif v_kind == VertexKind.AVG:
                edges = []
                for action, probability, target_vertex in transitions:
                    is_new = target_vertex not in idx
                    target_idx = get_or_create_idx(target_vertex)
                    if is_new:
                        worklist.append(target_idx)
                    edges.append((action, probability, target_idx))
                
                # Merge probabilities if they point to the exact same successor
                merged_transitions = {}
                for action, prob, target_idx in edges:
                    if target_idx in merged_transitions:
                        old_action, old_prob = merged_transitions[target_idx]
                        merged_transitions[target_idx] = (old_action + "+" + action, old_prob + prob)
                    else:
                        merged_transitions[target_idx] = (action, prob)
                
                # Format back to list
                final_avg_edges = []
                for target_idx, data in merged_transitions.items():
                    final_avg_edges.append((data[0], data[1], target_idx))
                    
                out_avg[current_idx] = final_avg_edges

        # build the final object
        g = cls(
            params=p,
            vertices=vertices,
            idx=idx,
            kinds=kinds,
            out_max=out_max,
            out_min=out_min,
            out_avg=out_avg,
        )
        
        # save sink indices for easy access during the math operations
        if ("S0", ()) in idx:
            g.s0_idx = idx[("S0", ())]
        if ("S1", ()) in idx:
            g.s1_idx = idx[("S1", ())]
            
        return g

    # Utility functions to query the graph
    def n(self) -> int:
        return len(self.vertices)

    def label(self, i: int) -> str:
        v = self.vertices[i]
        name = v[0]
        args = v[1]
        
        if name == "S0" or name == "S1":
            return name
            
        if name == "Max":
            sigma = args[0]
            b = args[1]
            return f"Max({sigma},{b})"
            
        return f"{name}({args[0]})"

    def transient_indices(self) -> List[int]:
        """return all vertices that are not sinks"""
        result = []
        for i in range(len(self.kinds)):
            if self.kinds[i] != VertexKind.S0 and self.kinds[i] != VertexKind.S1:
                result.append(i)
        return result

    def max_indices(self) -> List[int]:
        result = []
        for i in range(len(self.kinds)):
            if self.kinds[i] == VertexKind.MAX:
                result.append(i)
        return result

    def min_indices(self) -> List[int]:
        result = []
        for i in range(len(self.kinds)):
            if self.kinds[i] == VertexKind.MIN:
                result.append(i)
        return result

    def avg_indices(self) -> List[int]:
        result = []
        for i in range(len(self.kinds)):
            if self.kinds[i] == VertexKind.AVG:
                result.append(i)
        return result

    def vertex_table(self) -> Dict[str, int]:
        """provides counts by vertex class, used for reporting"""
        counts = {
            "V_max": 0, "V_min": 0, "V_avg": 0, "V_0": 0, "V_1": 0, "total": 0
        }
        for k in self.kinds:
            if k == VertexKind.MAX: counts["V_max"] += 1
            elif k == VertexKind.MIN: counts["V_min"] += 1
            elif k == VertexKind.AVG: counts["V_avg"] += 1
            elif k == VertexKind.S0: counts["V_0"] += 1
            elif k == VertexKind.S1: counts["V_1"] += 1
            counts["total"] += 1
        return counts

    def edge_count(self) -> int:
        count = 0
        for edges in self.out_max.values():
            count += len(edges)
        for edges in self.out_min.values():
            count += len(edges)
        for edges in self.out_avg.values():
            count += len(edges)
        return count

    def branching_max_indices(self) -> List[int]:
        """return MAX vertices with more than one choice"""
        branching = []
        for i in self.max_indices():
            if i in self.out_max and len(self.out_max[i]) > 1:
                branching.append(i)
        return branching

    def branching_min_indices(self) -> List[int]:
        branching = []
        for i in self.min_indices():
            if i in self.out_min and len(self.out_min[i]) > 1:
                branching.append(i)
        return branching