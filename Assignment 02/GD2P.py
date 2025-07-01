from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

def binary_search(values: List[float], target: float) -> float:
    left, right = 0, len(values) - 1
    result = -float('inf')
    
    while left <= right:
        mid = (left + right) // 2
        if values[mid] <= target:
            result = values[mid]
            left = mid + 1
        else:
            right = mid - 1
            
    return result

def gdp(Fj: Dict[Tuple, float], xi: str, Mxj_minus_xi: List[Dict[Any, float]], 
        states: List[Tuple], xi_idx: int) -> Dict[Any, Tuple[float, float]]:
    sorted_utils = {}
    for state in set(s[xi_idx] for s in states):
        relevant_states = [s for s in states if s[xi_idx] == state]
        utils = [Fj.get(s, 0) for s in relevant_states]
        sorted_utils[state] = sorted(utils, reverse=True)
    
    n = len(Mxj_minus_xi)
    m = sum(max(msg.values()) for msg in Mxj_minus_xi)
    
    pruned_ranges = {}
    for si in sorted_utils:
        Vi = sorted_utils[si]
        p = Vi[0] if Vi else 0
        max_state = next((s for s in states if s[xi_idx] == si and Fj.get(s, 0) == p), None)
        if max_state is None:
            continue
        b = 0
        for k, msg in enumerate(Mxj_minus_xi):
            var_idx = [i for i in range(len(states[0])) if i != xi_idx][k]
            b += msg.get(max_state[var_idx], 0)
        t = m - b
        q = binary_search(Vi, p - t)
        pruned_ranges[si] = (p, q)
    
    return pruned_ranges

def gd2p(Fj: Dict[Tuple, float], xi: str, Mxj_minus_xi: List[Dict[Any, float]], 
         states: List[Tuple], xi_idx: int) -> Dict[Any, Tuple[float, float]]:
    sorted_utils = {}
    for state in set(s[xi_idx] for s in states):
        relevant_states = [s for s in states if s[xi_idx] == state]
        utils = [Fj.get(s, 0) for s in relevant_states]
        sorted_utils[state] = sorted(utils, reverse=True)
    
    n = len(Mxj_minus_xi)
    m = sum(max(msg.values()) for msg in Mxj_minus_xi)
    
    pruned_ranges = {}
    for si in sorted_utils:
        Vi = sorted_utils[si]
        p = Vi[0] if Vi else 0
        max_state = next((s for s in states if s[xi_idx] == si and Fj.get(s, 0) == p), None)
        if max_state is None:
            continue
        b = 0
        for k, msg in enumerate(Mxj_minus_xi):
            var_idx = [i for i in range(len(states[0])) if i != xi_idx][k]
            b += msg.get(max_state[var_idx], 0)
        t = m - b
        q = binary_search(Vi, p - t)
        best_value = -float('inf')
        for assignment in states:
            if assignment[xi_idx] != si:
                continue
            utility = Fj.get(assignment, 0.0)
            if q <= utility <= p:
                msg_sum = sum(Mxj_minus_xi[i].get(assignment[vars_idx], 0)
                              for i, vars_idx in enumerate([j for j in range(len(states[0])) if j != xi_idx]))
                total = utility + msg_sum
                if total > best_value:
                    best_value = total
                    q = binary_search(Vi, best_value - m)
        pruned_ranges[si] = (p, q)
    
    return pruned_ranges

class FactorGraph:
    """Represents a DCOP as a factor graph with variable and function nodes."""
    def __init__(self):
        self.variables: Dict[str, Set[Any]] = {}
        self.functions: Dict[str, Tuple[List[str], Dict[Tuple, float]]] = {}
        self.var_to_funcs: Dict[str, Set[str]] = defaultdict(set)
        self.func_to_vars: Dict[str, List[str]] = {}

    def add_variable(self, name: str, domain: Set[Any]):
        """Add a variable node with its domain."""
        self.variables[name] = domain
        self.var_to_funcs[name] = set()

    def add_function(self, name: str, variables: List[str], utility_table: Dict[Tuple, float]):
        """Add a function node with its variables and utility table."""
        self.functions[name] = (variables, utility_table)
        self.func_to_vars[name] = variables
        for var in variables:
            self.var_to_funcs[var].add(name)

class MaxSumWithGD2P:
    """Implements the Max-Sum algorithm with Generic Domain Dynamic Pruning (GD²P)."""
    def __init__(self, factor_graph: FactorGraph, max_iterations: int = 10, use_gdp: bool = False):
        self.graph = factor_graph
        self.max_iterations = max_iterations
        self.use_gdp = use_gdp
        self.messages: Dict[Tuple[str, str], Dict[Any, float]] = {}
        self.pruned_counts: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.initialize_messages()

    def initialize_messages(self):
        """Initialize all messages to zero for all possible state assignments."""
        for var in self.graph.variables:
            for func in self.graph.var_to_funcs[var]:
                self.messages[(var, func)] = {state: 0.0 for state in self.graph.variables[var]}
                self.pruned_counts[(var, func)] = (0, 0)
        for func in self.graph.functions:
            for var in self.graph.func_to_vars[func]:
                self.messages[(func, var)] = {state: 0.0 for state in self.graph.variables[var]}
                self.pruned_counts[(func, var)] = (0, 0)

    def variable_to_function_message(self, var: str, func: str) -> Dict[Any, float]:
        """Compute message from variable to function (Equation 2)."""
        message = {state: 0.0 for state in self.graph.variables[var]}
        for other_func in self.graph.var_to_funcs[var]:
            if other_func != func:
                for state in message:
                    message[state] += self.messages[(other_func, var)][state]
        return message

    def function_to_variable_message(self, func: str, var: str) -> Tuple[Dict[Any, float], float]:
        """Compute message from function to variable with GDP or GD²P (Equation 3)."""
        message = {state: 0.0 for state in self.graph.variables[var]}
        vars_in_func, utility_table = self.graph.functions[func]
        var_idx = vars_in_func.index(var)
        other_vars = [v for v in vars_in_func if v != var]
        domains = [self.graph.variables[v] for v in vars_in_func]
        states = self._cartesian_product(domains)
        incoming_msgs = [self.messages[(v, func)] for v in other_vars]
        pruning_func = gdp if self.use_gdp else gd2p
        pruned_ranges = pruning_func(utility_table, var, incoming_msgs, states, var_idx)
        
        pruned_count = 0
        total_count = 0
        for state in message:
            max_value = float('-inf')
            p, q = pruned_ranges.get(state, (float('inf'), -float('inf')))
            for assignment in states:
                if assignment[var_idx] != state:
                    continue
                total_count += 1
                utility = utility_table.get(assignment, 0.0)
                if q == p - (sum(max(msg.values()) for msg in incoming_msgs) - 
                             sum(msg.get(assignment[vars_in_func.index(v)], 0) 
                                 for v, msg in zip(other_vars, incoming_msgs))):
                    if not (q <= utility <= p):
                        pruned_count += 1
                        continue
                else:
                    if not (q < utility <= p):
                        pruned_count += 1
                        continue
                msg_sum = sum(self.messages[(other_var, func)][assignment[vars_in_func.index(other_var)]]
                              for other_var in other_vars)
                total = utility + msg_sum
                max_value = max(max_value, total)
            message[state] = max_value
        pruned_percentage = (pruned_count / total_count * 100) if total_count > 0 else 0
        self.pruned_counts[(func, var)] = (pruned_count, total_count)
        return message, pruned_percentage

    def _cartesian_product(self, domains: List[Set[Any]]) -> List[Tuple]:
        if not domains:
            return [()]
        result = []
        for value in domains[0]:
            for sub_result in self._cartesian_product(domains[1:]):
                result.append((value,) + sub_result)
        return result

    def compute_local_objective(self, var: str) -> Dict[Any, float]:
        Z = {state: 0.0 for state in self.graph.variables[var]}
        for func in self.graph.var_to_funcs[var]:
            for state in Z:
                Z[state] += self.messages[(func, var)][state]
        return Z

    def select_optimal_assignment(self) -> Dict[str, Any]:
        assignment = {}
        for var in self.graph.variables:
            Z = self.compute_local_objective(var)
            max_state = max(Z, key=Z.get)
            assignment[var] = max_state
        return assignment

    def run(self) -> Tuple[Dict[str, Any], float]:
        for iteration in range(self.max_iterations):
            var_to_func = {}
            for var in self.graph.variables:
                for func in self.graph.var_to_funcs[var]:
                    var_to_func[(var, func)] = self.variable_to_function_message(var, func)
            for (sender, receiver), msg in var_to_func.items():
                self.messages[(sender, receiver)] = msg
            
            func_to_var = {}
            total_pruned = 0
            total_count = 0
            for func in self.graph.functions:
                for var in self.graph.func_to_vars[func]:
                    msg, pruned_pct = self.function_to_variable_message(func, var)
                    func_to_var[(func, var)] = msg
                    p, t = self.pruned_counts[(func, var)]
                    total_pruned += p
                    total_count += t
            for (sender, receiver), msg in func_to_var.items():
                self.messages[(sender, receiver)] = msg
        avg_pruned_percentage = (total_pruned / total_count * 100) if total_count > 0 else 0
        return self.select_optimal_assignment(), avg_pruned_percentage