from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

def cartesian_product(domains: List[Set[Any]]) -> List[Tuple]:
    """Generate all possible combinations of variable assignments."""
    if not domains:
        return [()]
    result = []
    for value in domains[0]:
        for sub_result in cartesian_product(domains[1:]):
            result.append((value,) + sub_result)
    return result

def function_decomposing(Fk: Dict[Tuple, float], vars: List[str], domains: List[Set[Any]]) -> Dict[Tuple, float]:
    """Function Decomposing (FD) for computing function estimations (Algorithm 1)."""
    n = len(vars)
    fun_est = {}
    
    for i in range(n-1, -1, -1):
        if i == n-1:
            for state in Fk:
                fun_est[(i, state[i], -1, None)] = Fk[state]
        else:
            for state in Fk:
                max_val = float('-inf')
                for next_val in domains[i+1]:
                    key = (i+1, next_val, -1, None)
                    if key in fun_est:
                        max_val = max(max_val, fun_est[key])
                fun_est[(i, state[i], -1, None)] = max_val
    
    for j in range(n-1, -1, -1):
        for vj in domains[j]:
            for i in range(j-1, -1, -1):
                if i == j-1:
                    for state in Fk:
                        if state[j] == vj:
                            fun_est[(i, state[i], j, vj)] = fun_est.get((i, state[i], -1, None), 0)
                else:
                    max_val = float('-inf')
                    for vi in domains[i+1]:
                        key = (i+1, vi, j, vj)
                        if key in fun_est:
                            max_val = max(max_val, fun_est[key])
                    for state in Fk:
                        fun_est[(i, state[i], j, vj)] = max_val
    
    return fun_est

def state_pruning(Fk: Dict[Tuple, float], xi: str, Mxj_minus_xi: List[Dict[Any, float]], 
                  states: List[Tuple], xi_idx: int, fun_est: Dict[Tuple, float], 
                  vars_in_func: List[str], domains: List[Set[Any]]) -> Tuple[Dict[Any, float], float]:
    """State Pruning (SP) for computing function-to-variable message (Algorithm 2)."""
    n = len(vars_in_func)
    result = {state: 0.0 for state in domains[xi_idx]}
    lb = float('-inf')
    pruned_count = 0
    total_count = 0
    
    msg_est = [0] * n
    for i in range(n-2, -1, -1):
        if i >= xi_idx:
            continue
        msg_est[i] = msg_est[i+1] + max(Mxj_minus_xi[i].values() if i < len(Mxj_minus_xi) else [0])
    
    def fdsp_rec(i: int, assign: List[Any], msg_util: float, start: int) -> float:
        nonlocal lb, pruned_count, total_count
        if i >= n:
            return float('-inf')
        
        next_unassigned = start
        while next_unassigned < n and assign[next_unassigned] is not None:
            next_unassigned += 1
        
        if next_unassigned >= n:
            if assign[xi_idx] != assign[xi_idx]:
                return float('-inf')
            complete_assign = tuple(assign)
            util = Fk.get(complete_assign, 0.0) + msg_util
            if util > lb:
                lb = util
            return util
        
        ub = 0
        if next_unassigned > xi_idx:
            ub += fun_est.get((next_unassigned, assign[next_unassigned], -1, None), 0)
        else:
            for j in range(next_unassigned+1, n):
                if assign[j] is not None:
                    ub += fun_est.get((next_unassigned, assign[next_unassigned], j, assign[j]), 0)
                    break
            else:
                ub += fun_est.get((next_unassigned, assign[next_unassigned], -1, None), 0)
        ub += msg_util + msg_est[next_unassigned]
        
        total_count += 1
        if ub <= lb:
            pruned_count += 1
            return float('-inf')
        
        max_util = float('-inf')
        for vi in domains[next_unassigned]:
            assign[next_unassigned] = vi
            msg_util_new = msg_util + Mxj_minus_xi[next_unassigned].get(vi, 0) if next_unassigned < len(Mxj_minus_xi) else msg_util
            util = fdsp_rec(next_unassigned, assign, msg_util_new, next_unassigned+1)
            max_util = max(max_util, util)
            assign[next_unassigned] = None
        
        return max_util
    
    for vt in domains[xi_idx]:
        assign = [None] * n
        assign[xi_idx] = vt
        util = fdsp_rec(0, assign, 0.0, 0)
        result[vt] = util if util != float('-inf') else 0.0
    
    pruned_percentage = (pruned_count / total_count * 100) if total_count > 0 else 0
    return result, pruned_percentage

class FactorGraph:
    """Represents a DCOP as a factor graph with variable and function nodes."""
    def __init__(self):
        self.variables: Dict[str, Set[Any]] = {}
        self.functions: Dict[str, Tuple[List[str], Dict[Tuple, float]]] = {}
        self.var_to_funcs: Dict[str, Set[str]] = defaultdict(set)
        self.func_to_vars: Dict[str, List[str]] = {}
        self.fun_est: Dict[str, Dict[Tuple, float]] = {}

    def add_variable(self, name: str, domain: Set[Any]):
        self.variables[name] = domain
        self.var_to_funcs[name] = set()

    def add_function(self, name: str, variables: List[str], utility_table: Dict[Tuple, float]):
        self.functions[name] = (variables, utility_table)
        self.func_to_vars[name] = variables
        for var in variables:
            self.var_to_funcs[var].add(name)
        domains = [self.variables[v] for v in variables]
        self.fun_est[name] = function_decomposing(utility_table, variables, domains)

class MaxSumWithFDSP:
    """Implements Max-Sum with FDSP pruning."""
    def __init__(self, factor_graph: FactorGraph, max_iterations: int = 10):
        self.graph = factor_graph
        self.max_iterations = max_iterations
        self.messages: Dict[Tuple[str, str], Dict[Any, float]] = {}
        self.pruned_counts: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.initialize_messages()

    def initialize_messages(self):
        for var in self.graph.variables:
            for func in self.graph.var_to_funcs[var]:
                self.messages[(var, func)] = {state: 0.0 for state in self.graph.variables[var]}
                self.pruned_counts[(var, func)] = (0, 0)
        for func in self.graph.functions:
            for var in self.graph.func_to_vars[func]:
                self.messages[(func, var)] = {state: 0.0 for state in self.graph.variables[var]}
                self.pruned_counts[(func, var)] = (0, 0)

    def variable_to_function_message(self, var: str, func: str) -> Dict[Any, float]:
        message = {state: 0.0 for state in self.graph.variables[var]}
        for other_func in self.graph.var_to_funcs[var]:
            if other_func != func:
                for state in message:
                    message[state] += self.messages[(other_func, var)][state]
        return message

    def function_to_variable_message(self, func: str, var: str) -> Tuple[Dict[Any, float], float]:
        vars_in_func, utility_table = self.graph.functions[func]
        var_idx = vars_in_func.index(var)
        other_vars = [v for v in vars_in_func if v != var]
        domains = [self.graph.variables[v] for v in vars_in_func]
        states = cartesian_product(domains)
        incoming_msgs = [self.messages[(v, func)] for v in other_vars]
        message, pruned_percentage = state_pruning(utility_table, var, incoming_msgs, states, var_idx, 
                                                  self.graph.fun_est[func], vars_in_func, domains)
        self.pruned_counts[(func, var)] = (int(pruned_percentage * len(states) / 100), len(states))
        return message, pruned_percentage

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