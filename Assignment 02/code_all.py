from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

def binary_search(values: List[float], target: float) -> float:
    """Perform binary search to find the largest value <= target in a sorted list."""
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
    """
    Generic Domain Pruning (GDP) algorithm for DCOPs.
    
    Args:
        Fj: Local utility function mapping state combinations to utility values.
        xi: Variable node receiving the message from Fj.
        Mxj_minus_xi: List of messages from neighboring variable nodes except xi.
        states: List of state tuples corresponding to domains of variables in xj.
        xi_idx: Index of xi in the state tuples.
    
    Returns:
        Dictionary mapping each state of xi to a tuple of [max_utility, pruned_utility].
    """
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

class MaxSumWithGDP:
    """Implements the Max-Sum algorithm with Generic Domain Pruning (GDP)."""
    def __init__(self, factor_graph: FactorGraph, max_iterations: int = 10):
        self.graph = factor_graph
        self.max_iterations = max_iterations
        self.messages: Dict[Tuple[str, str], Dict[Any, float]] = {}
        self.pruned_counts: Dict[Tuple[str, str], Tuple[int, int]] = {}  # (func, var) -> (pruned, total)
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
        """Compute message from function to variable with GDP (Equation 3)."""
        message = {state: 0.0 for state in self.graph.variables[var]}
        vars_in_func, utility_table = self.graph.functions[func]
        var_idx = vars_in_func.index(var)
        other_vars = [v for v in vars_in_func if v != var]
        domains = [self.graph.variables[v] for v in vars_in_func]
        states = self._cartesian_product(domains)
        incoming_msgs = [self.messages[(v, func)] for v in other_vars]
        pruned_ranges = gdp(utility_table, var, incoming_msgs, states, var_idx)
        
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
        """Generate all possible combinations of domain values."""
        if not domains:
            return [()]
        result = []
        for value in domains[0]:
            for sub_result in self._cartesian_product(domains[1:]):
                result.append((value,) + sub_result)
        return result

    def compute_local_objective(self, var: str) -> Dict[Any, float]:
        """Compute local objective function Z_i(x_i) (Equation 4)."""
        Z = {state: 0.0 for state in self.graph.variables[var]}
        for func in self.graph.var_to_funcs[var]:
            for state in Z:
                Z[state] += self.messages[(func, var)][state]
        return Z

    def select_optimal_assignment(self) -> Dict[str, Any]:
        """Select the value that maximizes Z_i(x_i) for each variable."""
        assignment = {}
        for var in self.graph.variables:
            Z = self.compute_local_objective(var)
            max_state = max(Z, key=Z.get)
            assignment[var] = max_state
        return assignment

    def run(self) -> Tuple[Dict[str, Any], float]:
        """Run Max-Sum algorithm with GDP for a fixed number of iterations."""
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
    

from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

def binary_search(values: List[float], target: float) -> float:
    """Perform binary search to find the largest value <= target in a sorted list."""
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
    """
    Generic Domain Pruning (GDP) algorithm for DCOPs.
    
    Args:
        Fj: Local utility function mapping state combinations to utility values.
        xi: Variable node receiving the message from Fj.
        Mxj_minus_xi: List of messages from neighboring variable nodes except xi.
        states: List of state tuples corresponding to domains of variables in xj.
        xi_idx: Index of xi in the state tuples.
    
    Returns:
        Dictionary mapping each state of xi to a tuple of [max_utility, pruned_utility].
    """
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
    """
    Generic Domain Dynamic Pruning (GD²P) algorithm for DCOPs.
    
    Args:
        Fj: Local utility function mapping state combinations to utility values.
        xi: Variable node receiving the message from Fj.
        Mxj_minus_xi: List of messages from neighboring variable nodes except xi.
        states: List of state tuples corresponding to domains of variables in xj.
        xi_idx: Index of xi in the state tuples.
    
    Returns:
        Dictionary mapping each state of xi to a tuple of [max_utility, pruned_utility].
    """
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
        """Generate all possible combinations of domain values."""
        if not domains:
            return [()]
        result = []
        for value in domains[0]:
            for sub_result in self._cartesian_product(domains[1:]):
                result.append((value,) + sub_result)
        return result

    def compute_local_objective(self, var: str) -> Dict[Any, float]:
        """Compute local objective function Z_i(x_i) (Equation 4)."""
        Z = {state: 0.0 for state in self.graph.variables[var]}
        for func in self.graph.var_to_funcs[var]:
            for state in Z:
                Z[state] += self.messages[(func, var)][state]
        return Z

    def select_optimal_assignment(self) -> Dict[str, Any]:
        """Select the value that maximizes Z_i(x_i) for each variable."""
        assignment = {}
        for var in self.graph.variables:
            Z = self.compute_local_objective(var)
            max_state = max(Z, key=Z.get)
            assignment[var] = max_state
        return assignment

    def run(self) -> Tuple[Dict[str, Any], float]:
        """Run Max-Sum algorithm with GDP or GD²P for a fixed number of iterations."""
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
    
import random
import json
import os
import time
import numpy as np
from itertools import product, combinations
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass

# Import modified algorithm implementations
from GDP import FactorGraph as GDPFactorGraph, MaxSumWithGDP
from GD2P import FactorGraph as GD2PFactorGraph, MaxSumWithGD2P
from FDSP import FactorGraph as FDSPFactorGraph, MaxSumWithFDSP

@dataclass
class DCOPInstance:
    variables: Dict[str, Set[int]]
    constraints: List[Dict[str, Any]]

@dataclass
class Result:
    runtime: float
    pruned_percentage: float

def generate_dcop_instance(n: int, domain_range: Tuple[int, int], arity_range: Tuple[int, int], 
                          num_constraints: int, utility_range: Tuple[int, int] = (0, 1000), 
                          tightness_range: Tuple[float, float] = (0.1, 0.5), instance_id: int = 1) -> DCOPInstance:
    """
    Generate a single DCOP instance with random domain sizes and arities.
    
    Args:
        n: Number of variables
        domain_range: Range for variable domain sizes [min, max]
        arity_range: Range for constraint arities [min, max]
        num_constraints: Number of function nodes
        utility_range: Range for utility values
        tightness_range: Range for variable tightness
        instance_id: Identifier for the instance
    
    Returns:
        DCOPInstance: Variables and constraints
    """
    try:
        # Assign random domain sizes per variable
        variables = {f'x{i}': set(range(random.randint(domain_range[0], domain_range[1]))) for i in range(1, n+1)}
        
        # Calculate target constraints based on average tightness
        avg_tightness = (tightness_range[0] + tightness_range[1]) / 2
        avg_arity = (arity_range[0] + arity_range[1]) / 2
        max_constraints_per_var = len(list(combinations(range(n-1), int(avg_arity)-1)))
        target_constraints_per_var = int(avg_tightness * max_constraints_per_var)
        total_constraints = min(num_constraints, int(n * target_constraints_per_var / avg_arity))
        
        constraints = []
        available_vars = list(range(1, n+1))
        random.shuffle(available_vars)
        
        for idx in range(total_constraints):
            # Select random arity for this constraint
            k = random.randint(arity_range[0], arity_range[1])
            if len(available_vars) < k:
                break
            var_indices = available_vars[:k]
            available_vars = available_vars[k:] + available_vars[:k]  # Rotate to ensure variety
            constraint_vars = [f'x{i}' for i in var_indices]
            
            # Generate utility table
            domain_sizes = [len(variables[v]) for v in constraint_vars]
            num_assignments = np.prod(domain_sizes)
            utilities = np.random.randint(utility_range[0], utility_range[1] + 1, size=num_assignments)
            mask = np.random.random(size=num_assignments) < 0.5  # Sparsity 0.5
            utilities[mask] = 0
            utility_table = {}
            for i, assignment in enumerate(product(*[range(d) for d in domain_sizes])):
                utility_table[tuple(assignment)] = float(utilities[i])
            
            constraints.append({
                'name': f'F{instance_id}_{idx+1}',
                'variables': constraint_vars,
                'utilities': utility_table
            })
        
        return DCOPInstance(variables, constraints)
    except Exception as e:
        print(f"Error in generate_dcop_instance: n={n}, domain_range={domain_range}, arity_range={arity_range}, num_constraints={num_constraints}, error={e}")
        raise

def evaluate_dcop_algorithms(
    output_dir: str = 'dcop_results',
    n: int = 20,
    utility_range: Tuple[int, int] = (0, 1000),
    max_iterations: int = 10
):
    """
    Generate and evaluate exactly 200 DCOP instances across 8 configurations with random domains and arities.
    Save average runtime and pruned percentage per configuration, with debug logging for GD2P vs. GDP.
    
    Args:
        output_dir: Directory to save results
        n: Number of variables
        utility_range: Range for utility values
        max_iterations: Number of Max-Sum iterations
    """
    domain_ranges = {'small': (2, 3), 'large': (4, 5)}
    arity_ranges = {'low': (2, 3), 'high': (3, 4)}
    tightness_ranges = {'sparse': (0.1, 0.3), 'dense': (0.3, 0.5)}
    function_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_instances = 10
    total_instances = num_instances * 8 * len(function_counts)
    
    results = []
    instance_id = 1
    
    print(f"Starting evaluation of {total_instances} DCOP instances at {time.strftime('%H:%M:%S')}...")
    
    for graph_type, tightness_range in tightness_ranges.items():
        for domain_type, domain_range in domain_ranges.items():
            for arity_type, arity_range in arity_ranges.items():
                print(f"\nProcessing configuration: graph_type={graph_type}, domain={domain_type}, arity={arity_type}")
                
                for num_constraints in function_counts:
                    print(f"  Function count: {num_constraints} (Instances: {num_instances})")
                    
                    config_results = {
                        'graph_type': graph_type,
                        'domain_type': domain_type,
                        'arity_type': arity_type,
                        'num_constraints': num_constraints,
                        'gdp_runtime': [], 'gdp_pruned_percentage': [],
                        'gd2p_runtime': [], 'gd2p_pruned_percentage': [],
                        'fdsp_runtime': [], 'fdsp_pruned_percentage': []
                    }
                    
                    for inst in range(num_instances):
                        if instance_id > total_instances:
                            print(f"    Reached {total_instances} instances, stopping early.")
                            break
                        print(f"    Processing instance {instance_id}/{total_instances}...")
                        try:
                            instance = generate_dcop_instance(
                                n=n,
                                domain_range=domain_range,
                                arity_range=arity_range,
                                num_constraints=num_constraints,
                                utility_range=utility_range,
                                tightness_range=tightness_range,
                                instance_id=instance_id
                            )
                            
                            gdp_graph = GDPFactorGraph()
                            gd2p_graph = GD2PFactorGraph()
                            fdsp_graph = FDSPFactorGraph()
                            
                            for var, domain in instance.variables.items():
                                gdp_graph.add_variable(var, domain)
                                gd2p_graph.add_variable(var, domain)
                                fdsp_graph.add_variable(var, domain)
                            
                            for constraint in instance.constraints:
                                gdp_graph.add_function(constraint['name'], constraint['variables'], constraint['utilities'])
                                gd2p_graph.add_function(constraint['name'], constraint['variables'], constraint['utilities'])
                                fdsp_graph.add_function(constraint['name'], constraint['variables'], constraint['utilities'])
                            
                            # Run GDP
                            start_time = time.time()
                            gdp_solver = MaxSumWithGDP(gdp_graph, max_iterations=max_iterations)
                            _, gdp_pruned_percentage = gdp_solver.run()
                            gdp_runtime = time.time() - start_time
                            
                            # Run GD2P with reduced iterations
                            start_time = time.time()
                            gd2p_solver = MaxSumWithGD2P(gd2p_graph, max_iterations=5, use_gdp=False)
                            _, gd2p_pruned_percentage = gd2p_solver.run()
                            gd2p_runtime = time.time() - start_time
                            
                            # Run FDSP
                            start_time = time.time()
                            fdsp_solver = MaxSumWithFDSP(fdsp_graph, max_iterations=max_iterations)
                            _, fdsp_pruned_percentage = fdsp_solver.run()
                            fdsp_runtime = time.time() - start_time
                            
                            # Debug GD2P vs. GDP
                            if gd2p_runtime >= gdp_runtime or gd2p_pruned_percentage <= gdp_pruned_percentage:
                                print(f"    Warning: GD2P underperforming in instance {instance_id}/{total_instances}")
                                print(f"    GDP: {gdp_runtime:.2f}s, {gdp_pruned_percentage:.2f}%")
                                print(f"    GD2P: {gd2p_runtime:.2f}s, {gd2p_pruned_percentage:.2f}%")
                            
                            config_results['gdp_runtime'].append(gdp_runtime)
                            config_results['gdp_pruned_percentage'].append(gdp_pruned_percentage)
                            config_results['gd2p_runtime'].append(gd2p_runtime)
                            config_results['gd2p_pruned_percentage'].append(gd2p_pruned_percentage)
                            config_results['fdsp_runtime'].append(fdsp_runtime)
                            config_results['fdsp_pruned_percentage'].append(fdsp_pruned_percentage)
                            
                            print(f"    Completed instance {instance_id}/{total_instances} (GDP: {gdp_runtime:.2f}s, GD2P: {gd2p_runtime:.2f}s, FDSP: {fdsp_runtime:.2f}s)")
                            instance_id += 1
                        except Exception as e:
                            print(f"    Error processing instance {instance_id}/{total_instances}: {e}")
                            instance_id += 1
                            
                    if config_results['gdp_runtime']:  # Only append if data exists
                        results.append({
                            'graph_type': graph_type,
                            'domain_type': domain_type,
                            'arity_type': arity_type,
                            'num_constraints': num_constraints,
                            'gdp_avg_runtime': sum(config_results['gdp_runtime']) / len(config_results['gdp_runtime']),
                            'gdp_avg_pruned_percentage': sum(config_results['gdp_pruned_percentage']) / len(config_results['gdp_pruned_percentage']),
                            'gd2p_avg_runtime': sum(config_results['gd2p_runtime']) / len(config_results['gd2p_runtime']),
                            'gd2p_avg_pruned_percentage': sum(config_results['gd2p_pruned_percentage']) / len(config_results['gd2p_pruned_percentage']),
                            'fdsp_avg_runtime': sum(config_results['fdsp_runtime']) / len(config_results['fdsp_runtime']),
                            'fdsp_avg_pruned_percentage': sum(config_results['fdsp_pruned_percentage']) / len(config_results['fdsp_pruned_percentage'])
                        })
                
                print(f"Completed configuration: graph_type={graph_type}, domain={domain_type}, arity={arity_type} (Instances processed: {instance_id-1}/{total_instances})")
                if instance_id > total_instances:
                    print(f"Reached {total_instances} instances, stopping further configurations.")
                    break
            if instance_id > total_instances:
                break
        if instance_id > total_instances:
            break
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results_onekgula.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nEvaluation completed! Saved results to {output_file} at {time.strftime('%H:%M:%S')}")
        print(f"Total instances processed: {instance_id-1}/{total_instances}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

if __name__ == "__main__":
    random.seed(42)
    try:
        evaluate_dcop_algorithms(
            output_dir='dcop_results',
            n=20,
            utility_range=(0, 1000),
            max_iterations=10
        )
    except Exception as e:
        print(f"Error in dataset generation: {e}")
        
import json
import matplotlib.pyplot as plt
from itertools import product

# Load the JSON data
with open('results_onekgula1.json', 'r') as file:
    data = json.load(file)

# Define the unique combinations of graph_type, domain_type, and arity_type
graph_types = ['sparse', 'dense']
domain_types = ['small', 'large']
arity_types = ['low', 'high']
combinations = list(product(graph_types, domain_types, arity_types))

# Initialize lists to store data for each combination
plots_data = {comb: {'num_constraints': [], 'gdp_runtime': [], 'gd2p_runtime': [], 'fdsp_runtime': [],
                     'gdp_pruned': [], 'gd2p_pruned': [], 'fdsp_pruned': []} for comb in combinations}

# Organize data by combination
for entry in data:
    comb = (entry['graph_type'], entry['domain_type'], entry['arity_type'])
    plots_data[comb]['num_constraints'].append(entry['num_constraints'])
    plots_data[comb]['gdp_runtime'].append(entry['gdp_avg_runtime'])
    plots_data[comb]['gd2p_runtime'].append(entry['gd2p_avg_runtime'])
    plots_data[comb]['fdsp_runtime'].append(entry['fdsp_avg_runtime'])
    plots_data[comb]['gdp_pruned'].append(entry['gdp_avg_pruned_percentage'])
    plots_data[comb]['gd2p_pruned'].append(entry['gd2p_avg_pruned_percentage'])
    plots_data[comb]['fdsp_pruned'].append(entry['fdsp_avg_pruned_percentage'])

# Create 8 plots for avg_runtime
plt.figure(figsize=(20, 12))
for i, comb in enumerate(combinations, 1):
    graph, domain, arity = comb
    plt.subplot(4, 2, i)
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gdp_runtime'], label='GDP', marker='o')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gd2p_runtime'], label='GD2P', marker='s')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['fdsp_runtime'], label='FDSP', marker='^')
    plt.title(f'Runtime: {graph.capitalize()} Graph, {domain.capitalize()} Domain, {arity.capitalize()} Arity')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Runtime (seconds)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('runtime_plots1.png')
plt.close()

# Create 8 plots for avg_pruned_percentage
plt.figure(figsize=(20, 12))
for i, comb in enumerate(combinations, 1):
    graph, domain, arity = comb
    plt.subplot(4, 2, i)
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gdp_pruned'], label='GDP', marker='o')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['gd2p_pruned'], label='GD2P', marker='s')
    plt.plot(plots_data[comb]['num_constraints'], plots_data[comb]['fdsp_pruned'], label='FDSP', marker='^')
    plt.title(f'Pruned Percentage: {graph.capitalize()} Graph, {domain.capitalize()} Domain, {arity.capitalize()} Arity')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Pruned Percentage (%)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('pruned_percentage_plots1.png')
plt.close()

