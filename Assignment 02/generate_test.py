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