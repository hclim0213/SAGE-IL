"""
Copyright (c) 2022 Hocheol Lim.
"""
import sys
sys.path.append('/workspace/_ext')

from typing import List, Tuple
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark

from sage.benchmark.benchmarks_il_const import (
    similarity_constrained_penalized_logp_atomrings,
    similarity_constrained_penalized_logp_cyclebasis,
)

from sage.benchmark.general_tasks_const import (
    similarity_constrained_solubility_task,
    similarity_constrained_synthetic_accessibility_task
)

# GEGL Benchmarks

#   00 : Similarity_constrained penalized_Logp atomrings,
#   01 : Similarity_constrained penalized Logp cyclebasis,

# General Tasks

#   02 : Similarity_constrained Solubility,
#   03 : Similarity constrained Synthetic Accessibility

def load_benchmark(benchmark_id: int) -> Tuple[GoalDirectedBenchmark, List[int]]:
    
    benchmark = {
        
        # GEGL Benchmark
        
        0: similarity_constrained_penalized_logp_atomrings,
        1: similarity_constrained_penalized_logp_cyclebasis,
        
        # General Tasks
        
        2: similarity_constrained_solubility_task,
        3: similarity_constrained_synthetic_accessibility_task,

    }.get(benchmark_id)
    
    if benchmark_id in [0, 1]:
        scoring_num_list = [1]
    else:
        scoring_num_list = [1, 10, 100]
        
    return benchmark, scoring_num_list
