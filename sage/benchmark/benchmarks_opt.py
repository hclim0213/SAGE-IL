"""
Copyright (c) 2022 Hocheol Lim.
"""

import sys
sys.path.append('/workspace/_ext')

from typing import List, Tuple
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark

from sage.benchmark.benchmarks_il import (
    similarity_cation,
    similarity_anion,
    isomers_c8h15n2_cation,
    isomers_c9h14n_cation,
    isomers_c2h5o4s_anion,
    median_emim_bmim_cation,
    median_bf4_pf6_anion,
#    replacement_emim_cation,
#    replacement_ntf_anion,
#    scaffold_hop_emim_cation,
#    decoration_hop_ntf_anion,
)
from sage.benchmark.general_tasks_opt import (
    high_viscosity_cation_task,
    high_viscosity_anion_task,
    low_viscosity_cation_task,
    low_viscosity_anion_task,
    high_melting_point_cation_task,
    high_melting_point_anion_task,
    low_melting_point_cation_task,
    low_melting_point_anion_task,
)

from sage.benchmark.general_tasks_opt import (
    high_solubility_co2_cation_task,
    high_solubility_co2_anion_task,
    low_idac_co2_cation_task,
    low_idac_co2_anion_task,
    low_idac_artemisinin_cation_task,
    low_idac_betulin_cation_task,
    low_idac_caffeine_cation_task,
    low_idac_cianidanol_anion_task,
    low_idac_galantamine_anion_task,
    low_idac_glaucine_anion_task,
    low_idac_norgalanthamine_cation_task,
    low_idac_piperine_cation_task,
    low_idac_shikimate_anion_task,
    low_idac_ungiminorine_anion_task,
)

# Ionic Liquid Benchmarks

#   00 : Rediscovery / Cation / 1-Ethyl-3-methylimidazol-3-ium
#   01 : Rediscovery / Cation / Trihexyl(tetradecyl)phosphonium
#   02 : Rediscovery /  Anion / Bis(trifluoromethane)sulfonimide
#   03 : Rediscovery /  Anion / Tetrafluoroborate
#   04 : Similarity  / Cation / 1-Butylpyridinium
#   05 : Similarity  / Cation / 1-Butyl-1-methylpyrrolidinium
#   06 : Similarity  /  Anion / Hexafluorophosphate
#   07 : Similarity  /  Anion / Ethylsulphate
#   08 : Isomers     / Cation / C8H15N2+ (1-Butyl-1-methylpyrrolidinium)
#   09 : Isomers     / Cation / C9H14N+ (1-Butylpyridinium)
#   10 : Isomers     /  Anion / C2H5O4S- (Ethylsulphate)
#   11 : Median Sim. / Cation / 1-Ethyl-3-methylimidazol-3-ium and 1-Butyl-1-methylpyrrolidinium
#   12 : Median Sim. /  Anion / Tetrafluoroborate and Hexafluorophosphate

#   13 : Single Opt. / Cation / High Viscosity
#   14 : Single Opt. /  Anion / High Viscosity
#   15 : Single Opt. / Cation / Low Viscosity
#   16 : Single Opt. /  Anion / Low Viscosity
#   17 : Single Opt. / Cation / High Melting Point
#   18 : Single Opt. /  Anion / High Melting Point
#   19 : Single Opt. / Cation / Low Melting Point
#   20 : Single Opt. /  Anion / Low Melting Point
#   21 : Multi. Opt. / Cation / High Solubility for CO2
#   22 : Multi. Opt. /  Anion / High Solubility for CO2
#   23 : Multi. Opt. / Cation / Low IDAC for CO2
#   24 : Multi. Opt. /  Anion / Low IDAC for CO2

#   25 : Multi. Opt. / Cation / Low IDAC for Artemisinin
#   26 : Multi. Opt. / Cation / Low IDAC for Betulin
#   27 : Multi. Opt. / Cation / Low IDAC for Caffeine
#   28 : Multi. Opt. / Cation / Low IDAC for Norgalanthamine
#   29 : Multi. Opt. / Cation / Low IDAC for Piperine

#   30 : Multi. Opt. /  Anion / Low IDAC for Cianidanol
#   31 : Multi. Opt. /  Anion / Low IDAC for Galantamine
#   32 : Multi. Opt. /  Anion / Low IDAC for Glaucine
#   33 : Multi. Opt. /  Anion / Low IDAC for Shikimate
#   34 : Multi. Opt. /  Anion / Low IDAC for Ungiminorine

def load_benchmark(benchmark_id: int) -> Tuple[GoalDirectedBenchmark, List[int]]:
    benchmark = {
        
        # Ionic Liquid Benchmark
        
        0: similarity_cation(
            smiles="CCN1C=C[N+](=C1)C",
            name="1-Ethyl-3-methylimidazol-3-ium",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        1: similarity_cation(
            smiles="CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC",
            name="Trihexyl(tetradecyl)phosphonium",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        2: similarity_anion(
            smiles="C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
            name="Bis(trifluoromethane)sulfonimide",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        3: similarity_anion(
            smiles="[B-](F)(F)(F)F",
            name="Tetrafluoroborate",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        4: similarity_cation(
            smiles="CCCC[N+]1=CC=CC=C1",
            name="1-Butylpyridinium",
            fp_type="ECFP6",
            threshold=0.75,
            rediscovery=False,
        ),
        5: similarity_cation(
            smiles="CCCC[N+]1(CCCC1)C",
            name="1-Butyl-1-methylpyrrolidinium",
            fp_type="ECFP6",
            threshold=0.75,
            rediscovery=False,
        ),
        6: similarity_anion(
            smiles="F[P-](F)(F)(F)(F)F",
            name="Hexafluorophosphate",
            fp_type="ECFP6",
            threshold=0.75,
            rediscovery=False,
        ),
        7: similarity_anion(
            smiles="CCOS(=O)(=O)[O-]",
            name="Ethylsulphate",
            fp_type="ECFP6",
            threshold=0.75,
            rediscovery=False,
        ),
        8: isomers_c8h15n2_cation(),
        9: isomers_c9h14n_cation(),
        10: isomers_c2h5o4s_anion(),
        11: median_emim_bmim_cation(),
        12: median_bf4_pf6_anion(),

        # Single Optimization Tasks   
        13: high_viscosity_cation_task(),
        14: high_viscosity_anion_task(),
        15: low_viscosity_cation_task(),
        16: low_viscosity_anion_task(),
        17: high_melting_point_cation_task(),
        18: high_melting_point_anion_task(),
        19: low_melting_point_cation_task(),
        20: low_melting_point_anion_task(),
        
        # Multiple Optimization Tasks   
        21: high_solubility_co2_cation_task(),
        22: high_solubility_co2_anion_task(),
        23: low_idac_co2_cation_task(),
        24: low_idac_co2_anion_task(),
        
        # Augmented Multiple Optimization Tasks
        25: low_idac_artemisinin_cation_task(),
        26: low_idac_betulin_cation_task(),
        27: low_idac_caffeine_cation_task(),
        28: low_idac_norgalanthamine_cation_task(),
        29: low_idac_piperine_cation_task(),
        30: low_idac_cianidanol_anion_task(),
        31: low_idac_galantamine_anion_task(),
        32: low_idac_glaucine_anion_task(),
        33: low_idac_shikimate_anion_task(),
        34: low_idac_ungiminorine_anion_task(),
    }.get(benchmark_id)

    if benchmark_id in [
        4,
        5,
        6,
        7,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
    ]:
        scoring_num_list = [1, 10, 100]
    elif benchmark_id in [8]:
        scoring_num_list = [299]
    elif benchmark_id in [9]:
        scoring_num_list = [273]
    elif benchmark_id in [10]:
        scoring_num_list = [23]
    elif benchmark_id in [0, 1, 2, 3]:
        scoring_num_list = [1]
    else:
        scoring_num_list = [1, 10, 100]

    return benchmark, scoring_num_list  # type: ignore

