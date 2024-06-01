"""
Copyright (c) 2022 Hocheol Lim.
"""

import sys
import os
from sage.scoring.common_scoring_functions import (
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    RdkitScoringFunction,
    SMARTSScoringFunction,
    TanimotoScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from sage.scoring.score_modifier import (
    ClippedScoreModifier,
    GaussianModifier,
    MaxGaussianModifier,
    MinGaussianModifier,
)
from sage.scoring.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    ScoringFunction,
    MoleculewiseScoringFunction,
)

from sage.scoring.descriptors import (
    solubility,
    synthetic_accessibility,
)

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol

class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )
        return score



def similarity_constrained_solubility_task(
    smiles, name: str, threshold: float, fp_type: str = "ECFP4"
) -> GoalDirectedBenchmark:
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Solubility"

    specification = uniform_specification(1, 10, 100)
    
    objective = RdkitScoringFunction(descriptor=solubility)
    
    offset = -objective.score(smiles)
    
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0
    
    return GoalDirectedBenchmark(
        name="Similarity Constrained Solubility",
        objective=constrained_objective,
        contribution_specification=specification,
    )

def similarity_constrained_synthetic_accessibility_task(
    smiles, name: str, threshold: float, fp_type: str = "ECFP4"
) -> GoalDirectedBenchmark:
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Synthetic Accessbility"

    specification = uniform_specification(1, 10, 100)
    
    objective = RdkitScoringFunction(descriptor=synthetic_accessibility)
    
    offset = -objective.score(smiles)
    
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0
    
    return GoalDirectedBenchmark(
        name="Similarity Constrained Synthetic Accessbility",
        objective=constrained_objective,
        contribution_specification=specification,
    )
