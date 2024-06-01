"""
Copyright (c) 2022 Hocheol Lim.
"""

import sys
import os

from sage.scoring.common_scoring_functions import (
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
    RdkitScoringFunction_cation,
    RdkitScoringFunction_anion,
    score_rascore,
    high_viscosity_cation,
    low_viscosity_cation,
    high_viscosity_anion,
    low_viscosity_anion,
    high_melting_point_cation,
    low_melting_point_cation,
    high_melting_point_anion,
    low_melting_point_anion,
)

from sage.scoring.descriptors import (
    high_solubility_co2_cation,
    high_solubility_co2_anion,
    low_idac_co2_cation,
    low_idac_co2_anion,
    low_idac_artemisinin_cation,
    low_idac_betulin_cation,
    low_idac_caffeine_cation,
    low_idac_norgalanthamine_cation,
    low_idac_piperine_cation,
    low_idac_cianidanol_anion,
    low_idac_galantamine_anion,
    low_idac_glaucine_anion,
    low_idac_shikimate_anion,
    low_idac_ungiminorine_anion,
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

def high_viscosity_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Viscosity Cation",
        objective=RdkitScoringFunction_cation(descriptor=high_viscosity_cation),
        contribution_specification=specification,
    )

def low_viscosity_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Viscosity SPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_viscosity_cation),
        contribution_specification=specification,
    )
    
def high_melting_point_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Tm SPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=high_melting_point_cation),
        contribution_specification=specification,
    )

def low_melting_point_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Tm SPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_melting_point_cation),
        contribution_specification=specification,
    )

def high_viscosity_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Viscosity SPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=high_viscosity_anion),
        contribution_specification=specification,
    )

def low_viscosity_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Viscosity SPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_viscosity_anion),
        contribution_specification=specification,
    )
    
def high_melting_point_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Tm SPO Cation",
        objective=RdkitScoringFunction_anion(descriptor=high_melting_point_anion),
        contribution_specification=specification,
    )

def low_melting_point_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Tm SPO Cation",
        objective=RdkitScoringFunction_anion(descriptor=low_melting_point_anion),
        contribution_specification=specification,
    )

def high_solubility_co2_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Solubility CO2 MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=high_solubility_co2_cation),
        contribution_specification=specification,
    )

def high_solubility_co2_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High Solubility CO2 MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=high_solubility_co2_anion),
        contribution_specification=specification,
    )

def low_idac_co2_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC CO2 MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_co2_cation),
        contribution_specification=specification,
    )

def low_idac_co2_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC CO2 MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_co2_anion),
        contribution_specification=specification,
    )

def low_idac_artemisinin_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Artemisinin MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_artemisinin_cation),
        contribution_specification=specification,
    )

def low_idac_betulin_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Betulin MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_betulin_cation),
        contribution_specification=specification,
    )

def low_idac_caffeine_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Caffeine MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_caffeine_cation),
        contribution_specification=specification,
    )

def low_idac_norgalanthamine_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Norgalanthamine MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_norgalanthamine_cation),
        contribution_specification=specification,
    )

def low_idac_piperine_cation_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Piperine MPO Cation",
        objective=RdkitScoringFunction_cation(descriptor=low_idac_piperine_cation),
        contribution_specification=specification,
    )

def low_idac_cianidanol_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Cianidanol MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_cianidanol_anion),
        contribution_specification=specification,
    )

def low_idac_galantamine_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Galantamine MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_galantamine_anion),
        contribution_specification=specification,
    )

def low_idac_glaucine_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Glaucine MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_glaucine_anion),
        contribution_specification=specification,
    )

def low_idac_shikimate_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Shikimate MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_shikimate_anion),
        contribution_specification=specification,
    )

def low_idac_ungiminorine_anion_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low IDAC Ungiminorine MPO Anion",
        objective=RdkitScoringFunction_anion(descriptor=low_idac_ungiminorine_anion),
        contribution_specification=specification,
    )