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

from sage.scoring.descriptors import (
    IsomerScoringFunction_cation,
    RdkitScoringFunction_cation,
    SMARTSScoringFunction_cation,
    TanimotoScoringFunction_cation,
    IsomerScoringFunction_anion,
    RdkitScoringFunction_anion,
    SMARTSScoringFunction_anion,
    TanimotoScoringFunction_anion,
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
from guacamol.utils.descriptors import (
    AtomCounter,
    bertz,
    logP,
    mol_weight,
    num_aromatic_rings,
    num_rings,
    num_rotatable_bonds,
    qed,
    tpsa,
)

from sage.scoring.descriptors import (
    filters_set_cation,
    filters_set_anion,
)

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, RDConfig

class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset
        self.corrupt_score = -1.0

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )
        return score

def similarity_cation(
    smiles: str,
    name: str,
    fp_type: str = "ECFP6",
    threshold: float = 0.7,
    rediscovery: bool = False,
) -> GoalDirectedBenchmark:
    category = "rediscovery" if rediscovery else "similarity"
    benchmark_name = f"{name} {category}"

    modifier = ClippedScoreModifier(upper_x=threshold)
    scoring_function = TanimotoScoringFunction_cation(
        target=smiles, fp_type=fp_type, score_modifier=modifier
    )
    if rediscovery:
        specification = uniform_specification(1)
    else:
        specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=scoring_function,
        contribution_specification=specification,
    )

def similarity_anion(
    smiles: str,
    name: str,
    fp_type: str = "ECFP6",
    threshold: float = 0.7,
    rediscovery: bool = False,
) -> GoalDirectedBenchmark:
    category = "rediscovery" if rediscovery else "similarity"
    benchmark_name = f"{name} {category}"

    modifier = ClippedScoreModifier(upper_x=threshold)
    scoring_function = TanimotoScoringFunction_anion(
        target=smiles, fp_type=fp_type, score_modifier=modifier
    )
    if rediscovery:
        specification = uniform_specification(1)
    else:
        specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=scoring_function,
        contribution_specification=specification,
    )

def isomers_c8h15n2_cation(mean_function="arithmetic") -> GoalDirectedBenchmark:
    """
    Benchmark to try and get all C11H24 molecules there are.
    There should be 159 if one ignores stereochemistry.
    Args:
        mean_function: 'arithmetic' or 'geometric'
    """

    specification = uniform_specification(299)

    return GoalDirectedBenchmark(
        name="C11H24",
        objective=IsomerScoringFunction_cation("C8H11N2", mean_function=mean_function),
        contribution_specification=specification,
    )

def isomers_c9h14n_cation(mean_function="arithmetic") -> GoalDirectedBenchmark:
    """
    Benchmark to try and get 100 isomers for C7H8N2O2.
    Args:
        mean_function: 'arithmetic' or 'geometric'
    """

    specification = uniform_specification(273)

    return GoalDirectedBenchmark(
        name="C7H8N2O2",
        objective=IsomerScoringFunction_cation("C9H14N", mean_function=mean_function),
        contribution_specification=specification,
    )

def isomers_c2h5o4s_anion(mean_function="arithmetic") -> GoalDirectedBenchmark:
    """
    Benchmark to try and get 100 isomers for C9H10N2O2PF2Cl.
    Args:
        mean_function: 'arithmetic' or 'geometric'
    """

    specification = uniform_specification(23)

    return GoalDirectedBenchmark(
        name="C9H10N2O2PF2Cl",
        objective=IsomerScoringFunction_anion("C2H5O4S", mean_function=mean_function),
        contribution_specification=specification,
    )

def median_emim_bmim_cation(
    mean_cls=ArithmeticMeanScoringFunction,
) -> GoalDirectedBenchmark:
    t_emim = TanimotoScoringFunction_cation("CCN1C=C[N+](=C1)C", fp_type="ECFP6")
    t_bmim = TanimotoScoringFunction_cation("CCCC[N+]1(CCCC1)C", fp_type="ECFP6")
    median = mean_cls([t_emim, t_bmim])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Median molecules 1",
        objective=median,
        contribution_specification=specification,
    )

def median_bf4_pf6_anion(
    mean_cls=ArithmeticMeanScoringFunction,
) -> GoalDirectedBenchmark:
    # median mol between tadalafil and sildenafil
    t_bf4 = TanimotoScoringFunction_anion("[B-](F)(F)(F)F", fp_type="ECFP6")
    t_pf6 = TanimotoScoringFunction_anion("F[P-](F)(F)(F)(F)F", fp_type="ECFP6")
    median = mean_cls([t_bf4, t_pf6])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Median molecules 2",
        objective=median,
        contribution_specification=specification,
    )

#def replacement_emim_cation() -> GoalDirectedBenchmark:
#
#    smiles = "CCN1C=C[N+](=C1)C"
#    emim_cation = Chem.MolFromSmiles(smiles)
#    target_logp = logP(emim_cation)
#    target_tpsa = tpsa(emim_cation)
#
#    similarity = TanimotoScoringFunction(
#        smiles, fp_type="ECFP6", score_modifier=GaussianModifier(mu=0, sigma=0.1)
#    )
#    lp = RdkitScoringFunction(
#        descriptor=logP, score_modifier=GaussianModifier(mu=target_logp, sigma=0.2)
#    )
#    tp = RdkitScoringFunction(
#        descriptor=tpsa, score_modifier=GaussianModifier(mu=target_tpsa, sigma=5)
#    )
#    isomers = IsomerScoringFunction("C16H15F6N5O")
#
#    specification = uniform_specification(1, 10, 100)
#
#    return GoalDirectedBenchmark(
#        name="Replacement 1-Ethyl-3-methylimidazol-3-ium",
#        objective=ArithmeticMeanScoringFunction([similarity, lp, tp, isomers]),
#        contribution_specification=specification,
#    )
#
#def replacement_ntf_anion() -> GoalDirectedBenchmark:
#
#    smiles = "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"
#    ntf_anion = Chem.MolFromSmiles(smiles)
#    target_logp = logP(ntf_anion)
#    target_tpsa = tpsa(ntf_anion)
#
#    similarity = TanimotoScoringFunction(
#        smiles, fp_type="ECFP6", score_modifier=GaussianModifier(mu=0, sigma=0.1)
#    )
#    lp = RdkitScoringFunction(
#        descriptor=logP, score_modifier=GaussianModifier(mu=target_logp, sigma=0.2)
#    )
#    tp = RdkitScoringFunction(
#        descriptor=tpsa, score_modifier=GaussianModifier(mu=target_tpsa, sigma=5)
#    )
#    isomers = IsomerScoringFunction("C16H15F6N5O")
#
#    specification = uniform_specification(1, 10, 100)
#
#    return GoalDirectedBenchmark(
#        name="Replacement Bis(trifluoromethane)sulfonimide",
#        objective=ArithmeticMeanScoringFunction([similarity, lp, tp, isomers]),
#        contribution_specification=specification,
#    )
#
#def scaffold_hop_emim_cation() -> GoalDirectedBenchmark:
#    """
#    Keep the decoration, and similarity to start point, but change the scaffold.
#    """
#
#    smiles = "CCN1C=C[N+](=C1)C"
#
#    pharmacophor_sim = TanimotoScoringFunction(
#        smiles, fp_type="ECFP6", score_modifier=ClippedScoreModifier(upper_x=0.75)
#    )
#
#    deco = SMARTSScoringFunction(
#        "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1", inverse=False
#    )
#
#    # anti scaffold
#    scaffold = SMARTSScoringFunction(
#        "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12", inverse=True
#    )
#
#    scaffold_hop_obj = ArithmeticMeanScoringFunction([pharmacophor_sim, deco, scaffold])
#
#    specification = uniform_specification(1, 10, 100)
#
#    return GoalDirectedBenchmark(
#        name="Scaffold Hop",
#        objective=scaffold_hop_obj,
#        contribution_specification=specification,
#    )
#
#def decoration_hop_ntf_anion() -> GoalDirectedBenchmark:
#    smiles = "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"
#
#    pharmacophor_sim = TanimotoScoringFunction(
#        smiles, fp_type="ECFP6", score_modifier=ClippedScoreModifier(upper_x=0.85)
#    )
#    # change deco
#    deco1 = SMARTSScoringFunction("CS([#6])(=O)=O", inverse=True)
#    deco2 = SMARTSScoringFunction("[#7]-c1ccc2ncsc2c1", inverse=True)
#
#    # keep scaffold
#    scaffold = SMARTSScoringFunction(
#        "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12", inverse=False
#    )
#
#    deco_hop1_fn = ArithmeticMeanScoringFunction(
#        [pharmacophor_sim, deco1, deco2, scaffold]
#    )
#
#    specification = uniform_specification(1, 10, 100)
#
#    return GoalDirectedBenchmark(
#        name="Deco Hop",
#        objective=deco_hop1_fn,
#        contribution_specification=specification,
#    )
#
