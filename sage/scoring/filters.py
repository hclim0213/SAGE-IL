"""
Copyright (c) 2022 Hocheol Lim.
"""
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import Descriptors, rdMolDescriptors

import signal
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata

def handler(signum, frame):
    raise Exception()

def filter_cycle_len(mol: Mol, threshold=6) -> bool:
    import networkx as nx
    
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])

    if cycle_length > threshold:
        return False
    
    return True

def score_rascore(mol: Mol, threshold=0.5) -> bool:
    from RAscore import RAscore_XGB

    compound = str(Chem.MolToSmiles(mol))
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    score = xgb_scorer.predict(compound)
    
    return float(score)

def filter_cation(mol: Mol) -> bool:
    charge = Chem.GetFormalCharge(mol)
    if charge > 0:
        return True
    else:
        return False

def filter_anion(mol: Mol) -> bool:
    
    charge = Chem.GetFormalCharge(mol)
    if charge < 0:
        return True
    else:
        return False