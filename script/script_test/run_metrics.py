import sys
import numpy as np
from rdkit import Chem, SimDivFilters
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors

filename_target = sys.argv[1]
filename_train = sys.argv[2]

target_smiles = np.loadtxt(filename_target, dtype=str, delimiter="\n")
train_smiles = np.loadtxt(filename_train, dtype=str, delimiter="\n")

def smiles_to_mols(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def mols_to_smiles(mol_list):
    return [Chem.MolToSmiles(mol) for mol in mol_list]

def validity(mols):
    return len([mol for mol in mols if mol is not None]) / len(mols)

def uniqueness(smiles_list):
    return len(set(smiles_list)) / len(smiles_list)

def novelty(smiles_list, train_smiles_list):
    train_set = set(train_smiles_list)
    return len(set(smiles_list) - train_set) / len(smiles_list)

def count_anions_and_cations(mol_list):
    anion_count = 0
    cation_count = 0
    
    for mol in mol_list:
        charge = Chem.GetFormalCharge(mol)
        
        if charge > 0:
            cation_count += 1
        elif charge < 0:
            anion_count += 1
    
    total_count = len(mol_list)
    anion_ratio = anion_count / total_count
    cation_ratio = cation_count / total_count
    
    return anion_ratio, cation_ratio

def se_diversity(mols, threshold=0.65):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    lp = SimDivFilters.rdSimDivPickers.LeaderPicker()
    ids = lp.LazyBitVectorPick(fps, len(fps), threshold)
    
    return len(ids) / len(fps)

def calculate_metrics(smiles_list, train_smiles_list):
    mols = smiles_to_mols(smiles_list)
    valid_mols = [mol for mol in mols if mol is not None]
    valid_smiles = [smiles for smiles in mols_to_smiles(valid_mols) if smiles is not None]    

    unique_ratio = uniqueness(valid_smiles)
    novel_ratio = novelty(valid_smiles, train_smiles_list)
    se_div = se_diversity(valid_mols)
    
    anion_ratio, cation_ratio = count_anions_and_cations(valid_mols)

    metrics = {
        "validity": validity(mols),
        "uniqueness": unique_ratio,
        "novelty": novel_ratio,
        "SEDiv": se_div,
        "anion_ratio": anion_ratio,
        "cation_ratio": cation_ratio,
    }
    
    return metrics

print(filename_target, '\t', filename_train, '\t', calculate_metrics(target_smiles, train_smiles))