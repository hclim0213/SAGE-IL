"""
Copyright (c) 2022 Hocheol Lim.
"""
from typing import Callable, List

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from guacamol.utils.descriptors import mol_weight, logP, num_H_donors, tpsa, num_atoms, AtomCounter
from guacamol.utils.fingerprints import get_fingerprint
from sage.scoring.score_modifier import ScoreModifier, MinGaussianModifier, MaxGaussianModifier, GaussianModifier
from sage.scoring.scoring_function import ScoringFunctionBasedOnRdkitMol, MoleculewiseScoringFunction
from guacamol.utils.chemistry import smiles_to_rdkit_mol, parse_molecular_formula
from guacamol.utils.math import arithmetic_mean, geometric_mean

from sage.scoring.filters import (
    filter_cycle_len,
    filter_cation,
    filter_anion,
    score_rascore,
)

def filters_set_cation(mol: Mol) -> bool:
    flag = False
    
    flag_cation = filter_cation(mol)
    
    if flag_cation == True:
        flag = True
    
    return flag

def filters_set_anion(mol: Mol) -> bool:
    flag = False
    
    flag_anion = filter_anion(mol)
    
    if flag_anion == True:
        flag = True
    
    return flag


class RdkitScoringFunction_cation(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, descriptor: Callable[[Chem.Mol], float], score_modifier: ScoreModifier = None) -> None:

        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_cation(mol):
            return self.descriptor(mol)
        else:
            return float(-1.0)
        

class RdkitScoringFunction_anion(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, descriptor: Callable[[Chem.Mol], float], score_modifier: ScoreModifier = None) -> None:

        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_anion(mol):
            return self.descriptor(mol)
        else:
            return float(-1.0)
        

class TanimotoScoringFunction_cation(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, target, fp_type, score_modifier: ScoreModifier = None) -> None:

        super().__init__(score_modifier=score_modifier)

        self.target = target
        self.fp_type = fp_type
        target_mol = smiles_to_rdkit_mol(target)
        if target_mol is None:
            raise RuntimeError(f'The similarity target {target} is not a valid molecule.')

        self.ref_fp = get_fingerprint(target_mol, self.fp_type)

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_cation(mol):
            fp = get_fingerprint(mol, self.fp_type)
            return TanimotoSimilarity(fp, self.ref_fp)
        else:
            return float(-1.0)

class TanimotoScoringFunction_anion(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, target, fp_type, score_modifier: ScoreModifier = None) -> None:

        super().__init__(score_modifier=score_modifier)

        self.target = target
        self.fp_type = fp_type
        target_mol = smiles_to_rdkit_mol(target)
        if target_mol is None:
            raise RuntimeError(f'The similarity target {target} is not a valid molecule.')

        self.ref_fp = get_fingerprint(target_mol, self.fp_type)

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_anion(mol):
            fp = get_fingerprint(mol, self.fp_type)
            return TanimotoSimilarity(fp, self.ref_fp)
        else:
            return float(-1.0)

class IsomerScoringFunction_cation(MoleculewiseScoringFunction):

    def __init__(self, molecular_formula: str, mean_function='geometric') -> None:

        super().__init__()

        self.mean_function = self.determine_mean_function(mean_function)
        self.scoring_functions = self.determine_scoring_functions(molecular_formula)

    @staticmethod
    def determine_mean_function(mean_function: str) -> Callable[[List[float]], float]:
        if mean_function == 'arithmetic':
            return arithmetic_mean
        if mean_function == 'geometric':
            return geometric_mean
        raise ValueError(f'Invalid mean function: "{mean_function}"')

    @staticmethod
    def determine_scoring_functions(molecular_formula: str) -> List[RdkitScoringFunction_cation]:
        element_occurrences = parse_molecular_formula(molecular_formula)

        total_number_atoms = sum(element_tuple[1] for element_tuple in element_occurrences)

        # scoring functions for each element
        functions = [RdkitScoringFunction_cation(descriptor=AtomCounter(element),
                                          score_modifier=GaussianModifier(mu=n_atoms, sigma=1.0))
                     for element, n_atoms in element_occurrences]

        # scoring functions for the total number of atoms
        functions.append(RdkitScoringFunction_cation(descriptor=num_atoms,
                                              score_modifier=GaussianModifier(mu=total_number_atoms, sigma=2.0)))

        return functions

    def raw_score(self, smiles: str) -> float:
        
        if filters_set_cation(Chem.MolFromSmiles(smiles)):
            # return the average of all scoring functions
            scores = [f.score(smiles) for f in self.scoring_functions]
            if self.corrupt_score in scores:
                return self.corrupt_score
            return self.mean_function(scores)
        else:
            return float(-1.0)

class IsomerScoringFunction_anion(MoleculewiseScoringFunction):

    def __init__(self, molecular_formula: str, mean_function='geometric') -> None:

        super().__init__()

        self.mean_function = self.determine_mean_function(mean_function)
        self.scoring_functions = self.determine_scoring_functions(molecular_formula)

    @staticmethod
    def determine_mean_function(mean_function: str) -> Callable[[List[float]], float]:
        if mean_function == 'arithmetic':
            return arithmetic_mean
        if mean_function == 'geometric':
            return geometric_mean
        raise ValueError(f'Invalid mean function: "{mean_function}"')

    @staticmethod
    def determine_scoring_functions(molecular_formula: str) -> List[RdkitScoringFunction_anion]:
        element_occurrences = parse_molecular_formula(molecular_formula)

        total_number_atoms = sum(element_tuple[1] for element_tuple in element_occurrences)

        # scoring functions for each element
        functions = [RdkitScoringFunction_anion(descriptor=AtomCounter(element),
                                          score_modifier=GaussianModifier(mu=n_atoms, sigma=1.0))
                     for element, n_atoms in element_occurrences]

        # scoring functions for the total number of atoms
        functions.append(RdkitScoringFunction_anion(descriptor=num_atoms,
                                              score_modifier=GaussianModifier(mu=total_number_atoms, sigma=2.0)))

        return functions

    def raw_score(self, smiles: str) -> float:
        
        if filters_set_anion(Chem.MolFromSmiles(smiles)):
            # return the average of all scoring functions
            scores = [f.score(smiles) for f in self.scoring_functions]
            if self.corrupt_score in scores:
                return self.corrupt_score
            return self.mean_function(scores)
        else:
            return float(-1.0)

class SMARTSScoringFunction_cation(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, target: str, inverse=False) -> None:

        super().__init__()
        self.inverse = inverse
        self.smarts = target
        self.target = Chem.MolFromSmarts(target)

        assert target is not None

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_cation(mol):
            
            matches = mol.GetSubstructMatches(self.target)
            
            if len(matches) > 0:
                if self.inverse:
                    return 0.0
                else:
                    return 1.0
            else:
                if self.inverse:
                    return 1.0
                else:
                    return 0.0
        else:
            return float(-1.0)

class SMARTSScoringFunction_anion(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, target: str, inverse=False) -> None:

        super().__init__()
        self.inverse = inverse
        self.smarts = target
        self.target = Chem.MolFromSmarts(target)

        assert target is not None

    def score_mol(self, mol: Chem.Mol) -> float:
        
        if filters_set_anion(mol):
            
            matches = mol.GetSubstructMatches(self.target)
            
            if len(matches) > 0:
                if self.inverse:
                    return 0.0
                else:
                    return 1.0
            else:
                if self.inverse:
                    return 1.0
                else:
                    return 0.0
        else:
            return float(-1.0)

def score_rascore(mol: Mol) -> float:
    from RAscore import RAscore_XGB
    
    compound = str(Chem.MolToSmiles(mol))
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    score = xgb_scorer.predict(compound)
    return float(score)

def score_high_viscosity_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.max(viscosity_results)
    viscosity_row = np.argmax(viscosity_results)
    
    del clf
    del model
    return float(viscosity_score)

def score_low_viscosity_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.min(viscosity_results)
    viscosity_row = np.argmin(viscosity_results)
    
    del clf
    del model
    return float(viscosity_score)

def score_high_viscosity_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.max(viscosity_results)
    viscosity_row = np.argmax(viscosity_results)
    
    del clf
    del model
    return float(viscosity_score)

def score_low_viscosity_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.min(viscosity_results)
    viscosity_row = np.argmin(viscosity_results)
    
    del clf
    del model
    return float(viscosity_score)

def high_viscosity_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.max(viscosity_results)
    viscosity_row = np.argmax(viscosity_results)
    
    score = viscosity_score
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{viscosity_row+1:04d}"+"\t"+str(round(viscosity_score,3))+"' >> SAGE_IL_cation_high_viscosity.txt")
    del clf
    del model
    return float(score)

def low_viscosity_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.min(viscosity_results)
    viscosity_row = np.argmin(viscosity_results)
    
    score = 10 - viscosity_score
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{viscosity_row+1:04d}"+"\t"+str(round(viscosity_score,3))+"' >> SAGE_IL_cation_low_viscosity.txt")
    del clf
    del model
    return float(score)

def high_viscosity_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.max(viscosity_results)
    viscosity_row = np.argmax(viscosity_results)
    
    score = viscosity_score
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{viscosity_row+1:04d}"+"\t"+str(round(viscosity_score,3))+"' >> SAGE_IL_anion_high_viscosity.txt")
    del clf
    del model
    return float(score)

def low_viscosity_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Viscosity
    temp_model = 'IL_Viscosity_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Viscosity'].to_numpy()]
    
    viscosity_results = clf.predict(temp_fp)
    viscosity_score = np.min(viscosity_results)
    viscosity_row = np.argmin(viscosity_results)
    
    score = 10 - viscosity_score
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{viscosity_row+1:04d}"+"\t"+str(round(viscosity_score,3))+"' >> SAGE_IL_anion_low_viscosity.txt")
    
    del clf
    del model
    return float(score)

def score_high_melting_point_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.max(tm_results)
    tm_row = np.argmax(tm_results)

    del clf
    del model
    return float(tm_score)

def score_low_melting_point_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.min(tm_results)
    tm_row = np.argmin(tm_results)

    del clf
    del model
    return float(tm_score)

def score_high_melting_point_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.max(tm_results)
    tm_row = np.argmax(tm_results)

    del clf
    del model
    return float(tm_score)

def score_low_melting_point_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.min(tm_results)
    tm_row = np.argmin(tm_results)
    
    del clf
    del model
    return float(tm_score)

def high_melting_point_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.max(tm_results)
    tm_row = np.argmax(tm_results)
    
    score = tm_score + 273.15
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{tm_row+1:04d}"+"\t"+str(round(tm_score,3))+"' >> SAGE_IL_cation_high_melting_point.txt")
    del clf
    del model
    return float(score)

def low_melting_point_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.min(tm_results)
    tm_row = np.argmin(tm_results)
    
    score = 500 - tm_score
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{tm_row+1:04d}"+"\t"+str(round(tm_score,3))+"' >> SAGE_IL_cation_low_melting_point.txt")

    del clf
    del model
    return float(score)

def high_melting_point_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.max(tm_results)
    tm_row = np.argmax(tm_results)
    
    score = tm_score + 273.15
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{tm_row+1:04d}"+"\t"+str(round(tm_score,3))+"' >> SAGE_IL_anion_high_melting_point.txt")
    
    del clf
    del model
    return float(score)

def low_melting_point_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    # Density
    temp_model = 'IL_Tm_LGBM'
    clf = pickle.load(open('/home/il_qspr/'+temp_model+'.pkl', 'rb'))
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp = temp_fp_2[:, pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)['Tm'].to_numpy()]
    
    tm_results = clf.predict(temp_fp)
    tm_score = np.min(tm_results)
    tm_row = np.argmin(tm_results)
    
    score = 500 - tm_score
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{tm_row+1:04d}"+"\t"+str(round(tm_score,3))+"' >> SAGE_IL_anion_low_melting_point.txt")
    
    del clf
    del model
    return float(score)

def score_high_solubility_co2_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    #temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 Solubility
    temp_model_co2_solubility = 'IL_CO2_Solubility_LGBM'
    clf_co2_solubility = pickle.load(open('/home/il_qspr/'+temp_model_co2_solubility+'.pkl', 'rb'))
    
    temp_fp_co2_solubility = temp_fp_2[:, il_qspr_mask['CO2_Solubility'].to_numpy()]
    co2_solubility_results = clf_co2_solubility.predict(temp_fp_co2_solubility)
    co2_solubility_score_ = np.max(co2_solubility_results)
    co2_solubility_row = np.argmax(co2_solubility_results)
    
    del clf_co2_solubility
    del model
    return float(co2_solubility_score_)

def score_high_solubility_co2_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    #temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 Solubility
    temp_model_co2_solubility = 'IL_CO2_Solubility_LGBM'
    clf_co2_solubility = pickle.load(open('/home/il_qspr/'+temp_model_co2_solubility+'.pkl', 'rb'))
    
    temp_fp_co2_solubility = temp_fp_2[:, il_qspr_mask['CO2_Solubility'].to_numpy()]
    co2_solubility_results = clf_co2_solubility.predict(temp_fp_co2_solubility)
    co2_solubility_score_ = np.max(co2_solubility_results)
    co2_solubility_row = np.argmax(co2_solubility_results)
    
    del model
    del clf_co2_solubility
    return float(co2_solubility_score_)

def high_solubility_co2_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    #temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 Solubility
    temp_model_co2_solubility = 'IL_CO2_Solubility_LGBM'
    clf_co2_solubility = pickle.load(open('/home/il_qspr/'+temp_model_co2_solubility+'.pkl', 'rb'))
    
    temp_fp_co2_solubility = temp_fp_2[:, il_qspr_mask['CO2_Solubility'].to_numpy()]
    co2_solubility_results = clf_co2_solubility.predict(temp_fp_co2_solubility)
    co2_solubility_score_ = np.max(co2_solubility_results)
    co2_solubility_row = np.argmax(co2_solubility_results)
    
    co2_solubility_score = co2_solubility_score_
    
    # Viscosity
    temp_model_viscosity = 'IL_Viscosity_LGBM'
    clf_viscosity = pickle.load(open('/home/il_qspr/'+temp_model_viscosity+'.pkl', 'rb'))
    
    temp_fp_viscosity = temp_fp_2[:, il_qspr_mask['Viscosity'].to_numpy()]
    viscosity_results = clf_viscosity.predict(temp_fp_viscosity)
    viscosity_score_ = viscosity_results[co2_solubility_row]
    
    if viscosity_score_ <= 1:
        viscosity_score = 1.0
    elif viscosity_score_ <= 3:
        viscosity_score = float(1-(viscosity_score_-1)/2)
    else:
        viscosity_score = 0.0
    
    # Density
    temp_model_density = 'IL_Density_XGB'
    clf_density = pickle.load(open('/home/il_qspr/'+temp_model_density+'.pkl', 'rb'))
    
    temp_fp_density = temp_fp_2[:, il_qspr_mask['Density'].to_numpy()]
    density_results = clf_density.predict(temp_fp_density)
    density_score_ = density_results[co2_solubility_row]
    
    if density_score_ <= 3:
        density_score = 1.0
    elif density_score_ <= 3.3:
        density_score = float(1-(density_score_-3)/0.3)
    else:
        density_score = 0.0
    
    # Tm
    temp_model_tm = 'IL_Tm_LGBM'
    clf_tm = pickle.load(open('/home/il_qspr/'+temp_model_tm+'.pkl', 'rb'))

    temp_fp_tm = temp_fp_2[:, il_qspr_mask['Tm'].to_numpy()]
    tm_results = clf_tm.predict(temp_fp_tm)
    tm_score_ = tm_results[co2_solubility_row]
    
    if tm_score_ <= 25:
        tm_score = 1.0
    elif tm_score_ <= 100:
        tm_score = float(1-(tm_score_-25)/75)
    else:
        tm_score = 0.0
    
    # RAscore
    ra_score = float(score_rascore(mol))
    
    score = np.mean([co2_solubility_score, viscosity_score, density_score, tm_score, ra_score])
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{co2_solubility_row+1:04d}"+"\t"+str(round(co2_solubility_score_,3))+"\t"+str(round(viscosity_score_,3))+"\t"+str(round(density_score_,3))+"\t"+str(round(tm_score_,3))+"\t"+str(round(ra_score,3))+"' >> SAGE_IL_cation_high_solubility_co2.txt")
    
    del model
    del clf_co2_solubility
    del clf_viscosity
    del clf_density
    del clf_tm
    return float(score)

def high_solubility_co2_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    #solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    #temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 Solubility
    temp_model_co2_solubility = 'IL_CO2_Solubility_LGBM'
    clf_co2_solubility = pickle.load(open('/home/il_qspr/'+temp_model_co2_solubility+'.pkl', 'rb'))
    
    temp_fp_co2_solubility = temp_fp_2[:, il_qspr_mask['CO2_Solubility'].to_numpy()]
    co2_solubility_results = clf_co2_solubility.predict(temp_fp_co2_solubility)
    co2_solubility_score_ = np.max(co2_solubility_results)
    co2_solubility_row = np.argmax(co2_solubility_results)
    
    co2_solubility_score = co2_solubility_score_
    
    # Viscosity
    temp_model_viscosity = 'IL_Viscosity_LGBM'
    clf_viscosity = pickle.load(open('/home/il_qspr/'+temp_model_viscosity+'.pkl', 'rb'))
    
    temp_fp_viscosity = temp_fp_2[:, il_qspr_mask['Viscosity'].to_numpy()]
    viscosity_results = clf_viscosity.predict(temp_fp_viscosity)
    viscosity_score_ = viscosity_results[co2_solubility_row]
    
    if viscosity_score_ <= 1:
        viscosity_score = 1.0
    elif viscosity_score_ <= 3:
        viscosity_score = float(1-(viscosity_score_-1)/2)
    else:
        viscosity_score = 0.0
    
    # Density
    temp_model_density = 'IL_Density_XGB'
    clf_density = pickle.load(open('/home/il_qspr/'+temp_model_density+'.pkl', 'rb'))
    
    temp_fp_density = temp_fp_2[:, il_qspr_mask['Density'].to_numpy()]
    density_results = clf_density.predict(temp_fp_density)
    density_score_ = density_results[co2_solubility_row]
    
    if density_score_ <= 3:
        density_score = 1.0
    elif density_score_ <= 3.3:
        density_score = float(1-(density_score_-3)/0.3)
    else:
        density_score = 0.0
    
    # Tm
    temp_model_tm = 'IL_Tm_LGBM'
    clf_tm = pickle.load(open('/home/il_qspr/'+temp_model_tm+'.pkl', 'rb'))

    temp_fp_tm = temp_fp_2[:, il_qspr_mask['Tm'].to_numpy()]
    tm_results = clf_tm.predict(temp_fp_tm)
    tm_score_ = tm_results[co2_solubility_row]
    
    if tm_score_ <= 25:
        tm_score = 1.0
    elif tm_score_ <= 100:
        tm_score = float(1-(tm_score_-25)/75)
    else:
        tm_score = 0.0
    
    # RAscore
    ra_score = float(score_rascore(mol))
    
    score = np.mean([co2_solubility_score, viscosity_score, density_score, tm_score, ra_score])
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{co2_solubility_row+1:04d}"+"\t"+str(round(co2_solubility_score_,3))+"\t"+str(round(viscosity_score_,3))+"\t"+str(round(density_score_,3))+"\t"+str(round(tm_score_,3))+"\t"+str(round(ra_score,3))+"' >> SAGE_IL_anion_high_solubility_co2.txt")
    
    del model
    del clf_co2_solubility
    del clf_viscosity
    del clf_density
    del clf_tm
    return float(score)

def score_low_idac_co2_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)

    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 IDAC
    temp_model_co2_idac = 'IL_IDAC_XGB'
    clf_co2_idac = pickle.load(open('/home/il_qspr/'+temp_model_co2_idac+'.pkl', 'rb'))
    
    temp_fp_co2_idac = temp_fp_3[:, il_qspr_mask['IDAC'].to_numpy()]
    co2_idac_results = clf_co2_idac.predict(temp_fp_co2_idac)
    co2_idac_score_ = np.min(co2_idac_results)
    co2_idac_row = np.argmin(co2_idac_results)

    del model
    del clf_co2_idac
    return float(co2_idac_score_)

def score_low_idac_co2_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 IDAC
    temp_model_idac = 'IL_IDAC_XGB'
    clf_co2_idac = pickle.load(open('/home/il_qspr/'+temp_model_idac+'.pkl', 'rb'))
    
    temp_fp_co2_idac = temp_fp_3[:, il_qspr_mask['IDAC'].to_numpy()]
    co2_idac_results = clf_co2_idac.predict(temp_fp_co2_idac)
    co2_idac_score_ = np.min(co2_idac_results)
    co2_idac_row = np.argmin(co2_idac_results)
    
    del model
    del clf_co2_idac
    return float(co2_idac_score_)

def low_idac_co2_cation(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_cation(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    #cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(anion_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), np.tile(temp_fp_1, (il_iter, 1)), anion_fp, np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 IDAC
    temp_model_co2_idac = 'IL_IDAC_XGB'
    clf_co2_idac = pickle.load(open('/home/il_qspr/'+temp_model_co2_idac+'.pkl', 'rb'))
    
    temp_fp_co2_idac = temp_fp_3[:, il_qspr_mask['IDAC'].to_numpy()]
    co2_idac_results = clf_co2_idac.predict(temp_fp_co2_idac)
    co2_idac_score_ = np.min(co2_idac_results)
    co2_idac_row = np.argmin(co2_idac_results)
    
    co2_idac_score = 5 - co2_idac_score_
    
    # Viscosity
    temp_model_viscosity = 'IL_Viscosity_LGBM'
    clf_viscosity = pickle.load(open('/home/il_qspr/'+temp_model_viscosity+'.pkl', 'rb'))
    
    temp_fp_viscosity = temp_fp_2[:, il_qspr_mask['Viscosity'].to_numpy()]
    viscosity_results = clf_viscosity.predict(temp_fp_viscosity)
    viscosity_score_ = viscosity_results[co2_idac_row]
    
    if viscosity_score_ <= 1:
        viscosity_score = 1.0
    elif viscosity_score_ <= 3:
        viscosity_score = float(1-(viscosity_score_-1)/2)
    else:
        viscosity_score = 0.0
    
    # Density
    temp_model_density = 'IL_Density_XGB'
    clf_density = pickle.load(open('/home/il_qspr/'+temp_model_density+'.pkl', 'rb'))
    
    temp_fp_density = temp_fp_2[:, il_qspr_mask['Density'].to_numpy()]
    density_results = clf_density.predict(temp_fp_density)
    density_score_ = density_results[co2_idac_row]
    
    if density_score_ <= 3:
        density_score = 1.0
    elif density_score_ <= 3.3:
        density_score = float(1-(density_score_-3)/0.3)
    else:
        density_score = 0.0
    
    # Tm
    temp_model_tm = 'IL_Tm_LGBM'
    clf_tm = pickle.load(open('/home/il_qspr/'+temp_model_tm+'.pkl', 'rb'))

    temp_fp_tm = temp_fp_2[:, il_qspr_mask['Tm'].to_numpy()]
    tm_results = clf_tm.predict(temp_fp_tm)
    tm_score_ = tm_results[co2_idac_row]
    
    if tm_score_ <= 25:
        tm_score = 1.0
    elif tm_score_ <= 100:
        tm_score = float(1-(tm_score_-25)/75)
    else:
        tm_score = 0.0
    
    # RAscore
    ra_score = float(score_rascore(mol))
    
    score = np.mean([co2_idac_score, viscosity_score, density_score, tm_score, ra_score])
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Anion_"+f"{co2_idac_row+1:04d}"+"\t"+str(round(co2_idac_score_,3))+"\t"+str(round(viscosity_score_,3))+"\t"+str(round(density_score_,3))+"\t"+str(round(tm_score_,3))+"\t"+str(round(ra_score,3))+"' >> SAGE_IL_cation_low_idac_co2.txt")
    
    del model
    del clf_co2_idac
    del clf_viscosity
    del clf_density
    del clf_tm
    return float(score)

def low_idac_co2_anion(mol: Mol) -> float:
    import sys
    sys.path.append('/workspace/_ext')
    sys.path.append('/home/MFBERT')
    
    import os
    import uuid
    import subprocess
    import pandas as pd
    import re
    import time
    
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.ensemble import VotingClassifier
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    flag = filter_anion(mol)
    
    if flag == False:
        return float(-1.0)
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    # Exp. Contidion
    condition = np.expand_dims([0.41732033, 0.00009649], axis=0)
    
    compound = Chem.MolToSmiles(mol)
    temp_maccs_keys = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
    temp_ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList()
    temp_FCFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024).ToList()
    temp_PCFP = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
    temp_MFBERT = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
    
    temp_fp_1 = np.expand_dims(np.concatenate((temp_maccs_keys, temp_ECFP6, temp_FCFP4, temp_PCFP, temp_MFBERT)), axis=0)
    
    cation_fp = pd.read_csv('/home/il_qspr/IL_features_cation.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    #anion_fp = pd.read_csv('/home/il_qspr/IL_features_anion.tsv', sep='\t', header=0).iloc[:, 1:].to_numpy()
    solute_0_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_000'].iloc[:, 1:].to_numpy()
    solute_127_fp = pd.read_csv('/home/il_qspr/IL_features_solute.tsv', sep='\t', header=0)[lambda x: x['Name'] == 'Solute_127'].iloc[:, 1:].to_numpy()
    il_iter = np.shape(cation_fp)[0]
    
    temp_fp_2 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_0_fp, (il_iter, 1))), axis=1)
    temp_fp_3 = np.concatenate((np.tile(condition, (il_iter, 1)), cation_fp, np.tile(temp_fp_1, (il_iter, 1)), np.tile(solute_127_fp, (il_iter, 1))), axis=1)
    
    il_qspr_mask = pd.read_csv('/home/il_qspr/IL_QSPR_mask.tsv', sep='\t', header=0)
    
    # CO2 IDAC
    temp_model_idac = 'IL_IDAC_XGB'
    clf_co2_idac = pickle.load(open('/home/il_qspr/'+temp_model_idac+'.pkl', 'rb'))
    
    temp_fp_co2_idac = temp_fp_3[:, il_qspr_mask['IDAC'].to_numpy()]
    co2_idac_results = clf_co2_idac.predict(temp_fp_co2_idac)
    co2_idac_score_ = np.min(co2_idac_results)
    co2_idac_row = np.argmin(co2_idac_results)
    
    co2_idac_score = 5 - co2_idac_score_
    
    # Viscosity
    temp_model_viscosity = 'IL_Viscosity_LGBM'
    clf_viscosity = pickle.load(open('/home/il_qspr/'+temp_model_viscosity+'.pkl', 'rb'))
    
    temp_fp_viscosity = temp_fp_2[:, il_qspr_mask['Viscosity'].to_numpy()]
    viscosity_results = clf_viscosity.predict(temp_fp_viscosity)
    viscosity_score_ = viscosity_results[co2_idac_row]
    
    if viscosity_score_ <= 1:
        viscosity_score = 1.0
    elif viscosity_score_ <= 3:
        viscosity_score = float(1-(viscosity_score_-1)/2)
    else:
        viscosity_score = 0.0
    
    # Density
    temp_model_density = 'IL_Density_XGB'
    clf_density = pickle.load(open('/home/il_qspr/'+temp_model_density+'.pkl', 'rb'))
    
    temp_fp_density = temp_fp_2[:, il_qspr_mask['Density'].to_numpy()]
    density_results = clf_density.predict(temp_fp_density)
    density_score_ = density_results[co2_idac_row]
    
    if density_score_ <= 3:
        density_score = 1.0
    elif density_score_ <= 3.3:
        density_score = float(1-(density_score_-3)/0.3)
    else:
        density_score = 0.0
    
    # Tm
    temp_model_tm = 'IL_Tm_LGBM'
    clf_tm = pickle.load(open('/home/il_qspr/'+temp_model_tm+'.pkl', 'rb'))

    temp_fp_tm = temp_fp_2[:, il_qspr_mask['Tm'].to_numpy()]
    tm_results = clf_tm.predict(temp_fp_tm)
    tm_score_ = tm_results[co2_idac_row]
    
    if tm_score_ <= 25:
        tm_score = 1.0
    elif tm_score_ <= 100:
        tm_score = float(1-(tm_score_-25)/75)
    else:
        tm_score = 0.0
    
    # RAscore
    ra_score = float(score_rascore(mol))
    
    score = np.mean([co2_idac_score, viscosity_score, density_score, tm_score, ra_score])
    
    os.system("echo '"+str(compound)+"\t"+str(flag)+"\t"+"Cation_"+f"{co2_idac_row+1:04d}"+"\t"+str(round(co2_idac_score_,3))+"\t"+str(round(viscosity_score_,3))+"\t"+str(round(density_score_,3))+"\t"+str(round(tm_score_,3))+"\t"+str(round(ra_score,3))+"' >> SAGE_IL_anion_low_idac_co2.txt")
    
    del model
    del clf_co2_idac
    del clf_viscosity
    del clf_density
    del clf_tm
    return float(score)

def score_low_idac_artemisinin_cation(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_betulin_cation(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_caffeine_cation(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_norgalanthamine_cation(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_piperine_cation(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_cianidanol_anion(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_galantamine_anion(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_glaucine_anion(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_shikimate_anion(mol: Mol) -> float:
    import sys
    return float(1)
def score_low_idac_ungiminorine_anion(mol: Mol) -> float:
    import sys
    return float(1)

def low_idac_artemisinin_cation(mol: Mol) -> float:
    return float(1)
def low_idac_betulin_cation(mol: Mol) -> float:
    return float(1)
def low_idac_caffeine_cation(mol: Mol) -> float:
    return float(1)
def low_idac_norgalanthamine_cation(mol: Mol) -> float:
    return float(1)
def low_idac_piperine_cation(mol: Mol) -> float:
    return float(1)
def low_idac_cianidanol_anion(mol: Mol) -> float:
    return float(1)
def low_idac_galantamine_anion(mol: Mol) -> float:
    return float(1)
def low_idac_glaucine_anion(mol: Mol) -> float:
    return float(1)
def low_idac_shikimate_anion(mol: Mol) -> float:
    return float(1)
def low_idac_ungiminorine_anion(mol: Mol) -> float:
    return float(1)