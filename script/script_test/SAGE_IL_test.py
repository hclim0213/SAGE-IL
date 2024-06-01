import sys
sys.path.append('/workspace/_ext')


from rdkit import Chem

from sage.scoring.descriptors import(
    score_high_viscosity_cation,
    score_low_viscosity_cation,
    score_high_viscosity_anion,
    score_low_viscosity_anion,
    score_high_melting_point_cation,
    score_low_melting_point_cation,
    score_high_melting_point_anion,
    score_low_melting_point_anion,
    score_high_solubility_co2_cation,
    score_high_solubility_co2_anion,
    score_low_idac_co2_cation,
    score_low_idac_co2_anion,
    high_viscosity_cation,
    low_viscosity_cation,
    high_viscosity_anion,
    low_viscosity_anion,
    high_melting_point_cation,
    low_melting_point_cation,
    high_melting_point_anion,
    low_melting_point_anion,
    high_solubility_co2_cation,
    high_solubility_co2_anion,
    low_idac_co2_cation,
    low_idac_co2_anion,
)

cation_smiles = 'CC(C)N1C=C(C[C@@H](NC(=O)OC(C)(C)C)C(C)=O)[N+](=C1)C(C)C'
anion_smiles = 'C(CC)[B-](F)(F)F'

cation_mol = Chem.MolFromSmiles(cation_smiles)
anion_mol = Chem.MolFromSmiles(anion_smiles)

print('High Viscosity', 'Cation', score_high_viscosity_cation(cation_mol))
print('Low Viscosity', 'Cation', score_low_viscosity_cation(cation_mol))
print('High Viscosity', 'Anion', score_high_viscosity_anion(anion_mol))
print('Low Viscosity', 'Anion', score_low_viscosity_anion(anion_mol))
print('High Tm', 'Cation', score_high_melting_point_cation(cation_mol))
print('Low Tm', 'Cation', score_low_melting_point_cation(cation_mol))
print('High Tm', 'Anion', score_high_melting_point_anion(anion_mol))
print('Low Tm', 'Anion', score_low_melting_point_anion(anion_mol))
print('High Solubility CO2', 'Cation', score_high_solubility_co2_cation(cation_mol))
print('High Solubility CO2', 'Anion', score_high_solubility_co2_anion(anion_mol))
print('Low IDAC CO2', 'Cation', score_low_idac_co2_cation(cation_mol))
print('Low IDAC CO2', 'Anion', score_low_idac_co2_anion(anion_mol))