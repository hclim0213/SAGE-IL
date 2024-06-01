"""
Copyright (c) 2022 Hocheol Lim.
"""

from pathlib import Path
from typing import List

from sage.data.char_dict import SmilesCharDictionary


def load_pretrain_dataset(char_dict: SmilesCharDictionary, smiles_path: str, code='train') -> List[str]:
    
    smiles_path = smiles_path + '/' + code + '.txt'
    processed_dataset_path = (
        str(Path(smiles_path).with_suffix("")) + "_processed.smiles"
    )

    if Path(processed_dataset_path).exists():
        with open(processed_dataset_path, "r") as file:
            processed_dataset = file.read().splitlines()

    else:
        with open(smiles_path, "r") as file:
            dataset = file.read().splitlines()

        processed_dataset = list(filter(char_dict.is_allowed, dataset))
        with open(processed_dataset_path, "w") as file:
            file.writelines("\n".join(processed_dataset))

    return processed_dataset


def load_dataset(char_dict: SmilesCharDictionary, smiles_path: str) -> List[str]:
    processed_dataset_path = (
        str(Path(smiles_path).with_suffix("")) + "_processed.smiles"
    )

    if Path(processed_dataset_path).exists():
        with open(processed_dataset_path, "r") as file:
            processed_dataset = file.read().splitlines()

    else:
        with open(smiles_path, "r") as file:
            dataset = file.read().splitlines()

        processed_dataset = list(filter(char_dict.is_allowed, dataset))
        with open(processed_dataset_path, "w") as file:
            file.writelines("\n".join(processed_dataset))

    return processed_dataset