import sys
sys.path.append('/workspace/_ext')

import argparse
import datetime
import random
from pathlib import Path

import torch
from torch.optim import Adam

from sage.data import SmilesCharDictionary, load_dataset
from sage.memory import FragmentLibrary, MaxRewardPriorityMemory, Recorder
from sage.models.apprentice import LSTMGenerator, TransformerGenerator, TransformerDecoderGenerator
from sage.models.handlers import (
    ExplainerHandler,
    GeneticOperatorHandler,
    LSTMGeneratorHandler,
    TransformerGeneratorHandler,
    TransformerDecoderGeneratorHandler,
)
from sage.runners import Trainer, Generator
from sage.utils.load_funcs import (
    load_apprentice_handler,
    load_explainer,
    load_explainer_handler,
    load_genetic_experts,
    load_logger,
    load_neural_apprentice,
)

random.seed(404)
device = 'cpu'

# LSTM + Cation-reduced
char_dict_cation_reduced = SmilesCharDictionary(dataset='cation_reduced', max_smi_len=100)

neural_apprentice_LSTM = LSTMGenerator.load(load_dir='/workspace/_ext/models/cation_reduced/LSTM/240111_0044')
neural_apprentice_LSTM.to(device)
neural_apprentice_LSTM.eval()

optimizer_LSTM = Adam(neural_apprentice_LSTM.parameters(), lr=1e-3)
apprentice_handler_LSTM = LSTMGeneratorHandler(
    model=neural_apprentice_LSTM,
    optimizer=optimizer_LSTM,
    char_dict=char_dict_cation_reduced,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    LSTM_cation_reduced_1k, _, _ , _ = apprentice_handler_LSTM.sample(num_samples=1000, device=device)
    with open('LSTM_cation_reduced_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(LSTM_cation_reduced_1k))

# Transformer Decoder + Cation-reduced
char_dict_cation_reduced = SmilesCharDictionary(dataset='cation_reduced', max_smi_len=100)

neural_apprentice_TransformerDecoder = TransformerDecoderGenerator.load(load_dir='/workspace/_ext/models/cation_reduced/TransformerDecoder/240111_1608')
neural_apprentice_TransformerDecoder.to(device)
neural_apprentice_TransformerDecoder.eval()

optimizer_TransformerDecoder = Adam(neural_apprentice_TransformerDecoder.parameters(), lr=1e-3)
apprentice_handler_TransformerDecoder = TransformerDecoderGeneratorHandler(
    model=neural_apprentice_TransformerDecoder,
    optimizer=optimizer_TransformerDecoder,
    char_dict=char_dict_cation_reduced,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    TransformerDecoder_cation_reduced_1k, _, _ , _ = apprentice_handler_TransformerDecoder.sample(num_samples=1000, device='cpu')
    with open('TransformerDecoder_cation_reduced_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(TransformerDecoder_cation_reduced_1k))

# Transformer + Cation-reduced
char_dict_cation_reduced = SmilesCharDictionary(dataset='cation_reduced', max_smi_len=100)
dataset_cation_reduced = load_dataset(char_dict=char_dict_cation_reduced, smiles_path='/workspace/_ext/data/datasets/cation_reduced/all.txt')

neural_apprentice_Transformer = TransformerGenerator.load(load_dir='/workspace/_ext/models/cation_reduced/Transformer/240111_1036')
neural_apprentice_Transformer.to(device)
neural_apprentice_Transformer.eval()

optimizer_Transformer = Adam(neural_apprentice_Transformer.parameters(), lr=1e-3)
apprentice_handler_Transformer = TransformerGeneratorHandler(
    model=neural_apprentice_Transformer,
    optimizer=optimizer_Transformer,
    char_dict=char_dict_cation_reduced,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    context_smiles_cation_reduced_1k = random.choices(population=dataset_cation_reduced, k=1000)
    Transformer_cation_reduced_1k, _, _ , _ = apprentice_handler_Transformer.sample(num_samples=1000, context_smiles=context_smiles_cation_reduced_1k, device=device)
    with open('Transformer_cation_reduced_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(Transformer_cation_reduced_1k))

