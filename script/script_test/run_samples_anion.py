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

# LSTM + Anion
char_dict_anion = SmilesCharDictionary(dataset='anion', max_smi_len=100)

neural_apprentice_LSTM = LSTMGenerator.load(load_dir='/workspace/_ext/models/anion/LSTM/240111_1931')
neural_apprentice_LSTM.to(device)
neural_apprentice_LSTM.eval()

optimizer_LSTM = Adam(neural_apprentice_LSTM.parameters(), lr=1e-3)
apprentice_handler_LSTM = LSTMGeneratorHandler(
    model=neural_apprentice_LSTM,
    optimizer=optimizer_LSTM,
    char_dict=char_dict_anion,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    LSTM_anion_1k, _, _ , _ = apprentice_handler_LSTM.sample(num_samples=1000, device=device)
    with open('LSTM_anion_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(LSTM_anion_1k))

# Transformer Decoder + Anion
char_dict_anion = SmilesCharDictionary(dataset='anion', max_smi_len=100)

neural_apprentice_TransformerDecoder = TransformerDecoderGenerator.load(load_dir='/workspace/_ext/models/anion/TransformerDecoder/240112_0125')
neural_apprentice_TransformerDecoder.to(device)
neural_apprentice_TransformerDecoder.eval()

optimizer_TransformerDecoder = Adam(neural_apprentice_TransformerDecoder.parameters(), lr=1e-3)
apprentice_handler_TransformerDecoder = TransformerDecoderGeneratorHandler(
    model=neural_apprentice_TransformerDecoder,
    optimizer=optimizer_TransformerDecoder,
    char_dict=char_dict_anion,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    TransformerDecoder_anion_1k, _, _ , _ = apprentice_handler_TransformerDecoder.sample(num_samples=1000, device='cpu')
    with open('TransformerDecoder_anion_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(TransformerDecoder_anion_1k))

# Transformer + Anion
char_dict_anion = SmilesCharDictionary(dataset='anion', max_smi_len=100)
dataset_anion = load_dataset(char_dict=char_dict_anion, smiles_path='/workspace/_ext/data/datasets/anion/all.txt')

neural_apprentice_Transformer = TransformerGenerator.load(load_dir='/workspace/_ext/models/anion/Transformer/240111_2318')
neural_apprentice_Transformer.to(device)
neural_apprentice_Transformer.eval()

optimizer_Transformer = Adam(neural_apprentice_Transformer.parameters(), lr=1e-3)
apprentice_handler_Transformer = TransformerGeneratorHandler(
    model=neural_apprentice_Transformer,
    optimizer=optimizer_Transformer,
    char_dict=char_dict_anion,
    max_sampling_batch_size=8192,
)

for iter_ in range(10):
    context_smiles_anion_1k = random.choices(population=dataset_anion, k=1000)
    Transformer_anion_1k, _, _ , _ = apprentice_handler_Transformer.sample(num_samples=1000, context_smiles=context_smiles_anion_1k, device=device)
    with open('Transformer_anion_1k_'+str(iter_)+'.smi', 'w+') as file:
        file.write('\n'.join(Transformer_anion_1k))

