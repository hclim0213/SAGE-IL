"""
Copyright (c) 2022 Hocheol Lim.
"""

from typing import List, Optional, Union

import torch
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import Parallel
from tqdm import tqdm

from sage.memory import Recorder
from sage.runners.trainer import Trainer


class Generator(GoalDirectedGenerator):
    def __init__(
        self,
        trainer: Trainer,
        recorder: Recorder,
        num_steps: int,
        device: torch.device,
        scoring_num_list: List[int],
        num_jobs: int,
        dataset_type: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.recorder = recorder
        self.num_steps = num_steps
        self.device = device
        self.scoring_num_list = scoring_num_list
        self.dataset_type = dataset_type

        self.pool = Parallel(n_jobs=num_jobs)

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        self.trainer.init(
            scoring_function=scoring_function, device=self.device, pool=self.pool
        )
        for step in tqdm(range(self.num_steps)):
            smiles, scores = self.trainer.step(
                scoring_function=scoring_function, device=self.device, pool=self.pool
            )

            self.recorder.add_list(smiles=smiles, scores=scores)
            current_score = self.recorder.get_and_log_score()

            temp_file_smiles = open(self.recorder.save_dir+'step_'+str(step+1).zfill(3)+'_smiles.txt', 'w')
            temp_file_smiles.write('\n'.join(smiles))

            temp_file_scores = open(self.recorder.save_dir+'step_'+str(step+1).zfill(3)+'_scores.txt', 'w')
            temp_file_scores.write('\n'.join(map(str, scores)))

            temp_file_smiles.close()
            temp_file_scores.close()
            
            if self.dataset_type == "guacamol":
                if current_score == 1.0:
                    break
                if round(current_score,3) == 1.0:
                    break
            
            if self.dataset_type == "cation_reduced" or self.dataset_type == "anion_reduced":
                if round(current_score,3) == 1.0:
                    break

        self.recorder.log_final()
        self.trainer.log_fragments()
        best_smiles, best_scores = self.recorder.get_topk(top_k=number_molecules)
        return best_smiles

