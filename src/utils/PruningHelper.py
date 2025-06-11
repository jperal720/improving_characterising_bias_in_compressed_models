import math
import torch
import torch.nn as nn
from typing import List, Tuple

class PruningHelper():
    def get_total_steps(total_samples: int, batch_size:int, num_folds: int, num_epochs:int, train_ratio: float=0.75):
        """
        Calculates all of the training steps taken given relevant data to our training

        :total_samples: Total samples in training dataset
        :batch_size: Batch size used in training
        :num_folds: Number of folds used in Cross-Validation
        :num_epochs: Number of epochs used in trainer
        :train_ratio: Ratio of samples used for training during CV - default: 0.75.

        :Returns:
            total_steps: int
        """

        samples_per_fold = int(total_samples * train_ratio)
        steps_per_epoch = math.ceil(samples_per_fold / batch_size)
        total_steps = int(steps_per_epoch * num_epochs * num_folds)

        return total_steps

    def get_pruning_schedule(total_steps:int, pruning_start:int = 1000, pruning_interval:int = 500) -> List[int]:
        """
        Generates a list of timesteps in which pruning will take place

        :total_steps: number of total steps that will be taken by trainer
        :pruning_start: step at which pruning starts
        :pruning_interval: step interval in which pruning will occur
        
        :Returns:
            List of steps of numbers where pruning should occur
        """

        return list(range(pruning_start, total_steps+1, pruning_interval))


    def get_pruning_rate(pruning_schedule: List[int], global_target_sparsity: float = 0.9) -> float:
        """
        Calculates the pruning rate with respect to the pruning schedule and target_sparsity provided

        :pruning_schedule: pruning schedule provided by the get_pruning_schedule function
        :global_target_sparsity: The desired global sparsity for our model post-training

        Returns:
            pruning_rate: float
        """

        num_pruning = len(pruning_schedule)
        if num_pruning == 0:
            raise ValueError("Non pruning steps on scheduler!")
        
        remaining_weights = 1 - global_target_sparsity
        pruning_rate = 1 - math.pow(remaining_weights, 1/num_pruning)
        pruning_rate = round(pruning_rate, 6)
        return pruning_rate
    
    def get_sparsity(model):
        """
        Calculates the global unstructured sparsity of our model

        :model: PyTorch model

        Returns:
            total_sparsity: int
        """
        total_active_w = 0
        total_w = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_active_w += torch.sum(module.weight == 0).item()
                total_w += module.weight.numel()

        total_sparsity = 100 * (total_active_w / total_w)

        return total_sparsity