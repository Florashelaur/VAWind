import os

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience, checkpoint_dir, delta=0.0):
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_loss, model):
        score = -validation_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(self.checkpoint_dir, "checkpoint.pth"),
            )
            return

        self.counter += 1
        print(f"EarlyStopping: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
