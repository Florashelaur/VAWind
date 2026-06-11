import importlib
import json
import os
import time

import numpy as np
import torch
from torch import nn

from data_provider.data_factory import data_provider
from exp.exp_basic import ExpBasic
from utils.metrics import metric
from utils.tools import EarlyStopping


class ExpMain(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        module = importlib.import_module(f"models.{args.model}")
        self.model = module.Model(args).float().to(self.device)

    @staticmethod
    def _raw_loss(dataset, predictions, targets, criterion):
        predictions = dataset.denormalize_target_tensor(predictions)
        targets = dataset.denormalize_target_tensor(targets)
        return criterion(predictions, targets)

    def _evaluate_loss(self, dataset, loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                predictions, _ = self.model(batch_x)
                losses.append(
                    self._raw_loss(
                        dataset, predictions, batch_y, criterion
                    ).item()
                )
        self.model.train()
        return float(np.mean(losses))

    def train(self, setting):
        train_data, train_loader = data_provider(self.args, "train")
        val_data, val_loader = data_provider(self.args, "val")
        test_data, test_loader = data_provider(self.args, "test")

        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.args.train_epochs * len(train_loader)),
        )
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            checkpoint_dir=checkpoint_dir,
        )

        for epoch in range(1, self.args.train_epochs + 1):
            start = time.time()
            train_losses = []
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                optimizer.zero_grad()
                predictions, _ = self.model(batch_x)
                loss = self._raw_loss(
                    train_data, predictions, batch_y, criterion
                )
                loss.backward()
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), clip_value=1e3
                )
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            val_loss = self._evaluate_loss(val_data, val_loader, criterion)
            test_loss = self._evaluate_loss(test_data, test_loader, criterion)
            train_loss = float(np.mean(train_losses))
            print(
                f"Epoch {epoch:03d} | train {train_loss:.6f} | "
                f"val {val_loss:.6f} | test {test_loss:.6f} | "
                f"{time.time() - start:.1f}s"
            )
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device)
        )
        return self.model

    def test(self, setting):
        test_data, test_loader = data_provider(self.args, "test")
        predictions = []
        targets = []
        attention = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                output, weights = self.model(batch_x)
                predictions.append(output.cpu().numpy())
                targets.append(batch_y.numpy())
                if weights is not None:
                    attention.append(weights.cpu().numpy())

        predictions = test_data.denormalize_target(
            np.concatenate(predictions, axis=0)
        )
        targets = test_data.denormalize_target(
            np.concatenate(targets, axis=0)
        )
        scores = metric(predictions, targets)

        output_dir = os.path.join(self.args.results, setting)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "predictions.npy"), predictions)
        np.save(os.path.join(output_dir, "targets.npy"), targets)
        if attention:
            np.save(
                os.path.join(output_dir, "attention.npy"),
                np.concatenate(attention, axis=0),
            )
        with open(
            os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(scores, handle, indent=2)

        print(json.dumps(scores, indent=2))
        return scores
