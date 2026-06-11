import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ZhongshanDataset(Dataset):
    SPLITS = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        root_path,
        data_path,
        flag,
        seq_len,
        pred_len,
        target,
        num_components,
        component_prefix,
    ):
        if flag not in self.SPLITS:
            raise ValueError(f"Unknown split: {flag}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        csv_path = os.path.join(root_path, data_path)
        frame = pd.read_csv(csv_path)
        self._validate_frame(
            frame, target, num_components, component_prefix, csv_path
        )

        component_columns = [
            f"{component_prefix}{index}" for index in range(1, num_components + 1)
        ]
        variable_columns = [
            column
            for column in frame.columns
            if column not in {"date", target, *component_columns}
        ]
        variable_columns.append(target)
        input_columns = variable_columns + component_columns

        values = frame[input_columns].to_numpy(dtype=np.float32)
        num_train = int(len(values) * 0.7)
        num_test = int(len(values) * 0.2)
        num_val = len(values) - num_train - num_test
        borders_start = [
            0,
            num_train - seq_len,
            num_train + num_val - seq_len,
        ]
        borders_end = [num_train, num_train + num_val, len(values)]
        split_index = self.SPLITS[flag]

        train_values = values[:num_train]
        self.mean = train_values.mean(axis=0)
        self.std = train_values.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        normalized = (values - self.mean) / self.std

        start = borders_start[split_index]
        end = borders_end[split_index]
        self.data = normalized[start:end]
        self.target_index = len(variable_columns) - 1
        self.target_mean = float(self.mean[self.target_index])
        self.target_std = float(self.std[self.target_index])
        self.num_variables = len(variable_columns)
        self.input_columns = input_columns

    @staticmethod
    def _validate_frame(
        frame, target, num_components, component_prefix, csv_path
    ):
        required = {"date", target}
        required.update(
            f"{component_prefix}{index}"
            for index in range(1, num_components + 1)
        )
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing)}"
            )
        if len(frame) < 10:
            raise ValueError(f"{csv_path} does not contain enough rows")
        if frame[list(required)].isnull().any().any():
            raise ValueError(f"{csv_path} contains missing values")

    def __getitem__(self, index):
        input_end = index + self.seq_len
        target_end = input_end + self.pred_len
        x = self.data[index:input_end]
        y = self.data[input_end:target_end, self.target_index : self.target_index + 1]
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def denormalize_target_tensor(self, values):
        return values * self.target_std + self.target_mean

    def denormalize_target(self, values):
        return values * self.target_std + self.target_mean
