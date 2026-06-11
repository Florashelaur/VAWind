import argparse
import os
import random

import numpy as np
import torch

from exp.exp_main import ExpMain


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAWind with a replaceable InitPredictor"
    )
    parser.add_argument(
        "--model",
        choices=["LSTM", "TCN", "CNN_LSTM"],
        default="TCN",
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--root_path", default="./data")
    parser.add_argument(
        "--target",
        choices=["wind_speed_10m", "wind_speed_100m"],
        default="wind_speed_10m",
    )
    parser.add_argument("--component_prefix", default="IMF_")
    parser.add_argument("--num_components", type=int, default=9)
    parser.add_argument("--num_variables", type=int, default=12)

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--label_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, choices=[3, 6, 9, 12], default=12)
    parser.add_argument("--use_noise", type=int, choices=[0, 1], default=1)
    parser.add_argument("--individual", type=int, choices=[0, 1], default=1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--results", default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.use_gpu = torch.cuda.is_available() and not args.cpu

    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    setting = (
        f"{data_name}_{args.target}_{args.model}_"
        f"sl{args.seq_len}_pl{args.pred_len}_noise{args.use_noise}"
    )
    experiment = ExpMain(args)
    experiment.train(setting)
    experiment.test(setting)


if __name__ == "__main__":
    main()
