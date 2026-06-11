import torch


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu}" if args.use_gpu else "cpu"
        )
        print(f"Using device: {self.device}")
