from torch.utils.data import DataLoader

from data_provider.data_loader import ZhongshanDataset


def data_provider(args, flag):
    dataset = ZhongshanDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target=args.target,
        num_components=args.num_components,
        component_prefix=args.component_prefix,
    )
    if dataset.num_variables != args.num_variables:
        raise ValueError(
            f"--num_variables={args.num_variables}, but the CSV contains "
            f"{dataset.num_variables} meteorological variables"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag == "train",
        num_workers=args.num_workers,
        drop_last=flag == "train",
        pin_memory=args.use_gpu,
    )
    print(f"{flag}: {len(dataset)} samples")
    return dataset, loader
