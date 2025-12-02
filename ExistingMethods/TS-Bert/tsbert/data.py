import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def make_loader(
    npz_path: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    oversample_anomalies: bool = False,
):
    """
    npz_path: path to your .npz file, e.g. ExistingMethods/TS-Bert/data/nab_windows_trainvaltest.npz
    split:    "train", "val", or "test"
    batch_size: batch size for DataLoader
    shuffle:  standard DataLoader shuffling (ignored if sampler is used)
    oversample_anomalies: if True and split == "train", use a WeightedRandomSampler
                           to oversample the minority (anomaly) class.
    """

    data = np.load(npz_path)

    # X: [N, T, D]
    X = data[split].astype("float32")

    # y: [N], from "<split>_labels" if present; otherwise dummy zeros
    y_key = f"{split}_labels"
    if y_key in data:
        y = data[y_key].astype("float32")
    else:
        # for purely unsupervised pretrain (not your case here, but safe fallback)
        y = np.zeros(X.shape[0], dtype="float32")

    X_tensor = torch.from_numpy(X)       # [N, T, D]
    y_tensor = torch.from_numpy(y)       # [N]

    dataset = TensorDataset(X_tensor, y_tensor)

    # === Oversample anomalies only for the training split ===
    if oversample_anomalies and split == "train":
        # y is float {0.0, 1.0} -> convert to int indices
        y_int = y.astype(int)

        # Count how many of each class
        class_counts = np.bincount(y_int)  # e.g., [num_normals, num_anomalies]

        # Avoid division by zero in case a class is missing
        class_counts[class_counts == 0] = 1

        # Weight = 1 / count -> minority gets higher weight
        class_weights = 1.0 / class_counts  # shape [num_classes]

        # Map each sample to its class weight
        sample_weights = class_weights[y_int]  # shape [N]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),  # one epoch ≈ N samples, but balanced
            replacement=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,     # sampler and shuffle cannot both be set
        )
    else:
        # Regular loader (for val/test or when not oversampling)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return loader
