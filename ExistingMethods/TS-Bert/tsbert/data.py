import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class NpzWindows(Dataset):
    def __init__(self, npz_path, split, for_pretrain=False):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data[f"{split}"].astype("float32")           # shape [N, T, D]
        self.y = data.get(f"{split}_labels", None)            # optional
        self.for_pretrain = for_pretrain

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        y = None if self.y is None else torch.tensor(self.y[i]).float()
        return x, y

def make_loader(npz_path, split, bs, shuffle, for_pretrain):
    ds = NpzWindows(npz_path, split, for_pretrain=for_pretrain)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, drop_last=False)
