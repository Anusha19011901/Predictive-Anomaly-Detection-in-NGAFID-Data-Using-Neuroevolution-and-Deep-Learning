# add this at the very top of each script in scripts/
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, yaml, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from tsbert.model import TimeSeriesBERT
from tsbert.data import make_loader

def sigmoid(x): return 1/(1+np.exp(-x))

if __name__ == "__main__":
    import argparse; parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Existing Methods/TS-Bert/configs/base.yaml")
    parser.add_argument("--ckpt",   default="Existing Methods/TS-Bert/checkpoints/tsbert_finetuned.ckpt")
    parser.add_argument("--split",  default="test")
    parser.add_argument("--outfile", default="Existing Methods/TS-Bert/outputs/tsbert_window_scores.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    state = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    m = TimeSeriesBERT(cfg["num_features"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                       cfg["ff_mult"], cfg["dropout"], cfg["max_len"])
    m.load_state_dict(state, strict=False); m.eval()

    loader = make_loader(cfg["train_npz"], args.split, bs=64, shuffle=False, for_pretrain=False)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = m.classify(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            if y is not None: all_labels.append(y.numpy())
    scores = np.concatenate(all_scores)
    df = pd.DataFrame({"score": scores})
    if len(all_labels)>0:
        labels = np.concatenate(all_labels)
        df["label"] = labels
    os.makedirs(cfg["outdir"], exist_ok=True)
    df.to_csv(args.outfile, index=False)
    print(f"Wrote: {args.outfile}")
