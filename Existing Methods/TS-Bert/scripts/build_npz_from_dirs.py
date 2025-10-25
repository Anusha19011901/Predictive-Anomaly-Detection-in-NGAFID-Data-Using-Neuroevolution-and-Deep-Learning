import os, argparse, numpy as np, pandas as pd

def load_dir(dir_path):
    paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")])
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {dir_path}")
    X = [np.load(p).astype("float32") for p in paths]
    return np.stack(X, axis=0), [os.path.basename(p) for p in paths]

def map_labels(filenames, labels_csv):
    if labels_csv is None:
        return None
    df = pd.read_csv(labels_csv)   # columns: filename,label  (label in {0,1})
    m = {row.filename: float(row.label) for _, row in df.iterrows()}
    y = [m.get(fn) for fn in filenames]
    if any(v is None for v in y):
        miss = [fn for fn,v in zip(filenames, y) if v is None]
        raise ValueError(f"Missing labels for {len(miss)} files, e.g., {miss[:5]}")
    return np.array(y, dtype="float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir",   required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--labels_csv", default=None,
                    help="CSV with columns: filename,label (optional)")
    ap.add_argument("--out", default="data/ngafid_windows_trainvaltest.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    Xtr, ftr = load_dir(args.train_dir)
    Xva, fva = load_dir(args.val_dir)
    Xte, fte = load_dir(args.test_dir)

    ytr = map_labels(ftr, args.labels_csv)
    yva = map_labels(fva, args.labels_csv)
    yte = map_labels(fte, args.labels_csv)

    save_kwargs = dict(train=Xtr, val=Xva, test=Xte)
    if ytr is not None: save_kwargs["train_labels"] = ytr
    if yva is not None: save_kwargs["val_labels"]   = yva
    if yte is not None: save_kwargs["test_labels"]  = yte

    np.savez_compressed(args.out, **save_kwargs)
    T, D = Xtr.shape[1], Xtr.shape[2]
    print(f"Saved {args.out}")
    print(f"train: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}  -> window_length(T)={T}, num_features(D)={D}")
    if ytr is not None:
        print(f"labels present; class balance (train): pos={float(ytr.mean()):.3f}")

if __name__ == "__main__":
    main()
