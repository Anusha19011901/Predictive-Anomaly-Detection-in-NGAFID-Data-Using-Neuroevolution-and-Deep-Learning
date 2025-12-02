import  glob, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

def read_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.loc[:, num.notna().any(axis=0)]
    num = num.ffill().bfill()
    return num

def load_windows(windows_dir, labels_csv):
    # labels_per_window.csv columns: file,label,start_timestamp,end_timestamp
    lab = pd.read_csv(labels_csv)
    m = dict(zip(lab["file"], lab["label"]))
    X_list, y_list, T_set, D_set = [], [], set(), set()

    csvs = sorted(glob.glob(os.path.join(windows_dir, "window_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No window_*.csv found in {windows_dir}")

    for fp in csvs:
        fname = os.path.basename(fp)
        if fname not in m:
            raise ValueError(f"Missing label for {fname} in {labels_csv}")
        df = pd.read_csv(fp)
        num = read_numeric(df)
        X = num.to_numpy().astype("float32")
        T_set.add(X.shape[0]); D_set.add(X.shape[1])
        X_list.append(X); y_list.append(float(m[fname]))

    # Handle unequal T across windows by trimming to the minimum T (common in NAB)
    T = min(T_set); D = list(D_set)[0] if len(D_set)==1 else min(D_set)
    X_list = [x[:T, :D] for x in X_list]
    X = np.stack(X_list, axis=0)   # [N,T,D]
    y = np.array(y_list, dtype="float32")
    return X, y, T, D

def standardize_on_train_normals(X_train, y_train):
    normals = X_train[y_train == 0]
    if normals.size == 0:
        raise RuntimeError("No normal (label==0) samples in train; cannot compute normalization.")
    flat = normals.reshape(-1, normals.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0); std[std==0] = 1.0
    return mean, std

def apply_z(X, mean, std): return ((X - mean) / std).astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--labels_csv",  required=True)
    ap.add_argument("--out", default="data/nab_windows_trainvaltest.npz")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    X, y, T, D = load_windows(args.windows_dir, args.labels_csv)

    # Split at window level with stratification
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    X_tr,  X_va, y_tr,  y_va  = train_test_split(X_trv, y_trv, test_size=args.val_size, stratify=y_trv, random_state=args.seed)

    # Normalize using TRAIN NORMALS only
    mean, std = standardize_on_train_normals(X_tr, y_tr)
    X_tr = apply_z(X_tr, mean, std); X_va = apply_z(X_va, mean, std); X_te = apply_z(X_te, mean, std)

    # Shuffle each split for convenience
    def shuffle_pair(Xs, ys, seed):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Xs))
        return Xs[idx], ys[idx]
    X_tr, y_tr = shuffle_pair(X_tr, y_tr, args.seed)
    X_va, y_va = shuffle_pair(X_va, y_va, args.seed+1)
    X_te, y_te = shuffle_pair(X_te, y_te, args.seed+2)

    np.savez_compressed(args.out,
                        train=X_tr, train_labels=y_tr,
                        val=X_va,  val_labels=y_va,
                        test=X_te, test_labels=y_te)
    print(f"Saved {args.out}")
    print(f"train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}  -> T={T}, D={D}")
    print(f"Train class balance (anomaly=1): {y_tr.mean():.3f}")

if __name__ == "__main__":
    main()
