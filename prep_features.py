# prep_features.py
import json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

AFTER = Path("dataset/after")
ARTF = Path("artifacts/features"); ARTS = Path("artifacts/scalers")
ARTF.mkdir(parents=True, exist_ok=True); ARTS.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = sorted(AFTER.glob("*.csv"))[:2]  # first 2 AFTER flights

# -------- helpers --------

NUMERIC_KEEP_THRESHOLD = 0.90   # keep column if >=90% values become numeric after coercion
VAR_MIN = 1e-10
CORR_THR = 0.98
DROP_TAIL_FRAC = 0.12
MIN_FEATURES_FALLBACK = 8

LIKELY_NON_FEATURES = {
    "flight_id","subseq_id","tailnumber","aircraft_id","date","timestamp","time","phase",
    "maintenance_flag","label","split","id","Index"
}

def read_csv_robust(p: Path) -> pd.DataFrame:
    # low_memory=False for consistent dtype inference across chunks
    return pd.read_csv(p, low_memory=False)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # drop obvious non-feature columns by name (case-insensitive match)
    drop_names = [c for c in df.columns if c.lower() in {n.lower() for n in LIKELY_NON_FEATURES}]
    df = df.drop(columns=drop_names, errors="ignore")

    # attempt to coerce every column to numeric; non-numeric become NaN
    coerced = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    # keep columns that are mostly numeric
    frac_numeric = 1.0 - coerced.isna().mean()
    keep_cols = frac_numeric[frac_numeric >= NUMERIC_KEEP_THRESHOLD].index.tolist()
    kept = coerced[keep_cols]

    # drop all-NaN columns
    kept = kept.dropna(axis=1, how="all")

    return kept

def feature_selection(train_dfs: list[pd.DataFrame]) -> list[str]:
    df = pd.concat([coerce_numeric(x) for x in train_dfs], axis=0, ignore_index=True)

    # variance filter
    var = df.var(skipna=True)
    keep = var[var > VAR_MIN].index.tolist()
    df = df[keep]

    if df.shape[1] == 0:
        return []

    # correlation de-dup
    corr = df.corr(method="pearson", min_periods=100).abs()
    to_drop = set()
    cols = list(df.columns)
    for i, c1 in enumerate(cols):
        if c1 in to_drop: continue
        for c2 in cols[i+1:]:
            if c2 in to_drop: continue
            v = corr.at[c1, c2] if (c1 in corr.index and c2 in corr.columns) else 0.0
            if pd.notna(v) and v > CORR_THR:
                drop = c1 if var.get(c1, 0) < var.get(c2, 0) else c2
                to_drop.add(drop)
    df = df.drop(columns=list(to_drop), errors="ignore")

    # predictability: MI between y(t) and X(t-1) (quick proxy)
    # build shift only on rows without NaNs after shift
    df_shift = df.shift(1)
    mask = ~(df.isna().any(axis=1) | df_shift.isna().any(axis=1))
    if mask.sum() < 200:
        # not enough clean rows; rank by variance as fallback
        ranked = df.var(skipna=True).sort_values(ascending=False).index.tolist()
        k = max(MIN_FEATURES_FALLBACK, int(len(ranked)*(1-DROP_TAIL_FRAC)))
        return ranked[:k]

    X = df_shift.loc[mask].values
    mi_scores = []
    for col in df.columns:
        y = df.loc[mask, col].values
        try:
            mi_arr = mutual_info_regression(X, y, discrete_features=False, random_state=0)
            mi = float(np.mean(mi_arr))
        except Exception:
            mi = 0.0
        mi_scores.append((col, mi))
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    k = max(MIN_FEATURES_FALLBACK, int(len(mi_scores)*(1-DROP_TAIL_FRAC)))
    selected = [c for c,_ in mi_scores[:k]]
    return selected

# -------- main --------

def main():
    if len(TRAIN_FILES) < 2:
        raise SystemExit("Need at least 2 AFTER CSVs in dataset/after for training feature selection.")

    train_dfs = [read_csv_robust(p) for p in TRAIN_FILES]

    feats = feature_selection(train_dfs)
    if not feats:
        # last-ditch fallback: pick top MIN_FEATURES_FALLBACK by variance after coercion
        df_all = pd.concat([coerce_numeric(x) for x in train_dfs], axis=0, ignore_index=True)
        ranked = df_all.var(skipna=True).sort_values(ascending=False).index.tolist()
        feats = ranked[:MIN_FEATURES_FALLBACK]
        if not feats:
            raise SystemExit("After coercion, found 0 usable numeric features. Check your CSVs.")

    # save features
    (ARTF / "selected_features.json").write_text(json.dumps(feats, indent=2))

    # fit scaler on the same two AFTER flights using ONLY these features
    num_train = []
    for p in TRAIN_FILES:
        df = read_csv_robust(p)
        df = coerce_numeric(df)
        # keep only selected features (may miss some if a file lacks the col)
        cols_present = [c for c in feats if c in df.columns]
        if not cols_present:
            continue
        num_train.append(df[cols_present])
    if not num_train:
        raise SystemExit("Selected features not present in the first 2 AFTER files after coercion.")

    train_cat = pd.concat(num_train, axis=0, ignore_index=True)

    # fill remaining NaNs with column medians (for scaler fit only)
    train_cat = train_cat.fillna(train_cat.median(numeric_only=True))

    scaler = StandardScaler().fit(train_cat.values)
    joblib.dump({"mean": scaler.mean_, "scale": scaler.scale_, "features": feats}, ARTS / "standardizer.pkl")

    print(f"✅ Saved {len(feats)} features to artifacts/features/selected_features.json")
    print(f"✅ Saved scaler to artifacts/scalers/standardizer.pkl")

if __name__ == "__main__":
    main()
