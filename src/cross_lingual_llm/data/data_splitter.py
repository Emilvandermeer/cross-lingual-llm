"""Create the fixed 70/20/10 train/dev/test split and save it as CSV.

  * Split unit: groups of examples that share an exact English or target line
    (so parallel versions and exact duplicates never cross partitions).
  * Stratification: multi-label iterative stratification (Sechidis et al., 2011)
    on the English + target label vectors together.

Usage (everything is hardcoded: data/raw -> data/splits, English + Romanian, seed 42):
    uv run data_splitter.py
"""
import numpy as np
import pandas as pd

from data_loader import build_examples
from data_utils import EN_COLS, OUT_DIR, SEED, RO_COLS, text_key

SPLITS = ("train", "dev", "test")
RATIOS = (0.7, 0.2, 0.1)


def assign_groups(df):
    """Group id per row: connected components of rows sharing an EN or RO line."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, (en, ro) in enumerate(zip(df["en_text"], df["ro_text"])):
        for key in (("en", text_key(en)), ("ro", text_key(ro))):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    return np.array([find(i) for i in range(len(df))])


def iterative_stratify(Y, ratios, seed):
    """Assign each row of binary matrix Y to a fold (0..len(ratios)-1)."""
    rng = np.random.default_rng(seed)
    Y = (Y > 0).astype(int)
    n = len(Y)
    want_rows = np.array(ratios) * n                       # rows each fold still wants
    want_label = np.array(ratios)[:, None] * Y.sum(0)      # per-fold, per-label wants
    fold = -np.ones(n, dtype=int)

    while (fold < 0).any():
        left = np.flatnonzero(fold < 0)
        counts = Y[left].sum(0)
        present = np.flatnonzero(counts > 0)
        if len(present) == 0:                              # rows with no labels
            for i in left:
                f = int(np.argmax(want_rows))
                fold[i], want_rows[f] = f, want_rows[f] - 1
            break
        j = present[np.argmin(counts[present])]            # rarest remaining label
        for i in rng.permutation(left[Y[left, j] == 1]):
            best = np.flatnonzero(want_label[:, j] == want_label[:, j].max())
            if len(best) > 1:                              # tie-break: fold wanting most rows
                best = best[want_rows[best] == want_rows[best].max()]
            f = int(rng.choice(best))
            fold[i] = f
            want_label[f] -= Y[i]
            want_rows[f] -= 1
    return fold


def split_examples(df, seed):
    """Return {'train': df, 'dev': df, 'test': df}."""
    df = df.copy()
    df["group"] = assign_groups(df)
    # one label vector per group: EN + RO labels (union over members)
    group_y = df.groupby("group")[EN_COLS + RO_COLS].max()
    fold = iterative_stratify(group_y.to_numpy(), RATIOS, seed)
    df["split"] = df["group"].map(dict(zip(group_y.index, fold)))

    out = {}
    for k, name in enumerate(SPLITS):
        part = df[df["split"] == k].drop(columns=["group", "split"])
        out[name] = part.sort_values("id").reset_index(drop=True)
    return out


def check_no_leakage(splits):
    """Raise if an id or an exact line appears in more than one partition."""
    owner = {}
    for name, part in splits.items():
        keys = (list(zip(["id"] * len(part), part["id"]))
                + [("en", text_key(t)) for t in part["en_text"]]
                + [("ro", text_key(t)) for t in part["ro_text"]])
        for key in keys:
            if owner.setdefault(key, name) != name:
                raise AssertionError(f"Leakage: {key} in {owner[key]} and {name}")


def main():
    df = build_examples()
    splits = split_examples(df, SEED)
    check_no_leakage(splits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)

    # summary: one row per split with size and per-label counts
    summary = pd.DataFrame({name: part[EN_COLS + RO_COLS].sum() for name, part in splits.items()}).T
    summary.insert(0, "size", [len(p) for p in splits.values()])
    summary.insert(0, "seed", SEED)
    summary.index.name = "split"
    summary.to_csv(OUT_DIR / "summary.csv")
    print(f"Saved to {OUT_DIR}")
    print(summary[["seed", "size"]])
    print(f"type data: {type(df)}  shape: {df.shape}  columns: {list(df.columns)}")


if __name__ == "__main__":
    main()