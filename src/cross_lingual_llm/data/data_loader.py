"""Load the raw XED files and build matched English <-> target-language examples.

Result is a pandas DataFrame with one row per alignment
(en_file | ro_file | en_sentence_id | ro_sentence_id) and columns:
    id, en_text, ro_text, en_anger ... en_trust, ro_anger ... ro_trust

Raw files needed in data/raw:
    en-annotated.tsv, ro-projections.tsv, pairs-ro.txt
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from data_utils import (EN_COLS, LABELS, LANG, OUT_DIR, RAW_DIR, RO_COLS,
                        labels_to_vector, read_tsv_lines, text_key, unquote,
                        vector_to_names)

LABEL_RE = re.compile(r"\d(, \d)*")
MASK_RE = re.compile(r"\[[A-Z]+\]")        # [PERSON], [LOCATION] in English file
PAIR_ROW_RE = re.compile(r"^[a-z]{2,3}/\d{4}/")


def load_english(path):
    """en-annotated.tsv -> list of (text, label_str)."""
    return [(t, l) for t, l in read_tsv_lines(path) if l and LABEL_RE.fullmatch(l)]


def load_projected(path):
    """<lang>-projections.tsv -> {text: label_str}.
    Drops corrupted lines (alignment rows swallowed by a stray quote, 'nan')."""
    out = {}
    for text, labels in read_tsv_lines(path):
        if labels is None or PAIR_ROW_RE.match(text) or not LABEL_RE.fullmatch(labels):
            continue
        text = unquote(text)
        if text and text != "nan" and text not in out:
            out[text] = labels
    return out


def load_pairs(path):
    """pairs-<lang>.txt -> DataFrame, English-source rows only."""
    cols = ["en_file", "ro_file", "en_sid", "ro_sid", "en_text", "ro_text"]
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) == 6 and p[0].startswith("en/"):
                rows.append(p)
    return pd.DataFrame(rows, columns=cols)


def english_labels_for(pairs, english):
    """Return a Series of English label strings aligned to `pairs` (NaN = no unique match).
    Pairs hold raw English text; the annotated file is tokenised and name-masked."""
    exact = defaultdict(set)
    masked = []                                   # (regex, labels) for [PERSON] lines
    for text, labels in english:
        if MASK_RE.search(text):
            parts = text_key(MASK_RE.sub("\x00", text)).split("\x00")
            if sum(map(len, parts)) >= 8:         # skip near-empty patterns
                pattern = "^" + ".{1,40}?".join(re.escape(p) for p in parts) + "$"
                masked.append((re.compile(pattern), labels))
        else:
            exact[text_key(text)].add(labels)

    def lookup(raw_text):
        key = text_key(raw_text)
        found = exact.get(key) or {l for rx, l in masked if rx.match(key)}
        return next(iter(found)) if len(found) == 1 else np.nan   # unique only

    return pairs["en_text"].map(lookup)


def build_examples():
    """Return a DataFrame of matched English <-> Romanian examples."""
    english = load_english(RAW_DIR / "en-annotated.tsv")
    projected = load_projected(RAW_DIR / f"{LANG}-projections.tsv")
    pairs = load_pairs(RAW_DIR / f"pairs-{LANG}.txt")

    pairs["en_labels"] = english_labels_for(pairs, english)
    pairs["ro_labels"] = pairs["ro_text"].map(projected)
    df = pairs.dropna(subset=["en_labels", "ro_labels"]).copy()

    df["id"] = (df["en_file"] + "|" + df["ro_file"] + "|"
                + df["en_sid"] + "|" + df["ro_sid"])
    df = df.drop_duplicates("id")
    df["en_text"] = df["en_text"].str.strip()
    df["ro_text"] = df["ro_text"].str.strip()

    en_y = np.vstack(df["en_labels"].map(labels_to_vector))
    ro_y = np.vstack(df["ro_labels"].map(labels_to_vector))
    out = pd.concat([df[["id", "en_text", "ro_text"]].reset_index(drop=True),
                     pd.DataFrame(en_y, columns=EN_COLS),
                     pd.DataFrame(ro_y, columns=RO_COLS)], axis=1)
    return out


# ---- helpers for reading saved splits in the experiments -----------------
def load_split(split):
    """Read one saved split ('train' | 'dev' | 'test') -> DataFrame."""
    return pd.read_csv(OUT_DIR / f"{split}.csv", keep_default_na=False)


def get_examples(df, side):
    """Single-language view. side = 'en' or 'ro'.
    Returns DataFrame: id, text, 8 label columns, and 'labels' (list of names)."""
    out = df[["id", f"{side}_text"]].rename(columns={f"{side}_text": "text"})
    y = df[[f"{side}_{l}" for l in LABELS]].to_numpy()
    for j, l in enumerate(LABELS):
        out[l] = y[:, j]
    out["labels"] = [vector_to_names(row) for row in y]
    return out.reset_index(drop=True)