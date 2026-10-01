"""Small helpers shared by the data loader and the splitter."""
import re
from pathlib import Path

import numpy as np

# Fixed run configuration (paths are relative to this file, so it runs from anywhere)
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"        # en-annotated.tsv, ro-projections.tsv, pairs-ro.txt
OUT_DIR = ROOT / "data" / "splits"     # train.csv, dev.csv, test.csv, summary.csv
LANG = "ro"                            # assigned language (English is always included)
SEED = 69

# Fixed label order. XED ids 1..8 map to these in this exact order.
LABELS = ["anger", "anticipation", "disgust", "fear",
          "joy", "sadness", "surprise", "trust"]
EN_COLS = [f"en_{l}" for l in LABELS]      # English label columns
RO_COLS = [f"ro_{l}" for l in LABELS]    # target-language label columns - romanian


def labels_to_vector(label_str):
    """'1, 4, 7' -> array([1,0,0,1,0,0,1,0])  (neutral 0 is ignored)."""
    y = np.zeros(len(LABELS), dtype=int)
    for tok in label_str.split(","):
        k = int(tok)
        if 1 <= k <= 8:
            y[k - 1] = 1
    return y


def vector_to_names(y):
    """[1,0,0,1,0,0,1,0] -> ['anger', 'fear', 'surprise']"""
    return [name for name, v in zip(LABELS, y) if v]


def text_key(text):
    """Matching key: lowercase, no leading dashes, no whitespace."""
    text = re.sub(r"^[\-\*\s\.]+", "", text.lower().strip())
    return re.sub(r"\s+", "", text)


def unquote(text):
    """Undo CSV-style quoting (outer quotes removed, doubled quotes collapsed)."""
    if len(text) > 1 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def read_tsv_lines(path):
    """Yield (text, labels) per line. Text may contain tabs, so we split on the
    last tab. Lines without a tab yield (line, None)."""
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "\t" in line:
                text, labels = line.rsplit("\t", 1)
                yield text, labels.strip()
            else:
                yield line, None