"""Helper file containg shared functions and constants."""
import re
from pathlib import Path

import numpy as np
import json

# Fixed run configuration (paths are relative to this file, so it runs from anywhere)
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"        # en-annotated.tsv, ro-projections.tsv, pairs-ro.txt
OUT_DIR = ROOT / "data" / "splits"     # train.csv, dev.csv, test.csv, summary.csv
LANG = "ro"                            # assigned language (English is always included)
SEED = 69
PRIMARY_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" # the primary model in use
SECONDARY_MODEL_ID = "google/gemma-2-2b-it" # the secondary model in use

# Fixed label order. XED ids 1..8 map to these in this exact order.
LABELS = ["anger", "anticipation", "disgust", "fear",
          "joy", "sadness", "surprise", "trust"]
EN_COLS = [f"en_{l}" for l in LABELS]      # English label columns
RO_COLS = [f"ro_{l}" for l in LABELS]    # target-language label columns - romanian

LLM_instruction = (
    "Classify the following text into one or more of these emotions: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
    "Output a JSON list containing only the permmited English label names. ")

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

def format_instruction(row, tokenizer, language="en"):
    
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    labels = [emotion for emotion in emotions if row[f"{language}_{emotion}"] == 1]
    LLM_instruction = (
        "Classify the following text into one or more of these emotions: "
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
        "Output a JSON list containing only the permmited English label names. ")
    target_output = str(labels).replace("'", '"')

    if "gemma" in tokenizer.name_or_path.lower():
        msg_dict = [
            {"role": "user", "content": f"{LLM_instruction}\n\nText: {row[f'{language}_text']}"},
            {"role": "assistant", "content": target_output}
        ]
    else:
        msg_dict = [
        {"role": "system", "content": LLM_instruction},
        {"role": "user", "content": row[f"{language}_text"]},
        {"role": "assistant", "content": target_output}
        ]

    prompt = tokenizer.apply_chat_template(msg_dict, tokenize=False)
    return {"text": prompt}

def parse_prediction(raw_output):

    clean_text = raw_output.strip().lower()
    match = re.search(r'\[.*?\]', clean_text, re.DOTALL)
    if match:
        clean_text = match.group(0)
    try:
        parsed_list = json.loads(clean_text)
        if not isinstance(parsed_list, list): #check for malformed labels
            return [], True
        valid_labels = list(set(label.strip() for label in parsed_list if label.strip() in LABELS))
        return valid_labels, False
    except json.JSONDecodeError:
        return [], True

def text_labels_to_vector(predicted_labels, valid_labels):
    y = np.zeros(len(valid_labels), dtype=int)
    for emotion in predicted_labels:
        if emotion in valid_labels:
            idx = valid_labels.index(emotion)
            y[idx] = 1
    return y