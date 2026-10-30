import argparse
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import json
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, hamming_loss
from preprocessing.data_utils import (
    PRIMARY_MODEL_ID,
    OUT_DIR,
    ROOT,
    LABELS,
    LLM_instruction
)

tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID)

def text_labels_to_vector(predicted_labels, valid_labels):
    y = np.zeros(len(valid_labels), dtype=int)
    for emotion in predicted_labels:
        if emotion in valid_labels:
            idx = valid_labels.index(emotion)
            y[idx] = 1
    return y

def parse_prediction(raw_output):

    clean_text = raw_output.strip().lower()
    parsed_list = json.loads(clean_text)
    if not isinstance(parsed_list, list): #check for malformed labels
        return [], True
    valid_labels = list(set(label.strip() for label in parsed_list if label.strip() in LABELS))
    return valid_labels, False

def test_model(language: str, model_type: str, df)->None:
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
    qwen = AutoModelForCausalLM.from_pretrained(
        PRIMARY_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
        )
    #! change the model type based on the three trained types
    model = PeftModel.from_pretrained(qwen, str(ROOT / "data" / "results" / "checkpoint-3531"))
    predictions = []
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        msg_dict = [
            {"role": "system", "content": LLM_instruction},
            {"role": "user", "content": row[f"{language[:2]}_text"]}
            ]
        prompt = tokenizer.apply_chat_template(msg_dict, tokenize=False, add_generation_prompt = True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        clean_labels, _ = parse_prediction(raw_output)
        predictions.append(clean_labels)

    y_pred = []
    for prediction in predictions:
        vec_pred = text_labels_to_vector(prediction, LABELS)
        y_pred.append(vec_pred)

    y_pred = np.array(y_pred, dtype=int)
    columns = [f"{language[:2]}_{emo}" for emo in LABELS]
    y_true = df[columns].to_numpy(dtype=int)

    f1_micro = f1_score(y_true, y_pred, average='micro')
    jac = jaccard_score(y_true, y_pred, average='samples')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    ham = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    print("------------------RESULTS------------------\n")
    print("Primary Evaluation Metrics:")
    print(f"micro-averaged F1: {f1_micro:.3f}")
    print(f"sample-averaged Jaccard index: {jac:.3f}\n")

    print("Secondary Evaluation Metrics:")
    print(f"macro-averaged F1: {f1_macro:.3f}")
    print(f"Hamming loss: {ham:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

def main()-> None:
    dev_df = pd.read_csv(OUT_DIR / "dev.csv")
    test_df = pd.read_csv(OUT_DIR / "test.csv")
    parser = argparse.ArgumentParser(description="Run model evaluation for cross-lingual emotion classification.")
    parser.add_argument(
        "--dataset", 
        type=str,
        choices=["dev", "test"], 
        required=True,
        help="Select the dataset used for testing."
    )
    parser.add_argument(
        "--language", 
        type=str,
        choices=["english", "romanian"],
        required=True,
        help="Select the testing dataset's language."
    )
    parser.add_argument(
        "--model", 
        type=str,
        choices=["en", "ro", "bilingual"], 
        required=True,
        help="Select the model used for testing."
    )
    args = parser.parse_args()
    if args.dataset == "dev":
        test_model(args.language, args.model, dev_df)
    else:
        test_model(args.language, args.model, test_df)

if __name__=="__main__":
    main()
