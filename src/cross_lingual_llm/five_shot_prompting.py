import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    set_seed,
    )
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, hamming_loss
from preprocessing.data_utils import (
    OUT_DIR,
    LABELS,
    PRIMARY_MODEL_ID,
    SECONDARY_MODEL_ID,
    LLM_instruction,
    parse_prediction,
    text_labels_to_vector,
    SEED
    )

train_df = pd.read_csv(OUT_DIR / "train.csv")
test_df = pd.read_csv(OUT_DIR / "test.csv")
five_samples = train_df.head(5)
set_seed(SEED)

def get_target_string(row, language):
    labels = [emo for emo in LABELS if row[f"{language}_{emo}"] == 1]
    return str(labels).replace("'", '"')

def format_five_shot_instruction(row, tokenizer, language="en"):
    msg_dict = []
    if "gemma" in tokenizer.name_or_path.lower():
        first_msg = five_samples.iloc[0]
        msg_dict.append({
            "role": "user",
            "content": f"{LLM_instruction}\n\nText: {first_msg[f'{language}_text']}"
        })
        msg_dict.append({
            "role": "assistant", 
            "content": get_target_string(first_msg, language)
        })

        for _, msg in five_samples.iloc[1:].iterrows():
            msg_dict.append({"role": "user", "content": f"Text: {msg[f'{language}_text']}"})
            msg_dict.append({"role": "assistant", "content": get_target_string(msg, language)})

        msg_dict.append({"role": "user", "content": f"Text: {row[f'{language}_text']}"})

    else:
        msg_dict.append({"role": "system", "content": LLM_instruction})

        for _, msg in five_samples.iterrows():
            msg_dict.append({"role": "user", "content": msg[f'{language}_text']})
            msg_dict.append({"role": "assistant", "content": get_target_string(msg, language)})

        msg_dict.append({"role": "user", "content": row[f'{language}_text']})

    return msg_dict

def test_model(model_id: str, language: str)->None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
        )

    predictions = []
    model.eval()
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        msg_dict = format_five_shot_instruction(row, tokenizer, language)
        prompt = tokenizer.apply_chat_template(msg_dict, tokenize=False, add_generation_prompt = True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Added so metrics stay the same
                temperature=None, # Added so metrics stay the same
                top_p=None # Added so metrics stay the same
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
    columns = [f"{language}_{emo}" for emo in LABELS]
    y_true = test_df[columns].to_numpy(dtype=int)

    f1_micro = f1_score(y_true, y_pred, average='micro')
    jac = jaccard_score(y_true, y_pred, average='samples')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    ham = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
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



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zero-Shot Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen", "gemma"],
        required=True
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["english", "romanian"],
        required=True
    )
    args = parser.parse_args()
    model_id = PRIMARY_MODEL_ID if args.model == "qwen" else SECONDARY_MODEL_ID
    test_model(model_id, args.language[:2])


if __name__ == "__main__":
    main()