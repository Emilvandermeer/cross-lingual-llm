import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from data.data_utils import OUT_DIR
from models_loader import PRIMARY_MODEL_ID, SECONDARY_MODEL_ID

primary_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID)
secondary_tokenizer = AutoTokenizer.from_pretrained(SECONDARY_MODEL_ID)
train_df = pd.read_csv(OUT_DIR / "train.csv")

def format_instruction(row, tokenizer=primary_tokenizer, language="en"):
    
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

def run_test(model: str):
    if model.lower() == "qwen":
        tokenizer = primary_tokenizer
        msg = "QWEN (PRIMARY)"
    elif model.lower() == "gemma":
        tokenizer = secondary_tokenizer
        msg = "GEMMA (SECONDARY)"
    else:
        raise ValueError("The project uses only Qwen and Gemma")
    train_dataset = Dataset.from_pandas(train_df)
    en_train_dataset = train_dataset.map(lambda row: format_instruction(row, tokenizer))
    ro_train_dataset = train_dataset.map(lambda row: format_instruction(row, tokenizer,"ro"))

    print("\n--- ENGLISH FORMATTED EXAMPLE ON " + msg + " ---")
    print(en_train_dataset[0]["text"])
    
    print("\n--- ROMANIAN FORMATTED EXAMPLE ON " + msg + " ---")
    print(ro_train_dataset[0]["text"])

def main() -> None:
    run_test("Qwen")
    run_test("gemma")

if __name__ == "__main__":
    main()