import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from data.data_utils import OUT_DIR

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
train_df = pd.read_csv(OUT_DIR / "train.csv")

def format_instruction(row, language="en"):
    
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    labels = [emotion for emotion in emotions if row[f"{language}_{emotion}"] == 1]
    LLM_instruction = (
        "Classify the following text into one or more of these emotions: "
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
        "Output a JSON list containing only the permmited English label names. ")
    target_output = str(labels).replace("'", '"')

    msg_dict = [
    {"role": "system", "content": LLM_instruction},
    {"role": "user", "content": row[f"{language}_text"]},
    {"role": "assistant", "content": target_output}
    ]

    prompt = tokenizer.apply_chat_template(msg_dict, tokenize=False)
    return {"text": prompt}

def main() -> None:
    train_dataset = Dataset.from_pandas(train_df)
    en_train_dataset = train_dataset.map(lambda row: format_instruction(row))
    ro_train_dataset = train_dataset.map(lambda row: format_instruction(row, language="ro"))

    print("\n--- ENGLISH FORMATTED EXAMPLE ---")
    print(en_train_dataset[0]["text"])
    
    print("\n--- ROMANIAN FORMATTED EXAMPLE ---")
    print(ro_train_dataset[0]["text"])

if __name__ == "__main__":
    main()