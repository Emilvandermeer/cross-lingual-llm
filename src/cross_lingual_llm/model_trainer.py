import argparse
import os

import pandas as pd
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

from preprocessing.data_utils import OUT_DIR, ROOT, SEED, format_instruction

load_dotenv()
train_df = pd.read_csv(OUT_DIR / "train.csv")
login(token=os.getenv("HF_TOKEN"))

PRIMARY_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SECONDARY_MODEL_ID = "google/gemma-2-2b-it"

hyperparams = {
    "adapter_targets":["q_proj", "v_proj"],
    "rank": 4,
    "alpha": 16,
    "dropout": 0.05,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "num_epochs": 3,
    "checkp_rule": "epoch",
    "seed": SEED,
    "max_seq_length": 256
}

def PEFT_training(language: str):

    #Quantization module
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    #LoRA module
    lora_config = LoraConfig(
        target_modules=hyperparams["adapter_targets"],
        r=hyperparams["rank"],
        lora_alpha=hyperparams["alpha"],
        lora_dropout=hyperparams["dropout"],
        )

    tr_arguments = SFTConfig(
        output_dir=str(ROOT / "data" / "results"),
        learning_rate=hyperparams["learning_rate"],
        per_device_train_batch_size=hyperparams["batch_size"],
        num_train_epochs=hyperparams["num_epochs"],
        save_strategy=hyperparams["checkp_rule"],
        seed=hyperparams["seed"],
        dataset_text_field="text",
        max_length=hyperparams["max_seq_length"]
    )

    primary_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(
            PRIMARY_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    print(f"Successfully loaded {PRIMARY_MODEL_ID}!")

    train_dataset = Dataset.from_pandas(train_df)
    if language == "bilingual":
        en_dataset = train_dataset.map(lambda row: format_instruction(row, primary_tokenizer))
        ro_dataset = train_dataset.map(lambda row: format_instruction(row, primary_tokenizer, "ro"))
        train_dataset = concatenate_datasets([en_dataset, ro_dataset])
        train_dataset = train_dataset.shuffle(seed=hyperparams["seed"])
    else:
        train_dataset = train_dataset.map(lambda row: format_instruction(row, primary_tokenizer, language))
    
    supervised_tr = SFTTrainer(
        model=qwen,
        processing_class=primary_tokenizer,
        train_dataset=train_dataset,
        args=tr_arguments,
        peft_config=lora_config,
    )

    print(f"Starting {language} PEFT training...")
    supervised_tr.train()
    print(f"Training complete.\nWeights saved at {str(ROOT / "data" / "results")}")

def main()->None:
    parser = argparse.ArgumentParser(description="Run PEFT training for cross-lingual emotion classification.")
    parser.add_argument(
        "--condition", 
        type=str,
        choices=["english", "romanian", "bilingual"], 
        required=True,
        help="Select the training condition to execute."
    )
    args = parser.parse_args()
    language = args.condition
    if language == "english" or language == "romanian":
        language = language[:2]

    PEFT_training(language)

if __name__=="__main__":
    main()