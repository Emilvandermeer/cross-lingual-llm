import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

PRIMARY_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SECONDARY_MODEL_ID = "google/gemma-2-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

primary_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    PRIMARY_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Successfully loaded {PRIMARY_MODEL_ID}!")