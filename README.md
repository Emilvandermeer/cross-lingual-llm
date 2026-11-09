# cross-lingual-llm


To process and split data run:
``` 
uv run src/cross_lingual_llm/preprocessing/data_splitter.py
```
This will produce 4 .csv files:

```
data/splits/dev.csv , data/splits/test.csv , data/splits/train.csv , data/splits/summary.csv
```
Information about each of the splits and their respective sizes + label distribution can be found in summary.csv

In the root directory, create a file called  ```.env``` and add this to it (This is needed so you can load the models):
```
HF_TOKEN="your hugginface token"
```
To train the primary model do:
```
uv run src/cross_lingual_llm/model_trainer.py --condition language
```
Instead of language, type either "english", "romanian" or "bilingual" (without quotation marks).
This will download the weights of Qwen and run PEFT training on it using the specified language. For the paper you need to run this command three times with all language options. I made it like this so the PC has time to empty its RAM before it starts another batch. It takes around 24GB of RAM and 7GB of VRAM to train one model. It will show a message where the final model is saved.

To evaluate the primary model do:
```
uv run src/cross_lingual_llm/inference.py --dataset dataset --language language --model model
```
Instead of dataset, type either "dev" or "test" (dev should only be used for hyperparameter tuning). Instead of language, type either "english" or "romanian", this is needed because the english and bilingual models are evaluated on both datasets. Instead of model type "en", "ro" or "bilingual". It takes around 5 minutes to evaluate a model and it uses up 15GB of RAM (because it loads the entire model to RAM).

<span style="color:red">To use the gemma model, you have to request access from google. To do so make sure you have a valid hugginface account, press this link and follow the steps:
[Gemma-2-2b (https://huggingface.co/google/gemma-2-2b)](https://huggingface.co/google/gemma-2-2b)</span><br>
To perform zero-shot prompting run:
```
uv run src/cross_lingual_llm/zero_shot_prompting.py --model model --language language
```
Instead of model type either "qwen" or "gemma". Instead of language type either "english" or "romanian"
Finally, to perform five-shot prompting run:
```
uv run src/cross_lingual_llm/five_shot_prompting.py --model model --language language
```
