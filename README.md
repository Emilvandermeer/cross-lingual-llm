# cross-lingual-llm

For now for the data to work, you should download the following files from that github repo on brightspace
```
en-annotated.tsv , ro-projections.tsv , pairs-ro.txt
```
These should be dragged and dropped into data/raw.<br>
<span style="color:red">need to do this in a better way, i believe that the pairs-ro will be too big to upload to github.</span>

To process and split data run:
``` 
uv run src/cross_lingual_llm/preprocessing/data_splitter.py
```

In the root directory, create a file called  ```.env``` and add this to it (This is needed so you can load the models):
```
HF_TOKEN="your hugginface token"
```
To train the primary model do:
```
uv run src/cross_lingual_llm/model_trainer.py --condition language
```
Instead of language, type either "english", "romanian" or "bilingual" (without quotation marks).
This will download the weights of Qwen and run PEFT training on it using the specified language. For the paper you need to run this command three times with all language options. I made it like this so my PC has time to empty its RAM before it starts another batch. It takes around 24GB of RAM and 7GB of VRAM to train one model. It will show a message where the final model is saved.

To evaluate the primary model do:
```
uv run src/cross_lingual_llm/inference.py --dataset dataset --language language --model model
```
Instead of dataset, type either "dev" or "test" (dev should only be used for hyperparameter tuning). Instead of language, type either "english" or "romanian", this is needed because the english and bilingual models are evaluated on both datasets. Instead of model type "en", "ro" or "bilingual". It takes like 5 minutes to evaluate a model and it uses up 15GB of RAM (because it loads the entire model to RAM).

<span style="color:red">To use the gemma model, you have to request access from google but it takes like 2 minutes.</span><br>
To perform zero-shot prompting run:
```
uv run src/cross_lingual_llm/zero_shot_prompting.py --model model --language language
```
Instead of model type either "qwen" or "gemma". Instead of language type either "english" or "romanian"
Finally, to perform five-shot prompting run:
```
uv run src/cross_lingual_llm/five_shot_prompting.py --model model --language language
```