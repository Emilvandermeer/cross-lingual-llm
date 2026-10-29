# cross-lingual-llm

For now for the data to work, you should download the following files from that github repo on brightspace
```
en-annotated.tsv , ro-projections.tsv , pairs-ro.txt
```
These should be dragged and dropped into data/raw.<br>
<span style="color:red">need to do this in a better way, i believe that the pairs-ro will be too big to upload to github.</span>

To process and split data run:
``` 
uv run src/cross_lingual_llm/data/data_splitter.py
```

In the root directory, create a file called  ```.env``` and add this to it (This is needed so you can load the models):
```
HF_TOKEN="your hugginface token"
```
To train the primary model do do:
```
uv run src/cross_lingual_llm/model_trainer.py --condition language
```
Instead of language, type either "english", "romanian" or "bilingual" (without quotation marks).
This will download the weights of Qwen and run PEFT training on it using the specified language. For the paper you need to run this command three times with all language options. I made it like this so my PC has time to empty its RAM before it starts another batch. It takes around 24GB of RAM and 7GB of VRAM to train one model. It will show a message where the weights are saved but I will try to push them to Git myself.

<span style="color:red">To use the gemma model, you have to request access from google but it takes like 2 minutes.</span><br>
Zero shot prompting should work for both models, both in English and Romanian but its not randomised so it outputs the same thing
every time.
```
uv run src/cross_lingual_llm/zero_shot_prompting.py
```