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
To load the models do:
```
uv run src/cross_lingual_llm/models_loader.py
```
This should download the weights of the 2 models, and run a tokenizer on the primary one. It will show
a msg if it fully loaded.

<span style="color:red">If it crashes because you reach a RAM limit, just comment out the "model" line.</span><br>
<span style="color:red">To use the gemma model, you have to request access from google but it takes like 2 minutes.</span><br>
Zero shot prompting should work for both models, both in English and Romanian but its not randomised so it outputs the same thing
every time.
```
uv run src/cross_lingual_llm/zero_shot_prompting.py
```