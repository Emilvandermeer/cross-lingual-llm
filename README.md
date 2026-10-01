# cross-lingual-llm

For now for the data to work, you should download the following files from that github repo on brightspace
```
en-annotated.tsv , ro-projections.tsv , pairs-ro.txt
```
These should be dragged and dropped into data/raw.
#! need to do this in a better way, i believe that the pairs-ro will be too big to upload to github


To process and split data run:
``` 
uv run src/cross_lingual_llm/data/data_splitter.py
```