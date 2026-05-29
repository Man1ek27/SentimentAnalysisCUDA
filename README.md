#Important:

This repo is split into two projects, one of which focuses on Sentiment Analysis of users' product reviews, and the second one which focuses on trying to fine-tune a local LLM to impersonate a person's writing style.

# SentimentAnalysisCUDA

Running envonment:
```uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
```

Loading Dataset
```
python src/download_data.py
```


Running Training
```
python -m src.train --model transformer --run-name transformer_run_01

--model [transformer|lstm]
--run-name [any name for the run, e.g. transformer_run_01]
--num-heads [number of heads for transformer, e.g. 4]
--num-layers [number of layers for transformer, e.g. 2]
--ff-dim [feedforward dimension for transformer, e.g. 128]
```