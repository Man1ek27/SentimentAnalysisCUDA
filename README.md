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

#Impersonating-LLM
Upload the notebook in notebooks/ to Colab and set up it's secrets. You're gonna need a Kaggle Account to execute this notebook.
kaggle_username = userdata.get('KAGGLE_USERNAME')
kaggle_key = userdata.get('KAGGLE_KEY')

Kaggle key can be found in Account - Your Api Keys.