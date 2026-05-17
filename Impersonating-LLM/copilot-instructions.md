# Impersonating LLM – Project Summary (Part 2)

## Project Goal
Fine-tune a small LLM (e.g. GPT‑2 124M) to impersonate the writing style of specific individuals based on their tweets.  
The codebase for this part of the project is hosted at:  
[https://github.com/Man1ek27/SentimentAnalysisCUDA](https://github.com/Man1ek27/SentimentAnalysisCUDA)
This codebase is split in two, because of an artifical requirement of the project. You're going to be responsible for Impersonation-LLM/ folder and the project assosiated with it.
---

## Datasets (ready‑to‑use, no scraping required)

### 1. Primary dataset: Donald Trump’s tweets
- **Trump’s Legacy** (Kaggle)  
  [https://www.kaggle.com/datasets/zusmani/trumps-legacy](https://www.kaggle.com/datasets/zusmani/trumps-legacy)  
  56,572 tweets before Twitter ban. Contains date, likes, retweets.
- **Complete Trump Tweets Archive** (Kaggle) – all presidential tweets, including deleted ones.
- **Trump Tweet Emotion Analysis** (GitHub) – 56k tweets, pre‑cleaned.
- **Trump Twitter Archive** (mkearney/trumptweets on GitHub) – full raw archive.

### 2. Sentiment140 (alternative for small‑sample impersonation)
- [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)  
  1.6 million tweets labelled with sentiment (0 = negative, 4 = positive).  
  Contains a `user` field; you can filter by a specific user to test style imitation with very few samples.  
  Sentiment label could be used as a conditioning prefix (e.g. “Positive tweet: ”).

### 3. Amazon Reviews
- [https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en)  
  Could be used to test imitation on a different text type.

---

## Fine‑tuning a Small LLM

### How it works
- Take a pre‑trained **causal language model** (GPT‑2, DistilGPT‑2, TinyLlama, etc.).
- Continue training it on your target text (tweets).  
- The model learns to predict the next token in the style of the new data.  
- This is **fine‑tuning**, not training from scratch.

### Execution options

You need to make this project compatibale for both of these options, but FIRST focus on Google Colab. Local PC method will be an additional fuctionality added later on.:

| Method | Hardware | Notes |
|--------|----------|-------|
| **Google Colab** (free GPU T4 16 GB) | Cloud GPU | Ideal for small models (DistilGPT‑2, GPT‑2 124M). No local setup. |
| **Local PC with NVIDIA GPU** (≥6 GB VRAM) | Own GPU | Full control, no time limits. Use QLoRA for larger models. |

### Recommended base models
- **DistilGPT‑2** (82M) – tiny, fast, fits easily in Colab.  
- **GPT‑2** (124M) – standard choice, excellent for style transfer.  
- **TinyLlama 1.1B** – better quality, works with LoRA on T4.

### First task:
First experiment: fine‑tune on Trump’s Legacy dataset (Kaggle).
Plenty of data; validate how well the model captures Trump’s writing style.
We want to create a notebooks/ file or something that keeps colab stuff seperate from other stuff.
We first want to make a colab notebook, no need to try and implement this locally.
Download the dataset in colab and try fine-tuning a local model to generate stuff from it.
Keep explanations of the code brief, simple and 1-2 lines at most.

---

## IMPLEMENTATION PLAN – Session 1

### Goal
Create Google Colab notebook for fine-tuning DistilGPT-2 on Trump's Legacy dataset to generate Trump-style tweets.

### Strategy
- **Model**: DistilGPT-2 (82M params) – fits Colab T4 16GB
- **Dataset**: Trump's Legacy (56k tweets) via Kaggle API
- **Approach**: Simple fine-tuning (3-5 epochs), no LoRA (reserved for future local GPU)
- **Code Style**: Brief explanations (1-2 lines per block)
- **Output**: `notebooks/01_trump_finetuning_colab.ipynb`

### Notebook Structure (8 cells)
1. **Setup**: Install packages, check GPU
2. **Kaggle Auth**: API credentials from Colab Secrets, download Trump dataset
3. **Exploration**: Show sample tweets, basic stats
4. **Data Prep**: Clean tweets, tokenize, train/val split (80/20), DataLoader
5. **Model Loading**: Load DistilGPT-2, training config
6. **Fine-tuning**: Custom training loop, loss tracking, checkpoint saving
7. **Generation**: Text generation with seed phrases (5-10 samples)
8. **Evaluation**: Perplexity comparison (optional)

### Dependencies
- torch >= 1.12
- transformers >= 4.30
- datasets >= 2.10
- pandas, kaggle

### Execution Order (8 Todos)
- [ ] setup-folders: Create notebooks/ directory
- [ ] build-cells-1-2: Env setup & Kaggle download
- [ ] build-cells-3-4: Data exploration & preparation
- [ ] build-cells-5-6: Model loading & fine-tuning loop
- [ ] build-cells-7-8: Text generation & evaluation
- [ ] save-notebook: Export as .ipynb file
- [ ] gitignore-update: Add dataset/checkpoint exclusions
- [ ] validate-notebook: Review and validate

### Progress
- **Status**: ✅ COMPLETED
- **Last Updated**: Session 1 - All tasks complete
- **Deliverable**: `notebooks/01_trump_finetuning_colab.ipynb` (15 cells, ready for use)