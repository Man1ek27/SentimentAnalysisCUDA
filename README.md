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
# TrumpTweetGenerationCUDA
### Colab ver.

Fine-tuning DistilGPT-2 on Trump's Legacy dataset to generate Trump-style tweets using CUDA acceleration.

### Running environment (Google Colab):

No local setup required. Simply open the notebook in Google Colab and run the first cell to install the remaining required packages:

```
!pip install -q torch transformers datasets pandas kaggle
```
```
## Informations & parameters
--notebook [Trump_Tweet_DistilGPT2_FineTuning.ipynb]
--model [distilgpt2]
--epochs [number of training epochs, default: 3]
--batch-size [batch size for training, default: 16]
--lr [learning rate, default: 5e-5]
```

### Running locally:

## Getting Started

### 1. Install Dependencies

Install the required packages using `pip`. If you are installing globally outside of a virtual environment, you may need the `--break-system-packages` flag:

```bash
pip install -r requirements_server.txt
```
The training script automatically downloads the Trump's Legacy dataset from Kaggle. To enable this, you need to set up your Kaggle API credentials.
Place your `kaggle.json` (downloaded from Kaggle -> Settings -> Create New Token) into the `.kaggle` directory:
```
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

To start the fine-tuning process, run:

```
python3 scripts/train_qwen_lora.py

# low VRAM optimization (Optional)
# The default configuration is optimized for larger GPUs. If you have limited VRAM (e.g.8 GB), to prevent Out of Memory (OOM) errors use:

python3 scripts/train_qwen_lora.py --batch-size 2 --grad-accum 16 --no-compile
```
