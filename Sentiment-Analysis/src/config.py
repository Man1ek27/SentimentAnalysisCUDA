"""Configuration file"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_PATH = DATA_DIR / 'amazon_reviews_multi_en_train.jsonl'
VAL_PATH = DATA_DIR / 'amazon_reviews_multi_en_validation.jsonl'
TEST_PATH = DATA_DIR / 'amazon_reviews_multi_en_test.jsonl'

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 5 
MAX_LENGTH = 128 

print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_DIR:', DATA_DIR)
print('TRAIN_PATH exists:', TRAIN_PATH.exists())
print('VAL_PATH exists:', VAL_PATH.exists())
print('TEST_PATH exists:', TEST_PATH.exists())