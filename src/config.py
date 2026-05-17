"""Configuration file"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

# Hyperparameters for Models
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 5 
MAX_LENGTH = 128 