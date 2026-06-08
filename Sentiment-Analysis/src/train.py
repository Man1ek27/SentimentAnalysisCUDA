import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import argparse
import csv
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AmazonSentimentDataset
from models.lstm_baseline import LSTMSentimentModel
from config import (
    DATA_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    MAX_LENGTH,
)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one full epoch over the dataset."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in dataloader:
        # Move batch data to the designated device (CPU or GPU)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Reset gradients from the previous step
        optimizer.zero_grad()
        
        # Compute predicted outputs by passing inputs to the model
        outputs = model(input_ids)
        
        # Calculate the batch loss
        loss = criterion(outputs, labels)
        
        # Compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step to update weights
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * input_ids.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += input_ids.size(0)
        
    return total_loss / total_samples, correct_predictions / total_samples

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model performance on the validation/test set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient computation to save memory and accelerate computation
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Compute predicted outputs by passing inputs to the model
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            # Accumulate metrics
            total_loss += loss.item() * input_ids.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += input_ids.size(0)
            
    return total_loss / total_samples, correct_predictions / total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm",
                        help="Which model to train: 'lstm' or 'transformer'")
    parser.add_argument("--num-heads", type=int, default=8, help="Transformer: number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer: number of encoder layers")
    parser.add_argument("--ff-dim", type=int, default=512, help="Transformer: feed-forward hidden dim")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for saving outputs")
    args = parser.parse_args()

    # Automatically select GPU (CUDA) if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected model: {args.model}")

    # Prepare run folder and output paths
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_{run_stamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RESULTS_DIR / run_name
    models_dir = run_dir / "models"
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "run_config.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve file paths using configuration
    print("Loading datasets...")
    train_file = DATA_DIR / "amazon_reviews_multi_en_train.jsonl"
    val_file = DATA_DIR / "amazon_reviews_multi_en_validation.jsonl"
    
    # Initialize Datasets
    train_dataset = AmazonSentimentDataset(train_file, max_length=MAX_LENGTH)
    val_dataset = AmazonSentimentDataset(val_file, vocab=train_dataset.vocab, max_length=MAX_LENGTH)
    
    # Initialize DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Extract dynamically built vocabulary size
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Instantiate the selected model
    if args.model == "lstm":
        model = LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        ).to(device)
    else:
        # Lazy import transformer only when requested
        from models.transformer import TransformerSentimentModel

        model = TransformerSentimentModel(
            vocab_size=vocab_size,
            embed_dim=EMBEDDING_DIM,
            num_heads=args.num_heads,
            ff_hidden_dim=args.ff_dim,
            num_layers=args.num_layers,
            output_dim=OUTPUT_DIM,
            max_length=MAX_LENGTH,
        ).to(device)
    
    # Define loss criteria and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Save run configuration for reproducibility
    run_config = {
        "model": args.model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "max_length": MAX_LENGTH,
        "vocab_size": vocab_size,
        "train_file": str(train_file),
        "val_file": str(val_file),
    }
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Prepare metrics CSV
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    
    # Core Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%")
        print("-" * 40)

        # Save metrics for this epoch
        with metrics_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = models_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "run_name": run_name,
            }, best_path)
            print(f"Saved best model to: {best_path}")

if __name__ == "__main__":
    main()