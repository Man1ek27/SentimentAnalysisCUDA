import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AmazonSentimentDataset
from models.lstm_baseline import LSTMSentimentModel
from config import (
    RAW_DATA_DIR, 
    BATCH_SIZE, 
    EPOCHS, 
    LEARNING_RATE, 
    EMBEDDING_DIM, 
    HIDDEN_DIM, 
    OUTPUT_DIM, 
    MAX_LENGTH
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
    # Automatically select GPU (CUDA) if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Resolve file paths using configuration
    print("Loading datasets...")
    train_file = RAW_DATA_DIR / "amazon_reviews_multi_en_train.jsonl"
    val_file = RAW_DATA_DIR / "amazon_reviews_multi_en_validation.jsonl"
    
    # Initialize Datasets
    train_dataset = AmazonSentimentDataset(train_file, max_length=MAX_LENGTH)
    val_dataset = AmazonSentimentDataset(val_file, vocab=train_dataset.vocab, max_length=MAX_LENGTH)
    
    # Initialize DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Extract dynamically built vocabulary size
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Instantiate the baseline model
    model = LSTMSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    ).to(device)
    
    # Define loss criteria and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Core Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc*100:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()