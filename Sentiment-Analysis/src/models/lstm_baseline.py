import torch
import torch.nn as nn

class LSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # Embedding layer: maps token IDs to dense vectors, ignoring padding tokens (ID 0)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer: processes word vectors sequentially
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        
        # Fully connected layer: maps the final hidden state to class logits
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # Pass token IDs through the embedding layer to get word vectors
        embedded = self.embedding(text)
        
        # Forward propagate through LSTM to get outputs and hidden states
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Extract the final hidden state of the last LSTM layer
        last_hidden = hidden[-1]
        
        # Map the hidden representation to output class probabilities (logits)
        return self.fc(last_hidden)