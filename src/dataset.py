import json
import torch
from torch.utils.data import Dataset
from collections import Counter
import re

class AmazonSentimentDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_length=128):
        self.max_length = max_length
        self.data = []
        
        # Load data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    'text': item['text'].lower(),   # Review text converted to lowercase
                    'label': int(item['label'])     # Rating label (0-4)
                })

        # Build vocabulary
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

        self.pad_token_id = self.vocab.get('<PAD>', 0)  # Padding token ID
        self.unk_token_id = self.vocab.get('<UNK>', 1)  # Unknown token ID for out-of-vocabulary words

    def build_vocab(self):
        words = []
        for item in self.data:
            words.extend(self.tokenize(item['text']))
        
        counter = Counter(words)
        common_words = counter.most_common(10000) # Limit vocabulary to the 10k most frequent words
        
        # Map words to unique IDs
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(common_words):
            vocab[word] = i + 2
        return vocab

    # Split text into individual words/tokens
    def tokenize(self, text):
        return re.findall(r'\w+', text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenize(item['text'])
        
        # Convert tokens to their respective vocabulary IDs
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Pad or truncate the sequence to match max_length
        if len(ids) < self.max_length:
            ids += [self.pad_token_id] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        # Return PyTorch tensors for word IDs and the corresponding label
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }