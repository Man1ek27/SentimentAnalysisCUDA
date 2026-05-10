import json
import torch
from torch.utils.data import Dataset
from collections import Counter
import re

class AmazonSentimentDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_length=128):
        self.max_length = max_length
        self.data = []
        
        # Wczytanie danych z pliku
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    'text': item['text'].lower(),   # recenzja
                    'label': int(item['label'])     # ocena (0-4)
                })

        # budowanie słownika
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

        self.pad_token_id = self.vocab.get('<PAD>', 0)  # padding token
        self.unk_token_id = self.vocab.get('<UNK>', 1)  # token dla nieznanych słów

    def build_vocab(self):
        words = []
        for item in self.data:
            words.extend(self.tokenize(item['text']))
        
        counter = Counter(words)
        common_words = counter.most_common(10000) # ograniczenie do 10k najczęstszych słów
        
        # mapowanie słów na ID
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(common_words):
            vocab[word] = i + 2
        return vocab

    # rozdzielanie na słowa
    def tokenize(self, text):
        return re.findall(r'\w+', text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenize(item['text'])
        
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        if len(ids) < self.max_length:
            ids += [self.pad_token_id] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        # zwracanie tensora z ID słów i etykiety
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }