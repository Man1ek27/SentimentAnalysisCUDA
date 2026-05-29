import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import TransformerSentimentModel

def main():
    m = TransformerSentimentModel(vocab_size=100, embed_dim=32, num_heads=4, ff_hidden_dim=64, num_layers=2, output_dim=3, max_length=16)
    total = sum(p.numel() for p in m.parameters())
    print('Transformer OK, params:', total)

if __name__ == '__main__':
    main()
