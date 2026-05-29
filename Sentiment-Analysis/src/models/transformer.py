import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
	def __init__(self, dropout=0.0):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, mask=None):
		# q,k,v: (batch, heads, seq_len, head_dim)
		d_k = q.size(-1)
		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
		if mask is not None:
			scores = scores.masked_fill(mask == 0, float('-1e9'))
		attn = torch.softmax(scores, dim=-1)
		attn = self.dropout(attn)
		out = torch.matmul(attn, v)
		return out, attn


class MultiHeadAttention(nn.Module):
	def __init__(self, embed_dim, num_heads, dropout=0.0):
		super().__init__()
		assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		self.attention = ScaledDotProductAttention(dropout=dropout)

	def forward(self, x, key_padding_mask=None):
		# x: (batch, seq_len, embed_dim)
		batch_size, seq_len, _ = x.size()

		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		if key_padding_mask is not None:
			mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
		else:
			mask = None

		attn_out, attn = self.attention(q, k, v, mask=mask)

		attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
		out = self.out_proj(attn_out)
		return out


class FeedForward(nn.Module):
	def __init__(self, embed_dim, hidden_dim, dropout=0.0):
		super().__init__()
		self.fc1 = nn.Linear(embed_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, embed_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.fc1(x)
		x = F.gelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x


class TransformerEncoderLayer(nn.Module):
	def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.0):
		super().__init__()
		self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout=dropout)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, key_padding_mask=None):
		# Self-attention + Add&Norm
		attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
		x = x + self.dropout(attn_out)
		x = self.norm1(x)

		# Feed-forward + Add&Norm
		ff_out = self.ff(x)
		x = x + self.dropout(ff_out)
		x = self.norm2(x)
		return x


class TransformerEncoder(nn.Module):
	def __init__(self, num_layers, embed_dim, num_heads, ff_hidden_dim, dropout=0.0):
		super().__init__()
		self.layers = nn.ModuleList([
			TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout=dropout)
			for _ in range(num_layers)
		])

	def forward(self, x, key_padding_mask=None):
		for layer in self.layers:
			x = layer(x, key_padding_mask=key_padding_mask)
		return x


class TransformerSentimentModel(nn.Module):
	"""Simple Transformer encoder-based classifier implemented from scratch.

	Constructor args:
	  - vocab_size: size of token vocabulary
	  - embed_dim: token embedding dimension
	  - num_heads: number of attention heads
	  - ff_hidden_dim: hidden dim for feed-forward
	  - num_layers: number of encoder layers
	  - output_dim: number of target classes
	  - max_length: maximum sequence length (for positional embeddings)
	  - dropout: dropout probability
	"""

	def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_hidden_dim=512, num_layers=2, output_dim=5, max_length=128, dropout=0.1):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
		self.pos_embed = nn.Embedding(max_length, embed_dim)
		self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout=dropout)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(embed_dim, output_dim)
		self.max_length = max_length

	def forward(self, input_ids):
		# input_ids: (batch, seq_len)
		batch_size, seq_len = input_ids.size()
		if seq_len > self.max_length:
			raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")

		token_emb = self.embed(input_ids) 
		positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
		pos_emb = self.pos_embed(positions)

		x = token_emb + pos_emb

		# key padding mask: 1 for real tokens, 0 for padding
		key_padding_mask = (input_ids != 0).to(input_ids.device)

		x = self.encoder(x, key_padding_mask=key_padding_mask)

		# mean pooling over real tokens
		mask = key_padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
		summed = (x * mask).sum(dim=1)  # (batch, embed_dim)
		denom = mask.sum(dim=1).clamp(min=1).to(x.dtype)
		pooled = summed / denom

		pooled = self.dropout(pooled)
		logits = self.fc(pooled)
		return logits


__all__ = ["TransformerSentimentModel"]
