"""
model.py
--------
GRU-based log anomaly detector with soft attention mechanism.
Outputs an anomaly score [0, 1] and per-token attention weights.
"""

import torch
import torch.nn as nn


class LogModel(nn.Module):
    """
    GRU encoder with a soft-attention pooling head.
    - Embedding  : vocab_size  → embed_dim
    - GRU        : embed_dim   → hidden_dim  (batch_first)
    - Attention  : hidden_dim  → scalar weight per time-step
    - FC head    : hidden_dim  → 1  (anomaly probability)
    """

    def __init__(self, vocab_size: int = 500, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.attn      = nn.Linear(hidden_dim, 1)
        self.fc        = nn.Linear(hidden_dim, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: LongTensor of shape (batch, seq_len)
        Returns:
            score      : FloatTensor (batch, 1)  — anomaly probability
            attn_weights: FloatTensor (batch, seq_len, 1) — per-token weight
        """
        embedded = self.embedding(x)               # (B, L, embed_dim)
        out, _   = self.rnn(embedded)              # (B, L, hidden_dim)

        # Soft attention
        raw_weights  = self.attn(out)              # (B, L, 1)
        attn_weights = torch.softmax(raw_weights, dim=1)  # normalised over seq

        # Context vector: weighted sum
        context = torch.sum(attn_weights * out, dim=1)    # (B, hidden_dim)

        score = self.sigmoid(self.fc(context))     # (B, 1)
        return score, attn_weights
