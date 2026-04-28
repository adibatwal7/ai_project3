#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rui.torch.transformer  —  local copy (bug-fixed)

Bug fixed vs reference:
  MultiHeadAttention.forward(): k and v were reshaped using q's seq_len,
  which breaks cross-attention when src and tgt have different lengths.
  Fixed to use each tensor's own sequence dimension.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Positional Embedding
# ---------------------------------------------------------------------------

def positional_encoding(seq_len, depth):
    d = depth / 2
    positions = np.arange(seq_len)[:, np.newaxis]        # (seq_len, 1)
    d_indices = np.arange(d)[np.newaxis, :] / d          # (1, d)
    angle_rates = 1 / (10000 ** d_indices)               # (1, d)
    angle_rads = positions * angle_rates                  # (seq_len, d)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )                                                     # (seq_len, 2*d)
    return torch.tensor(pos_encoding, dtype=torch.float32)


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_emb, padding_idx=0, seq_len=1024):
        super().__init__()
        self.d_emb = d_emb
        self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=padding_idx)
        self.register_buffer('pos_encoding', positional_encoding(seq_len=seq_len, depth=d_emb))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        self.embedding._fill_padding_idx_with_zero()

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= math.sqrt(self.d_emb)
        return x + self.pos_encoding[:seq_len, :].unsqueeze(0)


# ---------------------------------------------------------------------------
# Multi-Head Attention (bug-fixed)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, seq_len=1024,
                 qkv_bias=False, is_causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_model = embed_dim
        self.n_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        if self.is_causal:
            self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, q, k, v):
        b, q_len, _ = q.shape
        # BUG-FIX: use each tensor's own sequence length, not q's
        k_len = k.shape[1]
        v_len = v.shape[1]

        q = self.W_q(q)   # (b, q_len, d_model)
        k = self.W_k(k)   # (b, k_len, d_model)
        v = self.W_v(v)   # (b, v_len, d_model)

        q = q.view(b, q_len, self.n_heads, self.d_k).transpose(1, 2)  # (b, h, q_len, d_k)
        k = k.view(b, k_len, self.n_heads, self.d_k).transpose(1, 2)  # (b, h, k_len, d_k)
        v = v.view(b, v_len, self.n_heads, self.d_k).transpose(1, 2)  # (b, h, v_len, d_k)

        attn_scores = q @ k.transpose(2, 3)  # (b, h, q_len, k_len)

        if self.is_causal:
            mask_bool = self.mask.bool()[:q_len, :k_len]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.d_k ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ v).transpose(1, 2)            # (b, q_len, h, d_k)
        context_vec = context_vec.contiguous().view(b, q_len, self.d_model)
        return self.out_proj(context_vec)


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_emb, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_emb, d_ff)
        self.linear2 = nn.Linear(d_ff, d_emb)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, *, d_emb, n_heads, d_ff, dropout=0.1, prenorm=False):
        super().__init__()
        self.gsa = MultiHeadAttention(embed_dim=d_emb, num_heads=n_heads, dropout=dropout)
        self.ff = FeedForward(d_emb, d_ff)
        self.norm1 = nn.LayerNorm(d_emb)
        self.norm2 = nn.LayerNorm(d_emb)
        self.drop_shortcut = nn.Dropout(dropout)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            shortcut = x
            x = self.norm1(x)
        else:
            x = self.norm1(x)
            shortcut = x
        x = self.gsa(x, x, x)
        x = x + shortcut
        if self.prenorm:
            shortcut = x
            x = self.norm2(x)
        else:
            x = self.norm2(x)
            shortcut = x
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class Encoder(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout=0.1, prenorm=False):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff,
                         dropout=dropout, prenorm=prenorm)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, *, d_emb, n_heads, d_ff, seq_len=1024, dropout=0.1, prenorm=False):
        super().__init__()
        self.csa = MultiHeadAttention(embed_dim=d_emb, num_heads=n_heads,
                                      dropout=dropout, seq_len=seq_len, is_causal=True)
        self.ca = MultiHeadAttention(embed_dim=d_emb, num_heads=n_heads, dropout=dropout)
        self.ff = FeedForward(d_emb, d_ff)
        self.norm1 = nn.LayerNorm(d_emb)
        self.norm2 = nn.LayerNorm(d_emb)
        self.drop_shortcut = nn.Dropout(dropout)
        self.prenorm = prenorm

    def forward(self, x, context):
        if self.prenorm:
            shortcut = x
            x = self.norm1(x)
        else:
            x = self.norm1(x)
            shortcut = x
        x = self.csa(x, x, x)
        x = self.ca(q=x, k=context, v=context)
        x = x + shortcut
        if self.prenorm:
            shortcut = x
            x = self.norm2(x)
        else:
            x = self.norm2(x)
            shortcut = x
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class Decoder(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, seq_len=1024, dropout=0.1, prenorm=False):
        super().__init__()
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff,
                         seq_len=seq_len, dropout=dropout, prenorm=prenorm)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, context)
        return x


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff,
                 src_vocab_size, tgt_vocab_size, seq_len=1024,
                 dropout=0.1, prenorm=False):
        super().__init__()
        self.src_pos_embedding = PositionalEmbedding(
            vocab_size=src_vocab_size, d_emb=d_emb, seq_len=seq_len)
        self.tgt_pos_embedding = PositionalEmbedding(
            vocab_size=tgt_vocab_size, d_emb=d_emb, seq_len=seq_len)
        self.encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads,
                                d_ff=d_ff, dropout=dropout, prenorm=prenorm)
        self.decoder = Decoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads,
                                d_ff=d_ff, seq_len=seq_len, dropout=dropout, prenorm=prenorm)
        self.final_layer = nn.Linear(d_emb, tgt_vocab_size)

    def forward(self, x):
        src, tgt = x
        src_emb = self.src_pos_embedding(src)
        tgt_emb = self.tgt_pos_embedding(tgt)
        context = self.encoder(src_emb)
        out = self.decoder(tgt_emb, context)
        return self.final_layer(out)


if __name__ == "__main__":
    print("Transformer self-test")
else:
    print(f'Transformer imported from local file "{__name__}.py"')
