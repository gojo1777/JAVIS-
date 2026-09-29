import torch
import torch.nn as nn

from config import BLOCK_SIZE
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=N_EMBD,
            num_heads=N_HEAD,
            dropout=DROPOUT,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        T = x.size(1)

        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device
            ),
            diagonal=1
        ).bool()

        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        return self.dropout(output)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                N_EMBD,
                4 * N_EMBD
            ),

            nn.GELU(),

            nn.Linear(
                4 * N_EMBD,
                N_EMBD
            ),

            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attention = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(N_EMBD)
        self.feed_forward = FeedForward()

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


class MyAI(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            N_EMBD
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        self.ln = nn.LayerNorm(N_EMBD)

        self.output = nn.Linear(
            N_EMBD,
            vocab_size
        )

    def forward(self, idx):

        B, T = idx.shape

        if T > BLOCK_SIZE:
            raise ValueError(
                "Input is longer than BLOCK_SIZE"
            )

        positions = torch.arange(
            T,
            device=idx.device
        )

        token_embeddings = (
            self.token_embedding(idx)
        )

        position_embeddings = (
            self.position_embedding(positions)
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.output(x)

        return logits
