import torch
from torch import nn
import torch.nn.functional as F

class DAN(nn.Module):
    """
    Word-level DAN using pretrained embeddings from sentiment_data.WordEmbeddings
    (This is your Part 1 model; keep your existing implementation if you already have one.)
    """
    def __init__(self, embedding_layer, hidden_size=200, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding_layer = embedding_layer  # torch.nn.Embedding (possibly pretrained)
        emb_dim = embedding_layer.embedding_dim
        self.dropout = nn.Dropout(dropout)

        layers = []
        in_dim = emb_dim
        # num_layers means number of FF layers BEFORE output (>=1)
        for i in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 2)

    def forward(self, x):
        # x: [B, T] Long
        emb = self.embedding_layer(x)  # [B, T, D]
        avg = emb.mean(dim=1)          # [B, D]
        h = self.mlp(self.dropout(avg))
        logits = self.out(h)
        return F.log_softmax(logits, dim=1)


class SubwordDAN(nn.Module):
    """
    Part 2: Subword-based DAN (BPE). Random embeddings only.
    """
    def __init__(self, vocab_size, embed_dim=50, hidden_size=200, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        layers = []
        in_dim = embed_dim
        for i in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 2)

    def forward(self, x):
        # x: [B, T] Long (subword ids)
        emb = self.embedding(x)        # [B, T, D]
        avg = emb.mean(dim=1)          # [B, D]
        h = self.mlp(self.dropout(avg))
        logits = self.out(h)
        return F.log_softmax(logits, dim=1)

