import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        # TODO: tril buffer + dropout fields

    def forward(self, x):
        # input of size (batch, time-step, embed_dim/channels)
        # output of size (batch, time-step, head_size)
        # B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # calculate scaled dot-product attention
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # [B, T, T]
        # TODO: masked tril
        weights = F.softmax(weights, dim=-1)
        # TODO: dropout
        att = weights @ v  # [B, T, head_size]
        return att


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        # TODO: dropout field

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # TODO: dropout
        out = self.proj(out)  # [B, T, embed_dim]
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            # NOTE: understand why inner layer is 4x I/O dimension
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            # TODO: dropout
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = MultiHeadedAttention(embed_dim, num_heads, embed_dim // num_heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # input & ouput size [batch, time-step, embed_dim]
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, vocab_size):
        super().__init__()
        # TODO: vocab + position embeddings tables
        self.net = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(embed_dim),  # final layer norm
            nn.Linear(embed_dim, vocab_size)  # output logits
        )
        # TODO: better init in Karpathy's reference code, figure out its purpose
        self.apply(self._init_weights)

    # TODO: what does this do?
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        # output size [batch, time-step, vocab_size]
        # TODO: transform tokens to input embeddings
        x = tokens
        logits = self.net(x)

        # targets = indices to select within cross_entropy probabilities
        if targets is None:
            loss = None
        else:
            B, T = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # TODO: implement model sampling method
    def generate(self, tokens, max_new_tokens):
        pass
