import torch, math
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, embed_dim/channels)
        # output of size (batch, time-step, head_size)
        # B, T, C = x.shape
        T = x.shape[1]
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # calculate scaled dot-product attention
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # [B, T, T]
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [B, T, T]
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        att = weights @ v  # [B, T, head_size]
        return att


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # [B, T, embed_dim]
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # NOTE: understand why inner layer is 4x I/O dimension
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.mha = MultiHeadedAttention(embed_dim, num_heads, embed_dim // num_heads, block_size, dropout)
        self.ff = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # input & ouput size [batch, time-step, embed_dim]
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, vocab_size, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embeding_table = nn.Embedding(vocab_size, embed_dim)
        position_embedding_table = self._get_pos_embed_table(embed_dim)
        self.register_buffer("pet", position_embedding_table)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            *[Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)],
            nn.LayerNorm(embed_dim),  # final layer norm
            nn.Linear(embed_dim, vocab_size)  # output logits
        )
        # NOTE: better init in Karpathy's reference code, figure out its purpose
        self.apply(self._init_weights)
    
    # fixed sinusoidal positional embeddings
    def _get_pos_embed_table(self, embed_dim):
        pet = torch.zeros(self.block_size, embed_dim)
        pos = torch.arange(self.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pet[:,0::2] = torch.sin(pos * div_term)
        pet[:,1::2] = torch.cos(pos * div_term)
        return pet

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
        # both tokens & targets are B, T tensors containing indices
        x = self.token_embeding_table(tokens)
        x = x + self.pet[torch.arange(tokens.shape[1])].requires_grad_(False)
        logits = self.net(self.dropout(x)) # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, _ = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, tokens, max_new_tokens):
        for _ in range(max_new_tokens):
            # only process last block_size tokens
            token_block = tokens[:, -self.block_size:]
            logits, _ = self(token_block)
            # focus only on the last time step
            logits = logits[:, -1, :] # [B, C]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            tokens = torch.cat((tokens, next_token), dim=1) # (B, T+1)
        return tokens
