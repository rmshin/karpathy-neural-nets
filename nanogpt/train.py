import torch
import model
import time

with open("nanogpt/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]  # convert string to list of integers
decode = lambda l: "".join([itos[i] for i in l])  # convert list of integers to string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#### hyper-parameters ####
batch_size = 16
block_size = 64
num_heads = 8
num_layers = 8
embed_dim = 64
vocab_size = len(stoi)
dropout = 0.0
learning_rate = 1e-3
max_iter = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#### ---------------- ####

torch.manual_seed(1337)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


gpt = model.GPTLanguageModel(embed_dim, num_heads, num_layers, vocab_size, block_size, dropout).to(device)
print("\nModel size: ", sum(p.numel() for p in gpt.parameters())/1e6, 'M parameters\n')

# train model
print("---------- TRAINING START ----------")
start = time.perf_counter()
optimiser = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)
for i in range(max_iter):
    x, y = get_batch("train")
    _, loss = gpt(x, y)
    optimiser.zero_grad(True)
    loss.backward()
    optimiser.step()
    # print progress
    if i % 200 == 0 or i == max_iter - 1:
      print(f"progress: {i / max_iter * 100:.2f}%")
end = time.perf_counter()
print("---------- TRAINING FINISH ----------\n")
elapsed = end - start
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
print(f"Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

# evaluate final loss on train and val datasets
@torch.no_grad()
def estimate_loss():
    out = {}
    gpt.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            _, loss = gpt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt.train()
    return out

losses = estimate_loss()
print(f"Training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}\n")

# generate tokens for a batch
context = torch.zeros((1,1), dtype=torch.long, device=device)
new_tokens = gpt.generate(context, 1024).squeeze()
print("---------- START GENERATION ----------")
print(decode(new_tokens.tolist()), '\n')