# mlp model implementation
import torch, random

words = open('makemore/names.txt', 'r').read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}

# generate examples from dataset
block_size = 3
def build_dataset(words):
  xs, ys = [], []
  for w in words:
    context = [0] * block_size
    for c in (w + "."):
      xs.append(context)
      ys.append(stoi[c])
      context = context[1:] + [stoi[c]]
  X = torch.tensor(xs) # torch.Size([228146, 3])
  Y = torch.tensor(ys) # torch.Size([228146])
  return X, Y

X, Y = build_dataset(words)

# split dataset into training, dev/validation, and test (80:10:10)
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# create embeddings table (first layer)
embed_dim = 2 # 2D vector embeddings
C = torch.randn((len(chars) + 1, embed_dim), requires_grad=True) # torch.Size([27, 2])
emb = C[Xtr] # embed entire inputs tensor X, torch.Size([228146, 3, 2])

# create weights + biases (hidden layer)
num_neurons = 100 # hyperparameter for tuning
W1 = torch.randn((block_size * embed_dim, num_neurons), requires_grad=True) # torch.Size([6, 100])
b1 = torch.randn(num_neurons, requires_grad=True) # torch.Size([100])

# hidden layer 
h = emb.view(-1, emb.shape[1] * emb.shape[2]) @ W1 + b1 # flatten C to be able to multiply W
h = h.tanh() # non-linear activation, torch.Size([228146, 100])

# create final weights + biases for final probability distribution tensor (final layer)
W2 = torch.randn((num_neurons, len(chars) + 1), requires_grad=True)
b2 = torch.randn(len(chars) + 1, requires_grad=True)
logits = h @ W2 + b2 # torch.Size([228146, 27])

import torch.nn.functional as F
loss = F.cross_entropy(logits, Ytr)

def descend_gradient(X, Y, C, W1, b1, W2, b2):
  parameters = [C, W1, b1, W2, b2]
  # mini batching for faster training
  ix = torch.randint(0, X.shape[0], (32,)) # torch.Size([32])

  ### forward pass
  emb = C[X[ix]] # get embeddings only for batch, torch.Size([32, 3, 2])
  h = torch.tanh(emb.view(-1, emb.shape[1] * emb.shape[2]) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Y[ix])

  ### backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  ### update
  lr = 0.1
  for p in parameters:
    p.data += -lr * p.grad

# train model
for _ in range(300000):
  descend_gradient(Xtr, Ytr, C, W1, b1, W2, b2)

# compare loss for training & dev
emb_tr = C[Xtr] # torch.Size([32, 3, 2])
h_tr = torch.tanh(emb_tr.view(-1, emb_tr.shape[1] * emb_tr.shape[2]) @ W1 + b1) # torch.Size([32, 100])
logits_tr = h_tr @ W2 + b2 # torch.Size([32, 27])
loss_tr = F.cross_entropy(logits_tr, Ytr)
print(loss_tr)

emb_dev = C[Xdev]
h_dev = torch.tanh(emb_dev.view(-1, emb_tr.shape[1] * emb_tr.shape[2]) @ W1 + b1)
logits_dev = h_dev @ W2 + b2
loss_dev = F.cross_entropy(logits_dev, Ydev)
print(loss_dev)

# sample from model
g = torch.Generator().manual_seed(2147483647)
def generate_words(num_words, g):
  for _ in range(num_words):
    out = []
    context = [0] * block_size
    ix = 0
    while True:
      emb = C[torch.tensor([context])] # torch.Size([1, 3, 2])
      h = torch.tanh(emb.view(1, -1) @ W1 + b1) # torch.Size([1, 100])
      logits = h @ W2 + b2 # torch.Size([1, 27])
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix])
      if (ix == 0):
        break
      context = context[1:] + [ix]
    print("".join(out))

generate_words(20, g)