# bigram model implementation
import torch, string

words = open('makemore/names.txt', 'r').read().splitlines()

########## CLASSICAL STATISTICS MODEL ##########

chars = string.ascii_lowercase.split("")
ctoi = {c:i+1 for (i, c) in enumerate(chars)}
ctoi["."] = 0 # special char to represent start and end of word
itoc = {i:c for (c, i) in ctoi.items()}

# init with 1's to avoid 0 probability of zero frequency bigrams
N = torch.full((27, 27), 1, dtype=torch.int32)
for word in words:
  word = itoc[0] + word + itoc[0]
  for (c1, c2) in zip(word, word[1:]):
    i1 = ctoi[c1]
    i2 = ctoi[c2]
    N[i1, i2] += 1

# create probability distributions for every bigram combination
P = N.float()
P /= P.sum(1, keepdim=True) # ensure sum of row rather than column

# sample model
g = torch.Generator().manual_seed(2147483647)
for i in range(20):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    if ix == 0:
      break
    out.append(itoc[ix])
  print("".join(out))


########## NEURAL NETWORK MODEL ##########

# xs = inputs (first bigram letter index), ys = labels (associated second bigram letter index)
xs, ys = [], []
for word in words:
  word = itoc[0] + word + itoc[0]
  for (c1, c2) in zip(word, word[1:]):
    i1 = ctoi[c1]
    i2 = ctoi[c2]
    xs.append(i1)
    ys.append(i2)
ys = torch.tensor(ys)
num = len(xs)

# 27 neurons with 27 inputs each - each neuron represents probability distribution of one starting bigram letter
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(20):
  ### forward pass
  # each ix value in xs represents character index, W[ix] = "probability distribution" of itoc[ix]
  logits = torch.stack([W[ix] for ix in xs]) # builds probability distribution matrix
  counts = logits.exp()
  P_NN = counts / counts.sum(1, keepdim=True)
  loss = -P_NN[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # add regularisation

  ### backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  ### update
  W.data += -50 * W.grad

# sample model
g = torch.Generator().manual_seed(2147483647)
for i in range(20):
  out = []
  ix = 0
  while True:
    logits = torch.stack([W[ix]])
    counts = logits.exp()
    p = counts / counts.sum(1, keepdim=True)
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    if ix == 0:
      break
    out.append(itoc[ix])
  print("".join(out))