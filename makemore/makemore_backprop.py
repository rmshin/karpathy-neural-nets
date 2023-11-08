# manual backprop calculation exercises
import torch, random
import torch.nn.functional as F

words = open("makemore/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}

block_size = 3
def build_dataset(words):
    xs, ys = [], []
    for w in words:
        context = [0] * block_size
        for c in w + ".":
            xs.append(context)
            ys.append(stoi[c])
            context = context[1:] + [stoi[c]]
    X = torch.tensor(xs)  # torch.Size([228146, 3])
    Y = torch.tensor(ys)  # torch.Size([228146])
    return X, Y

# split dataset into training, dev/validation, and test (80:10:10)
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# utility function to compare manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

embed_dim = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP
vocab_size = len(itos)

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, embed_dim), generator=g)
# Layer 1
W1 = torch.randn((embed_dim * block_size, n_hidden), generator=g) * (5/3)/((embed_dim * block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.1 # b1 isn't actually relevant due to BN below
# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
  p.requires_grad = True

batch_size = 32
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb] # [32, 3, 10]
embcat = emb.view(emb.shape[0], -1) # [32, 30]
# Linear layer 1
h_prebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1 / batch_size * h_prebn.sum(0, keepdim=True)
bndiff = h_prebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1 / (batch_size - 1) * (bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
h_preact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(h_preact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(batch_size), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, h_preact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, h_prebn, bnmeani,
         embcat, emb]:
  t.retain_grad()

loss.backward()
print(loss)

#########################

# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one

# -----------------
# YOUR CODE HERE :)
# -----------------

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(batch_size), Yb] = -1.0 / batch_size
cmp('logprobs', dlogprobs, logprobs) # dL/dlogprobs

dprobs = (1 / probs) * dlogprobs
cmp('probs', dprobs, probs) # dL/dlogprobs * dlogprobs/dprobs

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv) # dL/dprobs * dprobs/dcounts_sum_inv

dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum) # dL/dcounts_sum_inv * dcounts_sum_inv/dcounts_sum

dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts) # dL/dprobs * dprobs/dcounts + dL/dcounts_sum * dcounts_sum/dcounts

dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits) # dL/dcounts * dcounts/dnorm_logits

dlogit_maxes = (-dnorm_logits.clone()).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes) # dL/dnorm_logits * dnorm_logits/dlogit_maxes

dlogits = dnorm_logits.clone() + F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
cmp('logits', dlogits, logits) # dL/dnorm_logits * dnorm_logits/dlogits + dL/dlogit_maxes * dlogit_maxes/dlogits

dh = dlogits @ W2.transpose(0, 1)
cmp('h', dh, h) # dL/dlogits @ W2^T

dW2 = h.transpose(0, 1) @ dlogits
cmp('W2', dW2, W2) # h^T @ dL/dlogits

db2 = dlogits.sum(0)
cmp('b2', db2, b2) 

dh_preact = (1 - h**2) * dh
cmp('hpreact', dh_preact, h_preact)

dbngain = (bnraw * dh_preact).sum(0)
cmp('bngain', dbngain, bngain)

dbnbias = dh_preact.sum(0)
cmp('bnbias', dbnbias, bnbias)

dbnraw = dh_preact * bngain
cmp('bnraw', dbnraw, bnraw)

dbnvar_inv = (dbnraw * bndiff).sum(0)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

dbndiff2 = torch.ones_like(bndiff2) * 1/(batch_size - 1) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff = 2 * bndiff * dbndiff2 + dbnraw * bnvar_inv
cmp('bndiff', dbndiff, bndiff)

dbnmeani = (-dbndiff).sum(0)
cmp('bnmeani', dbnmeani, bnmeani)

d_hprebn = 1 / batch_size * torch.ones_like(h_prebn) * dbnmeani + dbndiff
cmp('hprebn', d_hprebn, h_prebn)

dembcat = d_hprebn @ torch.transpose(W1, 0, 1)
cmp('embcat', dembcat, embcat)

dW1 = torch.transpose(embcat, 0, 1) @ d_hprebn
cmp('W1', dW1, W1)

db1 = d_hprebn.sum(0)
cmp('b1', db1, b1)

demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

dC = torch.zeros_like(C)
for n in range(Xb.shape[0]):
   for m in range(Xb.shape[1]):
      ix = Xb[n, m]
      dC[ix] += demb[n, m]
cmp('C', dC, C)

#########################

# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# forward pass

# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())

# backward pass

# -----------------
# YOUR CODE HERE :)
dlogits = None # TODO. my solution is 3 lines
# -----------------

#cmp('logits', dlogits, logits) # I can only get approximate to be true, my maxdiff is 6e-9

#########################

# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out
# BatchNorm paper: https://arxiv.org/abs/1502.03167

# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# now:
hpreact_fast = bngain * (h_prebn - h_prebn.mean(0, keepdim=True)) / torch.sqrt(h_prebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - h_preact).abs().max())

# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)

# -----------------
# YOUR CODE HERE :)
dhprebn = None # TODO. my solution is 1 (long) line
# -----------------

cmp('hprebn', dhprebn, h_prebn) # I can only get approximate to be true, my maxdiff is 9e-10