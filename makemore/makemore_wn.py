# wavenet model implementation
import torch, random, time
import torch.nn as nn

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = (
            torch.randn((in_features, out_features)) / in_features**0.5
        )  # Kaiming init
        self.bias = torch.zeros(out_features) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # BN params
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        # running estimate
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
              dim = 0
            elif x.ndim == 3:
              dim = (0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        # BN calculation
        yhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * yhat + self.beta
        # update running estimates
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class TanH:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(_):
        return []
    
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        return [self.weight]

class FlattenC:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        N, B, F = x.shape
        x = x.view(N, B // self.n, F * self.n)
        if x.shape[1] == 1:
          x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(_):
        return []

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        self.out = x
        return x    

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] 

words = open("makemore/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}

# generate examples from dataset
block_size = 8


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


torch.manual_seed(42)
embed_dim = 5  # vector embeddings dimension
n_hidden = 128  # hidden layer neuron size
# NOTE: number of flattens should correlate to block size (e.g. block size of 8 means 3 flattens of size 2 since 2^3 == 8)
model = Sequential(
    # embedding
    Embedding(len(stoi), embed_dim),
    FlattenC(2),
    Linear(embed_dim * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # first hidden layer
    FlattenC(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # second
    FlattenC(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # final out layer
    Linear(n_hidden, len(stoi), bias=False),
    BatchNorm1d(len(stoi)),
)


with torch.no_grad():
    # scale down final layer weights to achieve reasonable loss at initialisation
    model.layers[-1].gamma *= 0.1
    # apply gain to all other layers to counterbalance tanh compression
    for layer in model.layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3  # 5/3 due to tanh activation

parameters = model.parameters()
for p in parameters:
    p.requires_grad = True

# train model
start = time.perf_counter()
for i in range(300000):
    # mini batching for faster training
    ix = torch.randint(0, Xtr.shape[0], (32,))  # torch.Size([32])

    ### forward pass
    logits = model(Xtr[ix])
    loss = nn.functional.cross_entropy(logits, Ytr[ix])

    ### backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    ### update
    lr = 0.1 if i < 225000 else 0.01  # basic learning rate decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # print progress
    if i % 1000 == 0:
      print(f"{i / 300000 * 100:.2f}%")
end = time.perf_counter()
print(f"Total elapsed time: {time.strftime('%H:%M:%S', end - start)}")

# compare loss for training & dev
@torch.no_grad()
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    print(split, loss.item())


# put layers into eval mode
for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
split_loss("train")
split_loss("val")


# sample from model
g = torch.Generator().manual_seed(2147483647)
def generate_words(num_words, g):
    for _ in range(num_words):
        out = []
        context = [0] * block_size
        ix = 0
        while True:
            logits = model(torch.tensor([context]))
            probs = nn.functional.softmax(logits, dim=1)
            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
            context = context[1:] + [ix]
        print("".join(out))


generate_words(20, g)
