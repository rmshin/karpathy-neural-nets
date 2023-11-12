import torch, random
import torch.nn as nn

g = torch.Generator().manual_seed(2147483647)


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = (
            torch.randn((in_features, out_features), generator=g) / in_features**0.5
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
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
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


words = open("makemore/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}

# generate examples from dataset
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


embed_dim = 5  # vector embeddings dimension
n_hidden = 200  # hidden layer neuron size

C = torch.randn((len(stoi), embed_dim), generator=g)
layers = [
    Linear(embed_dim * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # first hidden layer
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # second
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # third
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    TanH(),
    # final out layer
    Linear(n_hidden, len(stoi), bias=False),
    BatchNorm1d(len(stoi)),
]


with torch.no_grad():
    # scale down final layer weights to achieve reasonable loss at initialisation
    layers[-1].gamma *= 0.1
    # apply gain to all other layers to counterbalance tanh compression
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3  # 5/3 due to tanh activation

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

# train model
for i in range(300000):
    # mini batching for faster training
    ix = torch.randint(0, Xtr.shape[0], (32,))  # torch.Size([32])

    ### forward pass
    emb = C[Xtr[ix]]  # torch.Size([32, 3, 2])
    x = emb.view(emb.shape[0], -1)
    for l in layers:
        x = l(x)

    loss = nn.functional.cross_entropy(x, Ytr[ix])

    ### backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    ### update
    lr = 0.1 if i < 225000 else 0.01  # basic learning rate decay
    for p in parameters:
        p.data += -lr * p.grad


# compare loss for training & dev
@torch.no_grad()
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]  # (N, block_size, embed_dim)
    x = emb.view(emb.shape[0], -1)  # (N, block_size * embed_dim)
    for layer in layers:
        x = layer(x)
    loss = nn.functional.cross_entropy(x, y)
    print(split, loss.item())


# put layers into eval mode
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
split_loss("train")
split_loss("val")


# sample from model
def generate_words(num_words, g):
    for _ in range(num_words):
        out = []
        context = [0] * block_size
        ix = 0
        while True:
            emb = C[torch.tensor([context])]  # torch.Size([1, 3, 2])
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            probs = nn.functional.softmax(x, dim=1)
            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
            context = context[1:] + [ix]
        print("".join(out))


generate_words(20, g)
