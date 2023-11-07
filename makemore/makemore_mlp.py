# mlp model implementation
import torch, random
import torch.nn.functional as F

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

# create embeddings table (first layer)
embed_dim = 5  # 2D vector embeddings
C = torch.randn((len(chars) + 1, embed_dim))  # torch.Size([27, 2])

# create weights + biases (hidden layer)
num_neurons = 200  # hyperparameter for tuning
W1 = (
    torch.randn((block_size * embed_dim, num_neurons))
    * (5 / 3)
    / ((block_size * embed_dim) ** 0.5)
)  # Kaiming initialisation, torch.Size([6, 100])
# b1 = torch.randn(num_neurons, requires_grad=True) * 0.01 # torch.Size([100]) - not needed with batch normalisation

# create weights + biases for final probability distribution tensor (final layer)
W2 = torch.randn((num_neurons, len(chars) + 1)) * 0.01  # torch.Size([100, 27])
b2 = torch.zeros(len(chars) + 1)  # torch.Size([27])

# for batch normalisation
bngain = torch.ones((1, num_neurons))
bnbias = torch.zeros((1, num_neurons))
# estimated running bn mean & std
bnmean_running = torch.zeros(
    (1, num_neurons)
)  # initialise zero mean due to Kaiming normal init
bnstd_running = torch.ones(
    (1, num_neurons)
)  # initialise to unit std due to Kaiming normal init


def descend_gradient(X, Y, C, W1, W2, b2, bngain, bnbias, i):
    parameters = [C, W1, W2, b2, bngain, bnbias]
    # mini batching for faster training
    ix = torch.randint(0, X.shape[0], (32,))  # torch.Size([32])

    ### forward pass
    emb = C[X[ix]]  # get embeddings only for batch, torch.Size([32, 3, 2])
    h_preact = emb.view(emb.shape[0], -1) @ W1
    bnmean_i = h_preact.mean(0, keepdim=True)
    bnstd_i = h_preact.std(0, keepdim=True)
    h_preact = bngain * (h_preact - bnmean_i) / bnstd_i + bnbias  # batch normalisation

    with torch.no_grad():
        global bnmean_running
        global bnstd_running
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean_i
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstd_i

    h = torch.tanh(h_preact)
    logits = h @ W2 + b2  # torch.Size([32, 27])
    loss = F.cross_entropy(logits, Y[ix])

    ### backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    ### update
    lr = 0.1 if i < 200000 else 0.01  # basic learning rate decay
    for p in parameters:
        p.data += -lr * p.grad


# train model
for p in [C, W1, W2, b2, bngain, bnbias]:
    p.requires_grad = True

for i in range(300000):
    descend_gradient(Xtr, Ytr, C, W1, W2, b2, bngain, bnbias, i)

# compare loss for training & dev
with torch.no_grad():
    emb_tr = C[Xtr]  # torch.Size([32, 3, 2])
    h_tr_preact = emb_tr.view(emb_tr.shape[0], -1) @ W1  # torch.Size([32, 100])
    h_tr_preact = bngain * (h_tr_preact - bnmean_running) / bnstd_running + bnbias
    h_tr = torch.tanh(h_tr_preact)  # torch.Size([32, 100])
    logits_tr = h_tr @ W2 + b2  # torch.Size([32, 27])
    loss_tr = F.cross_entropy(logits_tr, Ytr)
    print(loss_tr)

    emb_dev = C[Xdev]
    h_dev_preact = emb_dev.view(emb_dev.shape[0], -1) @ W1
    h_dev_preact = bngain * (h_dev_preact - bnmean_running) / bnstd_running + bnbias
    h_dev = torch.tanh(h_dev_preact)
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
            emb = C[torch.tensor([context])]  # torch.Size([1, 3, 2])
            h_preact = emb.view(1, -1) @ W1  # torch.Size([1, 100])
            h_preact = bngain * (h_preact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(h_preact)  # torch.Size([1, 100])
            logits = h @ W2 + b2  # torch.Size([1, 27])
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
            context = context[1:] + [ix]
        print("".join(out))


generate_words(20, g)
