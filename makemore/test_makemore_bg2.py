# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false
#
# # Worksheet 2b - Single-Layer Perceptron
#
# This is the third in a series of companion worksheets for for Andrej Karpathy's [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) videos.
#
# It corresponds to roughly the second half of the second video in the series, named "[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)".
#
# The rest of the worksheets are listed in the README [here](https://github.com/Russ741/karpathy-nn-z2h/).
#
# The overall objective of this worksheet is to write code that generates a word that is similar to a set of example words it is trained on.
# It does so using a single-layer neural network.


# %% [markdown] deletable=false editable=false
# ### Preamble: Load data
#
# Objective: Write a function that:
#  * Loads the remotely-hosted [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) file
# ([raw link](https://github.com/karpathy/makemore/raw/master/names.txt))
#  * Returns a list of strings
#    * Each string should be equal to the word from the corresponding line of names.txt
#    * The strings should not include line-break characters
#
# Note: In practice, the order of the strings in the returned list does not matter, but for the
# test to pass, they should be in the same order in the list as in words.txt.
#
# Hint: In the video, Karpathy has a local copy of words.txt.<br>
# One way to fetch words.txt is to use a function from the [requests](https://pypi.org/project/requests/) library.
#
# Video: [0:03:03](https://youtu.be/PaCmpygFfXo?t=183)

# %%
def load_words():
    return open('makemore/names.txt', 'r').read().splitlines()

# %% deletable=false editable=false
def test_words():
    if not isinstance(loaded_words, list):
        print(f"Expected words to be a list")
        return
    if (len_words := len(loaded_words)) != (expected_words := 32033):
        print(f"Expected {expected_words} elements in words, found {len_words} elements")
        return
    if (zeroth_word := loaded_words[0]) != (expected_zeroth := "emma"):
        print(f"Expected zeroth word in words to be '{expected_zeroth}', was '{zeroth_word}'")
        return
    if (final_word := loaded_words[-1]) != (expected_final := "zzyzx"):
        print(f"Expected final word in words to be '{expected_final}', was '{final_word}'")
        return
    print("load_words looks good. Onwards!")
loaded_words = load_words()
test_words()

# %% [markdown] deletable=false editable=false
# ### Step 1: Generate bigrams
#
# Objective: Populate the variable ```bigrams``` with a list of bigrams (2-element tuples) of adjacent characters in ```words```.
#
# Treat the start and end of each word as the character '.'
#
# Video: [0:06:24](https://youtu.be/PaCmpygFfXo?t=384) and [0:21:55](https://youtu.be/PaCmpygFfXo?t=1315)

# %%
def generate_bigrams(words):
    bigrams = []
    for w in words:
        w = "." + w + "."
        for bg in zip(w,w[1:]):
            bigrams.append(bg)
    return bigrams

# %% deletable=false editable=false
def test_generate_bigrams():
    bigrams = generate_bigrams(loaded_words)
    if not isinstance(bigrams, list):
        print(f"Expected bigrams to be a list")
        return
    if (start_m_ct := bigrams.count(('.', 'm'))) != (expected_start_m_ct := 2538):
        print(f"Expected {expected_start_m_ct} ('a', 'b') bigrams, found {start_m_ct}")
        return
    if (ab_ct := bigrams.count(('a', 'b'))) != (expected_ab_ct := 541):
        print(f"Expected {expected_ab_ct} ('a', 'b') bigrams, found {ab_ct}")
        return
    if (s_end_ct := bigrams.count(('s', '.'))) != (expected_s_end_ct := 1169):
        print(f"Expected {expected_s_end_ct} ('s', '.') bigrams, found {s_end_ct}")
        return
    print("generate_bigrams looks good. Onwards!")
test_generate_bigrams()

# %% [markdown] deletable=false editable=false
# ### Step 2: Map characters to indices
#
# Objective: Write a function that takes the following arguments:
# * a list of char, char tuples representing all of the bigrams in a word list
#
# And returns:
# * a dict (```stoi```) where
#   * the key is a character from ```words``` (including '.' for start/end),
#   * the value is a unique integer, and
#   * all the values are in the range from 0 to ```len(stoi) - 1``` (no gaps)
#
# We'll use these unique integers as an index to represent the characters in a Tensor in later steps
#
# Note that for this list of words, the same value of ```stoi``` could be generated without looking at the words at all,
# but simply by using all the lowercase letters and a period. This approach would be more efficient for this exercise,
# but will not generalize well conceptually to more complex models in future exercises.
#
# Video: [0:15:40](https://youtu.be/PaCmpygFfXo?t=940)

# %% deletable=false editable=false
import string

# %%
def get_stoi(bigrams):
    chars = set([char for bg in bigrams for char in bg])
    stoi = {c:i for (i, c) in enumerate(chars)}
    print(stoi)
    return stoi

def test_get_stoi():
    bigrams = [
        ('.', 'h'),
        ('h', 'i'),
        ('i', '.'),
        ('.', 'b'),
        ('b', 'y'),
        ('y', 'e'),
        ('e', '.'),
    ]
    expected_s = sorted(['.', 'h', 'i', 'b', 'y', 'e'])
    stoi = get_stoi(bigrams)
    if not isinstance(stoi, dict):
        print(f"Expected stoi to be a dict")
        return
    s = sorted(stoi.keys())
    if s != expected_s:
        print(f"Expected stoi keys to be {expected_s} when sorted, were {s}")
        return
    expected_i = list(range(len(s)))
    i = sorted(stoi.values())
    if i != expected_i:
        print(f"Expected stoi values to be {expected_i} when sorted, were {i}")
        return
    print("get_stoi looks good. Onwards!")
test_get_stoi()

# %% [markdown] deletable=false editable=false
# ### Step 3: Map indices to characters
#
# Objective: Write a function that takes the following arguments:
# * a dict (```stoi```) as defined in step 2
#
# And returns:
# * a dict (```itos```) where ```itos``` contains the same key-value pairs as ```stoi``` but with keys and values swapped.
#
# E.g. if ```stoi == {'.' : 0, 'b' : 1, 'z', 2}```, then ```itos == {0 : '.', 1 : 'b', 2 : 'z'}```
#
# Video: [0:18:49](https://youtu.be/PaCmpygFfXo?t=1129)

# %%
def get_itos(stoi):
    return {i:c for (c, i) in stoi.items()}

# %% deletable=false editable=false
import string

def test_get_itos():
    stoi = {elem:idx for idx, elem in enumerate(string.ascii_lowercase + ".")}
    itos = get_itos(stoi)
    if not isinstance(itos, dict):
        print(f"Expected stoi to be a dict")
        return
    for c in string.ascii_lowercase + ".":
        c_i = stoi[c]
        if (expected_c := itos[c_i]) != c:
            print(f"Expected itos[{c_i}] to be {expected_c}, was {c}")
    print("get_itos looks good. Onwards!")
test_get_itos()

# %% [markdown] deletable=false editable=false
# ### Step 4: Split bigrams into inputs and outputs
#
# Objective: Write a function that takes the following arguments:
# * a list ```bigrams``` as defined in step 1, and
# * a dict ```stoi``` as defined in step 2
#
# And returns:
# * a one-dimensional torch.Tensor ```x``` with all of the first characters in the tuples in ```bigrams```
# * a one-dimensional torch.Tensor ```y``` with all of the second characters in the tuples in ```bigrams```
# * Note: Both output tensors should be the same length as ```bigrams```
#
# Video: [1:05:25](https://youtu.be/PaCmpygFfXo?t=3925)

# %%
import torch

def get_x_and_y(bigrams, stoi):
    xs, ys = [], []
    for (c1, c2) in bigrams:
        i1 = stoi[c1]
        i2 = stoi[c2]
        xs.append(i1)
        ys.append(i2)
    return torch.tensor(xs), torch.tensor(ys)

# %% deletable=false editable=false
def test_get_x_and_y():
    bigrams = [
        ('.', 'h'),
        ('h', 'i'),
        ('i', '.'),
        ('.', 'b'),
        ('b', 'y'),
        ('y', 'e'),
        ('e', '.'),
    ]
    stoi = {
        '.': 0,
        'h': 1,
        'i': 2,
        'b': 3,
        'y': 4,
        'e': 5,
    }
    x, y = get_x_and_y(bigrams, stoi)
    if (x0 := x[0]) != (expected_x0 := 0):
        print(f"Expected x[0] to be {expected_x0}, was {x0}")
        return
    if (y0 := y[0]) != (expected_y0 := 1):
        print(f"Expected y[0] to be {expected_y0}, was {y0}")
        return
    if (x_sfe := x[-2]) != (expected_x_sfe := 4):
        print(f"Expected x[-2] to be {expected_x_sfe}, was {x_sfe}")
        return
    if (y_sfe := y[-2]) != (expected_y_sfe := 5):
        print(f"Expected y[-2] to be {expected_y_sfe}, was {y_sfe}")
    print("get_x_and_y looks good. Onwards!")
test_get_x_and_y()

# %% [markdown] deletable=false editable=false
# ### Step 5: Provide initial values for the model parameters
#
# Objective: Write a function that takes the following arguments:
# * a dict ```stoi``` as defined in step 2
#   * the length of ```stoi``` will be referred to as ```stoi_n```
#
# And returns:
# * a pytorch.Tensor ```W``` of shape (```stoi_n```, ```stoi_n```) where each element is randomly generated
# * a pytorch.Tensor ```b``` of shape (1, ```stoi_n```)
#   * The elements of ```b``` can be zero
#
# Video: [1:14:03](https://youtu.be/PaCmpygFfXo?t=4433)

# %%
import torch

def initialize_w_b(stoi):
    num_c = len(stoi.keys())
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((num_c, num_c), generator=g, requires_grad=True)
    b = torch.zeros(1, num_c, requires_grad=True)
    return W, b

# %% deletable=false editable=false
def test_initialize_w_b():
    stoi = {'q': 0, 'w': 1, 'e': 2, 'r': 3}
    expected_s_ct = 4
    W, b = initialize_w_b(stoi)
    if (w_len := len(W)) != expected_s_ct:
        print(f"Expected W to have {expected_s_ct} rows, had {w_len}")
        return
    for row_idx in range(w_len):
        if (row_len := len(W[row_idx])) != expected_s_ct:
            print(f"Expected W[{row_idx}] to have {expected_s_ct} columns, had {row_len}")
            return
        for col_idx in range(row_len):
            if (val := W[row_idx][col_idx]) == 0.0:
                print(f"Expected W[{row_idx}][{col_idx}] to be non-zero.")
                return
    if not W.requires_grad:
        print("W must be marked with requires_grad so its grad property will be populated by backpropagation for use in gradient descent.")
        return
    if not b.requires_grad:
        print("b must be marked with requires_grad so its grad property will be populated by backpropagation for use in gradient descent.")
        return
    if (b_shape := b.shape) != (expected_b_shape := (1, expected_s_ct)):
        print(f"Expected b to have shape {expected_b_shape}, had shape {b_shape}")
        return
    print("initialize_w_b looks good. Onwards!")
test_initialize_w_b()

# %% [markdown] deletable=false editable=false
# ### Step 6: Forward propagation
#
# Objective: Write a function that takes the following arguments:
# * a pytorch.Tensor ```x``` of training or testing inputs
# * pytorch.Tensors ```W``` and ```b``` representing the parameters of the model
#
# And returns:
# * a pytorch.Tensor ```y_hat``` of the model's predicted outputs for each input in x
#   * The predicted outputs for a given sample should sum to 1.0
#   * The shape of ```y_hat``` should be (```len(x)```, ```len(W)```)
#     * Note that ```len(W)``` represents the number of different characters in the word list
#
# Video: [1:15:12](https://youtu.be/PaCmpygFfXo?t=4512)

# %%
def forward_prop(xs, ws, bs):
    logits = torch.stack([ws[ix] for ix in xs]) + bs
    counts = logits.exp()
    p = counts / counts.sum(1, keepdim=True)
    return p

# %% deletable=false editable=false
def test_forward_prop():
    x = torch.tensor([
        1,
        0,
    ])

    W = torch.tensor([
        [0.1, 0.9, 0.2, 0.01],
        [0.04, 0.2, 1.6, 0.25],
        [0.02, 0.03, 0.7, 0.01],
    ], dtype=torch.float64)

    b = torch.tensor([
        0.01, 0.02, 0.03, 0.04
    ], dtype=torch.float64)

    expected_y_hat = torch.tensor([
        [0.1203, 0.1426, 0.5841, 0.1530],
        [0.1881, 0.4228, 0.2120, 0.1771],
    ], dtype=torch.float64)

    y_hat = forward_prop(x, W, b)

    if not torch.isclose(expected_y_hat, y_hat, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected y_hat for test case to be \n{expected_y_hat}\n, was \n{y_hat}")
        return
    print("forward_prop looks good. Onwards!")
test_forward_prop()

# %% [markdown] deletable=false editable=false
# ### Step 7: Loss calculation
# Objective: Write a function that takes the following arguments:
# * a pytorch.Tensor ```y_hat``` of predicted outputs for a particular set of inputs
# * a pytorch.Tensor ```y``` of actual outputs for the same set of inputs
#
# And returns:
# * a floating-point value representing the model's negative log likelihood loss for that set of inputs
#
# Video: [1:35:49](https://youtu.be/PaCmpygFfXo&t=5749)
# %%
def calculate_loss(y_hat, y):
    return -y_hat[torch.arange(y.size()[0]), y].log().mean()

# %% deletable=false editable=false
from math import exp

def test_calculate_loss():
    y = torch.tensor([2], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.0, 0.0, 1.0, 0.0]
    ])
    if abs((loss := calculate_loss(y_hat, y))) > 0.00001:
        print(f"Expected loss for first example to be 0.0, was {loss}")
        return

    y = torch.tensor([2, 0], dtype=torch.int64)
    y_hat = torch.tensor([
        [0.09, 0.09, exp(-0.5), 0.09],
        [exp(-0.1), 0.01, 0.02, 0.03]
    ])
    if abs((loss := calculate_loss(y_hat, y)) - (expected_loss := 0.3)) > 0.00001:
        print(f"Expected loss for second example to be {expected_loss}, was {loss}")
        return
    print("calculate_loss looks good. Onwards!")
test_calculate_loss()

# %% [markdown] deletable=false editable=false
# ### Step 8: Gradient descent
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```W``` and ```b``` representing the parameters of the model
# * a floating-point value ```learning_rate``` representing the overall size of adjustment to make to the parameters
#
# And returns:
# * the updated pytorch.Tensors ```W``` and ```b```
#   * Note: Updating the parameters in-place is desirable, but for ease of testing, please return them regardless.
#
# Video: [1:41:26](https://youtu.be/PaCmpygFfXo?t=6086)

# %%
def descend_gradient(ws, bs, l_rate):
    ws.data += -l_rate * ws.grad
    bs.data += -l_rate * bs.grad
    return ws, bs

# %% deletable=false editable=false
def test_descend_gradient():
    W = torch.tensor([
        [1.0, 2.0,],
        [3.0, -4.0],
        [-5.0, 6.0],
    ])
    W.grad = torch.tensor([
        [-2.0, 1.0],
        [0.0, -2.0],
        [4.0, 1.0]
    ])
    b = torch.tensor([
        1.0,
        2.0,
    ])
    b.grad = torch.tensor([
        -1.0,
        0.5,
    ])
    new_w, new_b = descend_gradient(W, b, 3.0)
    expected_new_w = torch.tensor([
        [7.0, -1.0],
        [3.0, 2.0],
        [-17.0, 3.0]
    ])
    if not new_w.equal(expected_new_w):
        print(f"Expected new W for test case to be \n{expected_new_w}\n, is \n{new_w}")
        return
    expected_new_b = torch.tensor([
        4.0,
        0.5,
    ])
    if not new_b.equal(expected_new_b):
        print(f"Expected new b for test case to be \n{expected_new_b}\n, is \n{new_b}")
        return
    print("descend_gradient looks good. Onward!")
test_descend_gradient()

# %% [markdown] deletable=false editable=false
# ### Step 9: Train model
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```x``` and ```y``` as described in Step 4
# * pytorch.Tensors ```W``` and ```b``` as described in Step 5
# * a floating-point value ```learning_rate``` representing the overall size of adjustment to make to the parameters
#
# Updates the values of W and b to fit the data slightly better
#
# And returns:
# * the loss as defined in Step 6
#
# Implementation note: this function should make use of several of the functions you've previously implemented.
#
# Video: [1:42:55](https://youtu.be/PaCmpygFfXo?t=6175)

# %%
def train_model(xs, ys, ws, bs, l_rate):
    y_hat = forward_prop(xs, ws, bs)
    loss = calculate_loss(y_hat, ys)
    loss.backward()
    descend_gradient(ws, bs, l_rate)
    return loss

# %% deletable=false editable=false
def test_train_model():
    x = torch.tensor([
        0,
        1,
        2,
    ])
    y = torch.tensor([
        1,
        2,
        0,
    ])
    W = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([
        0.1,
        0.2,
        0.3,
    ], dtype=torch.float64, requires_grad=True)

    loss = train_model(x, y, W, b, 2.0)

    expected_W = torch.tensor([
        [0.7996, 1.4452, 0.7552],
        [0.7996, 0.7785, 1.4219],
        [1.4663, 0.7785, 0.7552]
    ], dtype=torch.float64)
    if not torch.isclose(expected_W, W, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected W for test case to be \n{expected_W}\n, was \n{W}")
        return

    expected_b = torch.tensor([
        0.1654,
        0.2022,
        0.2323
    ], dtype=torch.float64)
    if not torch.isclose(expected_b, b, rtol = 0.0, atol = 0.0001).all():
        print(f"Expected b for test case to be \n{expected_b}\n, was \n{b}")
        return
    print("train_model looks good. Onward!")
test_train_model()

# %% [markdown] deletable=false editable=false
# ### Step 10: Generate words
# Objective: Write a function that takes the following arguments:
# * pytorch.Tensors ```W``` and ```b``` as described in Step 5
# * a dict ```stoi``` as described in Step 2
# * a dict ```itos``` as described in Step 3
# * a torch.Generator to use for pseudorandom selection of elements
#
# Repeatedly generates a probability distribution for the next letter to select given the last letter
#
# And returns
# * a string representing a word generated by repeatedly sampling the probability distribution
#
# Video: [1:54:31](https://youtu.be/PaCmpygFfXo?t=6871)

# %%
def generate_word(ws, bs, stoi, itos, g):
    out = []
    ix = stoi["."]
    while True:
        logits = torch.stack([ws[ix]]) + bs
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == stoi["."]:
            break
    return "".join(out)

# %% deletable=false editable=false
def test_generate_word():
    stoi = {
        '.': 0,
        'o': 1,
        'n': 2,
        'w': 3,
        'a': 4,
        'r': 5,
        'd': 6,
    }
    stoi_n = len(stoi)
    itos = {v:k for k,v in stoi.items()}

    W = torch.zeros((stoi_n, stoi_n), dtype=torch.float64)
    b = torch.zeros((1, stoi_n), dtype=torch.float64)
    for i in range(stoi_n - 1):
        W[i][i+1] = 1.0
    W[stoi_n - 1][0] = 1.0

    gen = torch.Generator()
    gen.manual_seed(2147476727)
    if (word := generate_word(W, b, stoi, itos, gen)) != (expected_word := "onward"):
        print(f"Expected word for test case to be {expected_word}, was {word}")
        return
    print(f"generate_word looks good. Onward!")
test_generate_word()

# %% [markdown] deletable=false editable=false
# ### Finale: Put it all together
#
# Objective: Write (and call) a function that:
# * generates the bigrams and character maps
# * repeatedly trains the model until its loss is acceptably small
#   * For reference, the "perfect" loss of the probability table approach is approximately 2.4241
# * uses the model to generate some made-up names

# %%
# TODO: Implement solution here

# %%
