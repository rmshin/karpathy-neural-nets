import random
from typing import Any
from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = 0

    def __call__(self, x):
        out = sum((wi * xi for (wi, xi) in zip(x, self.weights)), self.bias)
        return out.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    #  nout = num neurons in layer
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    # nlayers = list of layer sizes
    # e.g. [4, 4, 1] = two 4-neuron layers + one 1-neuron output
    def __init__(self, nin, nlayers, **kwargs):
        sizes = [nin] + nlayers
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
