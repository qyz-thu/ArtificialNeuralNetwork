from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return ((input - target) ** 2).mean(axis=0).sum() / 2

    def backward(self, input, target):
        return target - input


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self._saved_tensor = None

    def forward(self, input, target):
        h = (np.exp(input).transpose(1, 0) / np.exp(input).sum(axis=1)).transpose(1, 0)   # softmax
        return -(target * np.log(h)).sum()

    def backward(self, input, target):
        h = (np.exp(input).transpose(1, 0) / np.exp(input).sum(axis=1)).transpose(1, 0)
        return target - h
