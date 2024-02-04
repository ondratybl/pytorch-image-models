# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import softmax


class MetaModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(1, 2))
        self.model1 = model1
        self.model2 = model2
        assert model1.num_classes == model2.num_classes, 'Incompatible models due to num_classes missmatch'
        self.num_classes = model1.num_classes

    def forward(self, x):
        weights = softmax(self.weights, dim=-1)
        y = torch.add(torch.mul(self.model1(x), weights[0, 0]), torch.mul(self.model2(x), weights[0, 1]))
        output = softmax(y, dim=-1)
        return output

    def string(self):
        return f'Weight for the first model: {softmax(self.weights, dim=-1)[0][0].item()}'