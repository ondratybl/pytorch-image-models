# -*- coding: utf-8 -*-

import torch
from temperature_scaling import ModelWithTemperature


class MetaModel(torch.nn.Module):
    def __init__(self, model1, model2, num_classes, init_logit=0.0):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor([[init_logit]]))
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes

    def forward(self, x):

        # Weights for the two models
        weights = torch.nn.functional.softmax(torch.cat([self.logit, -self.logit], dim=1), dim=-1)

        # Standardized logits of the two models
        model1_logits = self.model1(x)
        model2_logits = self.model2(x)
        model1_logits_std = model1_logits #- torch.mean(model1_logits, dim=1, keepdim=True).repeat(1, self.num_classes)
        model2_logits_std = model2_logits #- torch.mean(model2_logits, dim=1, keepdim=True).repeat(1, self.num_classes)

        # Combined output
        y = torch.add(torch.mul(model1_logits_std, weights[0, 0]), torch.mul(model2_logits_std, weights[0, 1]))

        return torch.nn.functional.softmax(y, dim=-1)

    def string(self):
        return f'Logit for the first model: {self.logit[0].item()}, number of trainable params: {self.get_trainable_param_count()}.'

    def get_trainable_param_count(self):
        trainable_param_count = 0
        for param in self.parameters():
            if param.requires_grad:
                trainable_param_count += param.numel()
        return trainable_param_count