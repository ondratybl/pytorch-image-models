# -*- coding: utf-8 -*-

import torch


class MetaModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.rand(1, 1))
        self.model1 = model1
        self.model2 = model2
        assert model1.num_classes == model2.num_classes, 'Incompatible models due to num_classes missmatch'
        self.num_classes = model1.num_classes

        # Count number of trainable parameters
        trainable_param_count = 0
        for param in self.parameters():
            if param.requires_grad:
                trainable_param_count += param.numel()
        print(f'Number of trainable parameters for the meta model: {trainable_param_count}')

    def forward(self, x):

        # Weights for the two models
        weights = torch.nn.functional.softmax(torch.cat([self.logit, -self.logit], dim=1), dim=-1)

        # Standardized logits of the two models
        model1_logits = self.model1(x)
        model2_logits = self.model2(x)
        model1_logits_std = model1_logits - torch.mean(model1_logits, dim=1, keepdim=True).repeat(1, self.num_classes)
        model2_logits_std = model2_logits - torch.mean(model2_logits, dim=1, keepdim=True).repeat(1, self.num_classes)

        # Combined output
        y = torch.add(torch.mul(model1_logits_std, weights[0, 0]), torch.mul(model2_logits_std, weights[0, 1]))

        return torch.nn.functional.softmax(y, dim=-1)

    def string(self):
        return f'Weight for the first model: {torch.nn.functional.softmax(torch.cat([self.logit, -self.logit], dim=1), dim=-1)[0][0].item()}'
