# -*- coding: utf-8 -*-

import torch
from temperature_scaling import ModelWithTemperature


class MetaModel(torch.nn.Module):
    def __init__(self, model1, model2, temperature, init_logit=0.0):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor([[init_logit]]))
        self.model1 = ModelWithTemperature(model1, temperature)  # created model has trainable temperature
        self.model2 = ModelWithTemperature(model2, temperature)
        self.freeze_temperatures()
        assert model1.num_classes == model2.num_classes, 'Incompatible models due to num_classes missmatch'
        self.num_classes = model1.num_classes

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
        return f'Logit for the first model: {self.logit[0].item()}, number of trainable params: {self.get_trainable_param_count()}.'

    def get_trainable_param_count(self):
        trainable_param_count = 0
        for param in self.parameters():
            if param.requires_grad:
                trainable_param_count += param.numel()
        return trainable_param_count

    def set_and_freeze_temperatures(self, valid_loader):

        # Unfreeze temperatures
        for param_name, param in self.named_parameters():
            if 'temperature' in param_name:
                param.requires_grad = True

        # Optimize temperatures
        self.model1.set_temperature(valid_loader)
        self.model2.set_temperature(valid_loader)

        # Freeze temperatures
        self.freeze_temperatures()

    def freeze_temperatures(self):
        # Freeze temperatures
        for param_name, param in self.named_parameters():
            if 'temperature' in param_name:
                param.requires_grad = False

        print(f'Temperature for Model1: {self.model1.temperature[0].item()}')
        print(f'Temperature for Model2: {self.model2.temperature[0].item()}')

        print(self.string())