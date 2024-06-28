#!/usr/bin/env python
# coding: utf-8

# # FIM on MNIST
# 
# - evaluate FIM and NTK on MNIST with different model sizes and seeds

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import wandb
import argparse
import gc
from fisher import get_ntk_tenas_new, get_ntk_tenas_new_probs, jacobian_batch_efficient, cholesky_covariance


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual_conn=True, bn=True):
        super(ResidualBlock, self).__init__()
        self.residual_conn = residual_conn
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        if self.residual_conn:
            out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels, seed, residual_conn=True, num_classes=10):
        super(ResNet, self).__init__()
        torch.manual_seed(seed)
        self.in_channels = in_channels
        self.residual_conn = residual_conn
        self.conv1 = nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.layer1 = self._make_layer(in_channels, num_blocks, residual_conn, stride=1)
        self.fc = nn.Linear(in_channels, num_classes)

    def _make_layer(self, out_channels, num_blocks, residual_conn, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, residual_conn, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = torch.nn.functional.avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train(model, config):

    iteration = 0
    for epoch in range(n_epochs):
        for batch_idx, (input_train, label_train) in enumerate(train_loader):
    
            model.train()
            input_train, label_train = input_train.to(device), label_train.to(device)
            
            optimizer.zero_grad()
            output_train = model(input_train)
            loss = criterion(output_train, label_train)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

                model.eval()
                correct = 0
                total = 0

                # Evaluate
                with torch.no_grad():
                    for input_test, label_test in test_loader:
                        input_test, label_test = input_test.to(device), label_test.to(device)
                        output_test = model(input_test)
                        _, predicted = torch.max(output_test.data, 1)
                        total += label_test.size(0)
                        correct += (predicted == label_test).sum().item()
                
                print(f'Accuracy of the model on the test set: {100 * correct / total}%')

                results = config.copy()
                results.update({'iteration': iteration, 'test_accuracy': correct / total, 'train_loss': loss.item()})
                results.update(compute(model, input_train))
                wandb.log(results)
    
            iteration += 1


def get_ntk(model, input):

    cholesky = cholesky_covariance(model(input))
    jacobian = jacobian_batch_efficient(model, input)

    ntk_small = torch.mean(torch.matmul(jacobian, torch.transpose(jacobian, dim0=1, dim1=2)), dim=0).detach()
    ntk_large = torch.mean(torch.matmul(torch.transpose(jacobian, dim0=1, dim1=2), jacobian), dim=0).detach()

    A = torch.matmul(cholesky, jacobian).detach()

    del jacobian, cholesky
    gc.collect()
    torch.cuda.empty_cache()

    ntk_small_p = torch.mean(torch.matmul(A, torch.transpose(A, dim0=1, dim1=2)), dim=0).detach()
    ntk_large_p = torch.mean(torch.matmul(torch.transpose(A, dim0=1, dim1=2), A), dim=0).detach()
    
    del A
    gc.collect()
    torch.cuda.empty_cache()

    if torch.isnan(ntk_small).sum().item() > 0:
        print(f'NTK small n. of nan: {torch.isnan(ntk_small).sum().item()}')

    if torch.isnan(ntk_large).sum().item() > 0:
        print(f'NTK small n. of nan: {torch.isnan(ntk_large).sum().item()}')

    if torch.isnan(ntk_small_p).sum().item() > 0:
        print(f'NTK small n. of nan: {torch.isnan(ntk_small_p).sum().item()}')

    if torch.isnan(ntk_large_p).sum().item() > 0:
        print(f'NTK large n. of nan: {torch.isnan(ntk_large_p).sum().item()}')

    return ntk_small, ntk_large, ntk_small_p, ntk_large_p


def compute(model, input):

    # NTK
    ntk_small, ntk_large, ntk_small_p, ntk_large_p = get_ntk(model, input)
    ntk_small_eig, ntk_large_eig = torch.linalg.eigvalsh(ntk_small), torch.linalg.eigvalsh(ntk_large)
    ntk_small_eig_p, ntk_large_eig_p = torch.linalg.eigvalsh(ntk_small_p), torch.linalg.eigvalsh(ntk_large_p)
    
    ntk_small_cond = ntk_small_eig.max().item() / (ntk_small_eig.min().item()) if ntk_small_eig.min().item() > 0 else None
    ntk_large_cond = ntk_large_eig.max().item() / (ntk_large_eig.min().item()) if ntk_large_eig.min().item() > 0 else None
    ntk_small_cond_p = ntk_small_eig_p.max().item() / (ntk_small_eig_p.min().item()) if ntk_small_eig_p.min().item() > 0 else None
    ntk_large_cond_p = ntk_large_eig_p.max().item() / (ntk_large_eig_p.min().item()) if ntk_large_eig_p.min().item() > 0 else None

    # TENAS
    eig_tenas = get_ntk_tenas_new(model, model(input)).detach()
    eig_tenas_probs = get_ntk_tenas_new_probs(model, model(input)).detach()

    return {
        'ntk_small_fro': torch.linalg.matrix_norm(ntk_small, ord='fro').item(),
        'ntk_small_nuc': torch.linalg.matrix_norm(ntk_small, ord='nuc').item(),
        'ntk_small_sing': torch.linalg.matrix_norm(ntk_small, ord=2).item(),
        'ntk_small_cond': ntk_small_cond,

        'ntk_large_fro': torch.linalg.matrix_norm(ntk_large, ord='fro').item(),
        'ntk_large_nuc': torch.linalg.matrix_norm(ntk_large, ord='nuc').item(),
        'ntk_large_sing': torch.linalg.matrix_norm(ntk_large, ord=2).item(),
        'ntk_large_cond': ntk_large_cond,

        'ntk_small_p_fro': torch.linalg.matrix_norm(ntk_small_p, ord='fro').item(),
        'ntk_small_p_nuc': torch.linalg.matrix_norm(ntk_small_p, ord='nuc').item(),
        'ntk_small_p_sing': torch.linalg.matrix_norm(ntk_small_p, ord=2).item(),
        'ntk_small_p_cond': ntk_small_cond_p,

        'ntk_large_p_fro': torch.linalg.matrix_norm(ntk_large_p, ord='fro').item(),
        'ntk_large_p_nuc': torch.linalg.matrix_norm(ntk_large_p, ord='nuc').item(),
        'ntk_large_p_sing': torch.linalg.matrix_norm(ntk_large_p, ord=2).item(),
        'ntk_large_p_cond': ntk_large_cond_p,

        'tenas_max': eig_tenas.max().item(),
        'tenas_sum': eig_tenas.sum().item(),
        'tenas_sum2': torch.square(eig_tenas).sum().item(),
        'tenas_std': eig_tenas.std().item(),
        'tenas_cond': eig_tenas.max().item() / (eig_tenas.min().item()) if eig_tenas.min().item() > 0 else None,

        'tenas_p_max': eig_tenas_probs.max().item(),
        'tenas_p_sum': eig_tenas_probs.sum().item(),
        'tenas_p_sum2': torch.square(eig_tenas_probs).sum().item(),
        'tenas_p_std': eig_tenas_probs.std().item(),
        'tenas_p_cond': eig_tenas_probs.max().item() / (
            eig_tenas_probs.min().item()) if eig_tenas_probs.min().item() > 0 else None,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='Random seed')

    # wandb
    wandb.init(
        project=None,
        config=None,
        name="MNIST",
        tags=["MNIST"],
    )

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the dataset
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    n_epochs = 20
    for seed in range(10):
        for num_blocks in [1, 2]:
            for in_channels in [2, 3, 4, 5]:
                for initializer in (torch.nn.init.kaiming_uniform_, torch.nn.init.kaiming_normal_, nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.ones_, torch.nn.init.xavier_normal_, torch.nn.init.xavier_uniform_):
                    for residual_conn in [True, False]:

                        # Create model
                        model = ResNet(num_blocks=num_blocks, in_channels=in_channels, residual_conn=residual_conn, seed=seed).to(device)

                        def custom_weights_init(m):
                            if isinstance(m, (nn.Conv2d, nn.Linear)):  # TODO: add BatchNormalization
                                initializer(m.weight)
                                nn.init.zeros_(m.bias)

                        model.apply(custom_weights_init)
                        total_params, trainable_params = count_parameters(model)
                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                        print('---------------------')
                        print(f'Seed {seed}, num_blocks {num_blocks}, in_channels {in_channels}, initializer {str(initializer)}, residual_conn {residual_conn}')
                        print(f'Total parameters: {total_params}')
                        print(f'Trainable parameters: {trainable_params}')
                        print('---------------------')

                        config = {
                            'seed': seed,
                            'num_blocks': num_blocks,
                            'in_channels': in_channels,
                            'initializer': str(initializer),
                            'residual_conn': int(residual_conn),
                            'params': trainable_params,
                        }
                        train(model, config)
