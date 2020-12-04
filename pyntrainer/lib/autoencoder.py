import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from torch.utils.data import Dataset, DataLoader
from scipy.stats import iqr
import numpy as np
from scipy import stats

from abstract_dataset import AbstractDataset

from utils import create_histogram
from utils import htb
from utils import fetch_threshold

class Autoencoder(nn.Module):
    def __init__(self, layers=[], device=torch.device("cpu")):
        super().__init__()

        self.device = device

        self.encoding_layers = nn.ModuleList([])
        self.decoding_layers = nn.ModuleList([])

        reversed_layers = list(reversed(layers))

        for i in range(len(layers) - 1):
            self.encoding_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.decoding_layers.append(nn.Linear(reversed_layers[i], reversed_layers[i+1]))

    def encode(self, x):
        for i in range(len(self.encoding_layers)):
            x = F.relu(self.encoding_layers[i](x))

        return x

    def decode(self, x):
        for i in range(len(self.decoding_layers)):
            if i != len(self.decoding_layers) - 1:
                x = F.relu(self.decoding_layers[i](x))
            else:
                x = torch.sigmoid(self.decoding_layers[i](x))

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def save(self, filename):
        state = {
            'state_dict': self.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'loss': self.loss,
            'mu_tensor': self.mu_tensor,
            'optimal_threshold': self.optimal_threshold
        }

        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        
        self.load_state_dict(state['state_dict'])

        self.optimizer  = state['optimizer']
        self.loss       = state['loss']
        self.mu_tensor  = state['mu_tensor']
        self.optimal_threshold = state['optimal_threshold']

    def errors(self, x):
        x_hat = self.forward(x)

#        if self.loss == "mse":
#            err = (x_hat - x).pow(2).sum(dim=1)
#        elif self.loss == "aml":
#            err = (x_hat - x).pow(2).sum(dim=1) + (x_hat - self.mu_tensor).pow(2).sum(dim=1)
#        else:

        err = (x_hat - x).pow(2).sum(dim=1)
        return err.detach().numpy()

    def predict(self, x):
        errors = self.errors(x)

        bool_arr = errors >= self.optimal_threshold

        return np.array([-1 if elem else 1 for elem in bool_arr])
        

    def synthesize(self, x, num_samples=100, n_dim=20):
        z = self.encode(x)

        z_set = z[torch.randperm(len(z))[:num_samples]]

        for i in range(len(z_set)):
            rand_indices = torch.randperm(len(z_set[i]))[:n_dim]

            for r_i in rand_indices:
                z_set[i][r_i] = random.uniform(0, 1)

        return self.decode(z_set)

    def set_optimal_threshold(self, x, add_syn=True, num_samples=100, n_dim=5):
        errors = self.errors(x)

        if add_syn:
            syn_errors = self.errors(self.synthesize(x, num_samples=num_samples, n_dim=n_dim))

            errors = np.concatenate((errors, syn_errors), axis=0)

        # Calculate the number of bins according to Freedman-Diaconis rule
        bin_width   = 2 * iqr(errors) / np.power(len(errors), (1/3))
        num_bins    = (np.max(errors) - np.min(errors)) / bin_width

        hist, bins = create_histogram(errors, num_bins=num_bins, step=bin_width)
        occurences = [float(o) for o in hist.tolist()]

        breaks = htb(hist)

        possible_thresholds = []

        for b in breaks:
            t = fetch_threshold(bins, hist, b)
            possible_thresholds.append(t)

        self.optimal_threshold = max(possible_thresholds)

        return self.optimal_threshold

    def train(self, x, epochs=100, lr=0.005, batch_size=5, loss="mse", with_thresholding=True):
        self.loss = loss

        # Get the mean vector
        self.mu_tensor = torch.tensor(x.numpy()[0].mean(axis=0)).to(self.device)

        data = AbstractDataset(x)
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        num_iterations = len(x) / batch_size

        for epoch in range(epochs):
            curr_loss = 0
            for i, (inputs, labels) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.forward(inputs)

                temp_mu_tensor = torch.tensor(inputs.numpy()[0].mean(axis=0)).to(self.device)

                if self.loss == "mse":
                    loss = (output - labels).pow(2).sum().mean()
                elif self.loss == "aml":
                    loss = (output - labels).pow(2).sum().mean() + (output - temp_mu_tensor).pow(2).sum().mean()
                else:
                    loss = (output - labels).pow(2).sum().mean()

                curr_loss += loss
                loss.backward()
                self.optimizer.step()

            curr_loss = curr_loss / num_iterations
            print("=> Epoch: %i\tLoss: %0.5f" % (epoch + 1, curr_loss.item()))

        if with_thresholding:
            print("Setting optimal threshold...")

            if self.loss == "mse":
                add_syn = False
            elif self.loss == "aml":
                add_syn = True

            self.set_optimal_threshold(x, add_syn=add_syn)

            print("Optimal threshold: %0.4f" % (self.optimal_threshold))
