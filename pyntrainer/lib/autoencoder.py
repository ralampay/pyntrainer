import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from torch.utils.data import Dataset, DataLoader

from abstract_dataset import AbstractDataset

class Autoencoder(nn.Module):
    def __init__(self, layers=[]):
        super().__init__()

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
            'loss': self.loss
        }

        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        
        self.load_state_dict(state['state_dict'])

        self.optimizer  = state['optimizer']
        self.loss       = state['loss']

    def train(self, x, epochs=100, lr=0.005, batch_size=5, loss="mse"):
        self.loss = loss

        # Get the mean vector
        self.mu_tensor = torch.tensor(x.numpy()[0].mean(axis=0))

        data = AbstractDataset(x)
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        num_iterations = len(x) / batch_size

        for epoch in range(epochs):
            curr_loss = 0
            for i, (inputs, labels) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.forward(inputs)

                if self.loss == "mse":
                    loss = (output - labels).pow(2).sum().mean()
                elif self.loss == "aml":
                    loss = (output - labels).pow(2).sum().mean() + (output - self.mu_tensor).pow(2).sum().mean()
                else:
                    loss = (output - labels).pow(2).sum().mean()

                curr_loss += loss
                loss.backward()
                self.optimizer.step()

            curr_loss = curr_loss / num_iterations
            print("=> Epoch: %i\tLoss: %0.5f" % (epoch + 1, curr_loss.item()))
