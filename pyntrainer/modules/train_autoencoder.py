import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from torch import tensor
import torch

from lib.autoencoder import Autoencoder

class TrainAutoencoder:
  def __init__(self, params=None):
    # Parameters for autoencoder
    self.layers     = params.get('layers')
    self.epochs     = params.get('epochs')
    self.lr         = params.get('lr')
    self.batch_size = params.get('batch_size')
    self.device     = params.get('device')

    # Flow configurations
    self.cont       = params.get('cont')
    self.model_file = params.get('model_file')

    # Process parameters
    self.input_file = params.get('input_file')
    self.chunk_size = params.get('chunk_size')
    self.cf_file    = params.get('cf_file')

  def execute(self):
    print("Initializing autoencoder...")
    net = Autoencoder(layers=self.layers)

    net.to(self.device)

    print("Loading training data...")
    data = pd.DataFrame()

    for i, chunk in enumerate(pd.read_csv(self.input_file, header=None, chunksize=self.chunk_size)):
      print("Reading chunk: {}".format(i+1))
      print(chunk)
      data = data.append(chunk)

    tensor_data = torch.tensor(data.values).float()
    print(tensor_data)

    input_dimensionality = len(data.columns) - 1
    print("Input Dimensionality: {}".format(input_dimensionality))

    if self.cont:
      print("Loading model_file {}...".format(self.model_file))
      net.load(self.model_file)

    print("Training...")
    net.fit(tensor_data, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)

    print("Saving to {}...".format(self.model_file))
    net.save(self.model_file)
