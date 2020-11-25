import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd
from torch import tensor
import torch

from lib.autoencoder import Autoencoder

parser = argparse.ArgumentParser(description="PynTrainer: Stochastic Autoencoder trainer program")

parser.add_argument("--mode", help="Mode to be used", default="train")
parser.add_argument("--input-file", help="Input csv file for training", required=True)
parser.add_argument("--model-file", help="Output model file", type=str, const=1, nargs='?', default="output.pth.tar")
parser.add_argument("--chunk-size", help="Chunk size for reading large files", type=int, const=1, nargs='?', default=100)
parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+', required=True)
parser.add_argument("--loss", help='Loss function', type=str, default="mse")

args = parser.parse_args()

if __name__ == '__main__':
    input_file  = args.input_file
    model_file  = args.model_file
    chunk_size  = args.chunk_size
    layers      = args.layers
    mode        = args.mode
    loss        = args.loss

    data = pd.DataFrame()

    for chunk in pd.read_csv(input_file, header=None, chunksize=chunk_size):
        print("Reading chunk:")
        print(chunk)
        data = data.append(chunk)

    tensor_data = torch.tensor(data.values).float()

    input_dimensionality = len(data.columns)
    print("Input Dimensionality: %d" % (input_dimensionality))

    print("Initializing autoencoder...")
    net = Autoencoder(layers=layers)
    print(net)

    print(net.forward(tensor_data))

    if mode == "train":
        net.train(tensor_data, loss=loss)
        print(net)
        net.save(model_file)
    elif mode == "predict":
        pass
