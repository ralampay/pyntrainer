import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd
from torch import tensor
import torch

from lib.autoencoder import Autoencoder

parser = argparse.ArgumentParser(description="PynTrainer: Stochastic Autoencoder trainer program")

parser.add_argument("--mode", help="Mode to be used", choices=["train", "predict"], type=str, default="train")
parser.add_argument("--input-file", help="Input csv file for training")
parser.add_argument("--model-file", help="Output model file", type=str, const=1, nargs='?', default="output.pth.tar")
parser.add_argument("--chunk-size", help="Chunk size for reading large files", type=int, const=1, nargs='?', default=100)
parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+', required=True)
parser.add_argument("--loss", help='Loss function', type=str, default="mse")
parser.add_argument("--lr", help='Learning rate', type=float, default=0.001)
parser.add_argument("--epochs", help='Number of epochs', type=int, default=100)
parser.add_argument("--batch-size", help='Batch size', type=int, default=5)
parser.add_argument("--cont", help='Continue training from model file', type=bool, default=False)
parser.add_argument("--eval-file", help='File to evaluate. Should have the format x1,x2,x3...y', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    input_file  = args.input_file
    model_file  = args.model_file
    chunk_size  = args.chunk_size
    layers      = args.layers
    mode        = args.mode
    loss        = args.loss
    lr          = args.lr
    epochs      = args.epochs
    batch_size  = args.batch_size
    cont        = args.cont
    eval_file   = args.eval_file

    print("Initializing autoencoder...")
    net = Autoencoder(layers=layers)
    print(net)

    if mode == "train":
        print("Loading training data...")
        data = pd.DataFrame()

        for chunk in pd.read_csv(input_file, header=None, chunksize=chunk_size):
            print("Reading chunk:")
            print(chunk)
            data = data.append(chunk)

        tensor_data = torch.tensor(data.values).float()

        input_dimensionality = len(data.columns)
        print("Input Dimensionality: %d" % (input_dimensionality))

        if cont:
            net.load(model_file)

        print("Training...")
        net.train(tensor_data, epochs=epochs, lr=lr, batch_size=batch_size, loss=loss)
        net.save(model_file)

    elif mode == "predict":
        print("Loading training data...")
        data = pd.DataFrame()

        for chunk in pd.read_csv(eval_file, header=None, chunksize=chunk_size):
            print("Reading chunk...")
            data = data.append(chunk)

        print("Parsing evaluation file %s..." % (eval_file))

        positive_data = torch.tensor(data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1].values).float()
        negative_data = torch.tensor(data[data[len(data.columns) - 1] == 0].iloc[:,:len(data.columns) - 1].values).float()

        print("Number of positive data: %d" % (len(positive_data)))
        print("Number of negative data: %d" % (len(negative_data)))

        print("Loading model...")
        net.load(model_file)
        print(net)

        positive_predictions = net.predict(positive_data)
        negative_predictions = net.predict(negative_data)

        print("Positive predictions")
        print(positive_predictions)

        print("Negative predictions")
        print(negative_predictions)

        print("Positive Accuracy: %0.4f" % (len(positive_predictions[positive_predictions == 1]) / len(positive_predictions)))
        print("Negative Accuracy: %0.4f" % (len(negative_predictions[negative_predictions == -1]) / len(negative_predictions)))
