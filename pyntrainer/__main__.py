import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd
from torch import tensor
import torch
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from lib.autoencoder import Autoencoder
from math import sqrt

parser = argparse.ArgumentParser(description="PynTrainer: Stochastic Autoencoder trainer program")

parser.add_argument("--mode", help="Mode to be used", choices=["train", "predict", "eval"], type=str, default="train")
parser.add_argument("--input-file", help="Input csv file for training")
parser.add_argument("--model-file", help="Output model file", type=str, const=1, nargs='?', default="output.pth.tar")
parser.add_argument("--chunk-size", help="Chunk size for reading large files", type=int, const=1, nargs='?', default=100)
parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+')
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

    if mode == "eval":
        print("Loading training data...")
        data = pd.DataFrame()

        for chunk in pd.read_csv(input_file, header=None, chunksize=chunk_size):
            print("Reading chunk:")
            print(chunk)
            data = data.append(chunk)

        input_dimensionality = len(data.columns)
        print("Input Dimensionality: %d" % (input_dimensionality))

        positive_data = data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1]
        negative_data = data[data[len(data.columns) - 1] == 0].iloc[:,:len(data.columns) - 1]

        training_data            = positive_data.sample(frac=0.25)
        positive_validation_data = positive_data.drop(training_data.index)
        negative_validation_data = negative_data.copy()

        temp_positive = positive_data.copy()
        temp_positive[input_dimensionality] = 1

        temp_negative = negative_data.copy()
        temp_negative[input_dimensionality] = 0

        validation_data_with_labels = pd.concat([temp_positive, temp_negative], ignore_index=True)
        validation_data   = validation_data_with_labels.iloc[:,:len(data.columns) - 1]
        validation_labels = validation_data_with_labels.iloc[:,-1:].values

        print("Positive Data Points: %d" % (len(positive_data)))
        print("Negative Data Points: %d" % (len(negative_data)))
        print("Training Data Points: %d" % (len(training_data)))
        print("Validation Normal Data Points: %d" % (len(positive_validation_data)))
        print("Validation Outlier Data Points: %d" % (len(negative_validation_data)))

        # Convert to tensor
        positive_data   = torch.tensor(data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1].values).float()
        negative_data   = torch.tensor(data[data[len(data.columns) - 1] == 0].iloc[:,:len(data.columns) - 1].values).float()
        training_data   = torch.tensor(training_data.values).float()
        validation_data = torch.tensor(validation_data.values).float()

        print("Validation Data:")
        print(validation_data)

        if cont:
            net.load(model_file)

        print("Training...")
        net.train(training_data, epochs=epochs, lr=lr, batch_size=batch_size, loss=loss)
        net.save(model_file)

        print("Optimal Threshold: %0.4f" % (net.optimal_threshold))

        predictions = net.predict(validation_data)

        tn, fp, fn, tp = confusion_matrix(validation_labels, predictions).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        ts  = tp / (tp + fn + fp)
        pt  = (sqrt(tpr * (-tnr + 1)) + tnr - 1) / (tpr + tnr - 1)
        f1  = tp / (tp + 0.5 * (fp + fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = ((tp * tn) - (fp * fn))  / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        print(tabulate([[tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]], ["TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))
    elif mode == "train":
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

        validation_data   = torch.tensor(data.iloc[:,:len(data.columns) - 1].values).float()
        validation_labels = torch.tensor(data.iloc[:,-1:].values).int().detach().numpy()

        positive_data = torch.tensor(data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1].values).float()
        negative_data = torch.tensor(data[data[len(data.columns) - 1] == 0].iloc[:,:len(data.columns) - 1].values).float()

        print("Number of positive data: %d" % (len(positive_data)))
        print("Number of negative data: %d" % (len(negative_data)))

        print("Loading model...")
        net.load(model_file)

        predictions = net.predict(validation_data)

        positive_predictions = net.predict(positive_data)
        negative_predictions = net.predict(negative_data)

        tn, fp, fn, tp = confusion_matrix(validation_labels, predictions).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        ts  = tp / (tp + fn + fp)
        pt  = (sqrt(tpr * (-tnr + 1)) + tnr - 1) / (tpr + tnr - 1)
        f1  = tp / (tp + 0.5 * (fp + fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = ((tp * tn) - (fp * fn))  / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        print(tabulate([[tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]], ["TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))
