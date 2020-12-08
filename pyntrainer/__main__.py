import sys
import argparse
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd
from torch import tensor
import torch
from tabulate import tabulate

from lib.autoencoder import Autoencoder
from lib.utils import performance_metrics

# Existing implementations of anomaly detectors
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

parser = argparse.ArgumentParser(description="PynTrainer: Stochastic Autoencoder trainer program")

parser.add_argument("--mode", help="Mode to be used", choices=["train", "eval"], type=str, default="eval")
parser.add_argument("--input-file", help="Input csv file for training")
parser.add_argument("--model-file", help="Output model file", type=str, const=1, nargs='?', default="output.pth.tar")
parser.add_argument("--chunk-size", help="Chunk size for reading large files", type=int, const=1, nargs='?', default=5000)
parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+')
parser.add_argument("--lr", help='Learning rate', type=float, default=0.001)
parser.add_argument("--epochs", help='Number of epochs', type=int, default=100)
parser.add_argument("--batch-size", help='Batch size', type=int, default=50)
parser.add_argument("--cont", help='Continue training from model file', type=bool, default=False)
parser.add_argument("--eval-file", help='File to evaluate. Should have the format x1,x2,x3...y with y=1 if normal and y=0 if anomaly', type=str)
parser.add_argument("--neg-cont", help='Rate of positive contamination', type=float)
parser.add_argument("--add-syn", help='Add synthetic noise', type=bool, default=False)

args = parser.parse_args()

if __name__ == '__main__':
    input_file  = args.input_file
    model_file  = args.model_file
    chunk_size  = args.chunk_size
    layers      = args.layers
    mode        = args.mode
    lr          = args.lr
    epochs      = args.epochs
    batch_size  = args.batch_size
    cont        = args.cont
    eval_file   = args.eval_file
    add_syn     = args.add_syn
    neg_cont    = args.neg_cont

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("CUDA is available...")
    else:
        dev = "cpu"

    device = torch.device(dev)

    if mode == "eval":
        evaluation_results = []

        print("Loading training data...")
        data = pd.DataFrame()

        for i, chunk in enumerate(pd.read_csv(input_file, header=None, chunksize=chunk_size)):
            print("Reading chunk: %d" % (i+1))
            print(chunk)
            data = data.append(chunk)

        input_dimensionality = len(data.columns) - 1
        print("Input Dimensionality: %d" % (input_dimensionality))

        positive_data = data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1]
        negative_data = data[data[len(data.columns) - 1] == -1].iloc[:,:len(data.columns) - 1]

        training_data            = positive_data.sample(frac=0.70)
        positive_validation_data = positive_data.drop(training_data.index)

        if neg_cont and neg_cont > 0:
            print("Negative Contamination: %0.2f" % (neg_cont))
            num_negative = math.floor(neg_cont * (len(negative_data) + len(positive_validation_data)))
            negative_data = data.sample(frac=1, random_state=200)[data[len(data.columns) - 1] == -1].iloc[:num_negative,:len(data.columns) - 1]

        negative_validation_data = negative_data.copy()

        temp_positive = positive_validation_data.copy()
        temp_positive[input_dimensionality] = 1

        temp_negative = negative_data.copy()
        temp_negative[input_dimensionality] = -1

        validation_data_with_labels = pd.concat([temp_positive, temp_negative], ignore_index=True)
        validation_data   = validation_data_with_labels.iloc[:,:len(data.columns) - 1]
        validation_labels = validation_data_with_labels.iloc[:,-1:].values

        # Convert to tensor
        positive_data   = torch.tensor(positive_data.values).float().to(device)
        negative_data   = torch.tensor(negative_data.values).float().to(device)
        training_data   = torch.tensor(training_data.values).float()
        validation_data = torch.tensor(validation_data.values).float()

        print("Validation Data:")
        print(validation_data)

        ## MSE TRAINING ##
        print("Initializing autoencoder...")
        net = Autoencoder(layers=layers, device=device, add_syn=add_syn)
        net.to(device)

        print(net)

        print("Training...")
        net.train(training_data, epochs=epochs, lr=lr, batch_size=batch_size)

        predictions = net.predict(validation_data)

        tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

        r = ["AE-D", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]

        evaluation_results.append(r)

        print(tabulate([r], ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

        # Convert back to CPU before other methods
        validation_data = validation_data.cpu()

        ## ONE CLASS SVM TRAINING ##
        print("Training OneClassSVM...")
        clf = svm.OneClassSVM(gamma="auto")
        clf.fit(validation_data)

        predictions = clf.predict(validation_data)

        tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

        evaluation_results.append(
            ["OC-SVM", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]
        )

        ## ISOLATION FOREST TRAINING ##
        print("Training Isolation Forest...")
        clf = IsolationForest(random_state=0)
        clf.fit(validation_data)

        predictions = clf.predict(validation_data)

        tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

        evaluation_results.append(
            ["ISO-F", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]
        )

        ## LOCAL OUTLIER FACTOR ##
        print("Training Local Outlier Factor...")
        clf = LocalOutlierFactor(novelty=True)
        clf.fit(validation_data)

        predictions = clf.predict(validation_data)

        tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

        evaluation_results.append(
            ["LOC-OF", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]
        )

        ## ROBUST COVARIANCE ##
        print("Training Robust Covariance...")
        clf = EllipticEnvelope()
        clf.fit(validation_data)

        predictions = clf.predict(validation_data)

        tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

        evaluation_results.append(
            ["ROB-COV", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]
        )

        ## EVALUATE RESULTS ##
        print(tabulate(evaluation_results, ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

        ## DATASET METRICS ##
        len_training_data_points = len(training_data)
        len_positive_validations = len(positive_validation_data)
        len_negative_validations = len(negative_validation_data)
        len_validations          = len_positive_validations + len_negative_validations

        metrics_results = [
            ["Training Data Points", len_training_data_points],
            ["# Normal Points", len_positive_validations],
            ["# Anomalies", len_negative_validations],
            ["Contamination Percentage", math.floor((len_negative_validations / len_validations) * 100)]
        ]

        print(tabulate(metrics_results, ["METRIC", "VALUE"], tablefmt="grid"))
    elif mode == "train":
        print("Initializing autoencoder...")
        net = Autoencoder(layers=layers)
        print(net)

        print("Loading training data...")
        data = pd.DataFrame()

        for i, chunk in enumerate(pd.read_csv(input_file, header=None, chunksize=chunk_size)):
            print("Reading chunk: %d" % (i+1))
            print(chunk)
            data = data.append(chunk)

        tensor_data = torch.tensor(data.values).float()

        input_dimensionality = len(data.columns)
        print("Input Dimensionality: %d" % (input_dimensionality))

        if cont:
            net.load(model_file)

        print("Training...")
        net.train(tensor_data, epochs=epochs, lr=lr, batch_size=batch_size)
        net.save(model_file)
