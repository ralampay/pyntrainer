import sys
import argparse
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from torch import tensor
import torch
from tabulate import tabulate

from lib.autoencoder import Autoencoder
from lib.cnn_autoencoder import CnnAutoencoder
from lib.utils import performance_metrics
from lib.utils import load_images_from_dir
from lib.utils import cv2_to_tensor
from lib.evals import train_and_evaluate_classifier

# Existing implementations of anomaly detectors
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

import csv

# Modules
from modules.train_cnn import TrainCnn as ModuleTrainCnn

# https://github.com/yzhao062/pyod
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.vae import VAE
from pyod.models.mcd import MCD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.lscp import LSCP
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.sos import SOS

def main():
  parser = argparse.ArgumentParser(description="PynTrainer: Autoencoder trainer program")

  parser.add_argument("--mode", help="Mode to be used", choices=["train", "eval", "train-cnn", "eval-cnn"], type=str, default="eval")
  parser.add_argument("--input-file", help="Input csv file for training")
  parser.add_argument("--input-dir", help='Input directory for images for CNN', type=str)
  parser.add_argument("--eval-dir", help='Evaluation directory for images for CNN', type=str)
  parser.add_argument("--model-file", help="Output model file (i.e. pth file)", type=str, const=1, nargs='?', default="output.pth.tar")
  parser.add_argument("--chunk-size", help="Chunk size for reading large files", type=int, const=1, nargs='?', default=5000)
  parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+')
  parser.add_argument("--lr", help='Learning rate', type=float, default=0.001)
  parser.add_argument("--epochs", help='Number of epochs', type=int, default=100)
  parser.add_argument("--batch-size", help='Batch size', type=int, default=50)
  parser.add_argument("--cont", help='Continue training from model file', type=bool, default=False)
  parser.add_argument("--neg-cont", help='Rate of positive contamination', type=float)
  parser.add_argument("--add-syn", help='Add synthetic noise', type=bool, default=True)
  parser.add_argument("--printout", help='File for results of eval (csv)', type=str)
  parser.add_argument("--eval-cat", help='Category of algos for evaluation', choices=["linear", "prob", "nn", "ensemble", "proximity", "none"], type=str, default="none")
  parser.add_argument("--padding", help='Padding for CNN', type=int, default=1)
  parser.add_argument("--kernel-size", help='Kernel size for CNN', type=int, default=3)
  parser.add_argument("--num-channels", help='Num channels for CNN', type=int, default=3)
  parser.add_argument("--img-width", help='Image width for CNN', type=int, default=100)
  parser.add_argument("--img-height", help='Image height for CNN', type=int, default=100)
  parser.add_argument("--scale", help='Scale for CNN', type=int, default=2)

  args = parser.parse_args()

  input_file    = args.input_file
  input_dir     = args.input_dir
  eval_dir      = args.eval_dir
  model_file    = args.model_file
  chunk_size    = args.chunk_size
  layers        = args.layers
  mode          = args.mode
  lr            = args.lr
  epochs        = args.epochs
  batch_size    = args.batch_size
  cont          = args.cont
  add_syn       = args.add_syn
  neg_cont      = args.neg_cont
  printout      = args.printout
  eval_cat      = args.eval_cat
  padding       = args.padding
  kernel_size   = args.kernel_size
  num_channels  = args.num_channels
  img_width     = args.img_width
  img_height    = args.img_height
  scale         = args.scale

  if torch.cuda.is_available():
    dev = "cuda:0"
    print("CUDA is available...")
  else:
    dev = "cpu"

  device = torch.device(dev)

  if mode == "train-cnn":
    params = {
      'scale':        scale,
      'channel_maps': layers,
      'padding':      padding,
      'kernel_size':  kernel_size,
      'num_channels': num_channels,
      'img_width':    img_width,
      'img_height':   img_height,
      'device':       dev,
      'input_dir':    input_dir,
      'cont':         cont,
      'model_file':   model_file,
      'epochs':       epochs,
      'lr':           lr,
      'batch_size':   batch_size
    }

    module  = ModuleTrainCnn(params=params)

    module.execute()

  elif mode == 'eval-cnn':
    print("Initializing CNN autoencoder...")
    net = CnnAutoencoder(scale=scale, channel_maps=layers, padding=padding, kernel_size=kernel_size, num_channels=num_channels, img_width=img_width, img_height=img_height, device=dev)
    net.to(device)
    print(net)

    if cont:
      print("Loading model_file %s..." % (model_file))
      net.load(model_file)

    print("Loading images from %s..." % (input_dir))
    tensor_data = cv2_to_tensor(load_images_from_dir(input_dir, img_width, img_height))

    print("Training...")
    net.fit(tensor_data, epochs=epochs, lr=lr, batch_size=batch_size)

    print("Loading images for evaluation from %s..." % (eval_dir))
    eval_data = cv2_to_tensor(load_images_from_dir(eval_dir, img_width, img_height))

    print("Predicting...")
    y = net.predict(eval_data)

    for i, filename in enumerate(os.listdir(eval_dir)):
      f = os.path.join(eval_dir, filename)
      print("File: %s Prediction: %s" % (f, "NORMAL" if y[i] == 1 else "ANOMALY"))

  elif mode == "train":
    print("Initializing autoencoder...")
    net = Autoencoder(layers=layers)
    net.to(device)
    print(net)

    print("Loading training data...")
    data = pd.DataFrame()

    for i, chunk in enumerate(pd.read_csv(input_file, header=None, chunksize=chunk_size)):
      print("Reading chunk: %d" % (i+1))
      print(chunk)
      data = data.append(chunk)

    tensor_data = torch.tensor(data.values).float()

    input_dimensionality = len(data.columns) - 1
    print("Input Dimensionality: %d" % (input_dimensionality))

    if cont:
      print("Loading model_file %s..." % (model_file))
      net.load(model_file)

    print("Training...")
    net.fit(tensor_data, epochs=epochs, lr=lr, batch_size=batch_size)

    print("Saving to %s..." % (model_file))
    net.save(model_file)

  elif mode == "eval":
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
      print("Negative Contamination: %0.4f" % (neg_cont))
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

    ## AE-D TRAINING ##
    print("Initializing autoencoder...")
    net = Autoencoder(layers=layers, device=device, add_syn=add_syn)
    net.to(device)

    print(net)

    print("Training Stochastic Autoencoder...")
    net.fit(training_data, epochs=epochs, lr=lr, batch_size=batch_size)

    predictions = net.predict(validation_data)

    tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

    r = ["AE-D", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]

    evaluation_results.append(r)

    print("AE-D Results:")
    print(tabulate([r], ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

    # Convert back to CPU before other methods
    validation_data = validation_data.cpu()

    # Train only linear classifiers
    if eval_cat == "linear":
      print("Initiating training for linear detectors...")

      ## MCD ##
      print("Training MCD...")
      result = train_and_evaluate_classifier("MCD", MCD(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## ROBUST COVARIANCE ##
      print("Training Robust Covariance...")
      result = train_and_evaluate_classifier("ROB-COV", EllipticEnvelope(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## ONE CLASS SVM TRAINING ##
      print("Training OneClassSVM...")
      result = train_and_evaluate_classifier("OC-SVM", svm.OneClassSVM(gamma="auto"), validation_data, validation_labels)
      evaluation_results.append(result)

    elif eval_cat == "prob":
      ## ABOD ##
      print("Training ABOD...")
      result = train_and_evaluate_classifier("ABOD", ABOD(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## SOS ##
      print("Training SOS...")
      result = train_and_evaluate_classifier("SOS", SOS(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## COPOD ##
      print("Training COPOD...")
      result = train_and_evaluate_classifier("COPOD", COPOD(), validation_data, validation_labels)
      evaluation_results.append(result)

    elif eval_cat == "ensemble":
      ## ISOLATION FOREST TRAINING ##
      print("Training Isolation Forest...")
      result = train_and_evaluate_classifier("ISO-F", IsolationForest(random_state=0), validation_data, validation_labels)
      evaluation_results.append(result)

      ## LODA ##
      print("Training LODA...")
      result = train_and_evaluate_classifier("LODA", LODA(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## LSCP ##
      print("Training LSCP...")
      result = train_and_evaluate_classifier("LSCP", LSCP([LOF(), LOF()]), validation_data, validation_labels)
      evaluation_results.append(result)
    
    elif eval_cat == "proximity":
      ## LOCAL OUTLIER FACTOR ##
      print("Training Local Outlier Factor...")
      result = train_and_evaluate_classifier("LOC-OF", LocalOutlierFactor(novelty=True), validation_data, validation_labels)
      evaluation_results.append(result)

      ## CBLOF ##
      print("Training CBLOF...")
      result = train_and_evaluate_classifier("CBLOF", CBLOF(), validation_data, validation_labels)
      evaluation_results.append(result)

      ## HBOS ##
      print("Training HBOS...")
      result = train_and_evaluate_classifier("HBOS", HBOS(), validation_data, validation_labels)
      evaluation_results.append(result)

    elif eval_cat == "nn":
      ## VAE ##
      print("Training VAE...")
      result = train_and_evaluate_classifier("VAE", VAE(encoder_neurons=layers, decoder_neurons=layers.reverse()), validation_data, validation_labels)
      evaluation_results.append(result)

      ## SO_GAAL ##
      print("Training SO_GAAL...")
      result = train_and_evaluate_classifier("SO_GAAL", SO_GAAL(lr_d=lr, stop_epochs=epochs), validation_data, validation_labels)
      evaluation_results.append(result)

      ## MO_GAAL ##
      print("Training MO_GAAL...")
      result = train_and_evaluate_classifier("MO_GAAL", MO_GAAL(lr_d=lr, stop_epochs=epochs), validation_data, validation_labels)
      evaluation_results.append(result)

    ## EVALUATE RESULTS ##
    if eval_cat != "none":
      print("Aggregated Results:")
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

    ## EVALUATE RESULTS ##
    print(tabulate(metrics_results, ["Metric", "Value"], tablefmt="grid"))

    if printout:
      print("Saving results to %s" % (printout))
      df = pd.DataFrame(evaluation_results)
      df.to_csv(printout, header=None, index=False)

if __name__ == '__main__':
  main()
