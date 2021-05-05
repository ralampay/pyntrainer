import sys
import os
import pandas as pd
import math
import csv
from tabulate import tabulate
from torch import tensor
import torch

# Local libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.autoencoder import Autoencoder
from lib.utils import cv2_to_tensor
from lib.utils import performance_metrics
from lib.evals import train_and_evaluate_classifier

# Existing implementations of anomaly detectors
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

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

class Eval:
  def __init__(self, params=None):
    # Parameters for Autoencoder
    self.layers = params.get('layers')
    self.device = params.get('device')

    # Parameters for training
    self.input_file = params.get('input_file')
    self.lr         = params.get('lr')
    self.chunk_size = params.get('chunk_size')
    self.add_syn    = params.get('add_syn')
    self.epochs     = params.get('epochs')
    self.batch_size = params.get('batch_size')
    self.neg_cont   = params.get('neg_cont')
    self.eval_cat   = params.get('eval_cat')

    # Parameters for output
    self.printout = params.get('output')

  def execute(self):
    evaluation_results = []

    print("Loading training data...")
    data = pd.DataFrame()

    for i, chunk in enumerate(pd.read_csv(self.input_file, header=None, chunksize=self.chunk_size)):
      print("Reading chunk: %d" % (i+1))
      #print(chunk)
      data = data.append(chunk)

    input_dimensionality = len(data.columns) - 1
    print("Input Dimensionality: %d" % (input_dimensionality))

    positive_data = data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1]
    negative_data = data[data[len(data.columns) - 1] == -1].iloc[:,:len(data.columns) - 1]

    training_data            = positive_data.sample(frac=0.70)
    positive_validation_data = positive_data.drop(training_data.index)

    if self.neg_cont and self.neg_cont > 0:
      print("Negative Contamination: %0.4f" % (self.neg_cont))
      num_negative = math.floor(self.neg_cont * (len(negative_data) + len(positive_validation_data)))
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
    positive_data   = torch.tensor(positive_data.values).float().to(self.device)
    negative_data   = torch.tensor(negative_data.values).float().to(self.device)
    training_data   = torch.tensor(training_data.values).float()
    validation_data = torch.tensor(validation_data.values).float()

    print("Validation Data:")
    print(validation_data)

    ## AE-D TRAINING ##
    print("Initializing autoencoder...")
    net = Autoencoder(layers=self.layers, device=self.device, add_syn=self.add_syn)
    net.to(self.device)

    print(net)

    print("Training Stochastic Autoencoder...")
    net.fit(training_data, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)

    predictions = net.predict(validation_data)

    tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

    r = ["AE-D", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]

    evaluation_results.append(r)

    print("AE-D Results:")
    print(tabulate([r], ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

    # Convert back to CPU before other methods
    validation_data = validation_data.cpu()

    # Train only linear classifiers
    if self.eval_cat == "linear":
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

    elif self.eval_cat == "prob":
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

    elif self.eval_cat == "ensemble":
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
    
    elif self.eval_cat == "proximity":
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

    elif self.eval_cat == "nn":
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
    if self.eval_cat != "none":
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

    if self.printout:
      print("Saving results to %s" % (self.printout))
      df = pd.DataFrame(evaluation_results)
      df.to_csv(self.printout, header=None, index=False)
