import sys
import os
import pandas as pd
import math
import csv
from tabulate import tabulate
from torch import tensor
import torch
from sklearn.manifold import TSNE

# Local libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.autoencoder import Autoencoder
from lib.utils import performance_metrics
from lib.evals import train_and_evaluate_classifier

import plotly.express as px

class TrainClassify:
  def __init__(self, params=None):
    # Parameters for Autoencoder
    self.layers = params.get('layers')
    self.device = params.get('device')

    # Parameters for training
    self.input_file   = params.get('input_file')
    self.lr           = params.get('lr')
    self.chunk_size   = params.get('chunk_size')
    self.add_syn      = params.get('add_syn')
    self.epochs       = params.get('epochs')
    self.batch_size   = params.get('batch_size')
    self.neg_cont     = params.get('neg_cont')
    self.cf_file      = params.get('cf_file')
    self.cont         = params.get('cont')
    self.model_file   = params.get('model_file')
    self.will_reduce  = params.get('will_reduce')

  def execute(self):
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

    # Continue training
    if self.cont and self.model_file:
      net.load(self.model_file)

    print("Training Stochastic Autoencoder...")
    net.fit(training_data, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size)

    predictions = net.predict(validation_data)

    tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

    r = ["AE-D", tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]

    print("AE-D Results:")
    print(tabulate([r], ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

    # Convert back to CPU before other methods
    validation_data = validation_data.cpu()
    print(validation_data)

    # Create prediction data
    #prediction_set = pd.concat([pd.DataFrame(validation_data.numpy()), pd.DataFrame(predictions)], axis=1)
    #print(prediction_set)

    # Turn into 3 dimensions
    if self.will_reduce:
      print("Reducing to 3 dimensions...")
      self.reduced_data = TSNE(n_components=3).fit_transform(validation_data.numpy())
      self.reduced_data_ground_truth = pd.concat([pd.DataFrame(self.reduced_data), pd.DataFrame(validation_labels)], axis=1)
      self.reduced_data_predictions  = pd.concat([pd.DataFrame(self.reduced_data), pd.DataFrame(predictions)], axis=1)

      self.reduced_data_ground_truth.columns = ['x1', 'x2', 'x3', 'y']
      self.reduced_data_predictions.columns  = ['x1', 'x2', 'x3', 'y']
