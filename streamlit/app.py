import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import tensor
sys.path.append(os.path.join(os.path.dirname(__file__), '../pyntrainer'))

from lib.autoencoder import Autoencoder
from lib.utils import performance_metrics
from lib.evals import train_and_evaluate_classifier

is_processing = False

def file_selector(folder_path='./data', label='Select File'):
  filenames = os.listdir(folder_path)
  selected_filename = st.selectbox(label, filenames)

  return os.path.join(folder_path, selected_filename)

st.title('Autoencoder Reconstruction Visualizer')
st.markdown("## Anomaly Detector")

input_file  = file_selector(label='Select Input File')

col1, col2 = st.beta_columns(2)

input_lr    = col1.number_input("Learning Rate", 0.00001, 0.9999, 0.001, format="%.3f", step=0.001)
epochs      = col1.number_input("Epochs", 1, 1000, 100, format="%d", step=1)
batch_size  = col2.number_input("Batch Size", 1, 1000, 50, format="%d", step=1)
chunk_size  = col2.number_input("Chunk Size", 1, 10000, 5000, format="%d", step=10)
layers      = list(map(int, st.text_input("Layers", "18,10").split(",")))

if torch.cuda.is_available():
  dev = "cuda:0"
  print("CUDA is available...")
else:
  dev = "cpu"

device = torch.device(dev)

if st.button('Start Process'):
  is_processing = True

  st.write("Loading data from %s..." % (input_file))

  data = pd.DataFrame()

  for i, chunk in enumerate(pd.read_csv(input_file, header=None, chunksize=chunk_size)):
    data = data.append(chunk)

  input_dimensionality = len(data.columns) - 1
  st.write("Input Dimensionality: %d" % (input_dimensionality))

  st.write("Initializing Autoencoder...")

  tensor_data = torch.tensor(data.values).float()

  net = Autoencoder(layers=layers)
  net.to(device)

  positive_data = data[data[len(data.columns) - 1] == 1].iloc[:,:len(data.columns) - 1]
  negative_data = data[data[len(data.columns) - 1] == -1].iloc[:,:len(data.columns) - 1]

  training_data            = positive_data.sample(frac=0.70)
  positive_validation_data = positive_data.drop(training_data.index)

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

  net.fit(training_data, epochs=epochs, lr=input_lr, batch_size=batch_size)

  # Graph errors
  error_data = pd.DataFrame(net.errs, columns=['errors'])
  st.markdown("## Error Graph")
  st.line_chart(error_data)

  # Reconstruct
  st.write("Reconstructing....")
  x_hat = net.forward(validation_data)

  st.write("Predicting....")
  predictions = net.predict(validation_data)

  # Display reconstructions
  datapoints = []
  for i in range(len(predictions)):
    datapoints.append(i)
    temp_data = np.hstack((
                  np.array(x_hat[i].data).reshape(len(x_hat[i]), 1),
                  np.array(validation_data[i].data).reshape(len(validation_data[i]), 1)
                ))

    temp  = pd.DataFrame(temp_data, columns=['Reconstructed', 'Actual'])
    classification = "Normal" if predictions[i] == 1 else "Anomaly" 

    st.markdown(" Data Point %d" % (i+1))

    if validation_labels[i] == predictions[i]:
      st.success("Prection: %s Actual: %s" % ("Normal" if validation_labels[i] == 1 else "Anoamly", "Normal" if predictions[i] == 1 else "Anomaly"))
    else:
      st.error("Prection: %s Actual: %s" % ("Normal" if validation_labels[i] == 1 else "Anoamly", "Normal" if predictions[i] == 1 else "Anomaly"))
    
    st.line_chart(temp)
