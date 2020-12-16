import numpy as np
from tabulate import tabulate
from utils import performance_metrics

PYOD_ALGOS = [
  "VAE",
  "MCD",
  "SO_GAAL",
  "MO_GAAL",
  "LSCP",
  "LODA",
  "CBLOF",
  "LOCI",
  "SOS"
]

def train_and_evaluate_classifier(name, clf, validation_data, validation_labels):
  clf.fit(validation_data)

  predictions = np.array(clf.predict(validation_data))

  if name in PYOD_ALGOS:
    predictions = np.where(predictions==1, -1, predictions)
    predictions = np.where(predictions==0, 1, predictions)

  tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc = performance_metrics(validation_labels, predictions)

  result = [name, tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc]

  print("%s results:" % (name))
  print(tabulate([result], ["ALGO", "TP", "TN", "FP", "FN", "TPR", "TNR", "PPV", "NPV", "TS", "PT", "ACC", "F1", "MCC"], tablefmt="grid"))

  return result
