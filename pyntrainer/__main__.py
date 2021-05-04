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

from lib.autoencoder import Autoencoder
from lib.cnn_autoencoder import CnnAutoencoder

# Modules
from modules.train_cnn import TrainCnn as ModuleTrainCnn
from modules.eval_cnn import EvalCnn as ModuleEvalCnn
from modules.train_autoencoder import TrainAutoencoder as ModuleTrainAutoencoder
from modules.eval import Eval as ModuleEval
from modules.train_classify import TrainClassify as ModuleTrainClassify

def main():
  parser = argparse.ArgumentParser(description="PynTrainer: Neural network trainer program")

  parser.add_argument("--mode", help="Mode to be used", choices=["train", "eval", "train-cnn", "eval-cnn", "train-classify"], type=str, default="eval")
  parser.add_argument("--input-file", help="Input csv file for training")
  parser.add_argument("--cf-file", help="Classification csv file for training results")
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
  parser.add_argument("--will-reduce", help='Will reduce for Train Classify module', type=bool, default=False)

  args = parser.parse_args()

  input_file    = args.input_file
  cf_file       = args.cf_file
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
  will_reduce   = args.will_reduce

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
      'batch_size':   batch_size,
      'eval_dir':     eval_dir
    }

    module  = ModuleEvalCnn(params=params)

    module.execute()

  elif mode == "train":
    params = {
      'layers':     layers,
      'epochs':     epochs,
      'lr':         lr,
      'batch_size': batch_size,
      'device':     dev,
      'cont':       cont,
      'model_file': model_file,
      'input_file': input_file,
      'chunk_size': chunk_size,
      'cf_file':    cf_file
    }

    module = ModuleTrainAutoencoder(params=params)

    module.execute()

  elif mode == "eval":
    params = {
      'layers':     layers,
      'neg_cont':   neg_cont,
      'input_file': input_file,
      'chunk_size': chunk_size,
      'device':     device,
      'add_syn':    add_syn,
      'epochs':     epochs,
      'lr':         lr,
      'batch_size': batch_size,
      'printout':   printout,
      'eval_cat':   eval_cat
    }

    module = ModuleEval(params=params)

    module.execute()

  elif mode == "train-classify":
    params = {
      'layers':       layers,
      'epochs':       epochs,
      'lr':           lr,
      'batch_size':   batch_size,
      'device':       dev,
      'cont':         cont,
      'model_file':   model_file,
      'input_file':   input_file,
      'chunk_size':   chunk_size,
      'cf_file':      cf_file,
      'cont':         cont,
      'will_reduce':  will_reduce
    }

    module = ModuleTrainClassify(params=params)

    module.execute()
    

if __name__ == '__main__':
  main()
