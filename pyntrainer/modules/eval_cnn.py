import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.cnn_autoencoder import CnnAutoencoder
from lib.utils import load_images_from_dir
from lib.utils import cv2_to_tensor

class EvalCnn:
  def __init__(self, params=None):
    # Parameters for CNN
    self.scale        = params.get('scale')
    self.channel_maps = params.get('channel_maps')
    self.padding      = params.get('padding')
    self.kernel_size  = params.get('kernel_size')
    self.num_channels = params.get('num_channels')
    self.img_width    = params.get('img_width')
    self.img_height   = params.get('img_height')
    self.device       = params.get('device')

    # Configuration for training data
    self.input_dir  = params.get('input_dir')

    # Flow configurations
    self.cont       = params.get('cont')
    self.model_file = params.get('model_file')

    # Parameters for training
    self.epochs     = params.get('epochs')
    self.lr         = params.get('lr')
    self.batch_size = params.get('batch_size')

  def execute(self):
    print("Initializing CNN autoencoder...")

    net = CnnAutoencoder(
            scale=self.scale,
            channel_maps=self.channel_maps,
            padding=self.padding,
            kernel_size=self.kernel_size,
            num_channels=self.num_channels,
            img_width=self.img_width,
            img_height=self.img_height,
            device=self.device
          )

    if self.cont:
      print("Loading model_file {}...".format(model_file))
      net.load(model_file)

    print("Loading images...")

    tensor_data = cv2_to_tensor(
                    load_images_from_dir(
                      self.input_dir, 
                      self.img_width, 
                      self.img_height
                    )
                  )

    print("Training...")
    
    net.fit(tensor_data, epochs=self.epochs)

    print("Loading images for evaluation from {}...".format(self.eval_dir))

    eval_data = cv2_to_tensor(
                  load_images_from_dir(
                    self.eval_dir, 
                    self.img_width, 
                    self.img_height
                  )
                )

    print("Predicting...")

    y = net.predict(eval_data)

    for i, filename in enumerate(os.listdir(self.eval_dir)):
      f = os.path.join(self.eval_dir, filename)

      print("File: {} Prediction: {}".format(f, "NORMAL" if y[i] == 1 else "ANOMALY"))
