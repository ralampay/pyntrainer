# PynTrainer

Autoencoder training program

## Installing Dependencies (via `pip`)

```
pip install -r requirements.txt
```

## Installing Dependencies (via `conda`)

1. Rename `environment.yml.dist` to `environment.yml`

2. Install the packages by running `conda env update --prefix [environment_location] --file environment.yml  --prune`

## Installing the Package

This will also install the cli utility `pyntrainer-cli`

```
pip install .
```

## Uninstalling

```
pip uninstall pyntrainer
```

## Sample Usage for CNN-Autoencoder Training

Trains a CNN based autoencoder with `BCELoss` function and auto-thresholding. Outputs a model (pth) file.

```
python -m pyntrainer --mode train-cnn --input-dir [input directory of images] --layers [array of numbers representing channel maps i.e. 3 16 8] --epochs 100 --batch-size 1 --img-width 50 --img-height 50 --model-file [file.pth]
```

## Sample Usage for CNN-Autoencoder Training and Evaluation

Trains a CNN based autoencoder with `BCELoss` function and auto-thresholding. Basic command would look like the following:

```
pyntrainer-cli --mode eval-cnn --input-dir [directory_of_positive_images] --eval-dir [directory_of_images_for_evaluation] --layers 3 16 8 --epochs 100 --batch-size 5 --img-width 100 --img-height 100
```

Important parameters/flags:

* `--input-dir`: The directory containing positive images for training
* `--eval-dir`: The directory containing images for evaluation. The program will say if the image is considered an anomaly or not.
* `--layers`: an array of integers representing the channel mappings of CNN. In the example above, we choose 3 because we consider RGB based images.
* `--img-width`: The width of the image to be resized to.
* `--img-height`: The height of the image to be resized to.
