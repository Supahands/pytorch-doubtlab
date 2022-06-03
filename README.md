# PyTorch-Doubtlab

<!-- Add buttons here -->

The [**Doubtlab library**](https://github.com/koaning/doubtlab) modified to work with PyTorch models

At the moment, the four reasons below are available:

- **ProbaReason**: assign doubt when a models' confidence-values are low for any label
- **WrongPredictionReason**: assign doubt when a model cannot predict the listed label
- **ShortConfidenceReason**: assign doubt when the correct label gains too little confidence
- **LongConfidenceReason**: assign doubt when a wrong label gains too much confidence

# Dataset Structure

The scripts expect the data to be arranged in the following structure:

    Dataset
    ├ train
    | ├ class 1
    | | ├ image_1.png
    | | └ image_2.png
    | ├ class 2
    | └ class 3
    ├ valid
    | ├ class 1
    | ├ class 2
    | └ class 3
    └ test
    | ├ class 1
    | ├ class 2
    | └ class 3

# Usage

To generate potentially mislabeled images, you will need to:

1) Train a torchvision model on your data
2) Use your model with doubtlab to identify mislabels

**Training**: Run the Torch_Image_Classification_Training script, following the dataset structure above. Modify the **number of classes** before loading the model. 

**Doubtlab**: Follow the example doubtlab usage in Torch_Doubtlab_Example