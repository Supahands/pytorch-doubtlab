# PyTorch-Doubtlab

<!-- Add buttons here -->

The [**Doubtlab library**](https://github.com/koaning/doubtlab) modified to work with PyTorch models

# Dataset Structure

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

Run the Torch_Image_Classification_Training script, following the dataset structure above. Modify the **number of classes** before loading the model.