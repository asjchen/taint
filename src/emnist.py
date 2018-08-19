# Produces a trained classifier on the EMNIST handwritten alphabet letters

import numpy as np
import pandas as pd 
from hyperparams import CLASSIFIER_CONFIGS

def emnist_csv_to_xy(file_stream, img_height=28, img_width=28, num_classes=26):
    """
    Converts a file object to the EMNIST CSV file to NumPy inputs and labels

    Args:
        file_stream: a file object to the EMNIST CSV file. The data should 
        not have headers and should most likely originate from the Kaggle 
        EMNIST page: https://www.kaggle.com/crawford/emnist/

        img_height: the images' height, which should be the same for all of 
        the images. In EMNIST, this value is 28.

        img_width: the images' width, which should be the same for all of 
        the images. In EMNIST, this value is 28.

        num_classes: the number of possible categories for the images. By
        default, this number is 26 (the number of letters in the alphabet).

    Returns:
        A tuple (X, y), where X is a NumPy array of dimensions [None, 
        img_height, img_width] and y is a NumPy array of dimensions 
        [None, num_classes] (with each row a one-hot vector)

    """
    raw_data = pd.read_csv(file_stream, header=None)
    X_flat = raw_data.values[:, 1:] / 255
    assert img_height * img_width == X_flat.shape[1]
    X = X_flat.reshape(X_flat.shape[0], img_height, img_width) 
    y_classes = raw_data.values[:, 0] - 1
    y = np.zeros((y_classes.shape[0], num_classes))
    y[np.arange(y_classes.shape[0]), y_classes] = 1
    return X, y
