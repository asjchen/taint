# Produces a trained classifier on the EMNIST handwritten alphabet letters

import argparse
import numpy as np
import pandas as pd 
import tensorflow as tf
from letter_classifier import LetterClassifier

def emnist_csv_to_xy(file_stream, img_height=28, img_width=28, num_classes=26):
    raw_data = pd.read_csv(file_stream, header=None)
    X_flat = raw_data.values[:, 1:] / 256
    assert img_height * img_width == X_flat.shape[1]
    X = X_flat.reshape(X_flat.shape[0], img_height, img_width) 
    y_classes = raw_data.values[:, 0] - 1
    y = np.zeros((y_classes.shape[0], num_classes))
    y[np.arange(y_classes.shape[0]), y_classes] = 1
    return X, y

def main():
    parser = argparse.ArgumentParser(description=('Produces a trained '
        'classifier on the EMNIST handwritten alphabet letters'))
    parser.add_argument('train_file', type=argparse.FileType('r'),
        help='File for the EMNIST training dataset')
    parser.add_argument('test_file', type=argparse.FileType('r'),
        help='File for the EMNIST testing dataset')
    args = parser.parse_args()
    train_X, train_y = emnist_csv_to_xy(args.train_file)
    test_X, test_y = emnist_csv_to_xy(args.test_file)
    config = { 
        'img_height': 28, 
        'img_width': 28,
        'num_classes': 26,
        'epochs': 20,
        'learning_rate': 0.001,
        'output_activation': 'relu',
        'batch_size': 2000,
        'log_per': 10000
    }

    with tf.Graph().as_default():
        classifier = LetterClassifier(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            classifier.train(session, train_X, train_y)

if __name__ == '__main__':
    main()