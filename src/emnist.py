# Produces a trained classifier on the EMNIST handwritten alphabet letters

import argparse
import numpy as np
import pandas as pd 
import tensorflow as tf
from letter_classifier import LetterClassifier
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

def main():
    """
    Takes shell inputs, processes the EMNIST CSV data, and trains
    an image classifier on that data.

    Args:
        None

    Returns:
        None

    """
    parser = argparse.ArgumentParser(description=('Produces a trained '
        'classifier on the EMNIST handwritten alphabet letters'))
    parser.add_argument('train_file', type=argparse.FileType('r'),
        help='File for the EMNIST training dataset')
    parser.add_argument('test_file', type=argparse.FileType('r'),
        help='File for the EMNIST testing dataset')
    parser.add_argument('-a', '--architecture', 
        choices=CLASSIFIER_CONFIGS.keys(), default='cnn_two_layer',
        help=('Classifier architecture to be used, one of {}, default '
            'is cnn_two_layer'.format(list(CLASSIFIER_CONFIGS.keys()))))
    parser.add_argument('-s', '--save_path', default='tmp/model.ckpt',
        help='Path to store TF model checkpoint')

    args = parser.parse_args()
    train_X, train_y = emnist_csv_to_xy(args.train_file)
    test_X, test_y = emnist_csv_to_xy(args.test_file)
    config = CLASSIFIER_CONFIGS[args.architecture]

    with tf.Graph().as_default():
        classifier = LetterClassifier(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
            scope='letter_classifier'))
        with tf.Session() as session:
            session.run(init)
            classifier.train(session, train_X, train_y)
            save_path = saver.save(session, args.save_path)
            
            test_pred_classes = classifier.eval(session, test_X)
            test_actual_classes = np.argmax(test_y, axis=1)
            num_same = np.sum(test_actual_classes == test_pred_classes)
            print('Test Accuracy: {}'.format(num_same / test_y.shape[0]))

if __name__ == '__main__':
    main()