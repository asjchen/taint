# Produces a trained classifier on the EMNIST handwritten alphabet letters

import arg_utils
import numpy as np
import pandas as pd 
import tensorflow as tf
from letter_classifier import LetterClassifier
from hyperparams import CLASSIFIER_CONFIGS
from emnist_utils import emnist_csv_to_xy

def main():
    """
    Takes shell inputs, processes the EMNIST CSV data, and trains an image 
    classifier on that data. Prints training progress, as well as the test
    accuracy.

    Args:
        None

    Returns:
        None

    """
    parser = arg_utils.make_train_classifier_parser()
    args = parser.parse_args()
    train_X, train_y = emnist_csv_to_xy(args.train_file)
    test_X, test_y = emnist_csv_to_xy(args.test_file)
    config = CLASSIFIER_CONFIGS[args.classifier_architecture]

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
