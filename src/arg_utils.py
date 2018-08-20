# Functions for adding argparse arguments to different scripts
# (Several arguments are shared among the scripts.)

import argparse
from hyperparams import CLASSIFIER_CONFIGS, ADVERSARY_CONFIGS

def add_emnist_datasets(parser):
    """
    Adds the required parser arguments representing the training and testing
    EMNIST CSV datasets. Most likely, these filenames should end in 
    emnist-letters-train-classifier.csv and emnist-letters-test.csv, 
    respectively. (The training file should result from splitting the combined
    EMNIST dataset from running split_data.py.) Again, both CSV files should
    have 785 columns.

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('train_file', type=argparse.FileType('r'),
        help='File for the EMNIST training dataset')
    parser.add_argument('test_file', type=argparse.FileType('r'),
        help='File for the EMNIST testing dataset')

def add_classifier_architecture_args(parser):
    """
    Adds the optional flag for choosing the classifier architecture for 
    categorizing handwritten alphabet letters. Check CLASSIFIER_CONFIGS in
    hyperparams.py to see the available options.

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('-c', '--classifier_architecture', 
        choices=CLASSIFIER_CONFIGS.keys(), default='cnn_two_layer',
        help=('Classifier architecture to be used, one of {}, default '
            'is cnn_two_layer'.format(list(CLASSIFIER_CONFIGS.keys()))))
    
def add_save_path_args(parser):
    """
    Adds the optional flag for changing the path in which the classifier
    saves its Tensorflow model parameters. 

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('-s', '--save_path', default='tmp/model.ckpt',
        help='Path to store TF model checkpoint')

def add_image_args(parser):
    """
    Adds the arguments for handling image input: the required filename of the 
    letter image (should contain a white letter on a black background) and 
    the optional flag to display the original and tainted grayscale images.

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('image_filename', 
        help=('Filename of the (color) input image; the letter should be '
            'white on black background'))
    parser.add_argument('-d', '--display_image', action='store_true',
        help='Displays the grayscale scaled images (both original and tainted')

def add_adversary_args(parser):
    """
    Adds optional flags for changing the adversary for the classifier: the 
    architecture of the adversary (check ADVERSARY_CONFIGS in hyperparams.py
    for the options), the classifier model saved as Tensorflow checkpoints, 
    and the (lowercase) target letter that the adversary will trick the 
    classifier into seeing.

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('-a', '--adversary_architecture', 
        choices=ADVERSARY_CONFIGS.keys(), default='gradient_descent',
        help=('Adversary architecture to be used, one of {}, default '
            'is gradient_descent'.format(list(ADVERSARY_CONFIGS.keys()))))
    parser.add_argument('-p', '--checkpoint', default='tmp/model.ckpt',
        help=('File with classifier model checkpoint if the model has '
            'already been trained'))
    parser.add_argument('-t', '--target_letter', default='m',
        help='Lowercase letter that serves as target class')

def add_output_args(parser):
    """
    Adds the argument for choosing the directory in which the output tainted 
    image will be saved to.

    Args:
        parser: an argparse.ArgumentParser object

    Returns:
        None

    """
    parser.add_argument('-o', '--output_directory', default='bin',
        help='Directory to place tainted image in')

def make_train_classifier_parser():
    """
    Constructs the shell argument parser (takes in inputs) to train the letter 
    classifier and save the model.

    Args:
        None

    Returns:
        parser: an argparse.ArgumentParser object to parser command line args

    """
    parser = argparse.ArgumentParser(description=('Produces a trained '
        'classifier on the EMNIST handwritten alphabet letters'))
    add_emnist_datasets(parser)
    add_classifier_architecture_args(parser)
    add_save_path_args(parser)
    return parser

def make_image_adversary_parser():
    """
    Constructs the shell argument parser (takes in inputs) to create an 
    adversarial example that fools a given letter classifier.

    Args:
        None

    Returns:
        parser: an argparse.ArgumentParser object to parser command line args

    """
    parser = argparse.ArgumentParser(description=('Takes an image and a '
        'classifier model and outputs an adversarial example'))
    add_image_args(parser)
    add_classifier_architecture_args(parser)
    add_adversary_args(parser)
    add_output_args(parser)
    return parser
