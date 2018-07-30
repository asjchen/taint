# Produces a trained classifier on the EMNIST handwritten alphabet letters

import argparse
import numpy as np
import pandas as pd 


def emnist_csv_to_xy(file_stream, img_height=28, img_width=28):
    raw_data = pd.read_csv(file_stream, header=None)
    X_flat = raw_data.values[:, 1:]
    assert img_height * img_width == X_flat.shape[1]
    X = X_flat.reshape(X_flat.shape[0], img_height, img_width)
    y = raw_data.values[:, 0]
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

if __name__ == '__main__':
    main()