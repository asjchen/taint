# Splits the training data: one half will be for the letter classifier itself,
# and the other half will be be for the GAN

import arg_utils
import numpy as np
import pandas as pd

def produce_new_filenames(old_filename, suffices=['classifier', 'gan']):
    root = old_filename
    if root.endswith('.csv'):
        root = root[: -4]
    return ['{}-{}.csv'.format(root, suffix) for suffix in suffices]

def split_data(old_filename):
    with open(old_filename, 'r') as f:
        combined_pd_data = pd.read_csv(f, header=None)
    combined_data = combined_pd_data.values
    shuffled_combined_data = combined_data[:, :]
    np.random.shuffle(shuffled_combined_data)
    boundary = shuffled_combined_data.shape[0] // 2
    first_half = shuffled_combined_data[: boundary, :]
    second_half = shuffled_combined_data[boundary: , :]
    return [first_half, second_half]

def write_data(np_data, filename):
    pd_data = pd.DataFrame(data=np_data)
    with open(filename, 'w') as f:
        pd_data.to_csv(f, index=False, header=False)

def main():
    parser = arg_utils.make_split_data_parser()
    args = parser.parse_args()
    new_filenames = produce_new_filenames(args.combined_data_filename)
    data_halves = split_data(args.combined_data_filename)
    for i in range(2):
        write_data(data_halves[i], new_filenames[i])

if __name__ == '__main__':
    main()
