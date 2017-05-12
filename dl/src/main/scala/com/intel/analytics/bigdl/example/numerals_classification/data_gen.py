import os
import sklearn.datasets
import sklearn.linear_model
import struct
import argparse


def write_float_array(array, f):
    for num in array:
        n = num.item()
        n = struct.pack('>f', n)
        f.write(n)

def create_data_file(filename, sample_size):
    with open(filename, 'wb') as f:
        count = struct.pack('>i', sample_size)
        f.write(count)
        featureSize = 4
        featureSize = struct.pack('>i', featureSize)
        f.write(featureSize)
        for i in range(sample_size):
            write_float_array(X[i], f)
        f.close()

def create_label_file(filename, sample_size):
    with open(filename, 'wb') as f:
        count = struct.pack('>i', sample_size)
        f.write(count)
        write_float_array(y, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='create vector data')
    parser.add_argument(
        'dir_name', help='directory to store the generated data')
    args = parser.parse_args()
    dirname = args.dir_name

    X, y = sklearn.datasets.make_classification(
        n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
        n_clusters_per_class=2, hypercube=False, random_state=0
    )

    # Split into train and test
    X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

    # Make target directory
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Create train data
    train_filename = os.path.join(dirname, 'train.bin')
    train_label_filename = os.path.join(dirname, 'train_label.bin')
    create_data_file(train_filename, 7500)
    create_label_file(train_label_filename, 7500)

    # Create test data
    test_filename = os.path.join(dirname, 'test.bin')
    test_label_filename = os.path.join(dirname, 'test_label.bin')
    create_data_file(test_filename, 2500)
    create_label_file(test_label_filename, 2500)