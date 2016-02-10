#!/usr/bin/env python

import numpy
import os
from sklearn.cluster import KMeans
import cPickle
import argparse


def load_sift(sift_file):
    """
    load sampled SIFT features into a matrix X
    :param mfcc_csv_file: path to the mfcc csv file
    :return: X. shape=(n_samples, n_features)
    """
    X = []
    i = 0
    for line in open(sift_file):
        line = line.strip()
        i += 1
        x = [float(val) for val in line.split(';') if val]
        X.append(x)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sift_file", help="path to the sift file")
    parser.add_argument("cluster_num", type=int, help="number of cluster")
    parser.add_argument("output_file", help="path to save the k-means model")
    args = parser.parse_args()

    # load SIFT features
    X = load_sift(args.sift_file)

    # perform kmeans
    print ">> training K-means on {0} samples...".format(len(X))
    km = KMeans(args.cluster_num)
    km.fit(X)

    # save model
    outfile = open(args.output_file, 'wb')
    cPickle.dump(km, outfile)
    print "K-means model saved at {0}!".format(args.output_file)


# Performs K-means clustering and save the model to a local file
if __name__ == '__main__':
    main()
