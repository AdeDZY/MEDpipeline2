#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import argparse


def load_test_data(feat_file_path, feat_dim):
    """
    Load all test data
    :param feat_file_path: the file that contains all features.
    Each line represents a video. Line starts with video_name, than a '\t', than the feature vector
    :return: X, the test feature vectors. shape=(n_sample, n_feat)
    """
    test_list = open("/home/ubuntu/hw1/list/test.video")
    videos = {}
    for line in test_list:
        video = line.strip()
        videos[video] = 0
    test_list.close()

    X = []
    for line in open(feat_file_path):
        line = line.strip()
        video, feats = line.split('\t')
        if video not in videos:
            continue
        if feats == '-1':
            X.append([0 for i in range(feat_dim)])
            continue

        x = [float(t) for t in feats.split(';')]
        X.append(x)

    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="path of the trained svm file")
    parser.add_argument("feat_file", help="feature file")
    parser.add_argument("feat_dim", type=int, help="dimension of features")
    parser.add_argument("output_file", help="path to save the prediction score")
    args = parser.parse_args()

    # load model
    clf = cPickle.load(open(args.model_file, 'rb'))

    # load data
    X = load_test_data(args.feat_file, args.feat_dim)

    # predict with the log probability
    print ">> Predicting..."
    T = clf.decision_function(X)

    # write results
    outfile = open(args.output_file, 'w')
    for score in T:
        outfile.write(str(score) + '\n')
    outfile.close()
    print ">> Prediction scores written to {0}!".format(args.output_file)

# Apply the SVM model to the testing videos; Output the score for each video
if __name__ == '__main__':
    main()

