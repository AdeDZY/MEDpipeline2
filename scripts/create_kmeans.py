#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import cPickle
from sklearn.cluster import KMeans
import argparse
import sys, os
from os import listdir
from os.path import isfile, join


def transform_feats(km, cluster_num, feats):
    """
    transform a video's features into one bag-of-word vector
    :param km: kmeans model
    :param cluster_num: int. the number of clusters
    :param feats: features for the video. shape=(n_samples, n_features)
    :return: a feature vector for this video. shape=(1, cluster_num)
    """
    labels = km.predict(feats)
    v = [0 for i in range(cluster_num)]
    for label in labels:
        v[label] += 1
    return v


def load_feats(feat_csv_file):
    """
    load sampled features into a matrix X
    :param feat_csv_file: path to the mfcc csv file
    :return: X. shape=(n_samples, n_features)
    """
    X = []
    i = 0
    for line in open(feat_csv_file):
        i += 1
        if i % 10 != 0:
            continue
        x = [float(val) for val in line.split(';') if val]
        X.append(x)
    return X


# Generate k-means features for videos; each video is represented by a single vector
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_csv_dir", help="dir of all feature csv files")
    parser.add_argument("kmeans_model", help="path to the kmeans model")
    parser.add_argument("cluster_num", type=int, help="number of cluster")
    parser.add_argument("output_file_path", type=str)
    parser.add_argument("--list_file", "-l", default="~/hw2/list/all.video")
    args = parser.parse_args()

    # open output file
    output_file = open(args.output_file_path, 'w')

    # load the kmeans model
    km = cPickle.load(open(args.kmeans_model, "rb"))

    # get all feature file names
    feat_files = [f for f in listdir(args.feat_csv_dir) if isfile(join(args.feat_csv_dir, f))]
    vid2filepath = {}
    for f in feat_files:
        video_name = f.split('_')[0]
        vid2filepath[video_name] = join(args.feat_csv_dir, f)

    # process each video
    n = 0
    for video_name in open(args.list_file):

        video_name = video_name.strip()

        if video_name not in vid2filepath:
            print "{}'s features not exist! Write vector as -1".format(video_name)
            output_file.write(video_name + "\t-1\n")
            n += 1
            continue

        feats = load_feats(vid2filepath[video_name])

        # transform
        v = transform_feats(km, args.cluster_num, feats)

        # write new feature
        output_str = ';'.join([str(t) for t in v])
        output_file.write(video_name + '\t')
        output_file.write(output_str + '\n')

        # output process
        n += 1
        if n % 50 == 0:
            print "{0} videos processed.".format(n)

    print "K-means features generated successfully! Featues are written into {0}!".format(args.output_file_path)


if __name__ == '__main__':
    main()


