#!/usr/bin/env python
__author__ = 'zhuyund'

import numpy
import os
import cPickle
import sys
import argparse
from os import listdir
from os.path import isfile, join

def avg_feats(feat_file_path):
    """
    Generate the average of MFCC features
    :param mfcc_file_path
    :return: vec
    """
    vec = []
    n = 0
    for line in open(feat_file_path):
        vals = line.split(';')
        for i, v in enumerate(vals):
            v = float(v)
            if len(vec) <= i:
                vec.append(v)
            else:
                vec[i] += v
        n += 1

    for i in range(len(vec)):
        vec[i] /= n

    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir", help="the dir of video features")
    parser.add_argument("output_file_path", help="the output file")
    args = parser.parse_args()

    # output file
    output_file = open(args.output_file_path, 'w')

    # get all feature files
    feat_files = [f for f in listdir(args.feat_csv_dir) if isfile(join(args.feat_dir, f))]

    # process each video
    for f in open(feat_files):
        video_name = f.split('.')[0]
        feat_file_path = join(args.file_dir, f)
        vec = avg_feats(feat_file_path)
        output_str = ';'.join([str(t) for t in vec])
        output_file.write(video_name + '\t')
        output_file.write(output_str + '\n')

    print "avg features generated successfully! Written into {0}!".format(args.output_file_path)


if __name__ == '__main__':
    main()

