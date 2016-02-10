#!/usr/bin/env python
import numpy as np
import os
import cv2
from os import listdir
from os.path import isfile, join

def main():

    keyframe_dir = "~/hw2/keyframes/"
    frames = [f for f in listdir(keyframe_dir) if isfile(join(keyframe_dir, f))]
    video2des = {}
    for frame_path in frames:
        frame = cv2.imread(join(keyframe_dir, frame_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        vid = frame_path.split('_')[0]
        video2des[vid] = video2des.get(vid, []).append(des)

    # write each video's sift descriptors into a file
    out_dir = "~/hw2/sift_features/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for vid in video2des:
        out_file = open(join(out_dir, vid + ".sift.csv"), 'w')
        for des in video2des[vid]:
            for d in des:
                outline = ';'.join([str(v) for v in d])
                out_file.write(outline)
                out_file.write('\n')
        out_file.close()

    print ">> SIFT features written to {0}!".format(out_dir)

