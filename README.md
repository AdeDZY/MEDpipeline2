# MEDpipeline2

## Intro
All scripts are in ./scripts. To check the usage of each script, please use `./scripts/{script_name} -h`

##Raw Feature Extraction
Run `./run.feature.sh`. This script will automatically do the following:

1. Select the first 30 seconds of the videos, downsize them to 160×120 pixels, and export 15 frames per second into ./keyframes/. This is for SIFT feature extraction.
2. Select the first 30 seconds of the videos, downsize them to 224×224 pixels, and export 15 frames per second into ./keyframes2/. This is for CNN feature extraction.
3. Call ./scripts/extractSift to extract SIFT features of each keyframe. Results will be written into ./sift_features/. Each file represnets a keyframe.
4. Call ./scripts/create_cnn.py to extract cnn features for each fream. Results will be in ./frame_cnn_fc7_feats/. Then concatenate cnn features from the same video into one file. Result will be in ./cnn_fc7_features/. Each file represents a video, and each line in the file represents a keyframe.

## SIFT BOW features
1.Type in command line: `./scripts/sample_sift.py` to sample SIFT features for each video to train BOW model. 

2.Train kmeans model for SIFT with 
```
./scripts/train_kmeans.py sift_features/ 500 sift.kmeans.500.model 
```

3.generate BOW representation for each keyframe with 
```
./scripts/create_kmeans.py sift_features/ sift.kmeans.500.model 500
```
Features will be in ./siftbow_features/. 1 file per keyframe.

4.Using the command below to averge keyframe feature to get video representations.
```
./scripts/pooling.py siftbow_features/ list/all.video siftbow_features/all_avg.vectors
```
You can also use max pooling by `-m max`. 


## Pooling CNN features

Pooling CNN features by
```
./scripts/pooling.py cnn_fc7_features/ list/all.video cnn_fc7_features/all_avg.vectors
```
You can also use max pooling by add `-m max`.

## Step 2. Train and Test SVM

```
./run.test.sh
```

Test results (AP) will be written in the log file. You can change the svm kernel and parameters in the script.
