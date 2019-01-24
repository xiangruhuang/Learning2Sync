# Learning Transformation Synchronization
![alt tag](https://www.cs.utexas.edu/~xrhuang/Learning2Sync/RecurrentNetwork1.png)
Python implementation of our approach on Learning Transformation Synchronization.

## Data Acquisition:
The datasets we used are "Redwood" and "Scannet". 
However, for space and license issues, this repository do not contain any data.
Please contact the owners of the datasets to obtain the data.

## Overview:
The whole pipeline of our approach contains four steps, which are

> I. Data Preprocessing.

> II. Relative Pose Estimation.

> III. Training Classifier.

> IV. Optimizing Parameters.

Notice that step I and II can be skipped if you would like to run other relative pose estimation code and fed the result as the input of step III.
Step III can be skipped if you would like to use classifiers pretrained on Redwood or Scannet. 
Step IV is the core step of our approach which optimizes parameters of both the weighting module and synchronization module.

## Step I. Data Preprocessing:
Since each dataset has different data format, we require the users to write code and preprocess each dataset
into a certain format.

Here we demonstrate the data format after our data processing procedure. 

An example file hierarchy we use to store processed depth images of each dataset looks like
  > processed_dataset/scannet/scene0000_01/0.mat

which corresponds to the first scan for scene id `scene0000_01` of `scannet` dataset.

Each .mat file contains three attributes [`vertex`, `validIdx_rowmajor`, `pose`]. 

  `vertex`: np.ndarray of shape (3, n), where n is the number of valid points in the depth image.

  `validIdx_rowmajor`: np.ndarray of shape (1, n).

  `pose`: np.ndarray of shape (4, 4) representing the ground truth absolute pose of this scan.

## Step II. Relative Pose Estimation:
1. The source code of relative pose estimation is stored in folder "src/relative_pose_estimation/"
A parallelization is recommended in order to get pairwise estimated relative pose.
We use Fast Global Registration and Super4PCS to
    obtain pairwise relative pose estimation.

2. By default, results will be stored in "relative_pose", e.g. 
  > "relative_pose/scannet/scene0000_01/0_2_fgr.mat"
  
corresponds to relative pose estimation from the first scan to the third scan for scene `scene0000_01` 
for the algorithm `Fast Global Registration`.

3. In the mat file, attribute `Tij` corresponds to a np.ndarray of shape (4, 4) representing the relative pose estimated.

## Step III. Training Classifiers:
1. We first generate images for each estimated pairwise relative pose, the code is stored in `src/generate_images`
2. We then train our classifiers using the generated images, the code is stored in `src/training_classifiers`
3. The results will be stored in `classification/`.

## Step IV. Optimizing Parameters:
1. Given a trained classifier, we collect features for each estimated pairwise relative pose, 
  please refer to `src/training_classifier/main.py`.
2. To further optimize parameters of the weighting module, please refer to `src/differentiable_sync/train.py`.
  (Use `python train.py -h` to see the instructions of how to run this code).

