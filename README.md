# Pose Estimation Using Tactile Sensing
Inferring the orientation of objects and the sensors using tactile sensors and deep learning.

This was my project for my Summer Research Internship, undertaken at the Bristol Robotics Laboratory (BRL) as part of the faculty of Engineering summer research internship scheme.

## Project Overview
The aim of the project was to collect data from optical tactile sensors (TacTips, developed in BRL) to use for training a CNN to estimate the pose of the object and of the sensors themselves.

![Data Collection](https://github.com/cwhite098/TacTip_Orientation/blob/main/data/cube/Thu_Jun_30_10-43-53_2022/external.png)

The above image shows the data collection apparatus that was used.

## Poses
The pose information was obtained by using Aruco markers and pose estimation. The code for this can be found in the markers folder. This includes code for generating and detecting the markers.

## Deep Learning
Deep learning has been shown to interpret the images obtained from the TacTips well. The code pertatining to this can be found in the neural_net folder. This includes code for training a network, for optimising hyperparameters and producing plots.

## CAD
As part of the project I designed some modifications to the TacTips as well as some objects for data collection. These CAD files can be found in the CAD folder.
