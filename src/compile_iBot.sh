#!/bin/bash
#This code compiles the system in case the hyperparameters or functions are edited or changed

g++ \
-L/home/ubuntu/opencv_installed/opencv-3.1.0/lib/  \
-I/home/ubuntu/opencv_installed/opencv-3.1.0/include/  \
video_demo.cpp  \
-o iBot_demo  \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_xfeatures2d  \
-lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_videoio -lopencv_imgcodecs \
-lopencv_flann -lopencv_features2d \
-I../vlfeat \
-L../vlfeat/bin/glnxa64/ \
-lvl -std=c++11
