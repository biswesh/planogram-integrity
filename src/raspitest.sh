#!/bin/bash
#This shell file tests the compilation of RaspiCam API

g++ -L/usr/local/lib -I/usr/local/include raspicam_test.cpp -o raspi_test -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_xfeatures2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_features2d -std=c++11 -lraspicam -lraspicam_cv

