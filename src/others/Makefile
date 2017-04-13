CPP = g++

# OpenCV 3.1.0
CPPFLAGS = -L/home/ubuntu/opencv_installed/opencv-3.1.0/lib/ \
	   -I/home/ubuntu/opencv_installed/opencv-3.1.0/include/

# Opencv 2.4.8
#CPPFLAGS = -L/home/krystof/libs/opencv-2.4.8/release/installed/libs \
#	   -I/home/krystof/libs/opencv-2.4.8/release/installed/include

all: opencv

biswesh: ${CPPFILE}
	$(CPP) $(CPPFLAGS) $^ -o $@ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_xfeatures2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_features2d


#biswesh_1: ${CPPFILE}
#        $(CPP) $(CPPFLAGS) $^ -o $@ -lopencv_core -lopencv_highgui -lopencv_imgproc

