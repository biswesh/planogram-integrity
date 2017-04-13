#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/ml.hpp"
#include <cmath>
#include "opencv2/core/types.hpp"
#include "opencv2/objdetect.hpp"
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <fstream>

using namespace cv;
using namespace std;

int main(){

	VideoCapture capture("/home/ubuntu/instanceMatch/iBot/testing/video_files/2017-03-30 17_24_02.219841.h264");
	capture.set(CV_CAP_PROP_FOURCC, CV_FOURCC('H', '2', '6', '4'));
	



	int counter = 0;
    Mat frame;
    while(capture.isOpened()){
    	capture >> frame;
        if(frame.empty()){
            break;
        }
        counter++;
    } 

    double num_frames = capture.get(CAP_PROP_FRAME_COUNT);
    cout << "frames : " << num_frames << endl;

    cout << "the number of frames : " << counter << endl;




	return 0;
}