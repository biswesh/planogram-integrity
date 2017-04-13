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

   Mat plano = imread("../testing/test_cases/multiple_swapped/1.jpg");
   Mat query = imread("../testing/test_cases/multiple_swapped/2.jpg");

   Mat src_f;
   Mat dest;
   int kernel_size = 31;
   double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
   cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);	
   plano.convertTo(src_f,CV_32F);
   cv::filter2D(src_f, dest, CV_32F, kernel);
   Mat viz;
   dest.convertTo(viz,CV_8U,1.0/255.0);
   namedWindow("image",CV_WINDOW_NORMAL);
   imshow("image",viz);
   waitKey(0);







	return 0;
}


