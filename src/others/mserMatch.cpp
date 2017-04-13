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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::ml;

int main(){

	Mat img_1 = imread("/home/ubuntu/imgMatchData/real_1.jpg",IMREAD_GRAYSCALE);
    Mat img_2= imread("/home/ubuntu/imgMatchData/real_2.jpg",IMREAD_GRAYSCALE);

    Ptr<MSER> ms = MSER::create(21, (int)(0.00002*img_1.cols*img_1.rows), (int)(0.05*img_1.cols*img_1.rows), 1, 0.7);
    vector<vector<Point> > regions;
    vector<cv::Rect> mser_bbox;
    ms->detectRegions(img_1, regions, mser_bbox);


	return 0;
}