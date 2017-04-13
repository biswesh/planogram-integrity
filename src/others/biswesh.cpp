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

using namespace cv;
using namespace std;

double eucledianNorm(std::vector<float> vec){
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++){
        sum  += pow(vec.at(i),2);

    }
    sum = sqrt(sum);
    return sum;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


int main(){

	//vector<float> som;


	//for(int i = 4; i < 8; i++){
    //    som.push_back(i);
	//}

	//double b = eucledianNorm(som);
	//cout << b << endl;

    Mat planogramImg = imread("../testing/test-data/multiple_swapped/1.jpg",IMREAD_GRAYSCALE);
    resize(planogramImg,planogramImg,Size(),0.5,0.5,INTER_LINEAR);
    Mat planoROI = planogramImg(Rect(0,0, 24, 24));
    cout << planoROI << endl;
    //Mat mDist32 = Mat(planogramImg.rows,planogramImg.cols,CV_32FC1);
    Mat mDist32 = Mat(planoROI.rows,planoROI.cols,CV_32FC1);
    planoROI.convertTo(mDist32,CV_32FC1,1.0/255.5);


    string ty =  type2str( planoROI.type() );
    printf("Matrix: %s %dx%d \n", ty.c_str(), planoROI.cols, planoROI.rows );
  

    
   



	return 0;
}