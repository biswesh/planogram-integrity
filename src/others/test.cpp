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
using namespace std;

int main(){


	Mat img_1 = imread("/home/ubuntu/imgMatchData/test_1.jpg",IMREAD_GRAYSCALE);
    Mat img_2= imread("/home/ubuntu/imgMatchData/test_2.jpg",IMREAD_GRAYSCALE);

    Rect rect1;
    rect1.x = 100;
    rect1.y = 100;
    rect1.width = 100;
    rect1.height = 100;

    Mat img_3 = img_1;

    cout << "img_3 : "<< img_1.isContinuous() << endl;

    //cout << "img3 rows : " << img_3.rows << "img3 cols : " << img_3.cols << endl;
    img_3 = img_1(rect1);

    cout << "img_3 : "<< img_3.isContinuous() << endl;

    for(int i = 0; i < img_3.rows; i++){
        for(int j = 0; j < img_3.cols; j++){
             img_3.at<double>(i,j) = 122;
        }

    }

    
    namedWindow( "image", WINDOW_NORMAL);  
    imshow("image",img_1);
    waitKey(0);



  


    //cout << "img1 rows : " << img_1.rows << "img1 cols : " << img_1.cols << endl;
    //cout << "img3 rows : " << img_3.rows << "img3 cols : " << img_3.cols << endl;

    //Mat a = [1 2 3;4 5 6;7 8 9];
    //Mat b = a;






    //Ptr<MSER> ms = MSER::create(21, (int)(0.00002*img_1.cols*img_1.rows), (int)(0.05*img_1.cols*img_1.rows), 1, 0.7);
    //vector<vector<Point> > regions;
    //vector<cv::Rect> mser_bbox;
    //ms->detectRegions(img_1, regions, mser_bbox);

    //for (int i = 0; i < regions.size(); i++){
    //    cout << "x : " << regions[i][4].x <<" y : " << regions[i][4].y << endl; 
    //}

    //cout << "vector size : " << regions.size() << endl;
    //cout << "vector element size : " << regions[2].size() << endl;


    
    //for (int i = 0; i < regions.size(); i++)
    //{
    //    rectangle(img_1, mser_bbox[i], CV_RGB(0, 255, 0));  
    //}
    
    //imshow("mser", img_1);
    //waitKey(0);

    return 0;
}