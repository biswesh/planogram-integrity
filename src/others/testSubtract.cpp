#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


using namespace cv;
using namespace std;


void Threshold_Demo(int, void*);
int main(){

	Mat img_1 = imread("/home/ubuntu/instanceMatch/client_data_test/lighting/1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//resize(img_1,img_1,Size(),0.3,0.3,INTER_LINEAR);
    //cout << "channels : " << img_1.channels() << endl;
    Mat img_2 = imread("/home/ubuntu/instanceMatch/client_data_test/lighting/2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    //resize(img_2,img_2,Size(),0.3,0.3,INTER_LINEAR);
    //cout << "channels : " << img_2.channels() << endl;
    //time3-330
    //time4-5
    //time4-430
    //time5-530

    Mat img_3;
    absdiff(img_1,img_2,img_3);

   
    //cout << img_2 << endl;
    //cout << "img_1 : " << (int)img_1.at<uchar>(800,800) << endl;
    //cout << "img_2 : " << (int)img_2.at<uchar>(800,800) << endl;
    //cout << "img_3 : " << (int)img_3.at<uchar>(800,800) << endl;

    //cout << "img_3 : " << img_3[0][0] << endl;
    

    Mat img_4;
    threshold(img_3, img_4, 20, 255 , THRESH_BINARY);
    //adaptiveThreshold(img_3, img_4, 255,ADAPTIVE_THRESH_MEAN_C , THRESH_BINARY, 7, 0);

    
    Mat img_5, img_6, img_7;
    //int a = connectedComponents(img_4,img_5);
    //cout << "connected components : " << (a - 1) << endl;
    connectedComponentsWithStats(img_4,img_5,img_6,img_7,8,CV_32S);

    //cout << "stat rows : " << img_6.rows << "stat.cols : " << img_6.cols << endl;



    namedWindow( "Image Frame", CV_WINDOW_NORMAL);
    imshow("Image Frame", img_4);
    waitKey(0); 
    imwrite("/home/ubuntu/bb_test.jpg",img_4);

}



