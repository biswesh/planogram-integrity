#include <iostream>

#include "opencv2/imgproc.hpp"
//#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/types.hpp"
//#include <opencv2/imgcodecs.hpp>
//#include "opencv2/ml.hpp"
//#include "opencv2/opencv/types.hpp"
//#include "types.hpp"

using namespace cv;
using namespace std;

int main(){

    Mat planogramImage, planogramImageColor; 
    Mat queryImage;
    planogramImageColor = imread("/home/ubuntu/instanceMatch/data/real_1.jpg");
    resize(planogramImageColor,planogramImageColor,Size(),0.3,0.3,INTER_LINEAR);
    cvtColor(planogramImageColor, planogramImage, cv::COLOR_BGR2GRAY);
    

    if(!planogramImage.data){
    	cout << "No image data. check whether the path is correct." << endl;
    	return -1;
    }

    cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    
    cout << "rows : " << planogramImage.rows << " cols : " << planogramImage.cols << std::endl;
    vector<KeyPoint> planogramKeypoint;
    //vector<cv::Point2f> planogramKeypoint; 
    Mat planogramDescriptor;

    //detector->detectAndCompute(planogramImage,Mat(),planogramKeypoint, planogramDescriptor); /*for other detectros and descriptors*/
    detector->detect(planogramImage, planogramKeypoint);
    detector->compute(planogramImage, planogramKeypoint, planogramDescriptor);
    cout << "number of keypoints : " << planogramKeypoint.size() << endl;
    //cout << "descriptor rows : " << planogramDescriptor.rows << " descriptors cols : " << planogramDescriptor.cols << endl;
    vector<cv::Point2f> planogramKeypointXY;
    cv::KeyPoint::convert(planogramKeypoint,planogramKeypointXY);
    //cout << "point xy size : " << planogramKeypointXY.size() << endl;

    int clusterCount = 10;
    Mat labels,centers;
    kmeans(planogramKeypointXY, clusterCount, labels, 
    	TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 3, KMEANS_PP_CENTERS, centers);

    //cout << "labels rows : " << labels.rows << " labels cols : " << labels.cols << endl;
    //cout << centers << endl;

    //circle(planogramImageColor, planogramKeypointXY[0], 100, Scalar(0, 0, 255),-1, 8);


    for(int i = 0; i < planogramKeypoint.size(); i++){

    	circle(planogramImageColor, planogramKeypointXY[i], 5, Scalar((labels.at<int>(i,0)) * (255/clusterCount), 
    		(labels.at<int>(i,0)) * (255/clusterCount),0),-1, 8);
            
    }
    

    for(int i = 0; i < centers.rows; i++){

    	circle(planogramImageColor, Point(centers.at<float>(i,0),centers.at<float>(i,1)), 8, Scalar(0,0,255),-1, 8);
            
    }
    namedWindow("clustered image",CV_WINDOW_NORMAL);    
    imshow("clustered image", planogramImageColor);
    waitKey( 0 );



    
    


	return 0;
}