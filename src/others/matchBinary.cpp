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

	Mat img_object = imread("/home/ubuntu/instanceMatch/data/test_9.jpg", IMREAD_GRAYSCALE);
    //resize(img_object,img_object,Size(),0.1,0.1,INTER_LINEAR);
    Mat img_scene = imread("/home/ubuntu/instanceMatch/data/test_10.jpg", IMREAD_GRAYSCALE);
    //resize(img_scene,img_scene,Size(),0.4,0.4,INTER_LINEAR);
    
    if(!img_object.data || !img_scene.data){
    	cout << "There's some problem in reading images" << endl;
    	return -1;
    }

    Mat descriptors_scene, descriptors_object;
    std::vector<KeyPoint> keypoints_scene, keypoints_object;

    cv::Ptr<cv::Feature2D> kaze = cv::KAZE::create(); 

    //Ptr<BRISK> detector = BRISK::create();

    //cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    //cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    //Ptr<xfeatures2d::LATCH> latch = xfeatures2d::LATCH::create(32,false,3);



    kaze->detectAndCompute(img_scene,Mat(),keypoints_scene, descriptors_scene);
    kaze->detectAndCompute(img_object,Mat(),keypoints_object, descriptors_object);
    
    cout << "biswesh : " << keypoints_object.size() << endl;

    //detector->detect(img_scene, keypoints_scene);
    //latch->compute(img_scene, keypoints_scene, descriptors_scene);
    //detector->detect(img_object, keypoints_object);
    //latch->compute(img_object, keypoints_object, descriptors_object);


    cout << descriptors_object.rows << " "  << descriptors_object.cols << endl;
    cout << descriptors_scene.rows << " " << descriptors_scene.cols << endl;

    //cout << "xy : " << descriptors_object[0][0] << endl;


    std::vector<DMatch> good_matches;
    std::vector<std::vector<cv::DMatch> > matches;
    //std::vector<DMatch> matches;

    
    cv::BFMatcher matcher(NORM_L2, false);
    matcher.knnMatch(descriptors_object, descriptors_scene, matches,2);

    //matcher.match(descriptors_object, descriptors_scene, matches);


    //for (int i = 0; i < matches.size(); ++i){
        //cout << "1 : " << matches[i].queryIdx << endl;
    //    cout << "2 : " << matches[i].distance << endl;
    //}



    /*double max_dist=0;
    double min_dist = 1000;

    for (int i =0; i < matches.size();i++){
    	double dist = matches[i][0].distance;
    	if(max_dist<dist) max_dist = dist;
    	if(min_dist>dist) min_dist = dist;
    }

    min_dist = min_dist + (max_dist- min_dist)* 0.3;*/


    for (int i = 0; i < matches.size(); ++i){
    	const float ratio = 0.8;
    	if (matches[i][0].distance < ratio * matches[i][1].distance){
    		good_matches.push_back(matches[i][0]);
        } 
    }

    cout << good_matches.size() << endl;

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;


    for(size_t i = 0; i < good_matches.size(); i++){
    	obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }


    std::vector<Point2f> proj;
    Mat H = findHomography(obj, scene, RANSAC);
 
    perspectiveTransform(obj,proj,H);



    std::vector<DMatch> very_good_matches;
    double tolerance = 8;

    //cout << proj.size() << endl;
    for(size_t i = 0; i < proj.size(); i++){
        double sumX = pow((scene[i].x - proj[i].x),2);
        double sumY = pow((scene[i].y - proj[i].y),2);
        double distObjProj = sumX + sumY;
        if(distObjProj <= pow(tolerance,2)){
            very_good_matches.push_back(good_matches[i]);
        }

    } 

    cout << "The number of very good inliers :" << very_good_matches.size() << endl;


    Mat img_matches;


    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
    	very_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
    	DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 

    namedWindow( "image Match", WINDOW_NORMAL);  
    imshow("image Match",img_matches);
    waitKey(0);
    //imwrite("/home/ubuntu/matching_13.jpg",img_matches);
    




	return 0;
}