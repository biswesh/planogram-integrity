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

bool descriptorDistance(std::vector<float> first,std::vector<float> second){

    vector<double> diffVec;
    double diffThreshold = 0.005;
    
    float elementDiff;
    double diffSquare, elementSum, diff, finalDiff;


    for(int i = 0; i < first.size(); i++){
        elementDiff = first.at(i) - second.at(i);
        diffSquare = pow(elementDiff,2);
        elementSum = first.at(i) + second.at(i);
        diff = diffSquare/elementDiff;
        diff = 2 * diff;
        finalDiff = diff / first.size();
        diffVec.push_back(finalDiff);

    }

    for(int i = 0; i < diffVec.size(); i++){
        if (diffVec.at(i) > diffThreshold){
             //cout << diffVec.at(i) << endl;
        }
       
        if (diffVec.at(i) < diffThreshold)
            return true;
    }
    return false;
}

int main(){

    //collecting image from memory

    Mat planogramSrc = imread("../testing/test-data/removed_swapped/1.jpg",IMREAD_GRAYSCALE);
    Mat querySrc = imread("../testing/test-data/removed/2.jpg", IMREAD_GRAYSCALE);

    //resizing image to half the size

    resize(planogramSrc,planogramSrc,Size(),0.5,0.5,INTER_LINEAR);
    resize(querySrc,querySrc,Size(),0.5,0.5,INTER_LINEAR);

    //rescaling the data

    Mat planogramImg = Mat(planogramSrc.rows,planogramSrc.cols,CV_8UC1);
    Mat queryImg = Mat(querySrc.rows,querySrc.cols,CV_8UC1);

    planogramSrc.convertTo(planogramImg,CV_8UC1,1.0/255.5);
    querySrc.convertTo(queryImg,CV_8UC1,1.0/255.5);

    std::vector<Point2f> nonmatchesHOG;
  


    HOGDescriptor hog(
        Size(24,24), //winSize
        Size(16,16), //blocksize
        Size(8,8), //blockStride,
        Size(8,8), //cellSize,
        9, //nbins,
        1, //derivAper,
        -1, //winSigma,
        HOGDescriptor::L2Hys, //histogramNormType,
        0.2, //L2HysThresh,
        false,//gammal correction,
        HOGDescriptor::DEFAULT_NLEVELS, //nlevels
        false //signed gradient
        );

    int xslide = 3;
    int yslide = 3;
    int windowSize = 3;
    double diffThreshold = 0.015;
    bool found;

    for(int i = ((8 * yslide)); i < ((planogramImg.rows - (8 * yslide))); i = (i + 8)){
        for(int j = ((8 * xslide)); j < ((planogramImg.cols - (8 * xslide))); j = (j + 8)){

            found = false;

            vector<float> planoHist;
            Mat planoROI = planogramImg(Rect(j,i, 24, 24));
            hog.compute(planoROI,planoHist,Size(), Size());

            

            vector<float> queryHist;
            Mat queryROI = queryImg(Rect(j,i, 24, 24));
            hog.compute(queryROI,queryHist,Size(), Size());

   


            if(eucledianNorm(planoHist) < 1e-8 && eucledianNorm(queryHist) < 1e-8){
                found = true;
                continue;
            }

            double a = compareHist(planoHist,queryHist,4);
            a = a/planoHist.size();
            //cout << a << endl;
            if (a < diffThreshold){
                found = true;
                continue;
            }

        
            if (found == false)
                nonmatchesHOG.push_back(Point2f(j+12,i+12));    
            
        }
    }

    int myradius=5;
    //cout << nonmatchesHOG.size() << endl;
    for (int i=0;i < nonmatchesHOG.size();i++)
        circle(planogramSrc,cvPoint(nonmatchesHOG[i].x,nonmatchesHOG[i].y),myradius,CV_RGB(100,0,0),-1,8,0);

    cout << nonmatchesHOG.size() << endl;

    //cout << "biswesh : " << endl;
    namedWindow("image", CV_WINDOW_NORMAL);// Create a window for display.
    imshow("image",planogramSrc);
    while (char(waitKey(1))!= 'q'){}


        //            if (i == 24 && j == 24 && k ==0 && l == 0){
      
    //for (std::vector<float>::const_iterator i = queryHist.begin(); i != queryHist.end(); ++i)
     //   std::cout << *i << ' ';
    //} 


   

  

    return 0;
}

