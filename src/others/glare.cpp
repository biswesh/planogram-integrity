#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <fstream>
 
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

struct ORBDetector
{
    Ptr<Feature2D> orb;
    ORBDetector()
    {
        orb = ORB::create();
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        orb->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

void cropImage(Mat &im, Mat &h)
{
    Size sz = im.size();
    vector<Point2f> srcPoints;
    srcPoints.push_back(Point2f(0,0));
    srcPoints.push_back(Point2f(sz.width,0));
    srcPoints.push_back(Point2f(sz.width,sz.height));
    srcPoints.push_back(Point2f(0,sz.height));
    
    vector<Point2f> dstPoints(4);
    perspectiveTransform(srcPoints, dstPoints, h);
    
    Rect r = boundingRect(dstPoints);
    r.x = max(r.x, 0);
    r.y = max(r.y, 0);
    
    if (r.x + r.width > sz.width)
    {
        r.width = sz.width - r.x;
    }
    
    if (r.y + r.height > sz.height)
    {
        r.height = sz.height - r.y;
    }

    
    im = im(r);

    
}


void registerImages(Mat &img1, Mat &img2, Mat &img1Reg, Mat &h)
{
    Mat img1Gray;
    cvtColor(img1, img1Gray, CV_BGR2GRAY);
    
    Mat img2Gray;
    cvtColor(img2, img2Gray, CV_BGR2GRAY);
    
    Size sz = img1Gray.size();
    
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;
    
    Mat descriptors1, descriptors2;
    
    ORBDetector orb;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    orb(img1Gray, Mat(), keypoints1, descriptors1);
    orb(img2Gray, Mat(), keypoints2, descriptors2);

    cout << "k1 : " << keypoints1.size() << "k2 : " << keypoints2.size() << endl;
    
    matcher->match(descriptors1, descriptors2, matches, Mat());
    
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;
    
    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }

    cout << "good_matches : " << good_matches.size() << endl;
    
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for(size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    
    h = findHomography( obj, scene, RANSAC );

    cout << "obj : " << obj.size() << "scene : " << scene.size() << endl;

    warpPerspective(img1, img1Reg, h, img1.size());

    namedWindow("image warped",CV_WINDOW_NORMAL);
    imshow("image warped",img1Reg);
    waitKey(0);

    imwrite("warped.jpg",img1Reg);



}


int main( int argc, char** argv)
{   
    //Read input images
    string filename1 = argv[1];
    string filename2 = argv[2];
    
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);

    //cout << img1.rows << " " << img1.cols << endl;
    //cout << img2.rows << " " << img2.cols << endl;

    namedWindow("image object",CV_WINDOW_NORMAL);
    imshow("image object",img1);
    waitKey(0);

    namedWindow("image object",CV_WINDOW_NORMAL);
    imshow("image object",img2);
    waitKey(0);
    
    Mat img3, h;
    registerImages(img1, img2, img3, h);
    
    /*Mat img2Gray;
    cvtColor(img2, img2Gray, CV_BGR2GRAY);
    
    
    Mat mask2;
    threshold(img2Gray,mask2, 254, 255, THRESH_BINARY);
    
    int morph_size = 15;
    Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2 * morph_size + 1, 2 * morph_size + 1));
    
    dilate(mask2,mask2,element, Point(-1,-1), 15);
    
    vector<Vec4i> hierarchy;
    vector< vector<Point> > contours0;
    
    findContours( mask2, contours0, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<Point> contour;
    
    double maxArea = 0;
    int maxContourIndex = 0;

    for( size_t k = 0; k < contours0.size(); k++ )
    {
        double area = cv::contourArea(contours0[k]);
        if ( area > maxArea)
        {
            maxArea = area;
            maxContourIndex = k;
        }
    }
    
    
    contour = contours0[maxContourIndex];
    Mat mask = Mat::zeros(mask2.rows, mask2.cols, CV_8UC3);

    int idx = 0;
    Scalar color( 255, 255, 255 );
    drawContours( mask, contours0, maxContourIndex, color, CV_FILLED, 8, hierarchy );
    
    Rect r = boundingRect(contour);
    Point center = (r.tl() + r.br()) / 2;

    Mat output;
    seamlessClone(img1,img2, mask, center, output, NORMAL_CLONE);
    
    cropImage(output,h);
    Mat img1Cropped = img1.clone();
    cropImage(img1Cropped,h);
    
    
    //imwrite(filename2 + ".cropped.jpg", img1Cropped);
    imwrite(filename1 + ".fixed.jpg", output);*/
    
    
    
    return 0;
}
