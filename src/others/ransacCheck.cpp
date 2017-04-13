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
using namespace cv::xfeatures2d;
using namespace cv::ml;

extern "C" {
    #include <vl/covdet.h>
    #include <vl/hog.h>
    #include <vl/sift.h>
    #include <vl/mathop.h>
}

struct Tuple {
        Mat outDescriptor;
        VlCovDet *covdet;
    };

float ransacCompare(cv:: Mat, cv:: Mat);
static void flip_descriptor (float * , float const *);
struct Tuple detectorDescriptorSift(cv:: Mat);
double pointDistance(Point2f, Point2f);

int main(){

	int patchResize = 3;


	Mat ransacPlano = imread("../testing/video-test/1.jpg",IMREAD_GRAYSCALE);
	//blur(ransacPlano,ransacPlano,Size(3,3),Point(-1,-1),BORDER_DEFAULT); 
	//resize(ransacPlano,ransacPlano,Size(),patchResize,patchResize,INTER_LINEAR);

    cout << ransacPlano.rows << " " << ransacPlano.cols << endl;

	//cout << ransacPlano.rows << " " << ransacPlano.cols << endl;
	Mat ransacQuery = imread("../testing/video-test/2.jpg",IMREAD_GRAYSCALE);
	//blur(ransacQuery,ransacQuery,Size(3,3),Point(-1,-1),BORDER_DEFAULT); 
	//resize(ransacQuery,ransacQuery,Size(),patchResize,patchResize,INTER_LINEAR);

    cout << ransacQuery.rows << " " << ransacQuery.cols << endl;
	
	

	float ratio = ransacCompare(ransacPlano,ransacQuery);

    cout << "ratio : " << ratio << endl;

	return 0;
}

float ransacCompare(cv:: Mat img_object, cv:: Mat img_scene){

    //myfile << subDir << ',' << frame_no << ',' << imgCount << ',' << hogArea << ',' << Area << ',';


    if(!img_object.data || !img_scene.data){
        cout << "There's some problem in reading images" << endl;
        return -1;
    }

    Mat descriptors_object, descriptors_scene;
    std::vector<KeyPoint> keypoints_object,keypoints_scene;

    /*Ptr<FeatureDetector> detector = ORB::create( 
        1000,    
        1.2f,
        8,
        31,
        0,
        2,
        ORB::HARRIS_SCORE,
        31,
        12);*/
    Ptr<FeatureDetector> detector = ORB::create();

    detector->detect(img_object, keypoints_object);
    detector->detect(img_scene, keypoints_scene);

    Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();


    
    /*Ptr<DescriptorExtractor> descriptorExtractor = ORB::create(
        1000,    
        1.2f,
        8,
        31,
        0,
        2,
        ORB::HARRIS_SCORE,
        31,
        12);*/
    descriptorExtractor->compute(img_object, keypoints_object, descriptors_object);
    descriptorExtractor->compute(img_scene, keypoints_scene, descriptors_scene);

    std::vector<Point2f> objXY;
    std::vector<Point2f> sceneXY;

    //cout << descriptors_object.rows << " " << descriptors_object.cols << endl;
    //cout << descriptors_scene.rows << " " << descriptors_scene.cols << endl;

    int myradius = 3; 

    for(int i = 0; i < descriptors_object.rows; i++){
        objXY.push_back(Point2f(keypoints_object[i].pt));
        circle(img_object,cvPoint(objXY[i].x,objXY[i].y),myradius,Scalar(255,255,255),-1,8); 
    }

    for(int j = 0; j < descriptors_scene.rows; j++){
        sceneXY.push_back(Point2f(keypoints_scene[j].pt)); 
        circle(img_scene,cvPoint(sceneXY[j].x,sceneXY[j].y),myradius,Scalar(255,255,255),-1,8);  
    }

    //namedWindow("image object",CV_WINDOW_NORMAL);
    //imshow("image object",img_object);
    //waitKey(0);

    //namedWindow("image scene",CV_WINDOW_NORMAL);
    //imshow("image scene",img_scene);
    //waitKey(0);

    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<DMatch> good_matches;

    cv::BFMatcher matcher(NORM_L2, false);

    cout << "obj XY: " << objXY.size() << " " << "sceneXY :" << sceneXY.size() << endl;

    //myfile << objXY.size() << ',' << sceneXY.size() << ',';

    if(objXY.size() < 6){
        cout << "no or very few features can be found in either the planogram or the query" << endl;

        //myfile << ',' << ',' << "-1" << endl;
      
        return -1;  
    }

    if(sceneXY.size() == 0){
        cout << "no points could be found in query image. There is change" << endl;
        //myfile << ',' << ',' << '0' << endl;
        return 0;
    }

    matcher.knnMatch(descriptors_object, descriptors_scene , matches,2);


    const float descriptorDistanceRatio = 0.95;

    for (int i = 0; i < matches.size(); ++i){
        if (matches[i][0].distance < descriptorDistanceRatio * matches[i][1].distance){
            good_matches.push_back(matches[i][0]);
        } 
    }

    //myfile << good_matches.size() << ',';

    cout << "good_matches : " << good_matches.size() << endl;
   

    if(good_matches.size() < 4){
        cout << "The size of second nearest neighbour filtered matches is less than 4. Terminating Compare." << endl;

        //myfile << ',' << '0' << endl;
      
        return 0;
    }

     std::vector<Point2f> obj,scene,proj;

    for(size_t i = 0; i < good_matches.size(); ++i){
        obj.push_back(objXY[good_matches[i].queryIdx]);
        scene.push_back(sceneXY[good_matches[i].trainIdx]); 
    }

    Mat H = findHomography(obj, scene, CV_RANSAC);

    if (H.empty()){
        cout << "Homography Matrix is empty. Ratio is zero. There is change." << endl;
        //myfile << ',' << '0' << endl;
    
        return 0;
    }

    perspectiveTransform(obj,proj,H);


    std::vector<DMatch> very_good_matches;
    double tolerance =  max(1,max(img_scene.rows,img_scene.cols)/10);
    double tolerance_2 = max(1,max(img_object.rows,img_object.cols)/10);


    //cout << img_scene.rows << " " << img_scene.cols << endl;
    //cout << img_object.rows << " " << img_object.cols << endl;
    //cout << "tolerance :" << tolerance << endl;
    //cout << "tolerance_2 : " << tolerance_2 << endl;


    for(size_t i = 0; i < proj.size(); i++){

        double distObjScene = pointDistance(scene[i],proj[i]);
    
        if(distObjScene <= pow(tolerance,2)){
            {
                double distObjProj = pointDistance(obj[i],proj[i]);
                if(distObjProj <= pow(tolerance_2,2)){
                     very_good_matches.push_back(good_matches[i]);
                }
            }
        }
    } 

    //myfile << very_good_matches.size() << ','; 

    cout << "very good matches : " << very_good_matches.size() << endl;

   

    if(very_good_matches.size() == 0){
        cout << "no very good matches. Terminating compare. There is change." << endl;
        //myfile << '0' << endl;
        return 0;
    }

    Mat img_matches;

    std::vector<Point2f> objMatch;
    std::vector<Point2f> sceneMatch;

    for(size_t i = 0; i < very_good_matches.size(); i++){
        objMatch.push_back(objXY[very_good_matches[i].queryIdx]);
        sceneMatch.push_back(sceneXY[very_good_matches[i].trainIdx]);
    }

    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
        very_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

   
    //string write_patch = videoFold + "/" + std::to_string(imgCount) + "/" + std::to_string(imgCount) + ".jpg";
    //imwrite(write_patch,img_matches);


    std::vector<cv::DMatch > unique_matches;
    std::vector<int> traversed;
    bool got;


    for(int r = 0; r < very_good_matches.size(); r++){
        got = false;
        if(r == 0){
            unique_matches.push_back(very_good_matches[r]);
            traversed.push_back(very_good_matches[r].trainIdx);
        }
        else{
            for(int s = 0; s < traversed.size();s++){
                if(very_good_matches[r].trainIdx == traversed[s]){
                    got = true;
                    break;
                }
            }
            if(got == true){
                continue;
            }
            else{
                traversed.push_back(very_good_matches[r].trainIdx);
                unique_matches.push_back(very_good_matches[r]);
            }
        }
    }

    cout << "unique_matches : " << unique_matches.size() << endl;
     
 
    //namedWindow("image Match",CV_WINDOW_NORMAL);
    //imshow("image Match",img_matches);
    //waitKey(0);

    float ratio = ((float)unique_matches.size()/(float)objXY.size());

    //myfile << ratio << endl;

    return ratio;

}



static void
flip_descriptor(float *dst, float const *src)
{
  int const BO = 8;
  int const BP = 4; 
  int i, j, t;
  for (j = 0 ; j < BP ; ++j) {
    int jp = BP - 1 - j;
    for (i = 0 ; i < BP ; ++i) {
      int o  = BO * i + BP*BO * j;
      int op = BO * i + BP*BO * jp;

      dst [op] = src[o];

      for (t = 1 ; t < BO ; ++t)
        dst [BO - t + op] = src [t + o];
    }
  }

}

struct Tuple
detectorDescriptorSift(cv::Mat inputImage){

    VlCovDetMethod method = VL_COVDET_METHOD_DOG;
    
    //VL_COVDET_METHOD_MULTISCALE_HESSIAN
    //VL_COVDET_METHOD_MULTISCALE_HESSIAN
    //VL_COVDET_METHOD_HARRIS_LAPLACE
    //VL_COVDET_METHOD_HESSIAN;
    //VL_COVDET_METHOD_DOG
    //VL_COVDET_METHOD_NUM
    VlCovDet *covdet = vl_covdet_new(method);
    //vl_covdet_set_transposed(covdet, VL_TRUE);
    

    inputImage.convertTo(inputImage, CV_32F,(1.0/255.0));
      
    float const *imgData;

    imgData = (float *)inputImage.data;

   
    double peakThreshold = 0.01;
    double edgethreshold = 10;
    double boundaryMargin = 2.0;

    
    vl_covdet_set_edge_threshold(covdet,edgethreshold);
    vl_covdet_set_peak_threshold(covdet,peakThreshold);

    vl_covdet_set_first_octave(covdet,0);
    vl_covdet_put_image(covdet,imgData,inputImage.cols,inputImage.rows);

    vl_covdet_detect(covdet);

  
    vl_covdet_drop_features_outside(covdet,boundaryMargin);


 
    vl_covdet_extract_affine_shape(covdet);
   
    vl_covdet_extract_orientations(covdet);

    

    VlCovDetFeature const * feature = (VlCovDetFeature*)vl_covdet_get_features(covdet);

    vl_size numFeatures = vl_covdet_get_num_features(covdet);
    

    vl_index i;
    vl_size dimension = 128;
    vl_index patchResolution = 15;
    vl_size w = 2* patchResolution + 1;
    vl_size patchSide = 2 * patchResolution + 1;
    double patchRelativeExtent = 7.5;
    double patchStep = (double)patchRelativeExtent / patchResolution;
    float tempDesc[128];
    float *desc;

    //cout << "final features number : " << numFeatures << endl;
  
    
    desc = (float *) malloc(sizeof(float) * numFeatures * 128);
    double patchRelativeSmoothing = 1;

    float *patch  = NULL;
    float *patchXY = NULL ;
    patch = (float *)malloc(sizeof(float) * w * w);
    patchXY =(float *) malloc(2 * sizeof(float) * w * w);

    VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0);
    float *temp = desc;

    vl_sift_set_magnif(sift, 3.0);

   



    for (i = 0 ; i < (signed)numFeatures; ++i){
         vl_covdet_extract_patch_for_frame(covdet, 
                                           patch, 
                                           patchResolution, 
                                           patchRelativeExtent, 
                                           patchRelativeSmoothing, 
                                           feature[i].frame);

         vl_imgradient_polar_f (patchXY, 
                                patchXY +1,
                                2, 
                                2 * patchSide, 
                                patch, 
                                patchSide, 
                                patchSide, 
                                patchSide);
         vl_sift_calc_raw_descriptor (sift,
                                      patchXY,
                                      tempDesc,
                                      (int)patchSide, (int)patchSide,
                                      (double)(patchSide-1) / 2, (double)(patchSide-1) / 2,
                                      (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) /
                                      patchStep,
                                      VL_PI / 2);
         
         flip_descriptor(desc, tempDesc);
         desc += dimension;
        }


    Mat dest(numFeatures, 128, CV_32F, temp);
 

    //Mat newDest = dest.clone();
    

    vl_sift_delete(sift);  
    
    free(patch);
    free(patchXY);
    //free(temp);

    Tuple r = {dest,covdet};

    return r;

}


double eucledianNorm(std::vector<float> vec){
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++){
        sum  += pow(vec.at(i),2);

    }
    sum = sqrt(sum);
    return sum;
}

double pointDistance(Point2f a, Point2f b){
	    double sumX = pow((a.x - b.x),2);
        double sumY = pow((a.y - b.y),2);
        double dist = sumX + sumY;

        return dist;
}

