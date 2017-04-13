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

extern "C" {
    #include <vl/covdet.h>
    #include <vl/sift.h>
    #include <vl/mathop.h>
}

static void flip_descriptor (float * , float const *);

struct Tuple detectorDescriptorSift(cv:: Mat);

struct Tuple {
        Mat outDescriptor;
        VlCovDet *covdet;
        
    };

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
    blur(inputImage,inputImage,Size(3,3),Point(-1,-1),BORDER_DEFAULT);   
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

    cout << "final features number : " << numFeatures << endl;
  
    
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






int main(){
    
    Mat img_object = imread("/home/ubuntu/instanceMatch/iBot/testing/test_cases/multiple_swapped/1.jpg");
    blur(img_object,img_object,Size(3,3),Point(-1,-1),BORDER_DEFAULT); 
    resize(img_object,img_object,Size(),1.0,1.0,INTER_LINEAR);
    //"/home/ubuntu/instanceMatch/iBot/testing/test-data-results/multiple_swapped/2.jpg/1/one.jpg"
    Mat img_scene = imread("/home/ubuntu/instanceMatch/iBot/testing/test_cases/multiple_swapped/2.jpg");
    //"/home/ubuntu/instanceMatch/iBot/testing/test-data-results/multiple_swapped/2.jpg/1/two.jpg"
    blur(img_object,img_object,Size(3,3),Point(-1,-1),BORDER_DEFAULT); 
    resize(img_scene,img_scene,Size(),1.0,1.0,INTER_LINEAR);
    
    if(!img_object.data || !img_scene.data){
        cout << "There's some problem in reading images" << endl;
        return -1;
    }

    cout << img_object.rows << " " << img_object.cols << endl;

    Mat descriptors_scene, descriptors_object;
   
    struct Tuple objectTuple = detectorDescriptorSift(img_object);
    vl_size a = vl_covdet_get_num_features(objectTuple.covdet);
    descriptors_object = objectTuple.outDescriptor.clone();
    VlCovDetFeature const * featureObject = (VlCovDetFeature*)vl_covdet_get_features(objectTuple.covdet);
    //vl_covdet_delete(objectTuple.covdet);



    
    struct Tuple sceneTuple = detectorDescriptorSift(img_scene);
    vl_size b = vl_covdet_get_num_features(sceneTuple.covdet);
    descriptors_scene = sceneTuple.outDescriptor.clone();
    VlCovDetFeature const * featureScene = (VlCovDetFeature*)vl_covdet_get_features(sceneTuple.covdet);
    //vl_covdet_delete(sceneTuple.covdet);




    std::vector<DMatch> good_matches;
    std::vector<std::vector<cv::DMatch> > matches;

 
   
    //std::vector<DMatch> matches;

    cv::BFMatcher matcher(NORM_L2, false);
    matcher.knnMatch(descriptors_object, descriptors_scene , matches,2);

    cout << "the size of matches : " << matches.size() << endl;

    /*for (int i = 0; i < matches.size(); ++i){
        cout << "1 : " << matches[i][0].queryIdx << endl;
        cout << "2 : " << matches[i][0].trainIdx << endl;
        cout << "3 : " << matches[i][0].distance << endl;
    }*/





    
    for (int i = 0; i < matches.size(); ++i){
        const float ratio = 0.8;
        if (matches[i][0].distance < ratio * matches[i][1].distance){
            good_matches.push_back(matches[i][0]);
        } 
    }


    if(!good_matches.size()){
        cout << "the matches are ambiguosly close. Terminating program. " << endl;
        return -1;
    }



    cout << "good matches size : " << good_matches.size() << endl;

    std::vector<Point2f> objXY;
    std::vector<Point2f> sceneXY;

    for(int i = 0; i < descriptors_object.rows; ++i){
        objXY.push_back(Point2f(featureObject[i].frame.x,featureObject[i].frame.y));
        //cout << featureObject[i].frame.x << " " << featureObject[i].frame.y << endl;

    }

    for(int j = 0; j < descriptors_scene.rows; ++j){
        sceneXY.push_back(Point2f(featureScene[j].frame.x,featureScene[j].frame.y));
        //cout << featureScene[j].frame.x << " " << featureScene[j].frame.y << endl;
    }

    for(int i =0;i < sceneXY.size();i++){
        circle(img_scene, sceneXY[i],1,CV_RGB(255,255,255),3);

        //cout << objXY[i].x << " " << objXY[i].y << endl;
    }

    for(int i =0;i < objXY.size();i++){
        circle(img_object, objXY[i],1,CV_RGB(255,255,255),3);

        //cout << objXY[i].x << " " << objXY[i].y << endl;
    }

    namedWindow("scene",CV_WINDOW_NORMAL);
    imshow("scene",img_scene);
    waitKey(0);

    namedWindow("object",CV_WINDOW_NORMAL);
    imshow("object",img_object);
    waitKey(0);


    

    
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    std::vector<Point2f> proj;
 
    
   for(size_t i = 0; i < good_matches.size(); ++i){
        obj.push_back(objXY[good_matches[i].queryIdx]);
        scene.push_back(sceneXY[good_matches[i].trainIdx]);
   }


   for(size_t i = 0; i < good_matches.size(); ++i){
      cout << "distance : " << good_matches[i].distance << endl;
   }



    //cout << objXY.size() << " " << sceneXY.size() << endl; 



    Mat warped_image;

    Mat H = findHomography(obj, scene, CV_RANSAC);
    
    perspectiveTransform(obj,proj,H);

    //for(int i =0; i < obj.size(); i++){
        //cout << scene[0].x << " " << scene[0].y << endl;
        //cout << proj[0].x << " " << proj[0].y << endl;
    //}



    std::vector<DMatch> very_good_matches;
    double tolerance = 35;

    for(size_t i = 0; i < proj.size(); i++){
        double sumX = pow((scene[i].x - proj[i].x),2);
        double sumY = pow((scene[i].y - proj[i].y),2);
        double distObjProj = sumX + sumY;
        if(distObjProj <= pow(tolerance,2)){
            very_good_matches.push_back(good_matches[i]);
            //cout << "very good distance : " << good_matches[i].distance << endl; 
            cout << "scene : " << scene[i].x << " " << scene[i].y << "proj : " << proj[i].x << " " << proj[i].y << endl;
         }

    } 

    cout << "The number of very good inliers :" << very_good_matches.size() << endl;


    Mat img_matches;

    std::vector<Point2f> objMatch;
    std::vector<Point2f> sceneMatch;

    for(size_t i = 0; i < very_good_matches.size(); ++i){
        objMatch.push_back(objXY[very_good_matches[i].queryIdx]);
        sceneMatch.push_back(sceneXY[very_good_matches[i].trainIdx]);
    }

    //for(size_t i = 0; i < very_good_matches.size(); ++i){
    //     cout << objMatch[i].x << " " << objMatch[i].y << " " << sceneMatch[i].x << " " << sceneMatch[i].y << endl;
    //}

    std::vector<KeyPoint> keypoints_scene, keypoints_object;


    cv::KeyPoint::convert(objXY,keypoints_object);
    cv::KeyPoint::convert(sceneXY,keypoints_scene);


    //for(size_t i = 0; i < very_good_matches.size(); ++i){
    //    cout << keypoints_object[i].pt.x << " " << keypoints_object[i].pt.y << " " << keypoints_scene[i].pt.x << " " << keypoints_scene[i].pt.y << endl;
    //}

    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
        very_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 
 

    namedWindow("image Match",CV_WINDOW_NORMAL);
    
    imshow("image Match",img_matches);
    waitKey(0);
    //imwrite("/home/ubuntu/matching_5.jpg",img_matches);*/
    //vl_covdet_delete(objectTuple.covdet);
    //vl_covdet_delete(sceneTuple.covdet); **/

    //imwrite("../testing/results/sift_results/all_tilted.jpg",img_matches);

    vl_covdet_delete(objectTuple.covdet);
    vl_covdet_delete(sceneTuple.covdet);


    return 0;
}

   
