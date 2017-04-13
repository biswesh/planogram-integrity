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

static void flip_descriptor (float * , float const *);
struct Tuple detectorDescriptorSift(cv:: Mat);
double eucledianNorm(std::vector<float>);
cv:: Mat resizeAndRescale(cv::Mat, float);
vector<vector<vector<float> > > extractHOG(cv::Mat, vl_size&, vl_size&, vl_size&);
double windowDistance(std::vector<float>,std::vector<float>);
double windowDistanceNorm(std::vector<float>,std::vector<float>);
std::vector<Point2f> formWindowAndCompare(vector<vector<vector<float> > >, vector<vector<vector<float> > >, 
    int, int, int, vl_size, vl_size, vl_size, float);
void showNonMatches(cv::Mat, cv::Mat, vector<Point2f>, float);
float ransacCompare(cv:: Mat, cv:: Mat);

struct Tuple {
        Mat outDescriptor;
        VlCovDet *covdet;
    };




int main(int argc, char ** argv){

//PARAMETER CHANGE
    float ImgResize = atof(argv[4]);
    int windowSize = 3;
    int yslide = atoi(argv[6]);
    int xslide = atoi(argv[5]);
    float hogThreshold = atof(argv[3]);


    Mat planogramSrc = imread(argv[1],IMREAD_GRAYSCALE);

    Mat querySrc = imread(argv[2],IMREAD_GRAYSCALE);

    Mat planogramImg = resizeAndRescale(planogramSrc,ImgResize);
    Mat queryImg = resizeAndRescale(querySrc,ImgResize);

    vector<vector<vector<float> > > planoVec, queryVec;
    
    vl_size hogWidth, hogHeight, hogDimension;

    planoVec = extractHOG(planogramImg,hogWidth,hogHeight,hogDimension);
    queryVec = extractHOG(queryImg,hogWidth,hogHeight,hogDimension);

    vector<Point2f> nonMatchVector = formWindowAndCompare(planoVec,queryVec,windowSize,xslide,yslide,
                hogWidth,hogHeight,hogDimension,hogThreshold);


    showNonMatches(planogramSrc,querySrc,nonMatchVector,ImgResize);

    return 0;

}


void showNonMatches(cv::Mat imgMatRaw, cv::Mat imgMatRaw_2, vector<Point2f> nonMatchVector, float ImgResize){

    Mat imgMat = imgMatRaw.clone();
    Mat imgMat_2 = imgMatRaw_2.clone();
     
    resize(imgMat,imgMat,Size(),ImgResize,ImgResize,INTER_LINEAR);
    resize(imgMat_2,imgMat_2,Size(),ImgResize,ImgResize,INTER_LINEAR);
    Mat imgMatCopy = imgMat.clone();
    int myradius=4;

    for (int i=0;i < nonMatchVector.size();i++){
        circle(imgMat,cvPoint(nonMatchVector[i].x,nonMatchVector[i].y),myradius,Scalar(0,0,255),-1,8,0);  
    }
    Mat mask;

    inRange(imgMat,Scalar(0,0,0), Scalar(0,0,0),mask);
    //imwrite("../mask.jpg",mask);
    Mat labels,stats,centroids;

    int a = connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
    cout << "connected components : " << a << endl;

    //string ty =  type2str( mask.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), mask.cols, mask.rows );

    //namedWindow("image", CV_WINDOW_NORMAL);
    //imshow("image",mask);
    //while (char(waitKey(1))!= 'q'){};

    //cout << stats.rows << " " << stats.cols << endl;
    //cout << centroids << endl;
    cout << stats << endl;
    //cout << stats.at<int>(1,0) << endl;
    //cout << stats.at<int>(1,1) << endl;
    //cout << stats.at<int>(1,2) << endl;
    //cout << stats.at<int>(1,3) << endl;
    //cout << stats.at<int>(1,4) << endl;

    float rescale = 1.0 / ImgResize;

    for(int i = 0; i < stats.rows; i++){
        cout << stats.at<int>(i,4) << endl;
        if(stats.at<int>(i,4) > 500 && stats.at<int>(i,4) < 25000){
           
            Mat ransacPlano = imgMatRaw(Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1), 
                rescale * stats.at<int>(i,2),rescale * stats.at<int>(i,3))).clone();
            Mat ransacQuery = imgMatRaw_2(Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1),
                rescale * stats.at<int>(i,2), rescale * stats.at<int>(i,3))).clone();
            float ratio = ransacCompare(ransacPlano,ransacQuery);
            cout << "the ratio of cropped image : " << ratio << endl;


            if (ratio == -1)
                    cout << stats.at<int>(i,4) << endl;
                   //rectangle(imgMat,Point(stats.at<int>(i,0),stats.at<int>(i,1)),
                    //Point((stats.at<int>(i,0) + stats.at<int>(i,2)),(stats.at<int>(i,1) + stats.at<int>(i,3))),
                    //Scalar(255,255,255),
                    //5,
                    //LINE_8,
                    //0);
            else if (ratio > 0.5)
                continue;
            else
                rectangle(imgMat,Point(stats.at<int>(i,0),stats.at<int>(i,1)),
                    Point((stats.at<int>(i,0) + stats.at<int>(i,2)),(stats.at<int>(i,1) + stats.at<int>(i,3))),
                    Scalar(255,255,255),
                    5,
                    LINE_8,
                    0);
        }
     
    }

    //imwrite("../testing/results/bounding_box_results/lighting.jpg",imgMat);

    namedWindow("image", CV_WINDOW_NORMAL);
    imshow("image",imgMat);
    waitKey(0);
    //while (char(waitKey(1))!= 'q'){};


}

float ransacCompare(cv:: Mat img_object, cv:: Mat img_scene){

    if(!img_object.data || !img_scene.data){
        cout << "There's some problem in reading images" << endl;
        return -1;
    }

    Mat descriptors_scene, descriptors_object;

    struct Tuple objectTuple = detectorDescriptorSift(img_object);
    vl_size a = vl_covdet_get_num_features(objectTuple.covdet);
    descriptors_object = objectTuple.outDescriptor.clone();
    VlCovDetFeature const * featureObject = (VlCovDetFeature*)vl_covdet_get_features(objectTuple.covdet);

    struct Tuple sceneTuple = detectorDescriptorSift(img_scene);
    vl_size b = vl_covdet_get_num_features(sceneTuple.covdet);
    descriptors_scene = sceneTuple.outDescriptor.clone();
    VlCovDetFeature const * featureScene = (VlCovDetFeature*)vl_covdet_get_features(sceneTuple.covdet);

    std::vector<Point2f> objXY;
    std::vector<Point2f> sceneXY;

    for(int i = 0; i < descriptors_object.rows; ++i){
        objXY.push_back(Point2f(featureObject[i].frame.x,featureObject[i].frame.y));
    }

    for(int j = 0; j < descriptors_scene.rows; ++j){
        sceneXY.push_back(Point2f(featureScene[j].frame.x,featureScene[j].frame.y));
    }

    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<DMatch> good_matches;

    cv::BFMatcher matcher(NORM_L2, false);

    if(descriptors_object.rows == 0 || descriptors_scene.rows == 0){
        cout << "no features can be found" << endl;
        return -1;  
    }
    cout << descriptors_scene.rows << endl;
    cout << "biswesh start : " << endl;
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
    cout << "good matches size : " << good_matches.size() << endl;

    if(good_matches.size() == 0){
        cout << "size of matches is 0. Terminating compare." << endl;
        return -1;
    }

     std::vector<Point2f> obj,scene,proj;

    for(size_t i = 0; i < good_matches.size(); ++i){
        obj.push_back(objXY[good_matches[i].queryIdx]);
        scene.push_back(sceneXY[good_matches[i].trainIdx]);
    }

    Mat H = findHomography(obj, scene, CV_RANSAC);

    if (H.empty()){
        cout << "Homography Matrix is empty" << endl;
        return -1;
    }
    perspectiveTransform(obj,proj,H);


    //for(size_t i = 0; i < good_matches.size(); ++i){
    //  cout << "distance : " << good_matches[i].distance << endl;
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
            //cout << "scene : " << scene[i].x << " " << scene[i].y << "proj : " << proj[i].x << " " << proj[i].y << endl;
         }
    } 

    if(!very_good_matches.size()){
        cout << "no very good matches. Terminating compare." << endl;
        return -1;
    }



    cout << "The number of very good inliers :" << very_good_matches.size() << endl;


    Mat img_matches;

    std::vector<Point2f> objMatch;
    std::vector<Point2f> sceneMatch;

    for(size_t i = 0; i < very_good_matches.size(); ++i){
        objMatch.push_back(objXY[very_good_matches[i].queryIdx]);
        sceneMatch.push_back(sceneXY[very_good_matches[i].trainIdx]);
    }


    std::vector<KeyPoint> keypoints_scene, keypoints_object;

    cv::KeyPoint::convert(objXY,keypoints_object);
    cv::KeyPoint::convert(sceneXY,keypoints_scene);

    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, 
        very_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 
 

    namedWindow("image Match",CV_WINDOW_NORMAL);
    imshow("image Match",img_matches);
    waitKey(0);

    return ((float)very_good_matches.size()/(float)matches.size());

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


cv:: Mat resizeAndRescale(cv::Mat imgMat, float ImgResize){

    resize(imgMat,imgMat,Size(),ImgResize,ImgResize,INTER_LINEAR);
    Mat scaledMat = Mat(imgMat.rows,imgMat.cols,CV_32F);
    scaledMat = imgMat.clone();
    scaledMat.convertTo(scaledMat,CV_32F,1.0/255);
   
    return scaledMat;
}

vector<vector<vector<float> > > extractHOG(cv::Mat dataMat, vl_size& hogWidth, vl_size& hogHeight, vl_size& hogDimension){

    float const *hogData;

    hogData = (float *)dataMat.data;


    vl_size numOrientations = 9;
    vl_size cellSize = 8;
    vl_size numChannels = 1;

    //VlHogVariantUoctti
    //VlHogVariantDalalTriggs

    VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE);

    vl_hog_put_image(hog, hogData, dataMat.cols, dataMat.rows, numChannels, cellSize);

    hogWidth = vl_hog_get_width(hog);
    hogHeight = vl_hog_get_height(hog);
    hogDimension = vl_hog_get_dimension(hog);
    float *finalHOG = NULL;

    finalHOG = (float *)vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float));
    vl_hog_extract(hog,finalHOG);
    //vector<vector<vector<float> > > hogVec;



    vector<vector<vector<float> > > hogVec(hogHeight, vector<vector<float> >(hogWidth, vector<float>(hogDimension)));

 
    for(int i = 0; i < hogDimension; i++){
        for(int j = 0; j < hogHeight; j++){
            for(int k = 0; k < hogWidth; k++){
                hogVec[j][k][i] = *finalHOG;
                finalHOG++;
            }
        }
    }
    finalHOG = (finalHOG - (hogWidth * hogHeight * hogDimension));
    
    free(finalHOG);
    vl_hog_delete(hog);
    return hogVec;
    

}

double windowDistance(std::vector<float> window1,std::vector<float> window2){
    double sum = 0.0;
    for(int k = 0; k < window1.size(); k++){
        double squaredDif = pow((window2.at(k) - window1.at(k)),2);
        double sumDivide = (squaredDif / (window1.at(k) + window2.at(k)));
        sumDivide = 2 * sumDivide;
        sum += sumDivide;
    }

    sum = sum / window1.size();

    return sum;

}

double windowDistanceNorm(std::vector<float> window1,std::vector<float> window2){
    double sum = 0;
    for(int x = 0; x < window1.size(); x++){
        double eleMult = window1.at(x) * window2.at(x);
        sum += eleMult;
    }

    double eucledianNormMult = eucledianNorm(window1) * eucledianNorm(window2);
    sum = sum / eucledianNormMult;
    return sum;
}

std::vector<Point2f> formWindowAndCompare(vector<vector<vector<float> > > planoHog,vector<vector<vector<float> > > queryHog, 
    int windowSize, int xslide, int yslide, vl_size hogWidth, vl_size hogHeight, vl_size hogDimension, float hogThreshold){
    std::cout<<hogWidth<<" "<<hogHeight<<" "<<hogDimension<<"\n";
     int windowForm = (windowSize - 1) / 2; //assertion required
     bool found;
     std::vector<Point2f> nonmatchesHOG;
     int count = 0;

    for(int i = (0 + (yslide + windowForm)); i < (hogHeight - (yslide + windowForm)); i++){
        for(int j = (0 + (xslide + windowForm)); j < (hogWidth - (xslide + windowForm)); j++){

            found = false;
            //planogram image window
            std::vector<float> window1;
            for(int k = -windowForm; k <= windowForm; k++){
                for(int l = -windowForm; l <= windowForm; l++){ 
                    for(int m = 0; m < hogDimension; m++){
                        window1.push_back(planoHog[i+k][j+l][m]);
                    } 

                }
            }

            //query image window
            for(int a = (i - yslide); a <= (i + yslide); a++){
                for(int b = (j - xslide); b <= (j + xslide); b++){

                    std::vector<float> window2;


                    for(int x = -windowForm; x <= windowForm; x++){
                        for(int y = -windowForm; y <= windowForm; y++){
                    
                            for(int z = 0; z < hogDimension; z++){
                                window2.push_back(queryHog[a+x][b+y][z]);
                            }
                        }
                    }

                    if(eucledianNorm(window1) < 1e-8 && eucledianNorm(window2) < 1e-8){
                        found = true;
                        break;
                    }


                    double a = 1.0-windowDistanceNorm(window1,window2);
                    //double a = compareHist(window1,window2,3);
                    //a = a/window1.size();
                    //cout << a << endl;

                    if (a < hogThreshold){
                        found = true;
                        break;
                    }


                }

                if (found == true){
                    break;
                }

            }

            if (found == false){
                 nonmatchesHOG.push_back(Point2f(4+(j*8),4+(i*8)));    
            }


        }


    }

    return nonmatchesHOG;
}









