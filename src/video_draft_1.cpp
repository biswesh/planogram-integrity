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
#include "opencv2/videoio.hpp"

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

double eucledianNorm(std::vector<float>);
cv:: Mat resizeAndRescale(cv::Mat, float);
vector<vector<vector<float> > > extractHOG(cv::Mat, vl_size&, vl_size&, vl_size&);
double windowDistance(std::vector<float>,std::vector<float>);
double windowDistanceNorm(std::vector<float>,std::vector<float>);
double absoluteDistance(std::vector<float>,std::vector<float>);
std::vector<Point2f> formWindowAndCompare(vector<vector<vector<float> > >, vector<vector<vector<float> > >, 
    int, int, int, vl_size, vl_size, vl_size, float);
int showNonMatches(cv::Mat, cv::Mat, vector<Point2f>, float, string, string, int,ofstream&);
float ransacCompare(cv:: Mat, cv:: Mat, string, int,ofstream&,int,int,int,string,bool,Rect,Rect);
double pointDistance(Point2f,Point2f,double,double);
vector<string> getDirElements(string);
void makeAfolder(string);
Rect enlargeROI(Mat frm, Rect boundingBox, int padding);
void planoProcess(Mat,Rect,int, Rect &, Rect &);
void queryProcess(Mat,Rect,int,int, Rect &, Rect &);




int main(int argc, char ** argv){

//PARAMETER CHANGE
    float ImgResize = atof(argv[2]);
    int windowSize = 3;
    int yslide = atoi(argv[4]);
    int xslide = atoi(argv[3]);
    float hogThreshold = atof(argv[1]);
    double initialFrame = 100;
    double captureInterval = 100;
    
    cout << "reading video files from directory ... " << endl; 

    string Maindir = string("../testing/video_files/");

    vector<string> videoFiles = vector<string>();
    videoFiles = getDirElements(Maindir);

    cout << "displaying video file names ... " << endl;

    for(int i = 0; i < videoFiles.size(); i++){
        cout << videoFiles[i] << endl;
    }

    string destDir = string("../testing/video-data-results/");

    ofstream myfile;
    myfile.open("result_stats.csv");
    myfile << "video_file_name,frame_number,patch_name,connected_component_area,bounding_box_area,initial_plano_featues,initial_query_features,good_matches,very_good_matches,ratio" << endl;

    cout << "entering capture mode ... " << endl;

    cout << "############## DIRECTORY START #########################" << endl;
    
    for (unsigned int i = 0;i < 1;i++){

        cout << "############## VIDEO FILE START #######################" << endl;


        cout << "videoFile name : " << videoFiles[i] << endl;
        string videoPath = Maindir + videoFiles[i];

        string destSubDirPath = destDir + videoFiles[i];

        makeAfolder(destSubDirPath);
    
        VideoCapture capture(videoPath);
        capture.set(CAP_PROP_POS_FRAMES, initialFrame);

        if( !capture.isOpened() ){
            throw "Error when reading steam"; 
        }
            
        Mat planogramSrc,querySrc,frame;

        cout << "capturing planogram Image : " << endl;

        capture.read(frame);
        Mat planogramCol = frame.clone();

        cvtColor(planogramCol, planogramSrc, CV_BGR2GRAY);
        blur(planogramSrc, planogramSrc, Size(3, 3), Point(-1,-1));

        Mat planogramImg = resizeAndRescale(planogramSrc,ImgResize);

        vl_size hogWidth, hogHeight, hogDimension;
        vector<vector<vector<float> > > planoVec;
        planoVec = extractHOG(planogramImg,hogWidth,hogHeight,hogDimension);

        double captureTime = initialFrame;

        while(capture.isOpened()){
            cout << "############## QUERY IMAGE START #######################" << endl;

            cout << "capturing query image..." << endl;

            captureTime += captureInterval;
            capture.set(CAP_PROP_POS_FRAMES,captureTime);

            double fram = capture.get(CAP_PROP_POS_FRAMES);

            cout << "captureFrame : " << captureTime << endl;
            cout << "frame index : " << fram << endl;

            //double fps = capture.get(CV_CAP_PROP_FPS);

            //cout << "fps : " << fps << endl;

            capture.read(frame);

            if(frame.empty()){
                cout << "frame is empty, going to the next video file : " << endl; 
                break; 
            }

            Mat queryCol = frame.clone();

            cvtColor(queryCol, querySrc, CV_BGR2GRAY);
            blur(querySrc, querySrc, Size(3, 3), Point(-1,-1));
            //imwrite("../query.jpg",querySrc);
            Mat queryImg = resizeAndRescale(querySrc,ImgResize); 
            vector<vector<vector<float> > > queryVec;
            queryVec = extractHOG(queryImg,hogWidth,hogHeight,hogDimension);

            vector<Point2f> nonMatchVector = formWindowAndCompare(planoVec,queryVec,windowSize,xslide,yslide,
                hogWidth,hogHeight,hogDimension,hogThreshold);

            string videoFolder = destSubDirPath + '/' + "frame_" + std::to_string((int)captureTime);

            string raw_plano = videoFolder + "/" + "plano.jpg";
            string raw_query = videoFolder + "/" + "query.jpg";

            //cout << "raw_plano : "  << raw_plano << endl;
            //cout << "raw_query : " << raw_query << endl; 

            imwrite(raw_plano,planogramCol);
            imwrite(raw_query,queryCol);
            /*namedWindow("plano",CV_WINDOW_NORMAL);
            imshow("plano",planogramCol);
            waitKey(0);
            namedWindow("query",CV_WINDOW_NORMAL);
            imshow("query",planogramCol);
            waitKey(0);*/

            makeAfolder(videoFolder);

            cout <<  "videoFolder : " << videoFolder << endl; 

            int integrity = showNonMatches(planogramSrc,querySrc,nonMatchVector,ImgResize,videoFolder,videoFiles[i],(int)captureTime,myfile);

            cout << "integrity : " << integrity << endl;

            if(integrity == 1){
                cout << "PLANOGRAM INTEGRITY MAINTAINED" << endl;
            }
            else if(integrity == 0){
                cout << "PLANOGRAM INTEGRITY BROKEN" << endl;
            }
            else{
                cout << "NOTHING CAN BE SAID ABOUT INTEGRITY" << endl;
            }

            myfile << "integrity" << ',' <<  integrity << endl;
            cout << "############## QUERY IMAGE END #######################" << endl;

        }
        cout << "############## VIDEO FILE END #######################" << endl;
    }

    cout << "############## DIRECTORY END #########################" << endl;
    

    myfile.close();

    return 0;
}



int showNonMatches(cv::Mat imgMatRaw, cv::Mat imgMatRaw_2, vector<Point2f> nonMatchVector, 
    float ImgResize, string videoFolder, string subDir, int frame_no,ofstream& myfile){

    Mat imgMat = imgMatRaw.clone();
    Mat imgMat_2 = imgMatRaw_2.clone();
     
    resize(imgMat,imgMat,Size(),ImgResize,ImgResize,INTER_LINEAR);
    resize(imgMat_2,imgMat_2,Size(),ImgResize,ImgResize,INTER_LINEAR);
    Mat imgMat_2Copy = imgMat_2.clone();
    int myradius=4;

    for (int i=0;i < nonMatchVector.size();i++){
        circle(imgMat_2,cvPoint(nonMatchVector[i].x,nonMatchVector[i].y),myradius,Scalar(0,0,255),-1,8);  
    }
    Mat mask;

    inRange(imgMat_2,Scalar(0,0,0), Scalar(0,0,0),mask);
    //imwrite("../mask.jpg",mask);
    Mat labels,stats,centroids;

    int a = connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
    //cout << "connected components : " << a << endl;

    //namedWindow("image", CV_WINDOW_NORMAL);
    //imshow("image",mask);
    //while (char(waitKey(1))!= 'q'){};

    //cout << stats.rows << " " << stats.cols << endl;
    //cout << centroids << endl;
    //cout << stats << endl;

    float rescale = 1.0 / ImgResize;
    //int patchResize = 3.0;

 
    bool omit;
    int planogramIntegrity = 1;

    for(int i = 1; i < stats.rows; i++){

        omit = false;

        int hogArea = stats.at<int>(i,4);
        cout << "###################PATCH START " << i << " ###################" << endl;
        cout << "connected components area : " << hogArea << endl;
        int Area = (stats.at<int>(i,2) * stats.at<int>(i,3));
        cout << "Bounding box Area : " << Area << endl;
        float hogAreaRatio = (float)hogArea/float(Area);
        cout << "hogAreaRatio : " << hogAreaRatio << endl;


        if(hogArea > 42000){
            cout << "unusually big blob detected. Omitting frame." << endl;
            omit = true;
            break;
        }

        int padding = 100; 
        int displace_tolerance = 50;

        if(hogArea > 1200 && hogArea <= 42000 && hogAreaRatio > 0.20){

            Rect ROI = Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1), 
                rescale * stats.at<int>(i,2),rescale * stats.at<int>(i,3));
            //Rect query_rect = Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1), 
            //    rescale * stats.at<int>(i,2),rescale * stats.at<int>(i,3));

            Rect plan_rect_bound, plan_rect_padded, query_rect_bound, query_rect_displace_padded;

            planoProcess(imgMatRaw,ROI,padding,plan_rect_bound,plan_rect_padded);
            cout << "plan bound : " << plan_rect_bound << endl;
            Mat ransacPlano = imgMatRaw(plan_rect_padded).clone();
            queryProcess(imgMatRaw_2,ROI,padding,displace_tolerance,query_rect_bound,query_rect_displace_padded);
            //cout << "querybox : " << query_rect_bound << endl;
            Mat ransacQuery = imgMatRaw_2(query_rect_displace_padded).clone();
            bool plano2query = true;
            string videoFolder1 = videoFolder + '/' + "plano2query/";
            makeAfolder(videoFolder1);


            float ratio_plano_to_query = ransacCompare(ransacPlano,ransacQuery,videoFolder1,i,myfile,Area,hogArea,frame_no,subDir,plano2query,plan_rect_bound,query_rect_bound);
            planoProcess(imgMatRaw_2,ROI,padding,plan_rect_bound,plan_rect_padded);
            ransacPlano = imgMatRaw_2(plan_rect_padded).clone();
            queryProcess(imgMatRaw,ROI,padding,displace_tolerance,query_rect_bound,query_rect_displace_padded);
            ransacQuery = imgMatRaw(query_rect_displace_padded).clone();
            plano2query = false;
            string videoFolder2 = videoFolder + '/' + "query2plano/";
            makeAfolder(videoFolder2);


            float ratio_query_to_plano = ransacCompare(ransacPlano,ransacQuery,videoFolder2,i,myfile,Area,hogArea,frame_no,subDir,plano2query,plan_rect_bound,query_rect_bound);

            bool noChange = (((ratio_plano_to_query == -1) || (ratio_plano_to_query > 0.50)) && ((ratio_query_to_plano == -1) || (ratio_query_to_plano > 0.50))); 

           
            if(noChange == true)
                continue;
            else{
                planogramIntegrity = 0;
                rectangle(imgMat_2,Point(stats.at<int>(i,0),stats.at<int>(i,1)),
                    Point((stats.at<int>(i,0) + stats.at<int>(i,2)),(stats.at<int>(i,1) + stats.at<int>(i,3))),
                    Scalar(255,255,255),
                    5,
                    LINE_8,
                    0);
            }

        }
     
    }
    cout << "###################PATCH END###################" << endl;


    

    string write_final = videoFolder + "/" + "diff.jpg";
  
    if(omit == true){
        planogramIntegrity = -1;
        for (int i = 0;i < nonMatchVector.size();i++){
             circle(imgMat_2Copy,cvPoint(nonMatchVector[i].x,nonMatchVector[i].y),myradius,Scalar(0,0,255),-1,8);  
        }
        imwrite(write_final,imgMat_2Copy);
    }
    else{
        imwrite(write_final,imgMat_2);
    }

    return planogramIntegrity;
         

    //imwrite("../testing/results/bounding_box_results/lighting.jpg",imgMat);

    //namedWindow("image", CV_WINDOW_NORMAL);
    //imshow("image",imgMat);
    //waitKey(0);
    //while (char(waitKey(1))!= 'q'){};


}

void planoProcess(Mat imgMatRaw,Rect ROI, int padding,Rect &plan_rect_bound, Rect &plan_rect_padded){

    Mat img = imgMatRaw.clone();
    cout << "plan_rect : " << ROI << endl;
    plan_rect_padded = enlargeROI(img,ROI,padding);
    cout << "plan_rect_padded : " << plan_rect_padded << endl;
    plan_rect_bound.width = ROI.width;
    plan_rect_bound.height = ROI.height;
    plan_rect_bound.x = ROI.x - plan_rect_padded.x;
    plan_rect_bound.y = ROI.y - plan_rect_padded.y; 
    cout << "plan_rect_bound : " << plan_rect_bound << endl;
    
}

void queryProcess(Mat imgMatRaw_2,Rect ROI, int padding, int displace_tolerance, Rect &query_rect_bound, Rect &query_rect_displace_padded){

    Mat img = imgMatRaw_2.clone();
    cout << "query_rect : " << ROI << endl;
    Rect query_rect_displace = enlargeROI(imgMatRaw_2,ROI,displace_tolerance);
    cout << "query_rect_displace : " << query_rect_displace << endl;
    query_rect_displace_padded = enlargeROI(imgMatRaw_2,query_rect_displace,padding);
    cout << "query_rect_displace_padded : " << query_rect_displace_padded << endl;
    query_rect_bound.width = query_rect_displace.width;
    query_rect_bound.height = query_rect_displace.height;
    query_rect_bound.x = query_rect_displace.x - query_rect_displace_padded.x;
    query_rect_bound.y = query_rect_displace.y - query_rect_displace_padded.y;
    cout << "query_rect_bound : " << query_rect_bound << endl;
    
}

Rect enlargeROI(Mat frm, Rect boundingBox, int padding) {
    Rect returnRect = Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
    if (returnRect.x < 0)returnRect.x = 0;
    if (returnRect.y < 0)returnRect.y = 0;
    if (returnRect.x+returnRect.width >= frm.cols)returnRect.width = frm.cols-returnRect.x;
    if (returnRect.y+returnRect.height >= frm.rows)returnRect.height = frm.rows-returnRect.y;
    return returnRect;
}

float ransacCompare(cv:: Mat img_1, cv:: Mat img_2, string videoFold, int imgCount,ofstream& myfile,int Area,int hogArea,int frame_no,string subDir, bool plano2query, Rect plan_rect_bound, Rect query_rect_bound){
    

    Mat img_object = img_1.clone();
    //blur(img_object,img_object, Size(3, 3), Point(-1,-1));    
    
    Mat img_scene = img_2.clone();
    //blur(img_scene, img_scene, Size(3, 3), Point(-1,-1));
    
    
    if(plano2query == true){
        cout << "plano2query ..." << endl; 
        myfile << "plano2query" << endl;
    }
    else{
        cout << "query2plano ..." << endl;
        myfile << "query2plano" << endl; 
    }
       

    myfile << subDir << ',' << frame_no << ',' << imgCount << ',' << hogArea << ',' << Area << ',';


    if(!img_object.data || !img_scene.data){
        cout << "There's some problem in reading images" << endl;
        return -1;
    }

    Mat descriptors_object, descriptors_scene,descriptors_object_before_filtering, descriptors_scene_before_filtering;
    std::vector<KeyPoint> keypoints_object,keypoints_scene,keypoints_object_before,keypoints_scene_before;

    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(img_object, keypoints_object_before);
    detector->detect(img_scene, keypoints_scene_before);
    
    Ptr<DescriptorExtractor> descriptorExtractor = ORB::create();
    descriptorExtractor->compute(img_object, keypoints_object_before, descriptors_object_before_filtering);
    descriptorExtractor->compute(img_scene, keypoints_scene_before, descriptors_scene_before_filtering);

    std::vector<Point2f> objXY;
    std::vector<Point2f> sceneXY;

    //cout << descriptors_object.rows << " " << descriptors_object.cols << endl;
    //cout << descriptors_scene.rows << " " << descriptors_scene.cols << endl;

    int myradius = 1; 

    cout << "keypoints_object_before : " << keypoints_object_before.size() << endl;
    cout << "keypoints_scene_before : " << keypoints_scene_before.size() << endl;
    cout << "descriptors_object_before_filtering_rows : " << descriptors_object_before_filtering.rows << "cols : " << descriptors_object_before_filtering.cols << endl;
    cout << "descriptors_scene_before_filtering_rows : " << descriptors_scene_before_filtering.rows << "cols : " << descriptors_scene_before_filtering.cols << endl;

    cout << "patch_object rows : " << img_object.rows << "cols : " << img_object.cols << endl;
    cout << "patch_scene rows : " << img_scene.rows << "cols : " << img_scene.cols << endl;

    string patchOne = videoFold + "/" + std::to_string(imgCount) + "/";
    makeAfolder(patchOne);
    string patchTwo = videoFold + "/" + std::to_string(imgCount) + "/";
    makeAfolder(patchTwo);

    string one = patchOne + "/one_original.jpg";
    imwrite(one,img_object);

   
    string two = patchTwo + "/two_original.jpg";
    imwrite(two,img_scene);

    for(int i = 0; i < descriptors_object_before_filtering.rows; i++){
        Point2f point (keypoints_object_before[i].pt);
        if(plan_rect_bound.contains(point)) {
            objXY.push_back(Point2f(keypoints_object_before[i].pt));
            keypoints_object.push_back(keypoints_object_before[i]);
            descriptors_object.push_back(descriptors_object_before_filtering.row(i));
            circle(img_object,cvPoint(objXY.back().x,objXY.back().y),myradius,Scalar(255,255,255),-1,8); 
        }
        /*else
            if(imgCount == 2){
                circle(img_object,point,myradius,Scalar(255,255,255),-1,8);
                cout << "point : " << point << endl;
            }*/
    }

    cout << "objXY : " << objXY.size() << endl;
    cout << "keypoints_object : " << keypoints_object.size() << endl;
   
    cout << "descriptors_object : " << descriptors_object.rows << " " << descriptors_object.cols << endl;
    



    //descriptors_object.resize(0,descriptors_object_before_filtering.cols);

    cout << "descriptors_object_after_resize : " << descriptors_object.rows << " " << descriptors_object.cols << endl;

    




    for(int j = 0; j < descriptors_scene_before_filtering.rows; j++){
        Point2f point (keypoints_scene_before[j].pt);
        if(query_rect_bound.contains(point)) {
            sceneXY.push_back(Point2f(keypoints_scene_before[j].pt)); 
            keypoints_scene.push_back(keypoints_scene_before[j]);
            descriptors_scene.push_back(descriptors_scene_before_filtering.row(j));
            circle(img_scene,cvPoint(sceneXY.back().x,sceneXY.back().y),myradius,Scalar(255,255,255),-1,8);  
        }
    }

    cout << "sceneXY : " << sceneXY.size() << endl;

     cout << "keypoints_scene : " << keypoints_scene.size() << endl;
     cout << "descriptors_scene : " << descriptors_scene.rows << " " << descriptors_scene.cols << endl;


    //descriptors_scene.resize(0,descriptors_scene_before_filtering.cols);

    cout << "descriptors_scene_after_resize : " << descriptors_scene.rows << " " << descriptors_scene.cols << endl;


 
    
 
    one = patchOne + "/one_keypoints.jpg";
    imwrite(one,img_object);

   
    two = patchTwo + "/two_keypoints.jpg";
    imwrite(two,img_scene);

    //namedWindow("image object",CV_WINDOW_NORMAL);
    //imshow("image object",img_object);
    //waitKey(0);

    //namedWindow("image scene",CV_WINDOW_NORMAL);
    //imshow("image scene",img_scene);
    //waitKey(0);

    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<DMatch> good_matches;

    cv::BFMatcher matcher(NORM_HAMMING, false);

    //cout << "obj XY: " << objXY.size() << " " << "sceneXY :" << sceneXY.size() << endl;

    myfile << objXY.size() << ',' << sceneXY.size() << ',';

    if(objXY.size() < 6){
        cout << "no or very few features can be found in planogram" << endl;

        myfile << ',' << ',' << "-1" << endl;
      
        return -1;  
    }

    if(sceneXY.size() == 0){
        cout << "no points could be found in query image. There is change" << endl;
        myfile << ',' << ',' << '0' << endl;
        return 0;
    }

    matcher.knnMatch(descriptors_object, descriptors_scene , matches,2);


    const float descriptorDistanceRatio = 0.95;

    for (int i = 0; i < matches.size(); ++i){
        if (matches[i][0].distance < descriptorDistanceRatio * matches[i][1].distance){
            good_matches.push_back(matches[i][0]);
        } 
    }

    myfile << good_matches.size() << ',';

    cout << "good_matches : " << good_matches.size() << endl;
   

    if(good_matches.size() < 12){
        cout << "Too less good matches. Nothing can be said." << endl;
        myfile << ',' << "-1" << endl;
        return -1;
    }

     std::vector<Point2f> obj,scene,proj;

    for(size_t i = 0; i < good_matches.size(); ++i){
        obj.push_back(objXY[good_matches[i].queryIdx]);
        scene.push_back(sceneXY[good_matches[i].trainIdx]); 
    }
    double tolerance =  max(1,max(img_scene.rows,img_scene.cols)/8);

    Mat H = findHomography(obj, scene, CV_RANSAC,tolerance);

    if (H.empty()){
        cout << "Homography Matrix is empty. Ratio is zero. There is change." << endl;
        myfile << ',' << '0' << endl;
    
        return 0;
    }
    perspectiveTransform(obj,proj,H);


    std::vector<DMatch> very_good_matches;
    
    double tolerance_2 = max(1,max(img_object.rows,img_object.cols)/8);


    //cout << img_scene.rows << " " << img_scene.cols << endl;
    //cout << img_object.rows << " " << img_object.cols << endl;
    cout << "tolerance :" << tolerance << endl;
    cout << "tolerance_2 : " << tolerance_2 << endl;

    double mean_x = 0.0;
    double mean_y = 0.0;

    for(size_t i = 0; i < proj.size(); i++){

        double distObjScene = pointDistance(scene[i],proj[i],0.0,0.0);
    
        if(distObjScene <= pow(tolerance,2)){
            {
                mean_x = (mean_x*i + (proj[i].x - obj[i].x))/(i+1);
                mean_y = (mean_y*i + (proj[i].y - obj[i].y))/(i+1);
            }
        }
    } 

    for(size_t i = 0; i < proj.size(); i++){

        double distObjScene = pointDistance(scene[i],proj[i],0.0,0.0);
    
        if(distObjScene <= pow(tolerance,2)){
            {                
                double distObjProj = pointDistance(obj[i],proj[i],mean_x,mean_y);
                if(distObjProj <= pow(tolerance_2,2)){
                     very_good_matches.push_back(good_matches[i]);
                }
            }
        }
    } 

    myfile << very_good_matches.size() << ','; 

    cout << "very good matches : " << very_good_matches.size() << endl;

   

    if(very_good_matches.size() == 0){
        cout << "no very good matches. Terminating compare. There is change." << endl;
        myfile << '0' << endl;
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

   
    string write_patch = videoFold + "/" + std::to_string(imgCount) + "/" + std::to_string(imgCount) + ".jpg";
    imwrite(write_patch,img_matches);


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

    float ratio = ((float)unique_matches.size()/(float)good_matches.size());

    myfile << ratio << endl;

    return ratio;

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
    //std::cout<<hogWidth<<" "<<hogHeight<<" "<<hogDimension<<"\n";
     int windowForm = (windowSize - 1) / 2; //assertion required
     bool found;
     std::vector<Point2f> nonmatchesHOG;
     int count = 0;

    for(int i = (0 + (yslide + windowForm)); i < (hogHeight - (yslide + windowForm)); i++){
        for(int j = (0 + (xslide + windowForm)); j < (hogWidth - (xslide + windowForm)); j++){

            found = false;
            //planogram image window
            std::vector<float> window1;
            float dimensionSum;
            float dimensionAvg;
            int numWindows;
            for(int m = 0; m < hogDimension; m++){
                dimensionSum = 0;
                numWindows = 0;
                for(int k = -windowForm; k <= windowForm; k++){
                    for(int l = -windowForm; l <= windowForm; l++){
                        dimensionSum += planoHog[i+k][j+l][m];
                        numWindows++;

                    }
                }
                dimensionAvg = dimensionSum / (float)numWindows;
                window1.push_back(dimensionAvg);
            }

            //cout << window1.size() << endl;

            //query image window
            for(int a = (i - yslide); a <= (i + yslide); a++){
                for(int b = (j - xslide); b <= (j + xslide); b++){

                    std::vector<float> window2;

                    for(int z = 0; z < hogDimension; z++){
                       dimensionSum = 0;
                       numWindows = 0;
                       for(int x = -windowForm; x <= windowForm; x++){
                          for(int y = -windowForm; y <= windowForm; y++){
                              dimensionSum += queryHog[a+x][b+y][z];
                              numWindows++;
                            }
                        }
                    dimensionAvg = dimensionSum / (float)numWindows;
                    //cout << "dimensionAvg : " << dimensionAvg << endl;
                    window2.push_back(dimensionAvg);
                    }
            //cout << window2.size() << endl;


              

                    if(eucledianNorm(window1) < 1e-8 && eucledianNorm(window2) < 1e-8){
                        found = true;
                        break;
                    }


                    //double a = 1.0-windowDistanceNorm(window1,window2);

                    double a = absoluteDistance(window1,window2);

                    //cout << "histogram distance : " << a << endl;
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

double absoluteDistance(std::vector<float> window1,std::vector<float> window2){

    double sum = 0.0;
    for(int k = 0; k < window1.size(); k++){
        double diff = abs(window1.at(k) - window2.at(k));
        sum += diff;
       
    }
    sum = sum / window1.size();
    //cout << "distance : " << sum << endl;
    return sum;
}

double pointDistance(Point2f a, Point2f b, double dis_x, double dis_y){
        double sumX = pow((a.x + dis_x - b.x),2);
        double sumY = pow((a.y + dis_y - b.y),2);
        double dist = sumX + sumY;

        return dist;
}

std::vector<string> getDirElements(string dir)
{
    std::vector<string> files;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if(string(dirp->d_name) != "." && string(dirp->d_name) != "..")
            files.push_back(string(dirp->d_name));
    }

    closedir(dp);
    return files;
}

void makeAfolder(string folderName){
    struct stat sb;
    char folderNameChar[1024];
    strcpy(folderNameChar, folderName.c_str());

    if((stat(folderNameChar, &sb) == 0 && S_ISDIR(sb.st_mode)) == false){
            mkdir(folderNameChar,ACCESSPERMS);
    }

}












