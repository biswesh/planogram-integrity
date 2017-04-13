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
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include "opencv2/videoio.hpp"
#include <raspicam/raspicam_cv.h>
#include <ctime>
#include <unistd.h>


#define PLANO_FROM_CAM

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

extern "C" {
    #include <vl/hog.h>
    #include <vl/mathop.h>
}



double eucledianNorm(std::vector<float>);
cv:: Mat resizeAndRescale(cv::Mat, float);
vector<vector<vector<float> > > extractHOG(cv::Mat, vl_size&, vl_size&, vl_size&);
double windowDistance(std::vector<float>,std::vector<float>);
double windowDistanceNorm(std::vector<float>,std::vector<float>);
std::vector<Point2f> formWindowAndCompare(vector<vector<vector<float> > >, vector<vector<vector<float> > >, 
    int, int, int, vl_size, vl_size, vl_size, float);
double pointDistance(Point2f,Point2f);
vector<string> getDirElements(string);
void makeAfolder(string);
std::string getDateTimeStamp();
int showNonMatches(cv::Mat, cv::Mat, vector<Point2f>, float, string, string,ofstream&,string);
float ransacCompare(cv:: Mat, cv:: Mat,int,ofstream&,int,int,string);



int main(int argc, char ** argv){

//PARAMETER CHANGE
 
    float ImgResize = atof(argv[2]);
    int windowSize = 3;
    int yslide = atoi(argv[4]);
    int xslide = atoi(argv[3]);
    float hogThreshold = atof(argv[1]);
    
    double framesToSkip = 100; 
    unsigned int delayForQuery = 10; //in seconds
    
    raspicam::RaspiCam_Cv Camera;    
    cout<<"Opening Camera..."<<endl;
    
    if (!Camera.open()) {
		cerr<<"Error opening the camera"<<endl;
		return -1;
		}

    string destDir = string("../testing/video-data-results/");
    string planoFolder = destDir + "/planogram/";
    makeAfolder(planoFolder);
    string queryFolder = destDir + "/query/";
    makeAfolder(queryFolder);


    ofstream myfile;
    myfile.open("../stats/reports.csv");
    myfile << "instance_name,patch_name,connected_component_area,bounding_box_area,initial_plano_featues,initial_query_features,good_matches,very_good_matches,ratio" << endl;

    cout << "entering capture mode ... " << endl;
    
    Camera.set(CV_CAP_PROP_POS_FRAMES,framesToSkip); 
    
  
    
    Mat planogramSrc,querySrc,frame;
    
    
    string tStampPlano = getDateTimeStamp();
    
    #ifdef PLANO_FROM_CAM
    cout << "capturing planogram Image from cam..." << endl;
    Camera.grab();
    Camera.retrieve(frame);
    planogramSrc = frame.clone();
    #else
    cout << "collecting planogram Image from disk..." << endl;
    vector<string> planoFiles = vector<string>();
    planoFiles = getDirElements(planoFolder);
    cout << planoFiles[0] << endl;
    string planoPath = planoFolder + '/' + planoFiles[0];
    planogramSrc = imread(planoPath);
    #endif
    
    cvtColor(planogramSrc, planogramSrc, CV_BGR2GRAY);
    string planoFile = planoFolder + "plano_" + tStampPlano + ".jpg";
    imwrite(planoFile,planogramSrc);        

    Mat planogramImg = resizeAndRescale(planogramSrc,ImgResize);

    vl_size hogWidth, hogHeight, hogDimension;
    vector<vector<vector<float> > > planoVec;
    planoVec = extractHOG(planogramImg,hogWidth,hogHeight,hogDimension);

    while(1){
		cout << "entering sleep mode..." << endl;
	    usleep(delayForQuery * 1000 * 1000);
	    cout << "Capture resumed..." << endl;
	    
	    string tStamp = getDateTimeStamp();
	    string instanceName = "instance_" + tStamp;
        string destSubDirPath = queryFolder + instanceName + '/'; 
        makeAfolder(destSubDirPath);
        string planoInstance = destSubDirPath + "plano_" + tStampPlano + ".jpg";
        imwrite(planoInstance,planogramSrc);
        
        cout << "capturing query image..." << endl;

        Camera.grab();
        Camera.retrieve(frame);

        if(frame.empty()){
             cout << "frame is empty" << endl; 
             break; 
        }

        querySrc = frame.clone();
        cvtColor(querySrc, querySrc, CV_BGR2GRAY);
        Mat queryImg = resizeAndRescale(querySrc,ImgResize); 
        vector<vector<vector<float> > > queryVec;
        queryVec = extractHOG(queryImg,hogWidth,hogHeight,hogDimension);

        vector<Point2f> nonMatchVector = formWindowAndCompare(planoVec,queryVec,windowSize,xslide,yslide,
                hogWidth,hogHeight,hogDimension,hogThreshold);


        int integrity = showNonMatches(planogramSrc,querySrc,nonMatchVector,ImgResize,destSubDirPath,tStamp,myfile,instanceName); 

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
        }
      
    myfile.close();
    Camera.release();

    return 0;
}



int showNonMatches(cv::Mat imgMatRaw, cv::Mat imgMatRaw_2, vector<Point2f> nonMatchVector, 
    float ImgResize, string destSubDirPath, string tStamp, ofstream& myfile, string instanceName){

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
    Mat labels,stats,centroids;

    int a = connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);

    float rescale = 1.0 / ImgResize;
 
    bool omit;
    int planogramIntegrity = 1;

    for(int i = 1; i < stats.rows; i++){

        omit = false;

        int hogArea = stats.at<int>(i,4);
        int Area = (stats.at<int>(i,2) * stats.at<int>(i,3));
        float hogAreaRatio = (float)hogArea/float(Area);


        if(hogArea > 36000){
            //cout << "unusually big blob detected. omitting frame" << endl;
            omit = true;
            break;
        }

        if(hogArea > 700 && hogArea < 26000 && hogAreaRatio > 0.20){
           
            Mat ransacPlano = imgMatRaw(Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1), 
                rescale * stats.at<int>(i,2),rescale * stats.at<int>(i,3))).clone();

            Mat ransacQuery = imgMatRaw_2(Rect(rescale * stats.at<int>(i,0), rescale * stats.at<int>(i,1),
                rescale * stats.at<int>(i,2), rescale * stats.at<int>(i,3))).clone();

            float ratio = ransacCompare(ransacPlano,ransacQuery,i,myfile,Area,hogArea,instanceName); 

            //cout << "the ratio of cropped image : " << ratio << endl;


            if (ratio == -1){
                continue;
            }
            else if (ratio > 0.7)
                continue;
            else{
                planogramIntegrity = 0;
                rectangle(imgMat,Point(stats.at<int>(i,0),stats.at<int>(i,1)),
                    Point((stats.at<int>(i,0) + stats.at<int>(i,2)),(stats.at<int>(i,1) + stats.at<int>(i,3))),
                    Scalar(255,255,255),
                    5,
                    LINE_8,
                    0);
            }
        }
     
    } 

    string write_final = destSubDirPath + "/query_" + tStamp + ".jpg";
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


}

float ransacCompare(cv:: Mat img_object, cv:: Mat img_scene,int patchCount,ofstream& myfile,int Area,int hogArea,string instanceName){

    myfile << instanceName << ','  << patchCount << ',' << hogArea << ',' << Area << ',';


    if(!img_object.data || !img_scene.data){
        cout << "There's some problem in reading images" << endl;
        return -1;
    }

    Mat descriptors_object, descriptors_scene;
    std::vector<KeyPoint> keypoints_object,keypoints_scene;

    Ptr<FeatureDetector> detector = ORB::create( 
        1000,    
        1.2f,
        8,
        31,
        0,
        2,
        ORB::HARRIS_SCORE,
        31,
        12);
    detector->detect(img_object, keypoints_object);
    detector->detect(img_scene, keypoints_scene);
    
    Ptr<DescriptorExtractor> descriptorExtractor = ORB::create(
        1000,    
        1.2f,
        8,
        31,
        0,
        2,
        ORB::HARRIS_SCORE,
        31,
        12);
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
    
    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<DMatch> good_matches;

    cv::BFMatcher matcher(NORM_L2, false);
    
    myfile << objXY.size() << ',' << sceneXY.size() << ',';

    if(objXY.size() < 6){
        //cout << "no or very few features can be found in either the planogram or the query" << endl;

        myfile << ',' << ',' << "-1" << endl;
      
        return -1;  
    }

    if(sceneXY.size() == 0){
        //cout << "no points could be found in query image. There is change" << endl;
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
   

    if(good_matches.size() < 4){
        //cout << "The size of second nearest neighbour filtered matches is less than 4. Terminating Compare." << endl;

        myfile << ',' << '0' << endl;
      
        return 0;
    }

     std::vector<Point2f> obj,scene,proj;

    for(size_t i = 0; i < good_matches.size(); ++i){
        obj.push_back(objXY[good_matches[i].queryIdx]);
        scene.push_back(sceneXY[good_matches[i].trainIdx]); 
    }

    Mat H = findHomography(obj, scene, CV_RANSAC);

    if (H.empty()){
        //cout << "Homography Matrix is empty. Ratio is zero. There is change." << endl;
        myfile << ',' << '0' << endl;
    
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

    myfile << very_good_matches.size() << ','; 

   

    if(very_good_matches.size() == 0){
        //cout << "no very good matches. Terminating compare. There is change." << endl;
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


                    double a = 1.0-windowDistanceNorm(window1,window2);

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

double pointDistance(Point2f a, Point2f b){
        double sumX = pow((a.x - b.x),2);
        double sumY = pow((a.y - b.y),2);
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

string getDateTimeStamp(){
	
	time_t t = time(0);
	struct tm * now = localtime(&t);
	string dateTime = std::to_string(now->tm_year + 1900) + '-' + std::to_string(now->tm_mon + 1) + '-' + std::to_string(now->tm_mday)
	      + '_' + std::to_string(now->tm_hour) + ':' + std::to_string(now->tm_min) + ':' + std::to_string(now->tm_sec);	
	return dateTime;
	}











