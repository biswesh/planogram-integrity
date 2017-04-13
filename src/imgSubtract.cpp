#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap(0);
    if(!cap.open(0))
        return 0;
    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
    int num_frames = 120;
    time_t start, end;  
    Mat frame;
    time(&start);
    for(int i = 0; i < num_frames; i++){
        cap >> frame;
        if( frame.empty() ) break; 
    }
    time(&end);
    double seconds = difftime (end, start);
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;
    cap.release();

    int count;
     
    Mat frameImg_1;
    Mat frameImg_2;
    Mat frameImg_abs;

    int count = 0;
    char str[80];
    
    while(1){
          Mat frameImg;
          cap >> frameImg;
          if( frameImg.empty() ) break;
          //if(count == 1){
          //    Mat frameImg_1 = frameImg.clone();
          //}
          //if(count == 2 * fps){
          //    Mat frameImg_2 = frameImg.clone();
          //    cap.release();
          //    frameImg_abs = abs(frameImg_1,frameImg_2);
          //    return 0;
          //}
          FileName = "img_ " 
          imsave("");
              
          imshow("Image Frame", frame);
          if( waitKey(10) == 27 ) break; 
    }
    return 0;
