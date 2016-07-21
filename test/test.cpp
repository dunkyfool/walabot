//#include "/usr/local/include/opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "iostream"
#include "sstream"
#include "string"
#include "iomanip"
#include "typeinfo"

using namespace std;
using namespace cv;

int main()
{
  cout << "123123" << endl;
  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
        return -1;

  double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
  double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
  cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
  Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
  //Size frameSize(800,600);
  VideoWriter video("output.mov", CV_FOURCC('m', 'p', '4', 'v'), 10, frameSize, true);
  //VideoWriter video("output.mov", cap.get(CV_CAP_PROP_FOURCC), cap.get(CV_CAP_PROP_FPS), frameSize);

  if (!video.isOpened())
  {
    cout << "ERROR: Fail to write the video" << endl;
    return -1;
  }

  Mat edges;
  Mat gray;
  Mat background;
  Rect myROI(10, 10, 300, 300);

  int num=0;
  stringstream serialNum;
  string path="data/";
  string name;
  string fullPath;

  vector<int> compression_params; //vector that stores the compression parameters of the image
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
  compression_params.push_back(98); //specify the compression quality

  bool mode = false;
  bool crop = false;
  namedWindow("edges",1);
  for(;;)
  {
        Mat frame;
        cap >> frame; // get a new frame from camera

        // resize frame
        //resize(frame, frame, Size(800, 600));

        cvtColor(frame, edges, COLOR_BGR2GRAY);
        //GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        //Canny(edges, edges, 0, 30, 3);

        int key = cv::waitKey(30) & 255;
        if(key == 27) break; //ESC
        else if (key==114) //R
        {
          cout << "save background image!!"<<endl;
          edges.copyTo(background);
        }
        else if (key==115) //S
        {
          mode = not mode;
        }
        else if (key==97) //A
        {
          crop = not crop;
        }
        else if (key!=255)
        {
          cout << key << endl;
        }

        if (mode)
        {
          //edges -= background;
          absdiff(edges,background,edges);
          cout << mean(edges)<<endl;
          threshold( edges, edges, 20, 255,THRESH_BINARY);

          num++;
          //cout << num <<endl;
          serialNum << path << setw(8) << setfill('0') << num << ".jpg";
          //cout << serialNum.str() <<endl;
          serialNum >> fullPath;
          serialNum.clear();
          //fullPath = path + name + ".jpg";
          cout << fullPath;
          //cout << typeid(fullPath).name()<<endl;
          //cout << typeid(name).name()<<endl;
          imwrite(fullPath,edges);
        }
        if (crop)
        {
          edges = edges(myROI);
        }
        // show frame
        cvtColor(edges, gray, CV_GRAY2RGB);
        imshow("edges", gray);
        // record
        //video.write(gray);
  }
    // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
