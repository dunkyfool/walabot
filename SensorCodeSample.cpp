#include <WalabotAPI.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <typeinfo>
#include <math.h>
#include <ctime>
#include "sstream"
#include "string"
#include "iomanip"
#include "typeinfo"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace std;

#define PI 3.14159265
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

string _name = "plastic_100_40_0_v1";
string path = "data/"+_name+"/";
string _path = "data/";
string _logname = "data/"+_name+".log";
const char *logname = _logname.c_str();

void PrintSensorTargets(SensorTarget* targets, int numTargets)
{
	int targetIdx;

//    #ifdef __LINUX__
//		printf("\033[2J\033[1;1H");
//	#else
//	    system("cls");
//	#endif
	if (numTargets > 0)
	{
		for (targetIdx = 0; targetIdx < numTargets; targetIdx++)
		{
			printf("Target #%d: \nX = %lf \nY = %lf \nZ = %lf \namplitude = %lf\n\n\n ",
				targetIdx,
				targets[targetIdx].xPosCm,
				targets[targetIdx].yPosCm,
				targets[targetIdx].zPosCm,
				targets[targetIdx].amplitude);
		}
	}
	else
	{
		printf("No target detected\n");
	}
	//sleep(1);
}

void showRTheta(int* rasterImage, int sizeX, int sizeY, int width, int height, Mat* new_frame)
{
	namedWindow("SHOW1",1);
	int bound = sizeX * sizeY;
	double ratioX = width/sizeX;
	double ratioY = height/sizeY;

	Mat frame = Mat::zeros(sizeY,sizeX, CV_8UC1);
		
        //GetRawImageSlice phi-R output image
	for (int i=0,j=0;i<bound;i++)
	{
	  int idx = i % sizeX;
	  j = i / sizeX;
	  frame.at<uchar>(j,idx) = rasterImage[i];
	}
	resize(frame,*new_frame,(*new_frame).size(),ratioY,ratioX,INTER_NEAREST);
	imshow("SHOW1", *new_frame);
}

void showThetaPhi(int* rasterImage2, int sizeX2, int sizeY2, int sizeZ2, int width2, int height2, Mat* new_frame2)
{
	namedWindow("SHOW2",1);
	int bound2 = sizeX2 * sizeY2 * sizeZ2;
	double ratioX2 = width2/sizeX2;
	double ratioY2 = height2/sizeY2;

	Mat frame2 = Mat::zeros(sizeY2,sizeX2, CV_8UC1);

	//GetRawImage output theta-phi mapping image
	///*
	for (int i=0,j=0;i<bound2;i++)
	{
	  int idx = i % sizeX2;
	  j = i / sizeX2;
	  j = j % sizeY2;
	  //cout << "j: "<< j<<endl;
	  //cout << "idx: "<< idx<<endl;
	  frame2.at<uchar>(j,idx) += rasterImage2[i];
	  //cout << rasterImage2[i] <<" ";
	  //if((i+1)%sizeX2==0) cout<<endl;
	}
	frame2.mul(0.00390625);	// value down to 0-255
	resize(frame2,*new_frame2,(*new_frame2).size(),ratioY2,ratioX2,INTER_NEAREST);
	imshow("SHOW2", *new_frame2);
	//*/
}

void showXY(int* rasterImage2, double maxInCm, double minInCm, double resICm, double maxIndegrees, double minIndegrees, double resIndegrees, double maxPhiInDegrees, double minPhiInDegrees, double resPhiInDegrees, int sizeX2, int sizeY2, int sizeZ2, int width2, int height2, Mat* new_frame3)
{
	namedWindow("SHOW3",1);
	int bound2 = sizeX2 * sizeY2 * sizeZ2;
        int max_X = 0;
    	int max_Y = 0;
    	int min_X = 9999;
    	int min_Y = 9999;
	//int rangeX = int(maxInCm*sin(maxIndegrees*PI/180)) - int(maxInCm*sin(minIndegrees*PI/180)) +1;
	//int rangeY = int(maxInCm*sin(maxPhiInDegrees*PI/180)) - int(maxInCm*sin(minPhiInDegrees*PI/180)) +1;

	int rangeX = round(maxInCm*sin(maxIndegrees*PI/180)) - round(maxInCm*sin(minIndegrees*PI/180)) +1;
	int rangeY = round(maxInCm*sin(maxPhiInDegrees*PI/180)) - round(maxInCm*sin(minPhiInDegrees*PI/180)) +1;

	//cout << "rangeX: " << rangeX <<endl;
	//cout << "rangeY: " << rangeY <<endl;
	//cout << round(maxInCm*sin(maxIndegrees*PI/180)) <<endl;
	//cout << round(maxInCm*sin(minIndegrees*PI/180)) <<endl;
	//cout << round(maxInCm*sin(maxPhiInDegrees*PI/180)) <<endl;
	//cout << round(maxInCm*sin(minPhiInDegrees*PI/180)) <<endl;
	//PAUSE 
	double ratioX3 = width2/rangeX;
	double ratioY3 = height2/rangeY;
	Mat frame3 = Mat::zeros(rangeY,rangeX, CV_8UC1);
		
        //GetRawImage output x-y mapping image
	for (int i=0,j=0,k=0;i<bound2;i++)
	{
	  int idx = i % sizeX2;
	  j = (i / sizeX2) %sizeY2;
	  k = (i / sizeX2) /sizeY2;
	  double R = minInCm + k * resICm;
	  double T = minIndegrees + j * resIndegrees;
	  double P = minPhiInDegrees + idx * resPhiInDegrees;

	  //cout << "P: " << P <<endl;

	  int X = int(R * sin(PI*T/180));
	  int Y = int(R * cos(PI*T/180) * sin(PI*P/180));

	  // -R ~ R => 0 ~ 2R
	  X += (rangeX / 2);
	  Y += (rangeY / 2);

	  if (X>max_X) max_X = X;
	  if (Y>max_Y) max_Y = Y;
	  if (X<min_X) min_X = X;
	  if (Y<min_Y) min_Y = Y;

	  //cout << "type of X: " << typeid(X).name() <<endl;
	  //cout << "type of Y: " << typeid(Y).name() <<endl;
	  //PAUSE

	  frame3.at<uchar>(Y,X) += rasterImage2[i];
	}
	frame3.mul(0.00390625);	// value down to 0-255
	resize(frame3,*new_frame3,(*new_frame3).size(),ratioY3,ratioX3,INTER_NEAREST);
	imshow("SHOW3", *new_frame3);
	/*
	cout << "rangeX: " << rangeX <<endl;
	cout << "rangeY: " << rangeY <<endl;
	cout << "max_X: " << max_X <<endl;
	cout << "min_X: " << min_X <<endl;
	cout << "max_Y: " << max_Y <<endl;
	cout << "min_Y: " << min_Y <<endl;
	cout <<endl;
        */
}

void saveSignal(int *ctr)
{
    // Get all pair
	WALABOT_RESULT res;
	AntennaPair * antennaPairs;
	int numPairs;
	res = Walabot_GetAntennaPairs(&antennaPairs, &numPairs);
	assert(res == WALABOT_SUCCESS);

    // file name
	for (int i=0;i<numPairs;i++)
	{
	  int tx = (*(antennaPairs+i)).txAntenna;
	  int rx =  (*(antennaPairs+i)).rxAntenna;

      if (tx ==1)
      {
        double* signal;
        double* timeAxis;
        int 	numSamples;
        res = Walabot_GetSignal(tx,rx,&signal,&timeAxis,&numSamples);
        assert(res == WALABOT_SUCCESS);

        stringstream buff;
        string pair;
        buff << "antenna/" << tx << "_" << rx << ".txt";
        buff >> pair;
        buff.clear();
        const char *pairName = pair.c_str();


        fstream fp;
        fp.open (pairName,ios::out|ios::trunc);
        fp.close();
        fp.open (pairName,ios::out|ios::app);

        // save 2048 signal (abs)
	    for (int j=0;j<2048;j++)
        {
	      fp << *(timeAxis+j) << " " << abs(*(signal+j)) << "\n";
        }

        fp.close();
      }
	}

    // relax
    //sleep(1);
    *ctr = *ctr + 1;
}

void showSignal()
{
	WALABOT_RESULT res;
	AntennaPair * antennaPairs;
	int numPairs;
	res = Walabot_GetAntennaPairs(&antennaPairs, &numPairs);
	assert(res == WALABOT_SUCCESS);
	//cout << "numPairs: " << numPairs <<endl;

	//cout << "type of antennaPairs: "<< typeid(*(antennaPairs+1)).name() <<endl;
	//cout << "txAntenna: "<< (*(antennaPairs+1)).txAntenna<<endl;
	//cout << "rxAntenna: "<< (*(antennaPairs+1)).rxAntenna<<endl;
	//cout << "txAntenna: "<< typeid((*(antennaPairs)).txAntenna).name()<<endl;
	//cout << "rxAntenna: "<< typeid(*antennaPairs.rxAntenna).name()<<endl;

	///*
	for (int i=0;i<numPairs;i++)
	{
	  cout << "Pair-" << i+1 << ":" <<endl;
	  cout << "txAntenna: "<< (*(antennaPairs+i)).txAntenna<<endl;
	  cout << "rxAntenna: "<< (*(antennaPairs+i)).rxAntenna<<endl;
	}
	cout<<endl;
	//*/
    PAUSE;

	/*
	int zz = 0;
	zz = 0;
	cout << "Pair-" << zz+1 << ":" <<endl;
	cout << "txAntenna: "<< (*(antennaPairs+zz)).txAntenna<<endl;
	cout << "rxAntenna: "<< (*(antennaPairs+zz)).rxAntenna<<endl;
	zz = 13;
	cout << "Pair-" << zz+1 << ":" <<endl;
	cout << "txAntenna: "<< (*(antennaPairs+zz)).txAntenna<<endl;
	cout << "rxAntenna: "<< (*(antennaPairs+zz)).rxAntenna<<endl;
	zz = 24;
	cout << "Pair-" << zz+1 << ":" <<endl;
	cout << "txAntenna: "<< (*(antennaPairs+zz)).txAntenna<<endl;
	cout << "rxAntenna: "<< (*(antennaPairs+zz)).rxAntenna<<endl;
	zz = 32;
	cout << "Pair-" << zz+1 << ":" <<endl;
	cout << "txAntenna: "<< (*(antennaPairs+zz)).txAntenna<<endl;
	cout << "rxAntenna: "<< (*(antennaPairs+zz)).rxAntenna<<endl;
	*/

	//Walabot_GetSignal
	double* signal;
	double* timeAxis;
	int 	numSamples;
	res = Walabot_GetSignal(17,2,&signal,&timeAxis,&numSamples);
	assert(res == WALABOT_SUCCESS);
    /*
    fstream fp;
    fp.open ("walabot.log",ios::out|ios::app);
	for (int i=0;i<numPairs;i++)
	{
	  //cout << "Pair-" << i+1 << ":" <<endl;
	  //cout << "txAntenna: "<< (*(antennaPairs+i)).txAntenna<<endl;
	  //cout << "rxAntenna: "<< (*(antennaPairs+i)).rxAntenna<<endl;
	  res = Walabot_GetSignal((*(antennaPairs+i)).txAntenna,(*(antennaPairs+i)).rxAntenna,&signal,&timeAxis,&numSamples);
	  assert(res == WALABOT_SUCCESS);
      for (int j=0;j<numSamples;j++)
      {
        fp << *(signal+j) << " ";
      }
      fp << "\n";
    }
    fp.close();
    */
	//cout << "numSamples: " << numSamples <<endl;
	///*
	double sum = 0;
	for (int i=0;i<numSamples;i++)
	{
	  sum += abs(*(signal+i));
	  //cout << "index: " << i+1 <<endl;
	  //cout << "signal: " << *(signal+i) <<endl;
	  //cout << "timeAxis: " << *(timeAxis+i) <<endl;
	}
	cout << "SUM: " << sum<<endl;
    //*/
    //PAUSE
}

void recordVideo(VideoWriter* video, Mat* new_frame)
{
	(*video).write(*new_frame);
}

void getCenter(Mat *frame, int thres, double *x, double *y)
{
  //namedWindow("SHOW4",1);
  Mat output;
  Mat mask = (*frame) >= thres;
  //mask out output
  (*frame).copyTo(output,mask);


  //locate object 
  int sumR=0,sumC=0,ctr=0,centerR=0,centerC=0;
  for (int i=0;i<(output).rows;i++)
  {
    for (int j=0;j<(output).cols;j++)
    {
      if ((int)(output).at<uchar>(i,j) != 0){
        //cout << (int)output.at<char>(i,j) << " ";
        sumR += i;
        sumC += j;
        ctr += 1;
      }
    }
    cout << endl;
  }
  //cout << sumR << "," << sumC << "," << ctr <<endl;
  //PAUSE


  // calculate center
  if (sumR!=0 || sumC!=0)
  {
    centerR = sumR / ctr;
    centerC = sumC / ctr;
  }
  cout << "Center:(" << centerR << "," << centerC << ")" <<endl;
  *x = (double)centerR/480;
  *y = (double)centerC/640;

  // draw X
  /*
  int pad=10;
  for (int i=0; i<pad;i++)
  {
    output.at<uchar>(centerR+i,centerC+i) = 100;
    output.at<uchar>(centerR-i,centerC-i) = 100;
    output.at<uchar>(centerR+i,centerC-i) = 100;
    output.at<uchar>(centerR-i,centerC+i) = 100;
  }
  */
  //show image
  //imshow("SHOW4", output);
}

//void saveImage(Mat *frame, int num, string path)
//{
//}

void log(Mat *frame)
{
	WALABOT_RESULT res;
	AntennaPair * antennaPairs;
	int numPairs;
	res = Walabot_GetAntennaPairs(&antennaPairs, &numPairs);
	assert(res == WALABOT_SUCCESS);

	//Walabot_GetSignal
	double* signal;
	double* timeAxis;
    double x;
    double y;
	int 	numSamples;
    fstream fp;
    fp.open (logname,ios::out|ios::app);
	for (int i=0;i<numPairs;i++)
	{
	  //cout << "Pair-" << i+1 << ":" <<endl;
	  //cout << "txAntenna: "<< (*(antennaPairs+i)).txAntenna<<endl;
	  //cout << "rxAntenna: "<< (*(antennaPairs+i)).rxAntenna<<endl;
	  res = Walabot_GetSignal((*(antennaPairs+i)).txAntenna,(*(antennaPairs+i)).rxAntenna,&signal,&timeAxis,&numSamples);
	  assert(res == WALABOT_SUCCESS);
      for (int j=0;j<numSamples;j++)
      {
        fp << *(signal+j) << " ";
      }
      //fp << "\n";
    }
    fp << "\n";
    //getCenter(frame,100,&x,&y);
    //fp << x << " " << y << "\n";
    fp.close();
}

void SensorCode_SampleCode()
{
	// --------------------
	// Variable definitions
	// --------------------
	WALABOT_RESULT res;

	// Walabot_GetSensorTargets - output parameters
	SensorTarget* targets;
	int numTargets;

	// Walabot_GetStatus - output parameters
	APP_STATUS appStatus;
	double calibrationProcess; // Percentage of calibration completed, if status is STATUS_CALIBRATING

   // Walabot_GetRawImageSlice - output parameters

	int*	rasterImage;
	int		sizeX;
	int		sizeY;
	double	sliceDepth;
	double	power;

	int*	rasterImage2;
	int		sizeX2;
	int		sizeY2;
	int	sizeZ2;
	double	power2;

	// Walabot_SetArenaR - input parameters
	//last(30,100,1)
	double minInCm = 30;
	double maxInCm = 100;
	double resICm = 1;

	// Walabot_SetArenaTheta - input parameters
	//last(-30,30,2)
	double minIndegrees = -30;
	double maxIndegrees = 30;
	double resIndegrees = 2;

	// Walabot_SetArenaPhi - input parameters
	//last(-30,30,2)
	double minPhiInDegrees = -30;
	double maxPhiInDegrees = 30;
	double resPhiInDegrees = 2;

	// ----------------------
	// Sample Code Start Here
	// ----------------------

	/*
		For an image to be received by the application, the following need to happen :
		1) Connect
		2) Configure
		3) Calibrate
		4) Start
		5) Trigger
		6) Get action
		7) Stop/Disconnect
	*/

	bool mtiMode = true;


	// Configure Walabot database install location
	#ifdef __LINUX__
	    res = Walabot_SetSettingsFolder((char*)"/var/lib/walabot");
	#else
	    res = Walabot_SetSettingsFolder("C:/ProgramData/Walabot/WalabotMaker");
	#endif
	
	assert(res == WALABOT_SUCCESS);
	//cout << "0" << endl;
	

	//	1) Connect : Establish communication with Walabot.
	//	==================================================
	//res = Walabot_Connect();
	res = Walabot_ConnectAny();
	assert(res == WALABOT_SUCCESS);
	//cout << "1" << endl;

	//  2) Configure : Set scan profile and arena
	//	=========================================

	// Set Profile - to Sensor. 
	//			Walabot recording mode is configure with the following attributes:
	//			-> Distance scanning through air; 
	//			-> high-resolution images
	//			-> slower capture rate 
	res = Walabot_SetProfile(PROF_SENSOR);
	assert(res == WALABOT_SUCCESS);
	//cout << "2" << endl;

	// Setup arena - specify it by Cartesian coordinates(ranges and resolution on the x, y, z axes); 
	//	In Sensor mode there is need to specify Spherical coordinates(ranges and resolution along radial distance and Theta and Phi angles).
	res = Walabot_SetArenaR(minInCm, maxInCm, resICm);
	assert(res == WALABOT_SUCCESS);
	//cout << "3" << endl;

	// Sets polar range and resolution of arena (parameters in degrees).
	res = Walabot_SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees);
	assert(res == WALABOT_SUCCESS);
	//cout << "4" << endl;

	// Sets azimuth range and resolution of arena.(parameters in degrees).
	res = Walabot_SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees);
	assert(res == WALABOT_SUCCESS);
	//cout << "5" << endl;

	FILTER_TYPE filterType = mtiMode ?
		FILTER_TYPE_MTI :		//Moving Target Identification: standard dynamic-imaging filter
		FILTER_TYPE_NONE;

	res = Walabot_SetDynamicImageFilter(filterType);
	assert(res == WALABOT_SUCCESS);
	//cout << "6" << endl;

	//	3) Start: Start the system in preparation for scanning.
	//	=======================================================
	res = Walabot_Start();
	assert(res == WALABOT_SUCCESS);
	//cout << "7" << endl;

	//	4) Trigger: Scan(sense) according to profile and record signals to be available
	//	for processing and retrieval.
	//	================================================================================
	if (!mtiMode) // if MTI mode is not set - start calibrartion
	{
		// calibrates scanning to ignore or reduce the signals
		res = Walabot_StartCalibration();
		assert(res == WALABOT_SUCCESS);
		//cout << "8" << endl;
	}

	/*
	#################################################################
	#								#
	#			Define opencv parameter(s)		#
	#								#
	#################################################################
	*/

	bool recording = true; 	// walabot start running flag
	bool mode = false;	// record video flag
	int width = 640;	// frame-w
	int height = 480;	// frame-h
	int width2 = 640;	// frame2-w
	int height2 = 480;	// frame2-h
  	Size frameSize(width, height);
	VideoWriter video("data/Thz.mov", CV_FOURCC('m','p','4','v'),10,frameSize,false);
	Mat new_frame = Mat::zeros(height, width, CV_8UC1);
	Mat new_frame2 = Mat::zeros(height2, width2, CV_8UC1);
	Mat new_frame3 = Mat::zeros(height2, width2, CV_8UC1);
	Mat new_frame4 = Mat::zeros(height2, width2, CV_8UC1);

	if (!video.isOpened())
	{
    	//  cout << "ERROR: Fail to write the video" << endl;
	  //return -1;
	}

    // default webcam
    VideoCapture cap(0);
    //image folder path
    //string path = "data/clay_100_40_0/";
    stringstream serialNum;
    string fullPath;
	int num = 0;
    int ctr = 0;
    //cout << "Clean the scean and press R to caliberate!!"<<endl;
    //res = Walabot_GetStatus(&appStatus, &calibrationProcess);
    //assert(res == WALABOT_SUCCESS);
	while (recording)
	{
		// calibrates scanning to ignore or reduce the signals
		res = Walabot_GetStatus(&appStatus, &calibrationProcess);
		assert(res == WALABOT_SUCCESS);
		//cout << "9" << endl;

		//	5) Trigger: Scan(sense) according to profile and record signals to be 
		//	available for processing and retrieval.
		//	====================================================================
		res = Walabot_Trigger();
		assert(res == WALABOT_SUCCESS);
		//cout << "10" << endl;

		//	6) 	Get action : retrieve the last completed triggered recording 
		//	================================================================
		//res = Walabot_GetSensorTargets(&targets, &numTargets);
		//assert(res == WALABOT_SUCCESS);
		//cout << "11" << endl;

		//res = Walabot_GetRawImageSlice(&rasterImage, &sizeX, &sizeY, &sliceDepth, &power);
		//assert(res == WALABOT_SUCCESS);
		//res = Walabot_GetRawImage(&rasterImage2, &sizeX2, &sizeY2, &sizeZ2, &power2);
		//assert(res == WALABOT_SUCCESS);
		//cout << "12" << endl;

		//	******************************
		//	TODO: add processing code here
		//	******************************

		/*-
		check value (print it all out)
		*/
		/*
		cout << "sizeX2: " << sizeX2 <<endl;
		cout << "sizeY2: " << sizeY2 <<endl;
		cout << "sizeZ2: " << sizeZ2 <<endl;
		cout << "sizeX: " << sizeX <<endl;
		cout << "sizeY: " << sizeY <<endl;
		cout << "slicedepth: "<< sliceDepth <<endl;
		cout << "total: "<< sizeX2*sizeY2*sizeZ2 <<endl;
		*/
		//cout << "type of rasterImage: " << typeid(rasterImage).name() <<endl;
		//cout << "type of sizeX: " << typeid(sizeX).name() <<endl;
		//cout << "type of sizeY: " << typeid(sizeY).name() <<endl;
		//cout << "type of sliceDepth: " << typeid(sliceDepth).name() <<endl;

        
		//showRTheta(rasterImage,sizeX,sizeY,width,height,&new_frame);
		//showThetaPhi(rasterImage2,sizeX2,sizeY2,sizeZ2,width2,height2,&new_frame2);
		//showXY(rasterImage2,maxInCm,minInCm,resICm,maxIndegrees,minIndegrees,resIndegrees,maxPhiInDegrees, minPhiInDegrees,resPhiInDegrees,sizeX2,sizeY2,sizeZ2,width2,height2,&new_frame3);
        //getCenter(&new_frame, 100);

	    // print (17,2) Signal
        //showSignal();

        saveSignal(&ctr);
        if (ctr==10) break;

        // Webcam
        //Mat webcam_frame;
        //cap >> webcam_frame;
	    //namedWindow("SHOW99",1);
	    //imshow("SHOW99", webcam_frame);


        /*
		// KEYBOARD INTERRUPT
		int key = cv::waitKey(30) & 255;
        
        clock_t begin;
        clock_t end;
		if(key == 27) break; 	//press ESC to break
		if(key == 114)		// press "R "to re-Caliberate
		{
		  cout << "Re-Caliberation!!" <<endl;
		  res = Walabot_StartCalibration();
		  assert(res == WALABOT_SUCCESS);
		}
		if(key == 115)		// press "S" to switch the mode
		{
		  mode = not mode;
		  cout << "Current mode is: " << mode<<endl;
          if (mode==true)
          {
            begin = clock();
          }
          else
          {
            end = clock();
            double t = double(end-begin)/CLOCKS_PER_SEC;
            cout << t << endl;
          }
		}
        if (key==97) // A save background image
        {
          cout << "save background image!!"<<endl;
          string tmp = _path + "background.jpg";
          imwrite(tmp,webcam_frame);
        }
		if(key != 255) cout << key <<endl;

		if(mode)
        {
          //recordVideo(&video,&new_frame);//video.write(new_frame2);
          log(&new_frame);
          num++;
          ctr++;
    	  //cout<<num<<endl;
          serialNum << path << setw(8) << setfill('0') << num << ".jpg";
          serialNum >> fullPath;
          serialNum.clear();
          //cout << fullPath;
          imwrite(fullPath,webcam_frame);
          //saveImage(&webcam_frame,num,path);
          if (ctr==20) {mode = not mode; ctr=0; cout<<"DONE"<<endl;}
        }
        */
		//PrintSensorTargets(targets, numTargets);
	}

	//	7) Stop and Disconnect.
	//	======================
	res = Walabot_Stop();
	assert(res == WALABOT_SUCCESS);
	//cout << "13" << endl;

	res = Walabot_Disconnect();
	assert(res == WALABOT_SUCCESS);
	//cout << "14" << endl;

	video.release();
}


#ifndef _SAMPLE_CODE_
int main()
{
    cout << "Start to capture information!!" << endl;
	SensorCode_SampleCode();
    cout << "Done" << endl;
}
#endif
