#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <typeinfo>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace std;

#define PI 3.14159265
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

void easyAdd()
{
  Mat a1 = Mat::zeros(5,5, CV_8UC1);
  Mat a2 = Mat::eye(5,5, CV_8UC1);
  Mat a3 = Mat::zeros(5,5, CV_8UC1);
  Mat mask;
  a3 = a1+a2*10;
  mask = a3 == 0;
  cout << a3 <<endl;
  cout << mask <<endl;
  cout << sum(a3)[0] << endl;

  Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
                                 -1,  5, -1,
                                  0, -1,  0);
  Mat kern_mask = kern > 0;
  Mat output;
  kern.copyTo(output,kern_mask);
  cout << kern << endl;
  cout << kern_mask << endl;
  cout << output << endl;

  //cout << "type of sum(a3) " << typeid(sum(a3)[0]).name() <<endl;
  //cout << mean(a3.col(1)) <<endl;
}

void minMaxid()
{
  Mat m = Mat::zeros(5,5,CV_8UC1);
  double minVal; 
  double maxVal; 
  Point minLoc; 
  Point maxLoc;

  int ctr = 0;
  for(int i=0;i<5;i++)
  {
    for(int j=0;j<5;j++)
    {
      m.at<uchar>(i,j) += ctr;
      ctr++;
    }
  }

  minMaxLoc( m, &minVal, &maxVal, &minLoc, &maxLoc );

  cout << "m: " << endl << m <<endl;
  cout << "min val : " << minVal << endl;
  cout << "max val: " << maxVal << endl;
  cout << "minLoc: " << minLoc.x << minLoc.y << endl;
  cout << "maxLoc: " << maxLoc.x << maxLoc.y << endl;
  cout << (int)m.at<uchar>(maxLoc.x,maxLoc.y) << endl;

  //double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type)
}

void simulation()
{
  Mat matx = (Mat_<char>(5,5) <<  0, 1, 2, 1, 0,
                                  1, 2, 3, 2, 1,
                                  2, 3, 4, 4, 3,
                                  1, 2, 3, 2, 1,
                                  0, 1, 2, 1, 0);
  Mat mask = matx >= 2;
  Mat output;
  matx.copyTo(output,mask);
  cout << output << endl;

  int sumR=0,sumC=0,ctr=0,centerR=0,centerC=0;
  //cout << output.rows<<endl;
  //cout << output.cols<<endl;
  for (int i=0;i<output.rows;i++)
  {
    for (int j=0;j<output.cols;j++)
    {
      if ((int)output.at<char>(i,j) != 0){
        cout << (int)output.at<char>(i,j) << " ";
        sumR += i;
        sumC += j;
        ctr += 1;
      }
    }
    cout << endl;
  }

  centerR = sumR / ctr;
  centerC = sumC / ctr;
  //cout << centerR <<endl;
  //cout << centerC <<endl;

  output.at<char>(centerR,centerC) = 100;
  cout << output << endl;
}

void log()
{
  fstream fp;
  fp.open ("walabot.log",ios::out|ios::app);
  fp << "Writing this to a file.\n";
  fp.close();
}

double fp()
{
  int a=1,b=21;
  double c = (double)a/21;
  return c;
}


int main()
{
  //easyAdd();
  //minMaxid();
  //simulation();
  //log(); 
  cout << fp()<<endl;
  return 0;
}

