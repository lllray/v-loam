#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include "cameraParameters.h"
#include "pointDefinition.h"

using namespace std;
using namespace cv;

bool systemInited = false;
double timeCur, timeLast;

const int imagePixelNum = imageHeight * imageWidth;
CvSize imgSize = cvSize(imageWidth, imageHeight);

IplImage *imageCur = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *imageLast = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);

int showCount = 0;
const int showSkipNum = 2;
const int showDSRate = 2;
CvSize showSize = cvSize(imageWidth / showDSRate, imageHeight / showDSRate);

IplImage *imageShow = cvCreateImage(showSize, IPL_DEPTH_8U, 1);
IplImage *harrisLast = cvCreateImage(showSize, IPL_DEPTH_32F, 1);

CvMat kMat = cvMat(3, 3, CV_64FC1, kImage);
CvMat dMat = cvMat(4, 1, CV_64FC1, dImage);

IplImage *mapx, *mapy;

const int maxFeatureNumPerSubregion = 2;
const int xSubregionNum = 12;
const int ySubregionNum = 8;
const int totalSubregionNum = xSubregionNum * ySubregionNum;
const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum;

const int xBoundary = 20;
const int yBoundary = 20;
const double subregionWidth = (double)(imageWidth - 2 * xBoundary) / (double)xSubregionNum;
const double subregionHeight = (double)(imageHeight - 2 * yBoundary) / (double)ySubregionNum;

const double maxTrackDis = 100;
const int winSize = 15;

IplImage *imageEig, *imageTmp, *pyrCur, *pyrLast;

CvPoint2D32f *featuresCur = new CvPoint2D32f[2 * MAXFEATURENUM];
CvPoint2D32f *featuresLast = new CvPoint2D32f[2 * MAXFEATURENUM];
char featuresFound[2 * MAXFEATURENUM];
float featuresError[2 * MAXFEATURENUM];

int featuresIndFromStart = 0;
//!maxFeatureNumPerSubregion=2
int featuresInd[2 * MAXFEATURENUM] = {0};

int totalFeatureNum = 0;
int subregionFeatureNum[2 * totalSubregionNum] = {0};

pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imageShowPubPointer;
cv_bridge::CvImage bridge;

void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
{
  timeLast = timeCur;
  timeCur = imageData->header.stamp.toSec() - 0.1163;

  IplImage *imageTemp = imageLast;
  imageLast = imageCur;
  imageCur = imageTemp;

  for (int i = 0; i < imagePixelNum; i++) {
    imageCur->imageData[i] = (char)imageData->data[i];
  }

  IplImage *t = cvCloneImage(imageCur);
  //!去除图像畸变
  cvRemap(t, imageCur, mapx, mapy);
  //cvEqualizeHist(imageCur, imageCur);
  cvReleaseImage(&t);
  //!缩小一点可能角点检测速度比较快
  cvResize(imageLast, imageShow);
  cvCornerHarris(imageShow, harrisLast, 3);

  CvPoint2D32f *featuresTemp = featuresLast;
  featuresLast = featuresCur;
  featuresCur = featuresTemp;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
  imagePointsLast = imagePointsCur;
  imagePointsCur = imagePointsTemp;
  imagePointsCur->clear();

  if (!systemInited) {
    systemInited = true;
    return;
  }

  int recordFeatureNum = totalFeatureNum;
  for (int i = 0; i < ySubregionNum; i++) {
    for (int j = 0; j < xSubregionNum; j++) {
      //!ind指向当前的subregion编号
      int ind = xSubregionNum * i + j;
      int numToFind = maxFeatureNumPerSubregion - subregionFeatureNum[ind];

      if (numToFind > 0) {
        int subregionLeft = xBoundary + (int)(subregionWidth * j);
        int subregionTop = yBoundary + (int)(subregionHeight * i);
        //!将当前的subregion框选出来
        CvRect subregion = cvRect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);
        cvSetImageROI(imageLast, subregion);

        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
                              &numToFind, 0.1, 5.0, NULL, 3, 1, 0.04);

        int numFound = 0;
        for(int k = 0; k < numToFind; k++) {
          featuresLast[totalFeatureNum + k].x += subregionLeft;
          featuresLast[totalFeatureNum + k].y += subregionTop;

          int xInd = (featuresLast[totalFeatureNum + k].x + 0.5) / showDSRate;
          int yInd = (featuresLast[totalFeatureNum + k].y + 0.5) / showDSRate;
          //!查看检测的角点中是否有匹配到的合适的特征点
          if (((float*)(harrisLast->imageData + harrisLast->widthStep * yInd))[xInd] > 1e-7) {
            featuresLast[totalFeatureNum + numFound].x = featuresLast[totalFeatureNum + k].x;
            featuresLast[totalFeatureNum + numFound].y = featuresLast[totalFeatureNum + k].y;
            featuresInd[totalFeatureNum + numFound] = featuresIndFromStart;

            numFound++;
            featuresIndFromStart++;
          }
        }
        totalFeatureNum += numFound;
        subregionFeatureNum[ind] += numFound;

        cvResetImageROI(imageLast);
      }
    }
  }

  cvCalcOpticalFlowPyrLK(imageLast, imageCur, pyrLast, pyrCur,
                         featuresLast, featuresCur, totalFeatureNum, cvSize(winSize, winSize), 
                         3, featuresFound, featuresError, 
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);

  for (int i = 0; i < totalSubregionNum; i++) {
    subregionFeatureNum[i] = 0;
  }

  ImagePoint point;
  int featureCount = 0;
  double meanShiftX = 0, meanShiftY = 0;
  for (int i = 0; i < totalFeatureNum; i++) {
    double trackDis = sqrt((featuresLast[i].x - featuresCur[i].x) 
                    * (featuresLast[i].x - featuresCur[i].x)
                    + (featuresLast[i].y - featuresCur[i].y) 
                    * (featuresLast[i].y - featuresCur[i].y));

    if (!(trackDis > maxTrackDis || featuresCur[i].x < xBoundary || 
      featuresCur[i].x > imageWidth - xBoundary || featuresCur[i].y < yBoundary || 
      featuresCur[i].y > imageHeight - yBoundary)) {
      //!计算当前特征点是哪个subregion中检测到的，ind是subregion的编号
      int xInd = (int)((featuresLast[i].x - xBoundary) / subregionWidth);
      int yInd = (int)((featuresLast[i].y - yBoundary) / subregionHeight);
      int ind = xSubregionNum * yInd + xInd;

      if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) {
        //!根据筛选准则将光流法匹配到的特征点进行筛选,这里featureCount是从0开始的，
        //!所以featuresCur[]和featuresLast[]只保存了邻近图像的特征点，很久之前的没有保存
        featuresCur[featureCount].x = featuresCur[i].x;
        featuresCur[featureCount].y = featuresCur[i].y;
        featuresLast[featureCount].x = featuresLast[i].x;
        featuresLast[featureCount].y = featuresLast[i].y;
        //!有些特征点被筛掉，所以这里featureCount不一定和i相等
        featuresInd[featureCount] = featuresInd[i];
        /*! 这一步将图像坐标系下的特征点[u,v]，变换到了相机坐标系下，即[u,v]->[X/Z,Y/Z,1],参考《14讲》式5.5
        * 不过要注意这里加了个负号。相机坐标系默认是z轴向前，x轴向右，y轴向下，图像坐标系默认在图像的左上角，
        * featuresCur[featureCount].x - kImage[2]先将图像坐标系从左上角还原到图像中心，然后加个负号，
        * 即将默认相机坐标系的x轴负方向作为正方向，y轴同理。所以此时相机坐标系z轴向前，x轴向左，y轴向上
        */
        point.u = -(featuresCur[featureCount].x - kImage[2]) / kImage[0];
        point.v = -(featuresCur[featureCount].y - kImage[5]) / kImage[4];
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);

        if (i >= recordFeatureNum) {
          point.u = -(featuresLast[featureCount].x - kImage[2]) / kImage[0];
          point.v = -(featuresLast[featureCount].y - kImage[5]) / kImage[4];
          imagePointsLast->push_back(point);
        }

        meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        //!subregionFeatureNum是根据当前帧与上一帧的特征点匹配数目来计数的
        subregionFeatureNum[ind]++;
      }
    }
  }
  totalFeatureNum = featureCount;
  meanShiftX /= totalFeatureNum;
  meanShiftY /= totalFeatureNum;

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);
  //!隔两张图像才输出一副图像，如0,1不要，2输出，3,4不要，5输出
  showCount = (showCount + 1) % (showSkipNum + 1);
  if (showCount == showSkipNum) {
    Mat imageShowMat(imageShow);
    bridge.image = imageShowMat;
    bridge.encoding = "mono8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPubPointer->publish(imageShowPointer);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "featureTracking");
  ros::NodeHandle nh;

  mapx = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  mapy = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  cvInitUndistortMap(&kMat, &dMat, mapx, mapy);

  CvSize subregionSize = cvSize((int)subregionWidth, (int)subregionHeight);
  imageEig = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);
  imageTmp = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);

  CvSize pyrSize = cvSize(imageWidth + 8, imageHeight / 3);
  pyrCur = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  pyrLast = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);

  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/raw", 1, imageDataHandler);

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::spin();

  return 0;
}
