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
    ros::spinOnce();
   // ROS_INFO("START imageDataHandler !");
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
  //cvShowImage("imageShow",imageShow);
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
  //cvShowImage("imageLast",imageLast);
  cvWaitKey(1);
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

        /*!在这个函数中，输入图像image必须是8位或者是32位，也就是IPL_DEPTH_8U 或者是 IPL_DEPTH_32F 单通道图像。
            第二和第三个参数是大小与输入图像相同的32位单通道图像。
            参数 temp_image 和 eig_image 在计算过程中被当做临时变量使用，计算结束后eig_image中的内容是有效的。特别的，每个函数包含了输入图像中对应的最小特征值。

            corners 是函数的输出，为检测到 32位（CvPoint2D32f）的角点数组，在调用 cvGoodFeaturesToTrack 函数之前要为该数组分配内存空间。

            corner_count 表示可以返回的最大角点数目，函数调用结束后，其返回实际检测到的角点数目。
            quality_level 表示一点呗认为是角点的可接受的最小特征值，实际用于过滤角点的最小特征值是quality_level与图像汇总最大特征值的乘积，所以quality_level的值不应该超过1，通常取值为（0.10或者是0.01）

            检测完之后还要进一步剔除掉一些距离较近的角点，min_distance 保证返回的角点之间的距离不小于min_distance个像素

            mask是可选参数，是一幅像素值为boolean类型的图像，用于指定输入图像中参与角点计算的像素点，若mask的值为NULL，值表示选择整个图像

            block_size 是计算导数的自相关矩阵是指定的领域，采用小窗口计算的结果比单点（也就是block_size 为1）计算的结果要好

            函数cvGoodFeaturesToTrack() 的输出结果为需找到的角点的位置数组。
            int use_harris CV_DEFAULT(0),
            double k CV_DEFAULT(0.04) );
                                    */
//        cvShowImage("imageLast",imageLast);
//        cvWaitKey(1);
        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
                              &numToFind, 0.1, 5.0, NULL, 3, 1, 0.04);
//        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
//                               &numToFind, 0.01, 0.1, NULL, 3);
//        ROS_INFO("FeatureTracking | feature num to find:=%d",numToFind);
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
//    ROS_INFO("FeatureTracking | totalFeatureNum:=%d",totalFeatureNum);
//    ROS_INFO("FeatureTracking | featuresIndFromStart:=%d",featuresIndFromStart);
    /*!
        prev在时间 t 的第一帧
        curr
        在时间 t + dt 的第二帧
        prev_pyr
        第一帧的金字塔缓存. 如果指针非 NULL , 则缓存必须有足够的空间来存储金字塔从层 1 到层 #level 的内容。尺寸 (image_width+8)*image_height/3 比特足够了
        curr_pyr
        与 prev_pyr 类似， 用于第二帧
        prev_features
        需要发现光流的点集
        curr_features
        包含新计算出来的位置的 点集
        count
        特征点的数目
        win_size
        每个金字塔层的搜索窗口尺寸
        level
        最大的金字塔层数。如果为 0 , 不使用金字塔 (即金字塔为单层), 如果为 1 , 使用两层，下面依次类推。
        status
        数组。如果对应特征的光流被发现，数组中的每一个元素都被设置为 1， 否则设置为 0。
        error
        双精度数组，包含原始图像碎片与移动点之间的差。为可选参数，可以是 NULL .
        criteria
        准则，指定在每个金字塔层，为某点寻找光流的迭代过程的终止条件。
        flags
        其它选项：
        CV_LKFLOW_PYR_A_READY , 在调用之前，第一帧的金字塔已经准备好
        CV_LKFLOW_PYR_B_READY , 在调用之前，第二帧的金字塔已经准备好
        CV_LKFLOW_INITIAL_GUESSES , 在调用之前，数组 B 包含特征的初始坐标 （Hunnish: 在本节中没有出现数组 B，不知是指的哪一个）*/
  cvCalcOpticalFlowPyrLK(imageLast, imageCur, pyrLast, pyrCur,
                         featuresLast, featuresCur, totalFeatureNum, cvSize(winSize, winSize), 
                         3, featuresFound, featuresError, 
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);
   // ROS_INFO("FeatureTracking | totalFeatureNum:=%d",totalFeatureNum);
   //std::cout<<"FeatureTracking | featuresFound:"<<featuresFound<<endl;
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

  //lx add to remove nan
//    Preprocessing<ImagePoint> test;
//    test.removeNan(imagePointsLast);
  //end add

  sensor_msgs::PointCloud2 imagePointsLast2;

  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);

  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);
  //!隔两张图像才输出一副图像，如0,1不要，2输出，3,4不要，5输出
  showCount = (showCount + 1) % (showSkipNum + 1);
  if (showCount == showSkipNum) {
    //lx change 0815
    Mat imageShowMat= cv::cvarrToMat(imageShow);
    //Mat imageShowMat(imageShow);
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

/*!
// 计算形变和非形变图像的对应（map）
// void cvInitUndistortMap( const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy );
// 参数说明
// intrinsic_matrix——摄像机的内参数矩阵(A) [fx 0 cx; 0 fy cy; 0 0 1].
// distortion_coeffs——形变系数向量[k1, k2, p1, p2]，大小为4x1或者1x4。
// mapx——x坐标的对应矩阵。
// mapy——y坐标的对应矩阵。
// 概述
// 函数cvInitUndistortMap预先计算非形变对应－正确图像的每个像素在形变图像里的坐标。这个对应可以传递给cvRemap函数（跟输入和输出图像一起）
*/
  cvInitUndistortMap(&kMat, &dMat, mapx, mapy);

  CvSize subregionSize = cvSize((int)subregionWidth, (int)subregionHeight);
  imageEig = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);
  imageTmp = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);

  CvSize pyrSize = cvSize(imageWidth + 8, imageHeight / 3);
  pyrCur = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  pyrLast = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);

  //cmu
  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/raw", 1, imageDataHandler);
  //kitti
  //ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/camera/image_raw", 1, imageDataHandler);
  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::spin();

  return 0;
}
