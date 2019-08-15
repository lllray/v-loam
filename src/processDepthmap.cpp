#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include "pointDefinition.h"
/*!
* 数组voRx[]~voTz[]用于保存连续的相机位姿,depthCloud不断把之前的点云点变换到最近时刻的相机位姿,
* 当再有新的点云到来时,进入函数syncCloudHandler(),该函数先将该帧点云变换到最近的相机位姿下,然后
* 添加到depthCloud中;当相机位姿发生变化时,进入函数voDataHandler(),先将depthCloud变换到新的相机
* 位姿下,然后进行滤波,然后发布出去
*/
const double PI = 3.1415926;

const int keepVoDataNum = 30;
double voDataTime[keepVoDataNum] = {0};
double voRx[keepVoDataNum] = {0};
double voRy[keepVoDataNum] = {0};
double voRz[keepVoDataNum] = {0};
double voTx[keepVoDataNum] = {0};
double voTy[keepVoDataNum] = {0};
double voTz[keepVoDataNum] = {0};
int voDataInd = -1;
int voRegInd = 0;

pcl::PointCloud<pcl::PointXYZI>::Ptr depthCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZ>::Ptr syncCloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud2(new pcl::PointCloud<pcl::PointXYZI>());

double timeRec = 0;
double rxRec = 0, ryRec = 0, rzRec = 0;
double txRec = 0, tyRec = 0, tzRec = 0;

bool systemInited = false;
double initTime;

int startCount = -1;
const int startSkipNum = 5;

ros::Publisher *depthCloudPubPointer = NULL;

void voDataHandler(const nav_msgs::Odometry::ConstPtr& voData)
{
  double time = voData->header.stamp.toSec();

  double roll, pitch, yaw;
  geometry_msgs::Quaternion geoQuat = voData->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

  double rx = voData->twist.twist.angular.x - rxRec;
  double ry = voData->twist.twist.angular.y - ryRec;
  double rz = voData->twist.twist.angular.z - rzRec;

  if (ry < -PI) {
    ry += 2 * PI;
  } else if (ry > PI) {
    ry -= 2 * PI;
  }

  double tx = voData->pose.pose.position.x - txRec;
  double ty = voData->pose.pose.position.y - tyRec;
  double tz = voData->pose.pose.position.z - tzRec;

  rxRec = voData->twist.twist.angular.x;
  ryRec = voData->twist.twist.angular.y;
  rzRec = voData->twist.twist.angular.z;

  txRec = voData->pose.pose.position.x;
  tyRec = voData->pose.pose.position.y;
  tzRec = voData->pose.pose.position.z;
//!因为是把世界坐标系旋转到当前坐标系,所以roll,pitch,yaw应该取负值,而绕x轴和绕y轴的旋转角度在发布与接收时已经被
//!添加了负值,所以旋转矩阵没变,而绕z轴的旋转角没取负值,所以在旋转矩阵里要把绕z轴角度取负值
  double x1 = cos(yaw) * tx + sin(yaw) * tz;
  double y1 = ty;
  double z1 = -sin(yaw) * tx + cos(yaw) * tz;

  double x2 = x1;
  double y2 = cos(pitch) * y1 - sin(pitch) * z1;
  double z2 = sin(pitch) * y1 + cos(pitch) * z1;

  tx = cos(roll) * x2 + sin(roll) * y2;
  ty = -sin(roll) * x2 + cos(roll) * y2;
  tz = z2;

//!voDataInd取值为0~29
  voDataInd = (voDataInd + 1) % keepVoDataNum;
  voDataTime[voDataInd] = time;

/*!
* rx~ry存的是R_lc中的旋转量,旋转方向是z->x->y,参考坐标系是上一帧,所以也就是说上一帧按照R_lc=ry*rx*rz(旋转方向自右向左)的
* 顺序旋转可以得到当前帧的坐标,在visualOdometry.cpp中可以看到,transform[0]~[2]存储的其实是R_cl中的旋转角度,而vo的
* twist中的旋转角度存的是angleSum[0]~[2] -= transform[0]~[2],有一个取负值的操作,取负之后得到的就是R_lc中的旋转角,
* R_lc和R_cl的区别就是:
* R_lc=ry*rx*rz
* R_cl=-rz*-rx*-ry(旋转顺序从右往左看)
* tx~tz存的就是T_lc的位移量,当前坐标系相对于上一帧坐标系,在当前坐标系下表示的位移增量,
*/
  voRx[voDataInd] = rx;
  voRy[voDataInd] = ry;
  voRz[voDataInd] = rz;
  voTx[voDataInd] = tx;
  voTy[voDataInd] = ty;
  voTz[voDataInd] = tz;

  double cosrx = cos(rx);
  double sinrx = sin(rx);
  double cosry = cos(ry);
  double sinry = sin(ry);
  double cosrz = cos(rz);
  double sinrz = sin(rz);

  if (time - timeRec < 0.5) {
    pcl::PointXYZI point;
    tempCloud->clear();
    double x1, y1, z1, x2, y2, z2;
    int depthCloudNum = depthCloud->points.size();
    for (int i = 0; i < depthCloudNum; i++) {
      point = depthCloud->points[i];

      x1 = cosry * point.x - sinry * point.z;
      y1 = point.y;
      z1 = sinry * point.x + cosry * point.z;

      x2 = x1;
      y2 = cosrx * y1 + sinrx * z1;
      z2 = -sinrx * y1 + cosrx * z1;
/*!
* tx~tz存的就是当前坐标系相对于上一帧坐标系,在当前坐标系下表示的位移增量,T_lc,因为是
* rx = voData->twist.twist.angular.x - rxRec;所以基准坐标系是上一帧,即rxRec.
* 所以上一帧的一个点P_l,通过P_c=R_cl*P_l+T_cl可以变换到当前坐标系,
* 并且R_cl=-rz*-rx*-ry  T_cl=-T_lc(即-tx,-ty,-tz),所以这里是减去tx,ty,tz
*/
      point.x = cosrz * x2 + sinrz * y2 - tx;
      point.y = -sinrz * x2 + cosrz * y2 - ty;
      point.z = z2 - tz;

      double pointDis = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      double timeDis = time - initTime - point.intensity;
      if (fabs(point.x / point.z) < 2 && fabs(point.y / point.z) < 1 && point.z > 0.5 && pointDis < 15 &&
          timeDis < 5.0) {
        tempCloud->push_back(point);
      }
    }

    depthCloud->clear();
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setInputCloud(tempCloud);
    downSizeFilter.setLeafSize(0.05, 0.05, 0.05);
    downSizeFilter.filter(*depthCloud);
    depthCloudNum = depthCloud->points.size();

    tempCloud->clear();
    for (int i = 0; i < depthCloudNum; i++) {
      point = depthCloud->points[i];

      if (fabs(point.x / point.z) < 1 && fabs(point.y / point.z) < 0.6) {
        point.intensity = depthCloud->points[i].z;
        point.x *= 10 / depthCloud->points[i].z;
        point.y *= 10 / depthCloud->points[i].z;
        point.z = 10;

        tempCloud->push_back(point);
      }
    }

    tempCloud2->clear();
    downSizeFilter.setInputCloud(tempCloud);
    downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
    downSizeFilter.filter(*tempCloud2);
    int tempCloud2Num = tempCloud2->points.size();

    for (int i = 0; i < tempCloud2Num; i++) {
      tempCloud2->points[i].z = tempCloud2->points[i].intensity;
      tempCloud2->points[i].x *= tempCloud2->points[i].z / 10;
      tempCloud2->points[i].y *= tempCloud2->points[i].z / 10;
      tempCloud2->points[i].intensity = 10;
    }

    sensor_msgs::PointCloud2 depthCloud2;
    pcl::toROSMsg(*tempCloud2, depthCloud2);
    depthCloud2.header.frame_id = "camera2";
    depthCloud2.header.stamp = voData->header.stamp;
    depthCloudPubPointer->publish(depthCloud2);
  }

  timeRec = time;
}

void syncCloudHandler(const sensor_msgs::PointCloud2ConstPtr& syncCloud2)
{
  if (startCount < startSkipNum) {
    startCount++;
    return;
  }

  if (!systemInited) {
    initTime = syncCloud2->header.stamp.toSec();
    systemInited = true;
  }

  double time = syncCloud2->header.stamp.toSec();
  double timeLasted = time - initTime;

  syncCloud->clear();
  pcl::fromROSMsg(*syncCloud2, *syncCloud);

  double scale = 0;
  int voPreInd = keepVoDataNum - 1;
  if (voDataInd >= 0) {
    while (voDataTime[voRegInd] <= time && voRegInd != voDataInd) {
      voRegInd = (voRegInd + 1) % keepVoDataNum;
    }

    voPreInd = (voRegInd + keepVoDataNum - 1) % keepVoDataNum;
    double voTimePre = voDataTime[voPreInd];
    double voTimeReg = voDataTime[voRegInd];

    if (voTimeReg - voTimePre < 0.5) {
      double scale =  (voTimeReg - time) / (voTimeReg - voTimePre);
      if (scale > 1) {
        scale = 1;
      } else if (scale < 0) {
        scale = 0;
      }
    }
  }
//!通过插值得到与点云对应的坐标系,这个坐标系下保存的rx2~rz2,tx2~tz2指的是点云对应的帧与voRegInd指向的帧之间的R,T关系
  double rx2 = voRx[voRegInd] * scale;
  double ry2 = voRy[voRegInd] * scale;
  double rz2 = voRz[voRegInd] * scale;

  double tx2 = voTx[voRegInd] * scale;
  double ty2 = voTy[voRegInd] * scale;
  double tz2 = voTz[voRegInd] * scale;

  double cosrx2 = cos(rx2);
  double sinrx2 = sin(rx2);
  double cosry2 = cos(ry2);
  double sinry2 = sin(ry2);
  double cosrz2 = cos(rz2);
  double sinrz2 = sin(rz2);

  pcl::PointXYZI point;
  double x1, y1, z1, x2, y2, z2;
  int syncCloudNum = syncCloud->points.size();
  for (int i = 0; i < syncCloudNum; i++) {
    point.x = syncCloud->points[i].x;
    point.y = syncCloud->points[i].y;
    point.z = syncCloud->points[i].z;
    point.intensity = timeLasted;
      //!把插值得到的坐标系下的点转换到voRegInd指向的那一帧
    x1 = cosry2 * point.x - sinry2 * point.z;
    y1 = point.y;
    z1 = sinry2 * point.x + cosry2 * point.z;

    x2 = x1;
    y2 = cosrx2 * y1 + sinrx2 * z1;
    z2 = -sinrx2 * y1 + cosrx2 * z1;

    point.x = cosrz2 * x2 + sinrz2 * y2 - tx2;
    point.y = -sinrz2 * x2 + cosrz2 * y2 - ty2;
    point.z = z2 - tz2;
//!将点云一直变换到最新的一帧坐标系下
    if (voDataInd >= 0) {
      int voAftInd = (voRegInd + 1) % keepVoDataNum;
      while (voAftInd != (voDataInd + 1) % keepVoDataNum) {
        double rx = voRx[voAftInd];
        double ry = voRy[voAftInd];
        double rz = voRz[voAftInd];

        double tx = voTx[voAftInd];
        double ty = voTy[voAftInd];
        double tz = voTz[voAftInd];

        double cosrx = cos(rx);
        double sinrx = sin(rx);
        double cosry = cos(ry);
        double sinry = sin(ry);
        double cosrz = cos(rz);
        double sinrz = sin(rz);

        x1 = cosry * point.x - sinry * point.z;
        y1 = point.y;
        z1 = sinry * point.x + cosry * point.z;

        x2 = x1;
        y2 = cosrx * y1 + sinrx * z1;
        z2 = -sinrx * y1 + cosrx * z1;

        point.x = cosrz * x2 + sinrz * y2 - tx;
        point.y = -sinrz * x2 + cosrz * y2 - ty;
        point.z = z2 - tz;

        voAftInd = (voAftInd + 1) % keepVoDataNum;
      }
    }

    double pointDis = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (fabs(point.x / point.z) < 2 && fabs(point.y / point.z) < 1.5 && point.z > 0.5 && pointDis < 15) {
      depthCloud->push_back(point);
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "processDepthmap");
  ros::NodeHandle nh;

  ros::Subscriber voDataSub = nh.subscribe<nav_msgs::Odometry> ("/cam_to_init", 5, voDataHandler);

  //未发布　激光点云？
  ros::Subscriber syncCloudSub = nh.subscribe<sensor_msgs::PointCloud2>
                                 ("/sync_scan_cloud_filtered", 5, syncCloudHandler);

  ros::Publisher depthCloudPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_cloud", 5);
  depthCloudPubPointer = &depthCloudPub;

  ros::spin();

  return 0;
}
