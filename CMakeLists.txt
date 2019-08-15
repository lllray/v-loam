cmake_minimum_required(VERSION 2.4.6)
#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
project(demo_lidar)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
add_subdirectory(isamlib)
link_libraries(isamlib)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS PCL
  INCLUDE_DIRS include
)
#rosbuild_init()


add_executable(featureTracking src/featureTracking.cpp)
TARGET_LINK_LIBRARIES(featureTracking boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(visualOdometry src/visualOdometry.cpp)
TARGET_LINK_LIBRARIES(visualOdometry boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(bundleAdjust src/bundleAdjust.cpp)
TARGET_LINK_LIBRARIES(bundleAdjust boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(processDepthmap src/processDepthmap.cpp)
TARGET_LINK_LIBRARIES(processDepthmap boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(stackDepthPoint src/stackDepthPoint.cpp)
TARGET_LINK_LIBRARIES(stackDepthPoint boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(transformMaintenance src/transformMaintenance.cpp)
TARGET_LINK_LIBRARIES(transformMaintenance boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(registerPointCloud src/registerPointCloud.cpp)
TARGET_LINK_LIBRARIES(registerPointCloud boost_system boost_filesystem  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
