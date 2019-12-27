cmake_minimum_required(VERSION 2.8)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

if(NOT ROS_BUILD_TYPE)
    set(ROS_BUILD_TYPE Release)
endif()

message("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -march=native")
add_definitions(-DCOMPILEDWITHC11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../../../
    ${PROJECT_SOURCE_DIR}/../../../include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../../../Thirdparty/caffe-segnet-cudnn7/include
    ${PROJECT_SOURCE_DIR}/../../../Thirdparty/caffe-segnet-cudnn7/src
    /usr/local/cuda/include
)

set(LIBS
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
    ${PROJECT_SOURCE_DIR}/../../../Thirdparty/caffe-segnet-cudnn7/build/lib/libcaffe.so
    ${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
    boost_system
    glog
)

rosbuild_add_executable(RGBD
    src/ros_rgbd.cc
)

target_link_libraries(RGBD
    ${LIBS}
)
