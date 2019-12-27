/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "KeyFrame.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include <condition_variable>
#include <thread>

using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    //定义点云的格式，使用的是XYZRGBA
    typedef pcl::PointXYZRGBA pointT;
    typedef pcl::PointCloud<pointT> pointCloud;
    
    PointCloudMapping( double resolution_ );
    
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
    // TODO:shutdown()是什么意思
    void isFinished();
    // TODO:viewer()是什么意思
    void viewer();
    // 新增
    //void public_cloud(pointCloud &cloud);
    //void Cloud_transform(pointCloud::Ptr &source, pointCloud::Ptr &out);

public:
    // TODO:生成点云地图吗
    pointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    // 全局地图
    pointCloud::Ptr globalMap;
    // 可视线程
    shared_ptr<thread>  viewerThread;
    
    // 标志位
    bool    shutDownFlag    =false;
    // 互斥锁，针对shutdown()函数
    // 构造函数，std::mutex不允许拷贝构造，也不允许 move 拷贝
    mutex   shutDownMutex;
    
    // 关键帧更新条件变量
    condition_variable  keyFrameUpdated;
    // 关键帧更新互斥锁
    mutex               keyFrameUpdateMutex;
    
    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;
    // 关键帧互斥锁
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;
    
    // NOTE:从0.04修改为了0.01
    double resolution = 0.01;
    // 点云体素
    pcl::VoxelGrid<pointT>  voxel;
    // 创建滤波器
    pcl::StatisticalOutlierRemoval<pointT> sor;
};

#endif // POINTCLOUDMAPPING_H
