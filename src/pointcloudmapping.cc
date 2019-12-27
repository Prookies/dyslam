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

#include "pointcloudmapping.h"
#include <pcl/visualization/cloud_viewer.h>
#include "Converter.h"

// 为什么要把其作为一个全局变量
//PointCloudMapping::pointCloud pcl_cloud;

// 构造函数
PointCloudMapping::PointCloudMapping(double resolution_)
{
    // NOTE:resolution为点云地图的分辨率
    this->resolution = resolution_;
    this->voxel.setLeafSize( resolution, resolution, resolution);
    this->sor.setMeanK(50);
    this->sor.setStddevMulThresh(1.0);

    globalMap = boost::make_shared<pointCloud>( );
    // 将viewer()与类对象进行绑定
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::isFinished()
{
    {
        // unique_lock不会产生死锁
        // 方便线程对互斥量上锁，且提供了更好的上锁和解锁控制
        unique_lock<mutex> lck(shutDownMutex);
        // 设置标志位
        shutDownFlag = true;
        // 唤醒，对关键帧进行更新
        keyFrameUpdated.notify_one();
    }
    // 开启可视化线程
    viewerThread->join();
}
// 插入关键帧
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    // 构造的时候帮忙上锁，析构的时候释放
    unique_lock<mutex> lck(keyframeMutex);
    // 添加关键帧
    keyframes.push_back( kf );
    // 添加RGB图像
    colorImgs.push_back( color.clone() );
    // 添加深度图像
    depthImgs.push_back( depth.clone() );
    // 唤醒，对关键帧进行更新
    keyFrameUpdated.notify_one();
}
// 生成点云
PointCloudMapping::pointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    pointCloud::Ptr tmp( new pointCloud() );
    // point cloud is null ptr 空指针
    // NOTE: m,n以3递增，为什么不以1递增，考虑到速度吗?
    for ( int m=0; m<depth.rows; m+=2 )
    {
        for ( int n=0; n<depth.cols; n+=2 )
        {
            // NOTE:depth的类型为float
            float d = depth.ptr<float>(m)[n];
            // 除去深度信息过大过小的像素
            if (d < 0.01 || d>10)
                continue;

            pointT p;
            // NOTE: 该像素深度并没有除去深度的比例因子
            // 貌似只是放大了x,y,z的倍数，如果用这样的地图定位会有影响吗
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }

    // 相机的变换位姿
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    pointCloud::Ptr cloud(new pointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    // 为真的话，表示没有点是无效的
    cloud->is_dense = false;
    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}

// 点云可视化
void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        // 互斥锁保证了线程间的同步，但是却将并行操作变成了串行操作
        // 所以我们要尽可能的减小锁定的区域，也就是使用unique_lock
        // 该程序分为多个代码块，分别进行保护，同时减少锁定区域
        {
            // 对停止进行加锁
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            // 如果标志位为1就跳出循环了，该线程就停止了
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            // 对关键帧更新进行加锁
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            // 等待唤醒
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated 
        size_t N=0;
        {
            // 对关键帧加锁，这样会不会影响到跟踪，跟踪线程不应受到点云建图的影响
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            pointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
            *globalMap += *p;
        }
        pointCloud::Ptr tmp(new pointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        cout<<"show global map, size="<<globalMap->points.size()<<endl;
        lastKeyframeSize = N;
    }

    // 当实时建图完毕后对，再次对全部关键帧进行优化
//    globalMap->clear();
//    for(size_t i=0;i<keyframes.size();i++)
//    {
//        cout<<"keyframe "<<i<<" ..."<<endl;
//        pointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
//        pointCloud::Ptr tmp(new pointCloud);
//        sor.setInputCloud(p);
//        sor.filter(*tmp);
//        (*globalMap) += *tmp;
//        viewer.showCloud(globalMap);

//    }
//    globalMap->is_dense = false;
//    pointCloud::Ptr tmp(new pointCloud);
//    voxel.setInputCloud(globalMap);
//    voxel.filter(*tmp);
//    tmp->swap(*globalMap);
//    viewer.showCloud(globalMap);
}
