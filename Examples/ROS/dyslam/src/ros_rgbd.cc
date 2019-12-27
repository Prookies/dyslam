#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core.hpp>
#include "../../../include/System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}
    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD);
    ORB_SLAM2::System* mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr<<endl<<"Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_setting"<<endl;
        ros::shutdown();
        return 1;
    }

    string model_file = "/home/lab202/catkin_ws/src/dyslam/prototxt/segnet_pascal.prototxt";
    string trained_file = "/home/lab202/catkin_ws/src/dyslam/model/segnet_pascal.caffemodel";
    //
    ORB_SLAM2::System SLAM(argv[1],argv[2],model_file,trained_file,ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth/image_rect_raw",1);
    // 这里应该订阅小车位姿的信息，包括平移向量和四元数，后面还可以订阅IMU的消息

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::spin();

    SLAM.Shutdown();

    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr &msgRGB, const sensor_msgs::ImageConstPtr &msgD)
{
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // Note: 应当判断下彩色图像和深度图像的时间戳是否相等

    // Note: 这里可能需要对小车位姿进行筛选，需要知道小车位姿发布的频率。

    mpSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, cv_ptrRGB->header.stamp.toSec());
}
