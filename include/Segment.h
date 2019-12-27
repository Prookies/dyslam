#ifndef SEGMENT_H
#define SEGMENT_H

#include "Segnet.h"
#include "Tracking.h"

namespace ORB_SLAM2 {

class Segment
{
public:
    // 定义构造函数
    Segment(const string &pascal_prototxt, const string &pascal_caffemodel, const string &strSettingPath);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    // Thread Synch
    void isNewImgArried();
    bool checkFinish();
    bool checkRequest();
    void RequestFinish();
    void ProduceImgSegment();
    void SetInitOK();
    void SetSegOK();

    Tracking* mpTracker;

    Classifier* classifier;

    float mImageWidth;
    float mImageHeight;

    cv::Mat mImg;
    cv::Mat mImgTemp;
    cv::Mat mImgSegment;
    cv::Mat mImgSegmentLatest;

    std::mutex mMutexGetNewImg;
    std::condition_variable mCondGetNewImg;
    std::mutex mMutexFinish;
    // 确保初始化完成
    std::mutex mMutexInitOK;
    std::condition_variable mCondInitOK;
    // 确保没帧语义分割完成
    std::mutex mMutexSegOK;
    std::condition_variable mCondSegOK;
    // 请求语义图像
    std::mutex mMutexReqImgS;
    bool mbReqImgS;
    bool mbFinishRequested;
    bool mbInitOK;
    bool mbSegOK;

    std::mutex mMutexNewImgSegment;
    std::condition_variable mCondNewImgSegment;
    bool mbGetNewImgSegment;
    bool mbNewImgFlag;
    bool mbSeg;
    double mSegmentTime;
    int imgIndex;

    //Parameters for caffe
    string model_file;
    string trained_file;
    string LUT_file;

    // 语义分割时间
    vector<float> mvTimeSeg;
};

}



#endif
