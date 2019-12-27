#include "Segment.h"
#include <fstream>

using namespace std;

namespace ORB_SLAM2 {
    Segment::Segment(const string &pascal_prototxt, const string &pascal_caffemodel, const string &strSettingPath):
        mbFinishRequested(false),mSegmentTime(0),imgIndex(0)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        // 模型结构
        model_file = pascal_prototxt;
        // 模型参数
        trained_file = pascal_caffemodel;

        mImageWidth = fSettings["Camera.width"];
        mImageHeight = fSettings["Camera.height"];

        if(mImageWidth<1 || mImageHeight<1)
        {
            mImageWidth = 640;
            mImageHeight = 480;
        }

        // 最后分割的图像，使用的是灰度图
        mImgSegmentLatest = cv::Mat(mImageHeight,mImageWidth,CV_8UC1);
        mbNewImgFlag = false;
        mbReqImgS = false;
        mbInitOK = false;
        mbSegOK = false;
        mbSeg = true;
    }
    void Segment::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }
    void Segment::isNewImgArried()
    {
        // 细粒度锁
        // 其提供lock()和unlock()借口，可以记录处于上锁还是没上锁的状态
        unique_lock<mutex> lock_NewImg(mMutexGetNewImg);
        while(!mbNewImgFlag)
        {
            mCondGetNewImg.wait(lock_NewImg);
        }
        mbNewImgFlag = false;
    }
    void Segment::Run()
    {
        classifier = new Classifier(model_file, trained_file);
        SetInitOK();
        cout<<"Load model ..."<<endl;
        // 先凑合着，后面再改进
        while(1)
        {
            // 检测是否有新帧
            isNewImgArried();
            if(mbSeg)
            {
                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                mImgSegment = classifier->Predict(mImg);
                cv::resize(mImgSegment, mImgSegment, cv::Size(mImageWidth,mImageHeight));
                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
                mSegmentTime = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
                mvTimeSeg.push_back(mSegmentTime);
                imgIndex++;
            }

            SetSegOK();
            if(checkFinish())
            {
                break;
            }
        }
    }

    bool Segment::checkFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    bool Segment::checkRequest()
    {
        unique_lock<mutex> lock(mMutexReqImgS);
        return mbReqImgS;
    }

    void Segment::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    void Segment::ProduceImgSegment()
    {
        unique_lock<mutex> lock_NewImS(mMutexNewImgSegment);
        mImgSegment.copyTo(mImgSegmentLatest);
    }

    void Segment::SetInitOK()
    {
        unique_lock<mutex> locker(mMutexInitOK);
        mbInitOK = true;
        mCondInitOK.notify_all();
    }

    void Segment::SetSegOK()
    {
        unique_lock<mutex> lock_SegOK(mMutexSegOK);
        mImgSegmentLatest = mImgSegment.clone();
        mbSegOK = true;
//        cout<<"Segment Set:The SegOK is"<<(int)mbSegOK<<endl;
        mCondSegOK.notify_all();
    }


}
