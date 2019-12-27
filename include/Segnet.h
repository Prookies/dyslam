#ifndef SEGNET_H
#define SEGNET_H

#include <caffe/caffe.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

using namespace caffe;

class Classifier{
public:
    Classifier(const string& model_file,
               const string& trained_file);

    cv::Mat Predict(const cv::Mat& img);
private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    boost::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    cv::Size output_geometry_;
    int num_channels_;
};

#endif
