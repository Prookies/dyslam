#include "Segnet.h"



Classifier::Classifier(const string& model_file,
                       const string& trained_file)
{
    Caffe::set_mode(Caffe::GPU);

    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1)<<"Network should have exactly one input.";
    CHECK_EQ(net_->num_inputs(), 1)<<"Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)<<"Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}

cv::Mat Classifier::Predict(const cv::Mat& img){
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    // 获得输出层的宽度，高度和通道数
    int width = output_layer->width();
    int height = output_layer->height();
    int channels = output_layer->channels();

    cv::Mat class_each_row(channels, width*height, CV_32FC1, const_cast<float*>(output_layer->cpu_data()));

    class_each_row = class_each_row.t();

    cv::Point maxId;
    double maxValue;
    cv::Mat prediction_map(height, width, CV_8UC1);
    for (int i=0;i<class_each_row.rows;i++)
    {
        // 查找class_each_row第i行,其只返回了最大值的指针和最大值位置的指针。
        cv::minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);
        prediction_map.at<uchar>(i) = maxId.x;
    }
    prediction_map = (prediction_map == 15);

    return prediction_map;

    // 只考虑分割人的情况
//    cv::Mat prediction_map = class_each_row.row(15);
//    prediction_map = prediction_map.reshape(1, height);
//    cv::Mat img_bw = (prediction_map > 0.35);
//    return img_bw;


}

void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for(int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
{
    cv::Mat sample;
    if(img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if(img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if(img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if(img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if(sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if(num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            <<"Input channels are not wrapping the input layer of the network.";


}
