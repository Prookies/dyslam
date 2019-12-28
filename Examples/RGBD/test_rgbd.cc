#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include "System.h"

using namespace std;

void LoadImages(const string &strAssociationFilename,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps);

void LoadGroundTruth(const string &sGroundTruthFilename,
                     vector<double> &vTimestamps,
                     vector<Eigen::Isometry3d,
                     Eigen::aligned_allocator<Eigen::Isometry3d>> &Ts);

int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);
  if(argc != 5)
  {
    cerr<<endl<<"Usage: ./rgbd_tum path_to_vocabulary "
                "path_to_settings path_to_sequence "
                "path_to_association"<<endl;
    return 1;
  }

  string model_file = "./prototxt/segnet_pascal.prototxt";
  string trained_file = "./model/segnet_pascal.caffemodel";
  string sGroundTruth_file = string(argv[3]) + "/groundtruth.txt";
  // Retrieve paths to images
  vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> Ts;
  vector<string> vstrImageFilenamesRGB;
  vector<string> vstrImageFilenamesD;
  vector<double> vTimestamps;
  string strAssociationFilename = string(argv[4]);
  LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
  // Check consistency in the number of images and depthmaps
  int nImages = vstrImageFilenamesRGB.size();
  if(vstrImageFilenamesRGB.empty())
  {
    cerr<<endl<<"No images found in provided path."<<endl;
    return 1;
  }
  else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
  {
    cerr<<endl<<"Different number of images for rgb and depth."<<endl;
    return 1;
  }

  // 读取真实位姿
  LoadGroundTruth(sGroundTruth_file,vTimestamps,Ts);


  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(argv[1], argv[2], model_file, trained_file, ORB_SLAM2::System::RGBD, true);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);
  //    // Vector for orbExtractor time
  //    vector<float> vTimesOrb;
  //    vTimesOrb.resize(nImages);
  //    // Vector for MovingChecking time
  //    vector<float> vTimesMoving;
  //    vTimesMoving.resize(nImages);
  //    // Vector for Segnet time
  //    vector<float> vTimesSegnet;
  //    vTimesSegnet.resize(nImages);

  cout<<endl<<"------"<<endl;
  cout<<"Start processing sequence ..."<<endl;
  cout<<"Images in the sequence: "<< nImages<<endl<<endl;

  cv::Mat imRGB, imD;
  for(int ni=0; ni<nImages; ni++)
  {
    // Read image and depthmap from file
    imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
    imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[ni];

    if(imRGB.empty())
    {
      cerr<<endl<<"Fail to load image at: "
         <<string(argv[3])<<"/"<<vstrImageFilenamesRGB[ni]<<endl;
      return 1;
    }

    // 真实位姿
    Eigen::Isometry3d Tcw_real = Ts[ni].inverse()*Ts[0];
    Eigen::Matrix3d Rcw_real(Tcw_real.rotation());
    Eigen::Vector3d tcw_real(Tcw_real.translation());

    cv::Mat cvTcw_real = (cv::Mat_<float>(4,4) <<
                          Rcw_real(0,0),Rcw_real(0,1),Rcw_real(0,2),tcw_real(0),
                          Rcw_real(1,0),Rcw_real(1,1),Rcw_real(1,2),tcw_real(1),
                          Rcw_real(2,0),Rcw_real(2,1),Rcw_real(2,2),tcw_real(2),
                          0,0,0,1);

    cout << "opencv real Tcw: " << endl << cvTcw_real << endl;


    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // Pass the image to the SLAM system
    cv::Mat Tcw = SLAM.TrackRGBD(imRGB, imD, tframe, cvTcw_real);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    vTimesTrack[ni] = ttrack;


    // 输出真实位姿
    {
//      Eigen::Matrix3d Rcw_vision = Eigen::Matrix3d::Zero();
//      Eigen::Vector3d tcw_vision = Eigen::Vector3d::Zero();

//      Rcw_vision << Tcw.at<float>(0,0),Tcw.at<float>(0,1),Tcw.at<float>(0,2),
//          Tcw.at<float>(1,0),Tcw.at<float>(1,1),Tcw.at<float>(1,2),
//          Tcw.at<float>(2,0),Tcw.at<float>(2,1),Tcw.at<float>(2,2);
//      tcw_vision << Tcw.at<float>(0,3),Tcw.at<float>(1,3),Tcw.at<float>(2,3);



//      cout << "当前帧真实位姿为：" << endl << Tcw_real.matrix() << endl;
//      cout << "当前帧位姿为: " << endl << Tcw << endl;
//      //        cout << "当前帧旋转为：" << endl << Rcw_vision << endl;
//      //        cout << "当前帧位移为: " << endl << tcw_vision.transpose() << endl;

//      Eigen::Quaterniond Qcw_vision(Rcw_vision);
//      Eigen::Quaterniond Qcw_real(Tcw_real.rotation());
//      Qcw_vision.normalize();
//      Qcw_real.normalize();

//      Eigen::Quaterniond q_e = Qcw_vision.inverse() * Qcw_real;

//      float error_R = q_e.x() * q_e.x() + q_e.y() * q_e.y() +
//          q_e.z() * q_e.z() + (1 - q_e.w()) * (1 - q_e.w());

//      float error_t = (Tcw_real.translation() - tcw_vision).norm();
//      cout << "Rotation error = " << error_R << endl;
//      cout << "translation error = " << error_t << endl;
    }


    cv::waitKey(0);

    // Wait to load the next frame
    double T=0;
    if(ni<nImages-1)
      T = vTimestamps[ni+1]-tframe;
    else if(ni>0)
      T = tframe - vTimestamps[ni-1];

    if(ttrack<T)
      usleep((T-ttrack)*1e6);
  }

  // Stop all threads
  SLAM.Shutdown();

  SLAM.ComputeTime();
  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for(int ni=0; ni<nImages; ni++)
  {
    totaltime+=vTimesTrack[ni];
  }

  cout<<"------"<<endl<<endl;
  cout<<"median tracking time: "<<vTimesTrack[nImages/2]<<endl;
  cout<<"mean tracking time: "<<totaltime/nImages<<endl;

  // 这里应该增加一个保存每个过程时间到文本的程序，
  // Save camera trajectory
  SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
  ifstream fAssociation;
  fAssociation.open(strAssociationFilename.c_str());
  while(!fAssociation.eof())
  {
    string s;
    getline(fAssociation, s);
    if(!s.empty())
    {
      stringstream ss;
      ss << s;
      double t;
      string sRGB, sD;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenamesRGB.push_back(sRGB);
      ss >> t;
      ss >> sD;
      vstrImageFilenamesD.push_back(sD);
    }
  }
  cout << "总共加载图像数据大小为：" << vstrImageFilenamesRGB.size() << endl;
}

void LoadGroundTruth(const string &sGroundTruthFilename,
                     vector<double> &vTimestamps,
                     vector<Eigen::Isometry3d,
                     Eigen::aligned_allocator<Eigen::Isometry3d>> &Ts) {
  ifstream fGroundTruth;
  fGroundTruth.open(sGroundTruthFilename.c_str());
  if (!fGroundTruth.is_open()) {
    LOG(ERROR) << "Failed to open img0 file: " << sGroundTruthFilename << endl;
    return;
  }
  string sline;
  double dTime = 0.0;
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  size_t i = 0;
  while (getline(fGroundTruth, sline) && !sline.empty()) {
    if (sline.at(0) == '#')
      continue;

    istringstream ssPoseData(sline);
    ssPoseData >> dTime >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >>
                                    q.w();


    if (fabs(vTimestamps[i] - dTime) > 0.008)
      continue;

    // cout << fixed << vTimestamps[i] << " " << dTime << endl;

    i++;
    // cout << t.transpose() << " " << q.coeffs().transpose() << endl;
    q.normalize();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(q);
    T.pretranslate(t);
    Ts.push_back(T);
  }
  cout << "总共加载位姿数据大小为：" << Ts.size() << endl;
}
