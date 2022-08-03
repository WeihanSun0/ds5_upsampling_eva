#include "dsviewer_interface.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "z2color.h"
using namespace std;

const string rootPath = "../../";
const string strDataPath = rootPath + "dat/real/hand_test";
// const string strGuide = strDataPath + "/00000007_rgb_gray_img.png";
const string strGuide = strDataPath + "/00000001_rgb_gray_img.png";
// const string strSparsePointCloud = strDataPath + "/00000007_flood_depth_pc.tiff";
// const string strSparsePointCloud = strDataPath + "/00000001_flood_depth_pc.exr";
const string strSparsePointCloud = strDataPath + "/00000001_spot_depth_pc.exr";
// const string strSparsePointCloud = strDataPath + "/tmp2.tiff";
const string strParam = rootPath + "dat/real/camParam/camera_calib/param.txt";

int main(int argc, char* argv[])
{
    cv::Mat imgRGB = cv::imread(strGuide);
    cv::Mat pcFlood = cv::imread(strSparsePointCloud, -1);
    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
    double minValue, maxValue;
    //! test tiff
    cout << pcFlood.type() << endl;
    cv::imwrite(strDataPath+"/test.tiff", pcFlood);
    cv::waitKey(100);
    cv::Mat testTiff = cv::imread(strDataPath+"/test.tiff", -1);
    // cv::Mat depthMap = pc2detph(testTiff, imgRGB.size(), cx, cy, fx, fy);
    //! end test tiff
    //* check value
    cv::Vec3f val3f =  pcFlood.at<cv::Vec3f>(0, 0);
    cout << val3f[0] << "," << val3f[1] << "," << val3f[2] << endl;
    //* end check value
    cv::Mat depthMap = pc2detph(pcFlood, imgRGB.size(), cx, cy, fx, fy);
#if 0 // for Ikeya-san
    save_depth2txt(depthMap, strDataPath+ "/00000001_spot_depth_pc.txt");
    exit(0);
#endif
    cv::Mat maskMap = cv::Mat::zeros(depthMap.size(), depthMap.type());
    maskMap.setTo(1.0, depthMap != 0.0);
    // cv::Mat depthMap = pcFlood;
    cv::minMaxLoc(depthMap, &minValue, &maxValue);
    cv::Mat colorDepthMap = z2colormap(depthMap, minValue, maxValue);
    // cv::Mat overlapImg = overlap(colorDepthMap, imgRGB, 0.5);
    // cv::imshow("color depthmap", colorDepthMap);
    cv::Mat markSparseImg = markSparseDepth(imgRGB, colorDepthMap, maskMap, 3);
    cv::imshow("overlap", markSparseImg);
    cv::waitKey(0);
    // cv::Mat zerosImg = cv::Mat::zeros(cv::Size(200, 200), CV_32FC1);
    // cv::imwrite("zeros.tiff", zerosImg);
    // zerosImg.setTo(NAN, zerosImg == 0.0);
    // cv::imwrite("nanimage.tiff", zerosImg);
    exit(0);
}