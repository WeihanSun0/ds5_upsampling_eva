/**
 * @file sample.cpp
 * @author Weihan.Sun (weihan.sun@sony.com)
 * @brief sample application
 * @version 0.1
 * @date 2022-07-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "dsviewer_interface.h"
#include "z2color.h"
#include "upsampling.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace std;

#define SHOW_TIME 

const string rootPath = "../../";
const string strDataPath = rootPath + "dat/sample";
const string strGuide = strDataPath + "/0001_rgb_img.png";
const string strFloodPc = strDataPath + "/0001_flood_pc.tiff";
const string strSpotPc = strDataPath + "/0001_spot_pc.tiff";
const string strParam = rootPath + "dat/real/camParam/camera_calib/param.txt";


void viewInputData(const cv::Mat& imgGuide, const cv::Mat& pcFlood, const cv::Mat& pcSpot, 
                    float cx, float cy, float fx, float fy)
{
 // convert point cloud to depthmap
    cv::Mat dmapFlood = pc2detph(pcFlood, imgGuide.size(), cx, cy, fx, fy);
    cv::Mat dmapSpot = pc2detph(pcSpot, imgGuide.size(), cx, cy, fx, fy);

    // show input 
    double minVal, maxVal;
    cv::minMaxLoc(dmapFlood, &minVal, &maxVal);
    cv::Mat colorDmapFlood = z2colormap(dmapFlood, minVal, maxVal);
    cv::minMaxLoc(dmapSpot, &minVal, &maxVal);
    cv::Mat colorDmapSpot = z2colormap(dmapSpot, minVal, maxVal);

    cv::Mat imgMaskFlood = cv::Mat::zeros(dmapFlood.size(), dmapFlood.type());
    cv::Mat imgMaskSpot = cv::Mat::zeros(dmapSpot.size(), dmapSpot.type());
    imgMaskFlood.setTo(1.0, dmapFlood != 0.0);
    imgMaskSpot.setTo(1.0, dmapSpot != 0.0);
    cv::Mat imgGuideMapFlood = markSparseDepth(imgGuide, colorDmapFlood, imgMaskFlood, 3);
    cv::Mat imgGuideMapSpot = markSparseDepth(imgGuide, colorDmapSpot, imgMaskSpot, 3);

    cv::imshow("input Flood", imgGuideMapFlood);
    cv::imshow("input Spot", imgGuideMapSpot);
    cv::waitKey(0);
}

cv::Mat visualization(const cv::Mat& guide, const cv::Mat& pcFlood, const cv::Mat& pcSpot, const cv::Mat& dense, const cv::Mat& conf,
                    float cx, float cy, float fx, float fy, int mode)
{
    cv::Mat dmapFlood = pc2detph(pcFlood, guide.size(), cx, cy, fx, fy);
    cv::Mat dmapSpot = pc2detph(pcSpot, guide.size(), cx, cy, fx, fy);

    // input 
    double minVal, maxVal;
    cv::minMaxLoc(dmapFlood, &minVal, &maxVal);
    cv::Mat colorDmapFlood = z2colormap(dmapFlood, minVal, maxVal);
    cv::minMaxLoc(dmapSpot, &minVal, &maxVal);
    cv::Mat colorDmapSpot = z2colormap(dmapSpot, minVal, maxVal);

    cv::Mat imgMaskFlood = cv::Mat::zeros(dmapFlood.size(), dmapFlood.type());
    cv::Mat imgMaskSpot = cv::Mat::zeros(dmapSpot.size(), dmapSpot.type());
    imgMaskFlood.setTo(1.0, dmapFlood != 0.0);
    imgMaskSpot.setTo(1.0, dmapSpot != 0.0);
    cv::Mat imgGuideMapFlood = markSparseDepth(guide, colorDmapFlood, imgMaskFlood, 3);
    cv::Mat imgGuideMapSpot = markSparseDepth(guide, colorDmapSpot, imgMaskSpot, 3);
    // output
    cv::minMaxLoc(dense, &minVal, &maxVal);
    cv::Mat colorDense = z2colormap(dense, minVal, maxVal);
    cv::Mat imgOverlapDense = overlap(colorDense, guide, 0.5);
    //
    vector<cv::Mat> vecImgs;
    vector<string> vecLabel;
    vecImgs.push_back(imgGuideMapFlood);
    vecLabel.push_back("flood");
    vecImgs.push_back(imgGuideMapSpot);
    vecLabel.push_back("spot");
    vecImgs.push_back(imgOverlapDense);
    vecLabel.push_back("dense");
    return mergeImages(vecImgs, vecLabel, cv::Size(3, 1));
}

int main(int argc, char* argv[])
{
    // read dat
    cv::Mat imgGuide = cv::imread(strGuide, cv::IMREAD_GRAYSCALE);
    cv::Mat pcFlood = cv::imread(strFloodPc, -1);
    cv::Mat pcSpot = cv::imread(strSpotPc, -1);

    // read camera parameters
    map<string, float> params;
    if(!read_param(strParam, params)) {
        cout << "open param failed" <<endl;
        exit(0);
    }
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);

    // upsampling
    upsampling dc;
    dc.set_cam_paramters(cx, cy, fx, fy);
    double fgs_lambda_flood = 24.0;
	double fgs_sigma_flood = 8;
	double fgs_lambda_spot = 700;
    double fgs_sigma_spot = 5;

    //show results
    cv::Mat dense, conf, imgShow;
    cv::namedWindow("show");
    char mode = '1'; // 1: flood 2: spot 3: flood + spot
#ifdef SHOW_TIME
    chrono::system_clock::time_point t_start, t_end; 
#endif
    while(1) {
#ifdef SHOW_TIME
        t_start = chrono::system_clock::now();
#endif 
        // 
        dc.set_upsampling_parameters(fgs_lambda_flood, fgs_sigma_flood, fgs_lambda_spot, fgs_sigma_spot); if (mode == '1') {
            dc.run(imgGuide, pcFlood, cv::Mat(), dense, conf);
        } else if (mode == '2') {
            dc.run(imgGuide, cv::Mat(), pcSpot, dense, conf);
        } else if (mode == '3') {
            dc.run(imgGuide, pcFlood, pcSpot, dense, conf);
        } else if (mode == 'q') {
            break;
        } else {

        }
#ifdef SHOW_TIME
        t_end = chrono::system_clock::now();
        double elapsed = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
        cout << "Upsampling total time = " << elapsed << " [ms]" << endl;
#endif
        imgShow = visualization(imgGuide, pcFlood, pcSpot, dense, conf, cx, cy, fx, fy, mode);
        cv::imshow("show", imgShow);
        char c= cv::waitKey(100);
        switch (c)
        {
        case '1':
            mode = '1';
            break;
        case '2':
            mode = '2';
            break;
        case '3':
            mode = '3';
            break;
        case 'q':
            mode = 'q';
            break;
        default:
            break;
        }
    }
    exit(0);
}
