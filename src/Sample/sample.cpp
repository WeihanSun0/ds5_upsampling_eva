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
#include <thread>
using namespace std;

#define SHOW_TIME 

const string rootPath = "../../";
const string strDataPath = rootPath + "dat/real/hand_test";
const string strParam = rootPath + "dat/real/camParam/camera_calib/param.txt";

float g_FPS = 0.0;

cv::Mat visualization(const cv::Mat& guide, const cv::Mat& dmapFlood, const cv::Mat& dmapSpot,
                        const cv::Mat& dense, const cv::Mat& conf)
{
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
    char szLabel[255];
    sprintf(szLabel, "flood (FPS = %.2f)", g_FPS);
    vecLabel.push_back(szLabel);
    vecImgs.push_back(imgGuideMapSpot);
    vecLabel.push_back("spot");
    vecImgs.push_back(imgOverlapDense);
    vecLabel.push_back("dense");
    return mergeImages(vecImgs, vecLabel, cv::Size(3, 1));
}

int main(int argc, char* argv[])
{
    // file names
    int frame_num = 0;
    char szFN[255];
    sprintf(szFN, "%s/%08d_rgb_gray_img.png", strDataPath.c_str(), frame_num);
    string strGuide = string(szFN);
    sprintf(szFN, "%s/%08d_spot_depth_pc.exr", strDataPath.c_str(), frame_num);
    string strSpotPc = string(szFN);
    sprintf(szFN, "%s/%08d_flood_depth_pc.exr", strDataPath.c_str(), frame_num);
    string strFloodPc = string(szFN);
    // read dat
    cv::Mat imgGuide = cv::imread(strGuide, -1);
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

    // get default parameters
    double fgs_lambda_flood, fgs_sigma_flood, fgs_lambda_spot, fgs_sigma_spot;
    int fgs_num_iter_flood, fgs_num_iter_spot;
    dc.get_default_upsampling_parameters(fgs_lambda_flood, fgs_sigma_flood, 
                                        fgs_lambda_spot, fgs_sigma_spot,
                                        fgs_num_iter_flood, fgs_num_iter_spot);
    int edge_dilate_size, canny_thresh1, canny_thresh2, flood_range;
    float edge_thresh;
    dc.get_default_preprocessing_parameters(edge_dilate_size, edge_thresh, canny_thresh1, canny_thresh2, flood_range);
    int iEdge_thresh = edge_thresh * 100;
    int iFgs_lambda_flood = (int)fgs_lambda_flood;
    int iFgs_simga_flood = (int)fgs_sigma_flood;
    //show results
    cv::Mat dense, conf, imgShow;
    cv::namedWindow("show");
    cv::createTrackbar("dilate size", "show", &edge_dilate_size, 20); // 3~20
    cv::createTrackbar("canny thresh1", "show", &canny_thresh1, 255); // 0~255
    cv::createTrackbar("canny thresh2", "show", &canny_thresh2, 255); // 0~255
    cv::createTrackbar("edge_thresh", "show", &iEdge_thresh, 400); // 0~200
    cv::createTrackbar("flood_range", "show", &flood_range, 40); 
    cv::createTrackbar("fgs_lambda", "show", &iFgs_lambda_flood, 1000);
    cv::createTrackbar("fgs_sigma", "show", &iFgs_simga_flood, 20);
    cv::createTrackbar("iter_num", "show", &fgs_num_iter_flood, 5);


    char mode = '1'; // 1: flood 2: spot 3: flood + spot
#ifdef SHOW_TIME
    chrono::system_clock::time_point t_start, t_end; 
#endif
    while(1) {
        edge_thresh = (float)iEdge_thresh/100;
        fgs_lambda_flood = (float)iFgs_lambda_flood/10;
        fgs_sigma_flood = (float)iFgs_simga_flood;
        dc.set_upsampling_parameters(fgs_lambda_flood, fgs_sigma_flood, 
                            fgs_lambda_spot, fgs_sigma_spot, fgs_num_iter_flood, fgs_num_iter_spot); 
        dc.set_preprocessing_parameters(edge_dilate_size, edge_thresh, 
                            canny_thresh1, canny_thresh2, flood_range);
#ifdef SHOW_TIME
        t_start = chrono::system_clock::now();
#endif 
        // 
        if (mode == '1') {
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
        double elapsed = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
        cout << "\033[31;43mUpsampling total time = " << elapsed << " [us]\033[0m" << endl;
        g_FPS = (float)1000/elapsed*1000;
#endif
        cv::Mat dmapFlood = dc.get_flood_depthMap();
        cv::Mat dmapSpot = dc.get_spot_depthMap();
        imgShow = visualization(imgGuide, dmapFlood, dmapSpot, dense, conf);
        cv::imshow("show", imgShow);
        char c= cv::waitKey(30);
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
        case 's':
            sprintf(szFN, "%s/%08d_dense_dmap.tiff", strDataPath.c_str(), frame_num);
            cv::imwrite(szFN, dense);
            sprintf(szFN, "%s/%08d_conf.tiff", strDataPath.c_str(), frame_num);
            cv::imwrite(szFN, conf);
            break;
        default:
            break;
        }
    }
    exit(0);
}
