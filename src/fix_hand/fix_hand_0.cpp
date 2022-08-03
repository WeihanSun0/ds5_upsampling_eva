/**
 * @file fix_hand_0.cpp
 * @author Weihan.Sun (weihan.sun@sony.com)
 * @brief Original upsampling(FGS) without fix hand
 * @version 0.1
 * @date 2022-07-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cstdio>
#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include "z2color.h" // draw color
#include "viewer.h" // visualize 3D
#include "data_loader.h" // use sim data
#include "mcv_evaluation.h" // error eva
#include "dsviewer_interface.h"
#include <chrono>
#include <math.h>
#include <numeric>

#define DUMP_IMAGES

// include SAT
#include "FGS/upsampling.h"
using namespace std;

int main(int argc, char * argv[])
{
#if 0
    if (argc != 7) {
        cout << "USAGE: app guide_image flood_dmap output " << endl;
        exit(0);
    }
    const string fn_guide = string(argv[1]);
    const string fn_sparse = string(argv[2]);
    const string fn_output = string(argv[3]);
#else
    const string rootPath = "../../dat/real/hand_test";
    const string fn_guide = rootPath + "/00000000_rgb_gray_img.png";
    const string fn_sparse = rootPath + "/00000000_flood_depth_pc.exr";
    const string fn_output = rootPath + "/00000000_upsampling_results_0.tiff";
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
#endif

    // read data
    cv::Mat imgGray = cv::imread(fn_guide, cv::IMREAD_GRAYSCALE);
    cv::Mat pcFlood = cv::imread(fn_sparse, -1);
    // find_error_candidate_points2(pcFlood, imgGray);
    // exit(0);

    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
    cv::Mat imgFlood = pc2detph(pcFlood, imgGray.size(), cx, cy, fx, fy);
    cv::Mat maskMap = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    maskMap.setTo(1.0, imgFlood != 0.0);
    double minValue, maxValue;
    cv::minMaxLoc(imgFlood, &minValue, &maxValue);
    cv::Mat colorDepthMap = z2colormap(imgFlood, minValue, maxValue);
    cv::Mat markSparseImg = markSparseDepth(imgGray, colorDepthMap, maskMap, 3);
    // cv::imshow("input match", markSparseImg); // * match input point with RGB

    // upsampling 
    int count = 0; // frame count
    upsampling dc;
    cv::Rect roi; // process region, full resolution for only definition
    cv::Mat dense, conf;  // result
    // time 
    std::chrono::system_clock::time_point t_start, t_end;
    float duration;
    
    while(1) { // use the 5th result for best performance
        // run fgs 
        t_start = std::chrono::system_clock::now();
        dc.run2(imgGray, imgFlood, dense);
        t_end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        // cv::imshow("dense origin", dense);
        // cv::waitKey(0);
#if 1
        double minV, maxV;
        cv::minMaxLoc(dense, &minV, &maxV);
        cv::Mat imgShow = z2colormap(dense, minV, maxV);
        vector<cv::Mat> vecMats;
        vector<string> vecLabels;
        vecMats.push_back(markSparseImg);
        vecLabels.push_back("input match");
        vecMats.push_back(imgShow);
        vecLabels.push_back("dense color");
        cv::Mat mergeImg = mergeImages(vecMats, vecLabels, cv::Size(2,1));
        cv::imshow("imshow", mergeImg);
        char a = cv::waitKey(0);
        if (a == 's') {
            cv::imwrite(fn_output, dense);
        }
#endif
        break;
    }
    exit(0);
}