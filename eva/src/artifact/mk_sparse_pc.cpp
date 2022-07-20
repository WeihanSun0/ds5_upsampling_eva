/**
 * @file mk_sparse_pc.cpp
 * @author Weihan.Sun (weihan.sun@sony.com)
 * @brief Create flood and spot depthmap 
 * @version 0.1
 * @date 2022-06-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "../common/data_loader.h"
#include <vector>
using namespace std;

string rootFolder = "../../";
string g_strInputFolder = rootFolder + "dat/source/CamWithBump";
string g_strOutputFolder = rootFolder + "dat/artifact/CamWIthBump";
int g_arrFov[] = {30, 45, 60, 90}; // fov
int g_frame_num = 50;
// file appendix
string g_fa_gt = "_gtD.png";
string g_fa_flood = "_flood.tiff"; //depthmap mm
string g_fa_spot = "_spot.tiff";
//span
int flood_span_x = 12;
int flood_span_y = 12;
int spot_span_x = 70;
int spot_span_y = 50;

cv::Mat createMask(int cols, int rows, int span_x, int span_y)
{
    cv::Mat mask = cv::Mat::zeros(cv::Size(cols, rows), CV_32FC1);
    int dot_num_x = cols/span_x - 1;
    int dot_num_y = rows/span_y - 1;
    int shift_x = (cols - int(cols/span_x) * span_x)/2;
    int shift_y = (rows - int(rows/span_y) * span_y)/2;
    for (int r = shift_y+span_y; r <= rows-span_y; r+=span_y) {
        for (int c = shift_x+span_x; c <= cols-span_x; c+=span_x) {
            mask.at<float>(r, c) = 1.0;
        }
    }
    return mask;
}

int main(int argc, char* argv[])
{
    char szFileNameIn[255];
    char szFileNameOut[255];
    int cols = 640;
    int rows = 480;
    // create mask
    cv::Mat mask_flood = createMask(cols, rows, flood_span_x, flood_span_y);
    cv::Mat mask_spot = createMask(cols, rows, spot_span_x, spot_span_y);
    cv::imwrite("../../dat/artifact/mask_flood.tiff", mask_flood);
    cv::imwrite("../../dat/artifact/mask_spot.tiff", mask_spot);

    for (int ifov = 0; ifov < sizeof(g_arrFov)/sizeof(int); ++ifov) {
        string strInputFolder = g_strInputFolder + "_" + to_string(g_arrFov[ifov]) + "/house0_round0/polar";
        string strOutputFolder = g_strOutputFolder + "_" + to_string(g_arrFov[ifov]);
        for (int fid = 0; fid < g_frame_num; ++fid) {
            sprintf(szFileNameIn, "%s/%05d%s", strInputFolder.c_str(), fid, g_fa_gt.c_str());
            cout << "processing -> " << szFileNameIn << endl;
            cv::Mat imgGtD = read_D(string(szFileNameIn));
            if (imgGtD.empty()) {
                cout << "cannot find " << szFileNameIn << endl;
            }
            cv::Mat imgFlood, imgSpot;
            imgGtD.copyTo(imgFlood);
            imgGtD.copyTo(imgSpot);
            imgFlood.setTo(0, mask_flood == 0);
            imgSpot.setTo(0, mask_spot == 0);
            // upsampling calculation use 32F
            imgFlood.convertTo(imgFlood, CV_32F);
            imgSpot.convertTo(imgSpot, CV_32F);
            sprintf(szFileNameOut, "%s/%05d%s", strOutputFolder.c_str(), fid, g_fa_flood.c_str());
            cv::imwrite(szFileNameOut, imgFlood);
            sprintf(szFileNameOut, "%s/%05d%s", strOutputFolder.c_str(), fid, g_fa_spot.c_str());
            cv::imwrite(szFileNameOut, imgSpot);
        }
    }
    return 0;
}