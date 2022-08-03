
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

int g_iedge_threshold_depth = 10;
int g_iedge_threshold_rgb = 30;
int g_idilate_rgb = 12;


cv::Mat getSobelAbs(cv::InputArray src_, int tapsize_sobel, float minval = 0.0f)
{
	const cv::Mat src = src_.getMat();
	std::vector<cv::Mat> t_sobels(2);
	cv::Mat h_sobels, v_sobels;
	cv::Sobel(src, h_sobels, CV_32F, 1, 0, tapsize_sobel);
	cv::Sobel(src, v_sobels, CV_32F, 0, 1, tapsize_sobel);

	cv::Mat dst = 0.5 * (abs(h_sobels) + abs(v_sobels));
	if (minval > 0.0f)
		cv::add(dst, minval, dst);
	return dst;
}

/**
 * @brief filter edge points by depth 
 * 
 * @param pcFlood 
 */
void filter_edge_by_depth(cv::Mat& pcFlood)
{
    float dist_threshold = 0.75;
    float threshold_edge = (float)g_iedge_threshold_depth/100; 
    int cols = pcFlood.cols;
    int rows = pcFlood.rows;
    // * create depth matirx image
    cv::Mat matrixImg = cv::Mat::zeros(cv::Size(cols, rows), CV_32FC1);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float val = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(val)) {
                val = 0.0;
            }
            matrixImg.at<float>(r,c) = val;
        }
    }
    // * extract edge of depth
    cv::Mat imgEdge;
    imgEdge = getSobelAbs(matrixImg, 3);
    // * filter the edge
    float nanF = std::nan("");
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dist = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(dist) || dist > dist_threshold) {
                continue;
            }
            if (imgEdge.at<float>(r, c) >= threshold_edge) {
                pcFlood.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF); 
                continue;
            }
        }
    }
}


void filter_edge_by_guide(cv::Mat& imgFlood, const cv::Mat& imgGray)
{
    float edge_threshold = (float)g_iedge_threshold_rgb;
    int dilateSize = g_idilate_rgb;
    cv::Mat imgEdge, imgEdgeMask, imgBoarder;
    imgEdge = getSobelAbs(imgGray, 3);
    imgEdgeMask = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgEdgeMask.setTo(1.0, imgEdge > edge_threshold);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(imgEdgeMask, imgBoarder, kernel);
#if 0
    cv::imshow("edge", imgEdgeMask);
    cv::imshow("boarder", imgBoarder);
    cv::waitKey(0);
#endif
    imgFlood.forEach<float>([imgBoarder](float& p, const int* pos) -> void{
        if(p != 0.0)  {
            int x = pos[1];
            int y = pos[0];
            if (imgBoarder.at<float>(y, x) != 0) {
                p = 0.0;
            }
        }
    });
}

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
    const string fn_output = rootPath + "/00000000_upsampling_results_2.tiff";
    const string fn_outputFolder = "D:\\workspace2\\Upsampling\\DS5\\mis_upsampling\\dat\\real\\hand_test\\tuning_method2\\";
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
#endif

    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);

    cv::namedWindow("imshow");
    cv::createTrackbar("edge_thresh_depth", "imshow", &g_iedge_threshold_depth, 120);
    cv::createTrackbar("edge_thresh_rgb", "imshow", &g_iedge_threshold_rgb, 200);
    cv::createTrackbar("boarder_dilate_rgb", "imshow", &g_idilate_rgb, 40);

    while(1) { // use the 5th result for best performance
        // read data
        cv::Mat imgGray = cv::imread(fn_guide, cv::IMREAD_GRAYSCALE);
        cv::Mat pcFlood = cv::imread(fn_sparse, -1);
        // * filter the edge points by depth
        filter_edge_by_depth(pcFlood);

        // main process
        cv::Mat imgFlood = pc2detph(pcFlood, imgGray.size(), cx, cy, fx, fy);
        //* filter the edge point by guide
        filter_edge_by_guide(imgFlood, imgGray);


        cv::Mat imgMask;
        imgMask = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
        imgMask.setTo(1.0, imgFlood != 0.0);

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
        char a = cv::waitKey(10);
        if (a == 's') {
            char szFn[255];
            sprintf(szFn, "%smethod2_%d_%d_%d.tiff", fn_outputFolder.c_str(), g_iedge_threshold_depth, g_iedge_threshold_rgb, g_idilate_rgb);
            cv::imwrite(szFn, dense);
        }
        if (a == 'q')
            break;
#endif
    }
    exit(0);
}