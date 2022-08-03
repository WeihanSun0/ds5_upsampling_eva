
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

const int width = 960;
const int height = 540;

//* global parameters
int g_rgb_feature_span = 11;        //! block avg. should be better
int g_rgb_half_block_size = 9;           //* block_size = half_size * 2 + 1
int g_idist_threshold = 75;      //* (m) move fix apply under this distance
int g_iedge_threshold_depth = 10;         //* edge or not justification, smaller -> more pixels 
int g_rangeSearch = 11;               //* search similar parts range 
int g_isimilarity_threshold = 995;  //* mimium similarity for moving fix


void getImgPos(const cv::Vec3f& xyz, float cx, float cy, float fx, float fy, int& u, int& v)
{
    float z = xyz[2];
    float uf = ((xyz[0] * fx / z) + cx);
    float vf = ((xyz[1] * fy / z) + cy);
    u = static_cast<int>(std::round(uf));
    v = static_cast<int>(std::round(vf));
}

void getSpacePos(int u, int v, float z, float cx, float cy, float fx, float fy, float& x, float& y)
{
    x = (u - cx) * z / fx;
    y = (v - cy) * z / fy;
}


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

void normal_feats(float feats[])
{
    float sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        sum += feats[i];
    }
    for (int i = 0; i < 8; ++i) {
        feats[i] /= sum;
    }
}

void extractGuideFeatures(const cv::Mat& imgGray, float featureMap[960][540][8])
{
    int span = g_rgb_feature_span;
    int width = imgGray.cols;
    int height = imgGray.rows;
    for (int r = 0; r < imgGray.rows; ++r) {
        for (int c = 0; c < imgGray.cols; ++c) {
            uchar cur_val = imgGray.at<uchar>(r, c);
            int count = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    if (i == 0 && j == 0) // self
                        continue;
                    int x = c + span*i;
                    int y = r + span*j;
                    uchar ref_val = 0;
                    if (x < 0 || x >= width || y < 0 || y >= height) {
                        ;
                    } else {
                        ref_val = imgGray.at<uchar>(y, x);
                    }
                    featureMap[c][r][count] = abs(ref_val - cur_val) + 0.001;
                    count += 1;
                }
            }
            // normalization
            normal_feats(featureMap[c][r]);
        }
    }
}

//* use block
void extractGuideFeatures_2(const cv::Mat& imgGray, float featureMap[960][540][8])
{
    int span = g_rgb_feature_span;
    int width = imgGray.cols;
    int height = imgGray.rows;
    cv::Mat imgInt;
    cv::integral(imgGray, imgInt);
    int blockSize = g_rgb_half_block_size;
    for (int r = blockSize; r < imgGray.rows-blockSize; ++r) {
        for (int c = blockSize; c < imgGray.cols-blockSize; ++c) {
            int cur_val = imgInt.at<int>(r+blockSize, c+blockSize) + imgInt.at<int>(r-blockSize, c-blockSize)
                            - imgInt.at<int>(r+blockSize, c-blockSize) - imgInt.at<int>(r-blockSize, c+blockSize);
            // uchar cur_val = imgGray.at<uchar>(r, c);
            int count = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    if (i == 0 && j == 0) // self
                        continue;
                    int x = c + span*i;
                    int y = r + span*j;

                    int ref_val = 0;
                    // if (x < 0 || x >= width || y < 0 || y >= height) {
                    if (x < blockSize || x >= width-blockSize || y < blockSize || y >= height-blockSize) {
                        ;
                    } else {
                        // ref_val = imgGray.at<uchar>(y, x);
                        ref_val = imgInt.at<int>(y+blockSize, x+blockSize) + imgInt.at<int>(y-blockSize, x-blockSize)
                            - imgInt.at<int>(y+blockSize, x-blockSize) - imgInt.at<int>(y-blockSize, x+blockSize);
                    }
                    featureMap[c][r][count] = abs(ref_val - cur_val) + 0.001;
                    count += 1;
                }
            }
            // normalization
            normal_feats(featureMap[c][r]);
        }
    }
}


void extractDepthFeatures(const cv::Mat& pcFlood, float featureMap[80][60][8])
{
    cv::Mat matrixImg = cv::Mat::zeros(pcFlood.size(), CV_32FC1);
    // * create matrix image
    for (int r = 0; r < pcFlood.rows; ++r) {
        for (int c = 0; c < pcFlood.cols; ++c) {
            matrixImg.at<float>(r, c) = pcFlood.at<cv::Vec3f>(r, c)[2];
        }
    }
    int width = matrixImg.cols;
    int height = matrixImg.rows;
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float cur_val = matrixImg.at<float>(r, c);
            if(isnan(cur_val)) // invalid value
                continue;
            int count = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    if (i == 0 && j == 0) // self
                        continue;
                    int x = c + i;
                    int y = r + j;
                    float ref_val = 0.0;
                    if (x < 0 || x >= width || y < 0 || y >= height) {
                        ;
                    } else {
                        ref_val = matrixImg.at<float>(y, x);
                        if (isnan(ref_val)) 
                            ref_val = 0.0;
                    }
                    featureMap[c][r][count] = abs(ref_val - cur_val) + 0.001;
                    count += 1;
                }
            }
            // normalization
            normal_feats(featureMap[c][r]);
        }
    }
}

double calcSimilarity(float f1[8], float f2[8])
{
    double M1 = 0.0, M2 = 0.0, Nom = 0.0;
    for (int i = 0; i < 8; ++i) {
        Nom += f1[i] * f2[i];
        M1 += f1[i] * f1[i];
        M2 += f2[i] * f2[i];
    }
    return Nom/sqrt(M1)/sqrt(M2);
}

//* add invalide feature
double calcSimilarity_1(float f1[8], float f2[8])
{
    double M1 = 0.0, M2 = 0.0, Nom = 0.0;
    float maxF1 = 0.0, maxF2 = 0.0;
    float invalid_thresh = 0.3;
    for (int i = 0; i < 8; ++i) {
        if (maxF1 < f1[i]) maxF1 = f1[i];
        if (maxF2 < f2[2]) maxF2 = f2[i];
        Nom += f1[i] * f2[i];
        M1 += f1[i] * f1[i];
        M2 += f2[i] * f2[i];
    }
    if (maxF1 < invalid_thresh || maxF2 < invalid_thresh) 
        return 0.0;
    return Nom/sqrt(M1)/sqrt(M2);
}

/**
 * @brief filter edge points by depth 
 * 
 * @param pcFlood 
 */
void filter_edge_by_depth(cv::Mat& pcFlood, float cx, float cy, float fx , float fy, 
                float featureMapDepth[80][60][8], float featureMapRGB[960][540][8])
{
    float dist_threshold = (float)g_idist_threshold/100;
    float threshold_edge = (float)g_iedge_threshold_depth/100; 
    int rangeSearch = g_rangeSearch;
    float similarity_threshold = (float)g_isimilarity_threshold/1000;
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
#if 0
    cv::Mat imgMask = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgMask.setTo(1.0, imgEdge <= threshold_edge);
    cv::imshow("edge", imgMask);
    cv::waitKey(0);
#endif
    // * filter the edge
    float nanF = std::nan("");
    for (int r = 0; r < rows; ++r) { // ! loop the edge directly
        for (int c = 0; c < cols; ++c) {
            float dist = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(dist) || dist > dist_threshold) { // no fix
                continue;
            }
            if (imgEdge.at<float>(r, c) >= threshold_edge) { // fix candidates
                //
                int u, v;
                getImgPos(pcFlood.at<cv::Vec3f>(r, c), cx, cy, fx, fy, u, v);
                if (u >= 0 && u < width && v >= 0 && v < height) {
                    //! todo find nearest similar point
                    float maxSim = 0.0;
                    int maxPos_x, maxPos_y;
                    for (int j = -rangeSearch; j <= rangeSearch; ++j) {
                        for (int i = -rangeSearch; i <= rangeSearch; ++i) {
                            int x = u + i;
                            int y = v + j; 
                            // float similarity = calcSimilarity(featureMapDepth[c][r], featureMapRGB[x][y]);
                            float similarity = calcSimilarity_1(featureMapDepth[c][r], featureMapRGB[x][y]);
                            if (similarity > maxSim) {
                                maxPos_x = x;
                                maxPos_y = y;
                                maxSim = similarity;
                            }
                        }
                    }
                    if (maxSim >= similarity_threshold) { // * fix
                        float x1, y1;
                        getSpacePos(maxPos_x, maxPos_y, dist, cx, cy, fx, fy, x1, y1);
                        pcFlood.at<cv::Vec3f>(r, c) = cv::Vec3f(x1, y1, dist);
                    } else { // * remove
                        pcFlood.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF);
                    }
                }
            }
        }
    }
}


float featureMapRGB[width][height][8] = {0};
float featureMapDepth[80][60][8] = {0};

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
    const string fn_output = rootPath + "/00000000_upsampling_results_3.tiff";
    const string fn_outputFolder = "D:\\workspace2\\Upsampling\\DS5\\mis_upsampling\\dat\\real\\hand_test\\tuning_method3\\";
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
#endif

    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);

    cv::Mat imgGray = cv::imread(fn_guide, cv::IMREAD_GRAYSCALE);
    cv::Mat pcFlood_org = cv::imread(fn_sparse, -1);
    cv::Mat pcFlood;
    // create window
    cv::namedWindow("imshow");
    cv::createTrackbar("edge_thresh_depth", "imshow", &g_iedge_threshold_depth, 120);
    cv::createTrackbar("feat span", "imshow", &g_rgb_feature_span, 20);
    cv::createTrackbar("feat block", "imshow", &g_rgb_half_block_size, 12);
    cv::createTrackbar("similarity", "imshow", &g_isimilarity_threshold, 1000);
    cv::createTrackbar("SearchR", "imshow", &g_rangeSearch, 40);
    while(1) { // use the 5th result for best performance
        // read data
        pcFlood_org.copyTo(pcFlood);
        //! need optimization
        extractGuideFeatures_2(imgGray, featureMapRGB);
        extractDepthFeatures(pcFlood, featureMapDepth);
        // * filter the edge points
        filter_edge_by_depth(pcFlood, cx, cy, fx, fy, featureMapDepth, featureMapRGB);

        // main process
        cv::Mat imgFlood = pc2detph(pcFlood, imgGray.size(), cx, cy, fx, fy);
        cv::Mat imgMask;
        imgMask = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
        imgMask.setTo(1.0, imgFlood != 0.0);

        cv::Mat maskMap = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
        maskMap.setTo(1.0, imgFlood != 0.0);
        double minValue, maxValue;
        cv::minMaxLoc(imgFlood, &minValue, &maxValue);
        cv::Mat colorDepthMap = z2colormap(imgFlood, minValue, maxValue);
        cv::Mat markSparseImg = markSparseDepth(imgGray, colorDepthMap, maskMap, 3);

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
            sprintf(szFn, "%smethod3_%d_%d_%d_%d_%d.tiff", fn_outputFolder.c_str(), \
                    g_iedge_threshold_depth, g_rgb_feature_span, g_rgb_half_block_size, \
                    g_rangeSearch, g_isimilarity_threshold);
            cv::imwrite(szFn, dense);
        }
        if ( a == 'q') {
            break;
        }
#endif
    }
    exit(0);
}