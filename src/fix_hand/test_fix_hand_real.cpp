
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

/**
 * @brief check block images of pcFlood and imgGray
 * 
 * @param pcFlood 
 * @param imgGray 
 */
void check_input_depth(const cv::Mat& pcFlood, const cv::Mat& imgGray)
{
    // depth
    int cols = pcFlood.cols;
    int rows = pcFlood.rows;
    int span = 10;
    cv::Mat blockImg = cv::Mat::zeros(cv::Size(cols * span, rows * span), CV_8UC3);
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
    double minVal, maxVal;
    cv::minMaxLoc(matrixImg, &minVal, &maxVal);
    cv::Mat colorMatrix = z2colormap(matrixImg, minVal, maxVal);
    colorMatrix.forEach<cv::Vec3b>([&blockImg, span](cv::Vec3b& p, const int* pos) -> void{
        int x = pos[1];
        int y = pos[0];
        for (int r = y * span; r < (y+1) * span; ++r) {
            for (int c = x * span; c < (x+1) * span; ++c) {
                blockImg.at<cv::Vec3b>(r, c) = p; 
            }
        }
    });
    cv::imshow("depth block", blockImg);
    // rgb image 
    cv::Mat imgResizeGray;
    cv::Mat imgBlockGray = cv::Mat::zeros(imgGray.size(), imgGray.type());
    cv::resize(imgGray, imgResizeGray, cv::Size(96, 54), 0, 0);
    // cv::resize(imgResizeGray, imgBlockGray, cv::Size(969, 540), 0, 0, cv::INTER_LINEAR);
    imgResizeGray.forEach<uchar>([&imgBlockGray, span](uchar& p, const int* pos) -> void {
        int x = pos[1];
        int y = pos[0];
        for (int r = y*span; r < (y+1)*span; ++r) {
            for (int c = x*span; c < (x+1)*span; ++c) {
                imgBlockGray.at<uchar>(r, c) = p;
            }
        }
    });
    cv::imshow("rgb block", imgBlockGray);
    cv::waitKey(0);
    return;
}

/**
 * @brief Gradient feature
 * 
 */
class GFeat
{
public:
    GFeat(){};
    float feats[8] = {0.0};
};

void normal_features(GFeat& f)
{
    float sum = accumulate(f.feats, f.feats+size(f.feats), 0);
    for (int i = 0; i < 8; ++i) {
        f.feats[i] /= sum;
    }
}

void normal_feats(float f[], int num = 8)
{
    float sum = 0.00;
    for (int i = 0; i < num; ++i) {
        sum += f[i];
    }
    for (int i = 0; i < num; ++i) {
        f[i] /= sum;
    }
}

bool isCandidate(float f[], int num = 8) 
{
    float f_threshold = 0.3;
    for (int i = 0; i < num; ++i) {
        if (f[i] >= f_threshold) 
            return true;
    }
    return false;
}

void find_error_candidate_points(const cv::Mat& pcFlood, const cv::Mat& imgGray)
{
    float feats[80][60][8] = {0.0};
    int cols = pcFlood.cols;
    int rows = pcFlood.rows;
    float dist_threshold = 0.5;
    // extract features
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dist = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(dist)) continue; // no data
            if (dist > dist_threshold) continue; // no differences
            //* extract features
            int count = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    if (i == 0 && j == 0) continue;
                    float p_dist = 0.0;
                    if ((r + j >= 0) && (r + j) < rows && (c + i >= 0) && (c + i) < cols) {
                        p_dist = pcFlood.at<cv::Vec3f>(r+j, c+i)[2];
                        if (isnan(p_dist))
                            p_dist = 0.0;
                    }
                    feats[c][r][count] = abs(dist - p_dist) + 0.00001; // + 0.1mm
                    count += 1;
                }
            }
            normal_feats(feats[c][r]);
        }
    }
    // find candidates
    cv::Mat pcCandidate;
    pcFlood.copyTo(pcCandidate);
    float nanF = std::nan("");
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dist = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(dist) || dist > dist_threshold) {
                pcCandidate.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF); 
                continue;
            }
            if (!isCandidate(feats[c][r])) {
                pcCandidate.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF); 
                continue;
            }
        }
    }

    // * check
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
    cv::Mat imgFlood = pc2detph(pcCandidate, imgGray.size(), cx, cy, fx, fy);
    cv::Mat imgMask;
    // imgMask = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    // imgMask.setTo(1.0, imgFlood != 0.0);
    cv::Mat maskMap = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    maskMap.setTo(1.0, imgFlood != 0.0);
    double minValue, maxValue;
    cv::minMaxLoc(imgFlood, &minValue, &maxValue);
    cv::Mat colorDepthMap = z2colormap(imgFlood, minValue, maxValue);
    cv::Mat markSparseImg = markSparseDepth(imgGray, colorDepthMap, maskMap, 3);
    cv::imshow("candidate", markSparseImg);
    cv::waitKey(0);
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

void find_error_candidate_points2(const cv::Mat& pcFlood, const cv::Mat& imgGray)
{
    float dist_threshold = 0.5;
    int cols = pcFlood.cols;
    int rows = pcFlood.rows;
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
    cv::Mat imgEdge;
    imgEdge = getSobelAbs(matrixImg, 3);
    cv::Mat imgEdgeGray, imgEdgeMask;
    imgEdgeGray = getSobelAbs(imgGray, 3);
    imgEdgeMask = cv::Mat::zeros(imgEdgeGray.size(), imgEdgeGray.type());
    imgEdgeMask.setTo(1.0, imgEdgeGray > 30.0);
    cv::Mat imgBoardGray;
    int dilateSize = 12;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(imgEdgeMask, imgBoardGray, kernel);
    // find candidates
    cv::Mat pcCandidate;
    pcFlood.copyTo(pcCandidate);
    float nanF = std::nan("");
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dist = pcFlood.at<cv::Vec3f>(r, c)[2];
            if (isnan(dist) || dist > dist_threshold) {
                pcCandidate.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF); 
                continue;
            }
            if (imgEdge.at<float>(r, c) < 0.3) {
                pcCandidate.at<cv::Vec3f>(r, c) = cv::Vec3f(nanF, nanF, nanF); 
                continue;
            }
        }
    }


    // * check
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
    cv::Mat imgFlood = pc2detph(pcCandidate, imgGray.size(), cx, cy, fx, fy);
    cv::Mat imgMask;
    // imgMask = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    // imgMask.setTo(1.0, imgFlood != 0.0);
    cv::Mat maskMap = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    maskMap.setTo(1.0, imgFlood != 0.0);
    double minValue, maxValue;
    cv::minMaxLoc(imgFlood, &minValue, &maxValue);
    cv::Mat colorDepthMap = z2colormap(imgFlood, minValue, maxValue);
    cv::Mat markSparseImg = markSparseDepth(imgGray, colorDepthMap, maskMap, 3);
    cv::imshow("candidate", markSparseImg);
    cv::imshow("edge", imgEdgeMask);
    cv::imshow("boarder", imgBoardGray);
    cv::waitKey(0);
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
    const string fn_output = rootPath + "/00000000_upsampling_results.tiff";
    const string strParam = "../../dat/real/camParam/camera_calib/param.txt";
#endif

    // read data
    cv::Mat imgGray = cv::imread(fn_guide, cv::IMREAD_GRAYSCALE);
    cv::Mat pcFlood = cv::imread(fn_sparse, -1);
    //? check pcFlood
    //check_input_depth(pcFlood, imgGray);
    //? find error candidates
    // find_error_candidate_points(pcFlood, imgGray);
    find_error_candidate_points2(pcFlood, imgGray);
    exit(0);

    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
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
    cv::imshow("match", markSparseImg); // * match input point with RGB
    cv::waitKey(0);
    cv::imwrite(rootPath + "tmp_match_input.png", markSparseImg);

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
        cv::imshow("dense", dense);
        cv::waitKey(0);
        cv::imwrite(fn_output, dense);
        break;
#if 0
        double minV, maxV;
        cv::minMaxLoc(dense, &minV, &maxV);
        cv::Mat imgShow = z2colormap(dense, minV, maxV);
        cv::imshow("dense", imgShow);
        cv::waitKey(0);
#endif
    }
    exit(0);
}