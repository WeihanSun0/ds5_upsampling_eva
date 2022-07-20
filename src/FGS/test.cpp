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
using namespace std;

const string rootFolder = "../../";
const string g_strDepthInputFolder = rootFolder + "dat/artifact/CamWithBump";
const string g_strSourceInputFolder = rootFolder + "dat/source/CamWithBump";
const string g_strMaskInputFolder = rootFolder + "dat/artifact";

//file appendix
const string fa_rgb = "_s0_denoised.png";
const string fa_rgb_noise = "_s0.png";
const string fa_flood = "_flood.tiff";
const string fa_spot = "_spot.tiff";
const string fa_gtDepth = "_gtD.png";
const string fa_camK = "_camK.txt";

int main() {
    int iFov = 30;
    int iFrameIdx = 0;
    char szFileName[255];
    sprintf(szFileName, "%s_%d/house0_round0/polar/%05d%s", g_strSourceInputFolder.c_str(), iFov, iFrameIdx, fa_rgb.c_str());
    cv::Mat imgGray = cv::imread(szFileName, cv::IMREAD_GRAYSCALE);
    sprintf(szFileName, "%s_%d/%05d%s", g_strDepthInputFolder.c_str(), iFov, iFrameIdx, fa_flood.c_str());
    cv::Mat imgFlood= cv::imread(szFileName, -1);
    sprintf(szFileName, "%s_%d/%05d%s", g_strDepthInputFolder.c_str(), iFov, iFrameIdx, fa_spot.c_str());
    cv::Mat imgSpot= cv::imread(szFileName, -1);
    sprintf(szFileName, "%s/mask_flood.tiff", g_strMaskInputFolder.c_str());
    cv::Mat imgMaskFlood = cv::imread(szFileName, -1);
    sprintf(szFileName, "%s/mask_spot.tiff", g_strMaskInputFolder.c_str());
    cv::Mat imgMaskSpot= cv::imread(szFileName, -1);
    sprintf(szFileName, "%s_%d/house0_round0/polar/%05d%s", g_strSourceInputFolder.c_str(), iFov, iFrameIdx, fa_gtDepth.c_str());
    cv::Mat imgGtDepth = read_D(string(szFileName));
    sprintf(szFileName, "%s_%d/house0_round0/polar/%05d%s", g_strSourceInputFolder.c_str(), iFov, iFrameIdx, fa_camK.c_str());
    cv::Mat matCamK = read_K(string(szFileName));
    // FGS
    double fgs_lambda = 24.0;
    double fgs_simga_color = 8.0;
    double fgs_lambda_attenuation = 0.5;
    int fgs_num_iter = 2;
    // canny
    int iCannyThreshold1 = 100;
    int iCannyThreshold2 = 200;
    // viewer
    bool bShowViz = false;
    myViz viz;
    // parameters
    int transparentValue = 40;
    int i_fgs_lambda = 24;
    int i_fgs_sigma_color = 800;
    bool useFlood = false;
    bool useSpot = true;

    cv::namedWindow("show");
    cv::createTrackbar("transVal", "show", &transparentValue, 100);
    cv::createTrackbar("lambda", "show", &i_fgs_lambda, 1000);
    cv::createTrackbar("sigma", "show", &i_fgs_sigma_color, 10000);
    cv::createTrackbar("thresh1", "show", &iCannyThreshold1, 255);
    cv::createTrackbar("thresho2", "show", &iCannyThreshold2, 255);
    //evaluation
    Evaluation eva;
    cv::Mat evaMask = cv::Mat::zeros(imgGtDepth.size(), imgGtDepth.type());
    evaMask(cv::Rect(35, 25, imgGtDepth.cols-70, imgGtDepth.rows-50)).setTo(1);
    // decleration
    cv::Mat imgDense, imgSparseResult, imgMaskResult, imgSmooth, imgEdge;
    cv::Mat imgOverlap_dense, imgOverlap_spot, imgOverlap_flood, imgOverlap_gtDepth, imgInput;// for show
    cv::Mat colorFlood, colorSpot, colorSparseResult, colorMaskResult; 
    cv::Mat pc_gt, pc_sparse, pc_dense;
    cv::Mat non0idx;
    while(1) {
        fgs_lambda = (double)i_fgs_lambda;
        fgs_simga_color = (double)i_fgs_sigma_color/100;
        auto filter = cv::ximgproc::createFastGlobalSmootherFilter(imgGray, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
        if (useFlood) {
            filter->filter(imgFlood.mul(imgMaskFlood), imgSparseResult);
            filter->filter(imgMaskFlood, imgMaskResult);
        }
        if (useSpot) {
            filter->filter(imgSpot.mul(imgMaskSpot), imgSparseResult);
            filter->filter(imgMaskSpot, imgMaskResult);
        }
        imgDense = imgSparseResult / imgMaskResult;
        imgDense.convertTo(imgDense, CV_64F);
        if (useFlood) {
            eva.set_gt(imgGtDepth);
            eva.set_est(imgDense);
        } else { // spot black board
            eva.set_gt(imgGtDepth.mul(evaMask));
            eva.set_est(imgDense.mul(evaMask));
        }
        eva.set_K(matCamK);
        eva.exec();
        double minValue, maxValue;
        int minIndex, maxIndex;
        // cv::minMaxLoc(imgDense, &minValue, &maxValue);
        cv::minMaxLoc(imgGtDepth, &minValue, &maxValue);
        cv::Mat colorDense = z2colormap(imgDense, minValue, maxValue);
        //gt
        cv::Mat colorGtDepth = z2colormap(imgGtDepth, minValue, maxValue);
        imgOverlap_gtDepth = overlap(colorGtDepth, imgGray, (float)transparentValue/100);

        if (useFlood) {
            colorFlood = z2colormap(imgFlood, minValue, maxValue);
            imgOverlap_flood = markSparseDepth(imgGray, colorFlood, imgMaskFlood, 2);
            imgInput = imgOverlap_flood;
        }
        else 
            imgOverlap_flood = cv::Mat::zeros(imgFlood.size(), CV_8UC3);
        if (useSpot) {
            colorSpot = z2colormap(imgSpot, minValue, maxValue);
            imgOverlap_spot = markSparseDepth(imgGray, colorSpot, imgMaskSpot, 2);
            imgInput = imgOverlap_spot;
        }
        else
            imgOverlap_spot = cv::Mat::zeros(imgSpot.size(), CV_8UC3);

        imgOverlap_dense = overlap(colorDense, imgGray, (float)transparentValue/100);
        if (0) { // colorization:  numerator / denominator
            cv::minMaxLoc(imgSparseResult, &minValue, &maxValue);
            colorSparseResult = z2colormap(imgSparseResult, minValue, maxValue);
            cv::minMaxLoc(imgMaskResult, &minValue, &maxValue);
            colorMaskResult = z2colormap(imgMaskResult, minValue, maxValue);
        }
        // smooth & edge
        filter->filter(imgGray, imgSmooth); // 8UC1
        cv::Canny(imgSmooth, imgEdge, iCannyThreshold1, iCannyThreshold2, 3, false);

        // show
        vector<cv::Mat> vecImgs;
        vector<string> vecLabels;
        vecImgs.push_back(imgGray);
        vecLabels.push_back("Guide");
        vecImgs.push_back(imgInput);
        vecLabels.push_back("Sparse");
        vecImgs.push_back(imgOverlap_dense);
        // vecLabels.push_back("Dense (RMSE=" + to_string(eva.calc_RMSE()) + ")");
        vecLabels.push_back("Dense (MAE=" + to_string(eva.calc_MAE()) + ")");
        vecImgs.push_back(imgOverlap_gtDepth);
        vecLabels.push_back("GT");
        vecImgs.push_back(imgSmooth);
        vecLabels.push_back("smooth");
        vecImgs.push_back(imgEdge);
        vecLabels.push_back("Canny");

        cv::Mat imgShow = mergeImages(vecImgs, vecLabels, cv::Size(4,2));
        cv::imshow("show", imgShow);
        // viz
        if (bShowViz) {
            if (useFlood) {
                cv::findNonZero(imgFlood, non0idx);
                cv::rgbd::depthTo3dSparse(imgFlood, matCamK, non0idx, pc_sparse);
            } else {
                cv::findNonZero(imgSpot, non0idx);
                cv::rgbd::depthTo3dSparse(imgSpot, matCamK, non0idx, pc_sparse);
            }
            viz.set(pc_sparse, cv::viz::Color::green(), 7);
            cv::rgbd::depthTo3d(imgDense, matCamK, pc_dense);
            viz.set(pc_dense, imgGray, 3);
            if(0) { // show gt
                cv::rgbd::depthTo3d(imgGtDepth, matCamK, pc_gt);
                viz.set(pc_gt, imgGray, 3);
            }
            viz.show(true); 
        }

        char key = cv::waitKey(100);
        if (key == 'q') 
            break;
        if (key == '1') {
            useSpot ^= 1; 
            useFlood ^= 1;
        }
        
    }
   return 0;
}