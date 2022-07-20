#include "dsviewer_interface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include "z2color.h" // draw color
#include "viewer.h" // visualize 3D
#include "data_loader.h"
#include "mcv_evaluation.h"
#include <list>

using namespace std;

const string rootFolder = "../../";
const string g_strDepthInputFolder = rootFolder + "dat/artifact/CamWithBump";
const string g_strSourceInputFolder = rootFolder + "dat/source/CamWithBump";
const string g_strMaskInputFolder = rootFolder + "dat/artifact";
const string g_strDataPath = rootFolder + "data/debug";

//file appendix
const string fa_rgb = "_s0_denoised.png";
const string fa_rgb_noise = "_s0.png";
const string fa_flood = "_flood.tiff";
const string fa_spot = "_spot.tiff";
const string fa_gtDepth = "_gtD.png";
const string fa_camK = "_camK.txt";

inline cv::Mat exe_fgs(const cv::Mat& guide, const cv::Mat& sparse, const cv::Mat& mask, 
    float fgs_lambda, float fgs_simga_color, float fgs_lambda_attenuation, float fgs_num_iter)
{
    auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
    cv::Mat imgSparseResult, imgMaskResult;
    filter->filter(sparse, imgSparseResult);
    filter->filter(mask, imgMaskResult);
    return imgSparseResult/imgMaskResult;
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

// remove boarder points
void fixHand_1(const cv::Mat& guide, const cv::Mat& sparse, const cv::Mat& mask, 
    int dilateSize, int wndSize, float gradThresh, int iExtPixel, 
    cv::Mat& fixSparse, cv::Mat& fixMask)
{
    cv::Mat imgEdge, imgBoarder, imgEdgeMask;
    imgEdge = getSobelAbs(guide, 3);
    imgEdgeMask = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgEdgeMask.setTo(1.0, imgEdge > 30.0);
    // dilate
    if(dilateSize == 0) dilateSize = 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(imgEdgeMask, imgBoarder, kernel);

    // cv::imshow("edge", imgEdgeMask);
    // cv::imshow("imgBoarder", imgBoarder);
    // cv::waitKey(0);
    // temperary
    sparse.copyTo(fixSparse, imgBoarder == 0);
    mask.copyTo(fixMask, imgBoarder == 0);
}

class Grad
{
public: 
    Grad(float g, float z, int x, int y, int ox, int oy, bool fix=false) : gradVal(g), pz(z), px(x), py(y), ori_x(ox), ori_y(oy), bfix(fix){};
    void fix() {
        bfix = true;
        pz += gradVal;
    };
    float gradVal; // gradient value
    float pz; // point depth
    bool bfix = false; // need fix
    int px; // point position
    int py;
    int ori_x; // orientation
    int ori_y;
};

void calcGrad(const cv::Mat& boarderPnts, const cv::Mat& sparse, list<Grad>& listGrad,
            int wndSize)
{
    int width = boarderPnts.cols;
    int height = boarderPnts.rows;
    // create boarder point list
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float z = boarderPnts.at<float>(r, c);
            if (z != 0.0) {
                listGrad.push_back(Grad(0.0, z, c, r, 0, 0));
            }
        }
    }
    cout << "init list size = " << listGrad.size() << endl;
    // calc gradient
    int num = 0;
    for (auto it = listGrad.begin(); it != listGrad.end(); ++it) {
        int cx = it->px;
        int cy = it->py;
        float z = it->pz;
        float maxGrad = 0.0;
        for (int y = cy-wndSize; y < cy+wndSize; ++y) {
            for (int x = cx-wndSize; x < cx+wndSize; ++x) {
                if (y < 0 || y >= height || x < 0 || x >= width) continue;
                float val = sparse.at<float>(y, x);
                if (val == 0.0) continue;
                float grad = val - z;
                if (maxGrad < abs(grad)) { //* update
                    maxGrad = abs(grad);
                    it->gradVal = grad;
                    it->ori_x = x - cx;
                    it->ori_y = y - cy;
                }
            }
        }
        num += 1;
    }
    cout << "loop list size = " << num << endl;
}


void findErrBoarderPnts(const cv::Mat& edge, list<Grad>& listGrad, float threshod)
{
    int num = 0;
    int width = edge.cols;
    int height = edge.rows;
    for(auto it = listGrad.begin(); it != listGrad.end(); ++it) {
        float grad = it->gradVal;
        if (abs(grad) < threshod) continue;
        // reserve extenstion
        int c = (int)(sqrt(pow(it->ori_x, 2.0) + pow(it->ori_y, 2.0))+0.5);
        float scaleX = (float)it->ori_x/c;
        float scaleY = (float)it->ori_y/c;
        for (int d = 1; d < c+1; ++d) {
            int x = (int)(it->px - scaleX * d + 0.5);
            int y = (int)(it->py - scaleY * d + 0.5);
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            if(edge.at<float>(y, x) != 0.0) { // crossed
                it->bfix = true;
                num += 1;
                break;
            }
        }
    }
    cout << "chose " << num << "/" << listGrad.size() <<  " points" << endl;
}

void drawCandidatePnts(const cv::Mat& img, list<Grad>& listGrad)
{
    cv::Mat colorImg;
    cv::cvtColor(img, colorImg, cv::COLOR_GRAY2BGR);
    int num = 0;
    for(auto it = listGrad.begin(); it != listGrad.end(); ++it) {
        if (it->bfix) {
            cv::circle(colorImg, cv::Point2d(it->px, it->py), 3, cv::Scalar(255, 0, 0));
            num += 1;
        }
    }
    cout << "drawn " << num << " points" << endl;
    cv::imshow("candidate", colorImg);
    cv::waitKey(0);
}

void fixErrBoarderPnts(const cv::Mat& sparse, const cv::Mat& mask, cv::Mat& fixSparse, cv::Mat& fixMask, 
                    const list<Grad>& listGrad)
{
    sparse.copyTo(fixSparse);
    mask.copyTo(fixMask);
    for (auto it = listGrad.begin(); it != listGrad.end(); ++it) {
        if (it->bfix) {
            int x = it->px;
            int y = it->py;
            fixSparse.at<float>(y, x) = it->gradVal + it->pz;
        }
    }
}

const float binaryThreshold =  80.0;
void fixHand_2(const cv::Mat& guide, const cv::Mat& sparse, const cv::Mat& mask, 
    int dilateSize, int wndSize, float gradThresh, int iExtPixel, 
    cv::Mat& fixSparse, cv::Mat& fixMask)
{
    cv::Mat imgEdge, imgBoarder, imgEdgeMask;
    imgEdge = getSobelAbs(guide, 3);
    imgEdgeMask = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgEdgeMask.setTo(1.0, imgEdge > 30.0);
    // dilate
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(imgEdgeMask, imgBoarder, kernel);
    // process boarder points
    cv::Mat imgStrongEdge = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgStrongEdge.setTo(1.0, imgEdge >= binaryThreshold);
    cv::Mat boarderPnts;
    sparse.copyTo(boarderPnts, imgBoarder == 1.0);
    list<Grad> listGrad;
    // calculate gradient
    calcGrad(boarderPnts, sparse, listGrad, wndSize);
    findErrBoarderPnts(imgStrongEdge, listGrad, gradThresh);
    fixErrBoarderPnts(sparse, mask, fixSparse, fixMask, listGrad);
    //* draw
    drawCandidatePnts(imgStrongEdge, listGrad);
    return;
    //!tmp
    sparse.copyTo(fixSparse, imgBoarder == 1.0);
    mask.copyTo(fixMask, imgBoarder == 1.0);
}

void makeShift(cv::Mat& sparse, cv::Mat& mask, const int shift_x, const int shift_y) 
{
    cv::Mat sparseSwap, maskSwap;
    int width = sparse.cols;
    int height = sparse.rows;
    sparseSwap = cv::Mat::zeros(sparse.size(), sparse.type());
    sparse(cv::Rect(0, 0, width-shift_x, height-shift_y)).copyTo(sparseSwap(cv::Rect(shift_x, shift_y, width-shift_x, height-shift_y)));
    maskSwap = cv::Mat::zeros(mask.size(), mask.type());
    mask(cv::Rect(0, 0, width-shift_x, height-shift_y)).copyTo(maskSwap(cv::Rect(shift_x, shift_y, width-shift_x, height-shift_y)));
    sparseSwap.copyTo(sparse);
    maskSwap.copyTo(mask);
}

int main(int argc, char* argv[])
{
    // read data
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

    cv::Mat imgRGB = imgGray; 
    cv::Mat depthMap = imgFlood;
    cv::Mat imgMask = imgMaskFlood;
    //! make shift manually
    makeShift(depthMap, imgMask, 17, 0);
    // flag
    bool bFixHand = true;
    // FGS
    double fgs_lambda = 24.0;
    double fgs_simga_color = 8.0;
    double fgs_lambda_attenuation = 0.5;
    int fgs_num_iter = 2;
    // 3d visualization
    myViz viz;
    // evaluation
    Evaluation eva;
    eva.set_gt(imgGtDepth);
    eva.set_K(matCamK);
    cv::Mat non0idx;
    // parameters
    int transparentValue = 40;
    int i_fgs_lambda = 24;
    int i_fgs_sigma_color = 800;
    // fix hand parameter
    int dilateKernelSize = 10;
    int wndSize = 13;
    int iGradThresh = 10; // cm
    float gradThresh = (float)iGradThresh/100;
    int iExtPixel = 15; 
    cv::namedWindow("show");
    cv::createTrackbar("transVal", "show", &transparentValue, 100);
    cv::createTrackbar("dilateSize", "show", &dilateKernelSize, 60);
    cv::createTrackbar("wndSize", "show", &wndSize, 80);
    cv::createTrackbar("gradThresh", "show", &iGradThresh, 60);
    cv::createTrackbar("extPixel", "show", &iExtPixel, 50);

    // decleration
    cv::Mat imgDense, imgDenseFix, imgSparseResult, imgMaskResult, imgSmooth, imgEdge;
    cv::Mat imgOverlap_dense, imgOverlap_denseFix, imgOverlap_sparse, imgOverlap_sparseFix;// for show
    cv::Mat colorSparse, colorSparseFix, colorSparseResult, colorMaskResult; 
    cv::Mat pc_gt, pc_sparse, pc_dense;

    double minV, maxV;
    cv::minMaxLoc(imgGtDepth, &minV, &maxV);
    cv::Mat colorGtDepth = z2colormap(imgGtDepth, minV, maxV);
    while(1) {
        //update parameter
        gradThresh = (float)iGradThresh/100;
        // normal fgs
        fgs_lambda = (double)i_fgs_lambda;
        fgs_simga_color = (double)i_fgs_sigma_color/100;
        imgDense = exe_fgs(imgRGB, depthMap, imgMask, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
        eva.set_est(imgDense);
        eva.exec();
        double original_error = eva.calc_MAE();

        // fix hand
        cv::Mat fixSparse, fixMask;
        cv::Mat imgEdge = getSobelAbs(imgGray, 3);
        fixHand_2(imgRGB, depthMap, imgMask, 
                dilateKernelSize, wndSize, gradThresh, iExtPixel,
                fixSparse, fixMask);
        // break;
        imgDenseFix = exe_fgs(imgRGB, fixSparse, fixMask, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
        eva.set_est(imgDenseFix);
        eva.exec();
        double fixed_error = eva.calc_MAE();
        // colorize images
        double minValue, maxValue;
        cv::minMaxLoc(imgDense, &minValue, &maxValue);
        cv::Mat colorDense = z2colormap(imgDense, minValue, maxValue);
        cv::Mat colorDenseFix = z2colormap(imgDenseFix, minValue, maxValue);
        colorSparse = z2colormap(depthMap, minValue, maxValue);
        colorSparseFix = z2colormap(fixSparse, minValue, maxValue);
        imgOverlap_sparse = markSparseDepth(imgRGB, colorSparse, imgMask, 2);
        imgOverlap_sparseFix = markSparseDepth(imgRGB, colorSparseFix, fixMask, 2);

        cv::Mat imgOverlap_gtDepth = overlap(colorGtDepth, imgGray, (float)transparentValue/100);
        // drawPoint(imgOverlap_sparseFix, listGrad);

        imgOverlap_dense = overlap(colorDense, imgRGB, (float)transparentValue/100);
        imgOverlap_denseFix = overlap(colorDenseFix, imgRGB, (float)transparentValue/100);

        vector<cv::Mat> vecImgs;
        vector<string> vecLabels;
        vecImgs.push_back(imgRGB);
        vecLabels.push_back("Guide");
        vecImgs.push_back(imgOverlap_sparse);
        vecLabels.push_back("Sparse");
        vecImgs.push_back(imgOverlap_dense);
        vecLabels.push_back("Dense (MAE=" + to_string(original_error) + ")");
        vecImgs.push_back(imgOverlap_gtDepth);
        vecLabels.push_back("GT");
        vecImgs.push_back(imgEdge);
        vecLabels.push_back("Edge");
        vecImgs.push_back(imgOverlap_sparseFix);
        vecLabels.push_back("Sparse Fixed");
        vecImgs.push_back(imgOverlap_denseFix);
        vecLabels.push_back("Dense Fixed (MAE=" + to_string(fixed_error) + ")");
        cv::Mat imgShow = mergeImages(vecImgs, vecLabels, cv::Size(4,2));
        cv::imshow("show", imgShow); 
        cv::imwrite(g_strDataPath + "/dense.tiff", imgDense);
        char key = cv::waitKey(100);
        if (key == 'q') 
            break;
        if (key == '1') {
            bFixHand ^= 1; 
        }
    }
    exit(0);
}