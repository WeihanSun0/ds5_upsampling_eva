#include "dsviewer_interface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include "z2color.h" // draw color
#include "viewer.h" // visualize 3D
#include <list>


using namespace std;

const string rootPath = "../../";
const string strDataPath = rootPath + "dat/real/hand_test";
const string strGuide = strDataPath + "/0001_rgb_img.png";
const string strSparsePointCloud = strDataPath + "/0001_flood_pc.tiff";
const string strParam = rootPath + "dat/real/camParam/camera_calib/param.txt";

inline cv::Mat exe_fgs(const cv::Mat& guide, const cv::Mat& sparse, const cv::Mat& mask, 
    float fgs_lambda, float fgs_simga_color, float fgs_lambda_attenuation, float fgs_num_iter)
{
    auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
    cv::Mat imgSparseResult, imgMaskResult;
    filter->filter(sparse.mul(mask), imgSparseResult);
    filter->filter(mask, imgMaskResult);
    return imgSparseResult/imgMaskResult;
}

class GRAD
{
public: 
    GRAD(){};
    GRAD(float _v, float _g, int _x, int _y, int _ori_x, int _ori_y): 
        val(_v), grad(_g), x(_x), y(_y), ori_x(_ori_x), ori_y(_ori_y), bFix(false){};
    float val = 0.0; // original value
    float grad = 0.0; // gradient 
    int x = 0; 
    int y = 0;
    int ori_x = 0; // orientation x
    int ori_y = 0; // orientation y
    bool bFix = false; // need fix
};

void drawPoint(cv::Mat& colorImg, const list<GRAD>& listPnts)
{
    //map<pair<int, int>, GRAD>::reverse_iterator it;
    int total = 0;
    int count = 0;
    auto it = listPnts.begin();
    while(it != listPnts.end()) {
        if (it->bFix == true) {
            count += 1;;
            cv::circle(colorImg, cv::Point(it->x, it->y), 7, cv::Scalar(255, 255, 255), 1);
        }
        total += 1;
        it++;
    }
    cout <<  "draw fixed num = " << count << "/" << total << "/" << listPnts.size() << endl;
}

void calcGradient(const cv::Mat& img, const cv::Mat & boarder, list<GRAD>& listGrad, int wndSize = 32) 
{
    int count = 0;
    int width = img.cols;
    int height = img.rows;
    boarder.forEach<float>([&listGrad, &img, wndSize, width, height](float & p, const int* position) -> void {
        if (p != 0.0) {
            int x = position[1];
            int y = position[0];
            float maxGrad = 0.0;
            int ori_x = 0, ori_y = 0;
            for (int r = y - wndSize; r < y + wndSize; ++r) { // y scan
                for (int c = x - wndSize; c < x + wndSize; ++c) { // x scan
                    if (r < 0 || r >= height || c < 0 || c >= width) { // out
                       continue; 
                    }
                    float val = img.at<float>(r,c);
                    if (val == 0.0) continue;
                    // found 
                    float grad = val - p; 
                    if (abs(grad) > maxGrad) {
                        maxGrad = grad;
                        ori_x = c - x;
                        ori_y = r - y;
                    }
                }
            }
            listGrad.push_back(GRAD(p, maxGrad, x, y, ori_x, ori_y));
        }
    });

}

void calcGradient1(const cv::Mat& img, list<GRAD>& listGrad, int wndSize = 32) 
{
    cv::Mat mask =  cv::Mat::zeros(img.size(), img.type());
    mask.setTo(1.0, img != 0.0);
    int count = 0;
    int width = img.cols;
    int height = img.rows;
    img.forEach<float>([&listGrad, &img, wndSize, width, height](float & p, const int* position) -> void {
        if (p != 0.0) {
            int x = position[1];
            int y = position[0];
            float maxGrad = 0.0;
            int ori_x = 0, ori_y = 0;
            for (int r = y - wndSize; r < y + wndSize; ++r) { // y scan
                for (int c = x - wndSize; c < x + wndSize; ++c) { // x scan
                    if (r < 0 || r >= height || c < 0 || c >= width) { // out
                       continue; 
                    }
                    float val = img.at<float>(r,c);
                    if (val == 0.0) continue;
                    // found 
                    float grad = val - p; 
                    if (abs(grad) > maxGrad) {
                        maxGrad = grad;
                        ori_x = c - x;
                        ori_y = r - y;
                    }
                }
            }
            listGrad.push_back(GRAD(p, maxGrad, x, y, ori_x, ori_y));
        }
    });
}

void findAndFix(list<GRAD>& listGrad, const cv::Mat& edge, cv::Mat& fixSparse, cv::Mat& fixMask,
    float fGradThresh = 0.1, // m
    int  iExtPixels = 20 // pixel
)
{

    int count = 0;
    int total = 0;
    auto it = listGrad.begin();
    while(it != listGrad.end()) {
        //! test
        it->bFix = true;
        fixSparse.at<float>(it->y, it->x) = 0.0;
        fixMask.at<float>(it->y, it->x) = 0.0;
        it++;
    }
}

void findAndFix1(list<GRAD>& listGrad, const cv::Mat& edge, cv::Mat& fixSparse, cv::Mat& fixMask,
    float fGradThresh = 0.1, // m
    int  iExtPixels = 20 // pixel
    )
{
    int count = 0;
    int total = 0;
    auto it = listGrad.begin();
    while(it != listGrad.end()) {
        total += 1;
        if (it->grad < fGradThresh) { // not boundary
            it++;
            continue;
        } 
        float c = sqrt(it->ori_x^2 + it->ori_y^2);
        // -grad orientation extent
        float scale_x = -float(it->ori_x) / c;
        float scale_y = -float(it->ori_y) / c;
        for (int i = 1; i < iExtPixels; ++i) {
            int ex = it->x + (int)(i * scale_x);
            int ey = it->y + (int)(i * scale_y);
            if (ex >= 0 && ex < edge.cols && ey >= 0 && ey < edge.rows) {
                if(edge.at<uchar>(ey, ex) != 0) // hit
                    it->bFix = true;
                    // it->val += it->grad; // fix
                    // fixSparse.at<float>(it->y, it->x) = it->val;
                    count += 1;
                    goto LABEL;
            }
        }
// goto
LABEL:
        it++;
    }
    cout << " fixed num = " << count << "/" << total << "/" << listGrad.size() << endl;
}

void drawGrad(cv::Mat& img, const list<GRAD>& listGrad, float thresh)
{
    auto it = listGrad.begin();
    while(it != listGrad.end()) {
        if (abs(it->grad) > thresh) {
            if(it->grad > 0) { // !RED to backgrand
                cv::arrowedLine(img, cv::Point(it->x, it->y), cv::Point(it->x+it->ori_x, it->y+it->ori_y), 
                    cv::Scalar(0, 0, 255), 1);
            }
            if(it->grad < 0) { // ?BLUE to foregrand 
                cv::arrowedLine(img, cv::Point(it->x, it->y), cv::Point(it->x+it->ori_x, it->y+it->ori_y), 
                    cv::Scalar(255, 0, 0), 1);
            }
        }
        it++;
    }
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

void fixHand(const cv::Mat& guide, const cv::Mat& sparse, const cv::Mat& mask, 
    int cannyThresh1, int cannyThresh2, int dilateSize, int wndSize, float gradThresh, int iExtPixel, 
    cv::Mat& fixSparse, cv::Mat& fixMask, list<GRAD>& listGrad)
{
    sparse.copyTo(fixSparse);
    mask.copyTo(fixMask);
    cv::Mat imgEdge, imgBoarder, imgEdgeMask;
    // detect edge
    // cv::Canny(guide, imgEdge, cannyThresh1, cannyThresh2);
    imgEdge = getSobelAbs(guide, 3);
    imgEdgeMask = cv::Mat::zeros(imgEdge.size(), imgEdge.type());
    imgEdgeMask.setTo(1.0, imgEdge > 30.0);
    // dilate
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(imgEdgeMask, imgBoarder, kernel);
#if 0 
    cv::Mat colorEdge, colorBoarder;
    cv::Mat colorMapping_edge, colorMapping_guide, colorMapping_boarder;
    cv::cvtColor(imgEdge, colorEdge, cv::COLOR_GRAY2BGR);
    cv::cvtColor(imgBoarder, colorBoarder, cv::COLOR_GRAY2BGR);
#endif
   // calc gradient
    cv::Mat imgBoarderPoint, imgBoarderMask;
    // imgBoarder.convertTo(imgBoarderMask, CV_32FC1);
    imgBoarderPoint = sparse.mul(imgBoarder);
    calcGradient(sparse, imgBoarderPoint, listGrad, wndSize);
    // find and fix pnts
#if 0 
    double minValue, maxValue;
    cv::minMaxLoc(sparse, &minValue, &maxValue);
    cv::Mat colorSparse = z2colormap(sparse, minValue, maxValue);
    colorMapping_edge = markSparseDepth(colorEdge, colorSparse, mask, 2);
    colorMapping_guide = markSparseDepth(guide, colorSparse, mask, 2);
    colorMapping_boarder = markSparseDepth(colorBoarder, colorSparse, mask, 2); 
    
    cv::Mat colorMapping_edgeFix, colorMapping_guideFix, colorMapping_boarderFix;
    cv::Mat colorSparseFix = z2colormap(fixSparse, minValue, maxValue);
    colorMapping_edgeFix = markSparseDepth(colorEdge, colorSparseFix, mask, 2);
    colorMapping_guideFix = markSparseDepth(guide, colorSparseFix, mask, 2);
    colorMapping_boarderFix = markSparseDepth(colorBoarder, colorSparseFix, mask, 2);
    // draw gradient
    cv::Mat colorGrad;
    colorMapping_edge.copyTo(colorGrad);
    drawGrad(colorGrad, listGrad, gradThresh);
    cv::imwrite(strDataPath+"/colorGrad_edge.png", colorGrad);
    colorMapping_boarder.copyTo(colorGrad);
    drawGrad(colorGrad, listGrad, gradThresh);
    cv::imwrite(strDataPath+"/colorGrad_boarder.png", colorGrad);
    colorMapping_guide.copyTo(colorGrad);
    drawGrad(colorGrad, listGrad, gradThresh);
    cv::imwrite(strDataPath+"/colorGrad_guide.png", colorGrad);
#endif
    findAndFix(listGrad, imgEdge, fixSparse, fixMask, gradThresh, iExtPixel);
#if 0 
    colorMapping_guide.copyTo(colorGrad);
    drawPoint(colorGrad, listGrad);
    cv::imwrite(strDataPath+"/colorGrad_fixPoints.png", colorGrad);
    // drawPoint(colorMap_fixed, listGrad);
    vector<cv::Mat> vecImgs;
    vector<string> vecLabels;
    vecImgs.push_back(colorMapping_edge);
    vecLabels.push_back("edge");
    vecImgs.push_back(colorMapping_boarder);
    vecLabels.push_back("boarder");
    vecImgs.push_back(colorMapping_guide);
    vecLabels.push_back("guid");
    vecImgs.push_back(colorMapping_edgeFix);
    vecLabels.push_back("edgeFix");
    vecImgs.push_back(colorMapping_boarderFix);
    vecLabels.push_back("boarderFixe");
    vecImgs.push_back(colorMapping_guideFix);
    vecLabels.push_back("guideFix");
    vecImgs.push_back(colorGrad);
    vecLabels.push_back("grad");
    cv::Mat tImg = mergeImages(vecImgs, vecLabels, cv::Size(3, 2));
    cv::imshow("fix", tImg);
    cv::waitKey(0);
#endif
}


int main(int argc, char* argv[])
{
    // read data
    cv::Mat imgRGB = cv::imread(strGuide, cv::IMREAD_GRAYSCALE);
    cv::Mat pcFlood = cv::imread(strSparsePointCloud, -1);
    map<string, float> params;
    if(!read_param(strParam, params))
        cout << "open param failed" <<endl;
    float cx, cy, fx, fy;
    get_rgb_params(params, cx, cy, fx, fy);
    cv::Mat depthMap = pc2detph(pcFlood, imgRGB.size(), cx, cy, fx, fy);
    cv::Mat imgMask = cv::Mat::zeros(depthMap.size(), CV_32F);
    imgMask.setTo(1.0, depthMap != 0.0);
    // flag
    bool bFixHand = true;
    // FGS
    double fgs_lambda = 24.0;
    double fgs_simga_color = 8.0;
    double fgs_lambda_attenuation = 0.5;
    int fgs_num_iter = 2;
    // canny
    int iCannyThreshold1 = 70;
    int iCannyThreshold2 = 170;
    // viewer
    bool bShowViz = false;
    myViz viz;
    // parameters
    int transparentValue = 40;
    int i_fgs_lambda = 24;
    int i_fgs_sigma_color = 800;
    // fix hand parameter
    int dilateKernelSize = 20;
    int wndSize = 12;
    int iGradThresh = 3; // cm
    float gradThresh = (float)iGradThresh/100;
    int iExtPixel = 15; 
    cv::namedWindow("show");
    cv::createTrackbar("transVal", "show", &transparentValue, 100);
    cv::createTrackbar("dilateSize", "show", &dilateKernelSize, 60);
    cv::createTrackbar("wndSize", "show", &wndSize, 80);
    cv::createTrackbar("gradThresh", "show", &iGradThresh, 60);
    cv::createTrackbar("extPixel", "show", &iExtPixel, 50);
    cv::createTrackbar("thresh1", "show", &iCannyThreshold1, 255);
    cv::createTrackbar("thresho2", "show", &iCannyThreshold2, 255);
    // decleration
    cv::Mat imgDense, imgDenseFix, imgSparseResult, imgMaskResult, imgSmooth, imgEdge;
    cv::Mat imgOverlap_dense, imgOverlap_denseFix, imgOverlap_sparse, imgOverlap_sparseFix;// for show
    cv::Mat colorSparse, colorSparseFix, colorSparseResult, colorMaskResult; 
    cv::Mat pc_gt, pc_sparse, pc_dense;
    while(1) {
        //update parameter
        gradThresh = (float)iGradThresh/100;
        // normal fgs
        fgs_lambda = (double)i_fgs_lambda;
        fgs_simga_color = (double)i_fgs_sigma_color/100;
        imgDense = exe_fgs(imgRGB, depthMap, imgMask, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
        // fix hand
        cv::Mat fixSparse, fixMask;
        list<GRAD> listGrad;
        fixHand(imgRGB, depthMap, imgMask, 
                iCannyThreshold1, iCannyThreshold2, dilateKernelSize, wndSize, gradThresh, iExtPixel,
                fixSparse, fixMask, listGrad);
        // break;
        imgDenseFix = exe_fgs(imgRGB, fixSparse, fixMask, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);

        // colorize images
        double minValue, maxValue;
        cv::minMaxLoc(imgDense, &minValue, &maxValue);
        cv::Mat colorDense = z2colormap(imgDense, minValue, maxValue);
        cv::Mat colorDenseFix = z2colormap(imgDenseFix, minValue, maxValue);
        colorSparse = z2colormap(depthMap, minValue, maxValue);
        colorSparseFix = z2colormap(fixSparse, minValue, maxValue);
        imgOverlap_sparse = markSparseDepth(imgRGB, colorSparse, imgMask, 2);
        imgOverlap_sparseFix = markSparseDepth(imgRGB, colorSparseFix, fixMask, 2);
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
        vecLabels.push_back("Dense");
        vecImgs.push_back(imgRGB);
        vecLabels.push_back("Guide");
        vecImgs.push_back(imgOverlap_sparseFix);
        vecLabels.push_back("Sparse Fixed");
        vecImgs.push_back(imgOverlap_denseFix);
        vecLabels.push_back("Dense Fixed");
        cv::Mat imgShow = mergeImages(vecImgs, vecLabels, cv::Size(3,2));
        cv::imshow("show", imgShow); 
        cv::imwrite(strDataPath + "/dense.tiff", imgDense);
        char key = cv::waitKey(100);
        if (key == 'q') 
            break;
        if (key == '1') {
            bFixHand ^= 1; 
        }
    }
    exit(0);
}