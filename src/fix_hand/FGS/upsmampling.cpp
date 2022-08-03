#include "upsampling.h"


void upsampling::run(
	const cv::Mat& guide,
	const cv::Mat& sparse,
	cv::Mat& dense
) 
{
    cv::Mat conf, mask;
    conf = cv::Mat::zeros(guide.size(), CV_32FC1);
    conf.setTo(1.0, sparse != 0.0);
    conf.copyTo(mask);
    auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide, 
                                                this->fgs_lambda_, this->fgs_sigma_color_, 
                                                this->fgs_lambda_attenuation_, this->fgs_num_iter_);
    cv::Mat imgSparseResult, imgMaskResult;
    filter->filter(sparse, imgSparseResult);
    filter->filter(mask, imgMaskResult);
    dense = imgSparseResult/imgMaskResult;
}

inline cv::Mat getUpsamplingRange(const cv::Mat& mask, int r)
{
	cv::Mat imgDilate;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(r, r));
	cv::dilate(mask, imgDilate, kernel);
	return imgDilate;
}


// * with NaN area
void upsampling::run2(
	const cv::Mat& guide,
	const cv::Mat& sparse,
	cv::Mat& dense
) 
{
    int range = 60;
    cv::Mat conf, mask;
    conf = cv::Mat::zeros(guide.size(), CV_32FC1);
    conf.setTo(1.0, sparse != 0.0);
    conf.copyTo(mask);
    cv::Mat rangeImg = getUpsamplingRange(mask, range);
    auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide, 
                                                this->fgs_lambda_, this->fgs_sigma_color_, 
                                                this->fgs_lambda_attenuation_, this->fgs_num_iter_);
    cv::Mat imgSparseResult, imgMaskResult;
    filter->filter(sparse, imgSparseResult);
    filter->filter(mask, imgMaskResult);
    dense = imgSparseResult/imgMaskResult;
    dense.setTo(std::nan(""), rangeImg == 0.0);
}