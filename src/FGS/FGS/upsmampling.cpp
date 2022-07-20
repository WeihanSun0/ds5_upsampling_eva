#include "upsampling.h"
// #include "FastBilateralSolver.h"


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
