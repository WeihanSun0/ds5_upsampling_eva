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
    auto filter = cv::ximgproc::createFastBilateralSolverFilter(guide,
                                                this->sigma_spatial, this->sigma_luma, this->sigma_chroma, 
                                                this->lambda, this->num_iter);
    cv::Mat imgSparseResult, imgMaskResult;
    filter->filter(sparse, conf, imgSparseResult);
    filter->filter(mask, conf, imgMaskResult);
    dense = imgSparseResult/imgMaskResult;
}
