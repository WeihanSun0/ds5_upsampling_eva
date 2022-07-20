
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

class upsampling
{
public:
	upsampling() {

	};
	~upsampling() {};

	void run(const cv::Mat& guide, const cv::Mat& sparse, cv::Mat& dense);

private:
	double fgs_lambda_= 24;
	double fgs_sigma_color_= 8;
	double fgs_lambda_attenuation_ = 0.25;
	int fgs_num_iter_ = 2;
};