
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
    float zoom = 0.125;
    int r_dilate = 4;
    int num_iter = 2;
    double sigma_spatial = 3; // 
    double sigma_luma = 2.5; // 24
    double sigma_chroma = 64; // 8.; // 24
    double lambda = 24.;
};