#pragma once

#include <opencv2/opencv.hpp>
#include "FastBilateralSolver.h"

struct PARAM {
    float zoom = 0.125;
    int r_dilate = 4;
    int num_iter = 2;
    double sigma_spatial = 3;
    double sigma_luma = 2.5;
    double sigma_chroma = 8.;
    double lambda = 16.;

    void reset() {
        zoom = 0.125;
        r_dilate = 4;
        sigma_spatial = 3;
        sigma_luma = 2.5;
        sigma_chroma = 8.;
        lambda = 16.;
        num_iter = 2;
    }
};

bool upsampling(
	const cv::Mat& sparse,
	const cv::Mat& guide,
	const PARAM& param,
	cv::Mat& dense
	);