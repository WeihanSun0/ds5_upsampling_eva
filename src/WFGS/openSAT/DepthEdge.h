#include <opencv2/opencv.hpp>

cv::Mat getSobelAbs(
	cv::InputArray src_,
	int tapsize_sobel = 3, 
	float minval = 0.0f/*1.0E-2f*/
);

cv::Mat getDoGmagnitude(
	cv::InputArray src_,
	const double blur_sigma = 5.0,
	const float alpha = 1.0f,
	cv::InputArray DoG_ = cv::noArray()/*float*/
);