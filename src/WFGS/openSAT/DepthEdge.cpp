#include <omp.h>
// #include "openSAT/DepthEdge.h"
#include "DepthEdge.h"

cv::Mat getSobelAbs(cv::InputArray src_, int tapsize_sobel, float minval)
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

cv::Mat getDoGmagnitude(cv::InputArray src_, const double blur_sigma, const float alpha, cv::InputArray DoG_) {
	const cv::Mat src = src_.getMat();
	cv::Mat dst = cv::Mat::ones(src.size(), CV_32FC1);

	const int tapsizeSobel = 3;
	const cv::Mat edge = getSobelAbs(src, tapsizeSobel);

	cv::Mat blur;
	if (blur_sigma > 0.0f) {
		int bs = static_cast<int>(blur_sigma) * 2 + 1;
		if (bs % 2 == 0) bs++;
		cv::GaussianBlur(edge, blur, cv::Size(bs, bs), blur_sigma);
	}
	else
		blur = edge;

	if (blur.channels() == 1) {
		dst -= alpha * blur;
	}
	else {
		cv::Mat ave = cv::Mat::zeros(src.size(), CV_32FC1);
		std::vector<cv::Mat> split;
		cv::split(blur, split);
		for (int k = 0; k < blur.channels(); k++)
			ave += split[k] / blur.channels();
		dst -= alpha * ave;
	}
	dst.setTo(0.0f, dst < 0.0f);

	if (!DoG_.empty()) {
		cv::Mat DoG = DoG_.getMat();
		DoG = blur.clone();
	}

	return dst;
}