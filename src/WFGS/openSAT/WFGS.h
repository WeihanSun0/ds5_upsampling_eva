#pragma once
#ifndef _WFGS
#define _WFGS
// weighted FGS Filter 													
// https://sites.google.com/site/globalsmoothing/				

#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

inline int L2_index(const cv::Vec3b& p1, const cv::Vec3b& p2) {
	return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

void HorizontalPass(const cv::Mat& w_h, const cv::Mat& src, cv::Mat& dst, const double& lambda);
void VerticalPass(const cv::Mat& w_v, const cv::Mat& src, cv::Mat& dst, const double& lambda);

class WFGS
{
public:
	WFGS() {};
	~WFGS() {};
	void Init(const int& channels);
	bool Calculate_weight(const cv::Mat& guide_);

	bool Execute(const cv::Mat& src, const cv::Mat& conf, cv::Mat& dst, cv::Mat& dst_conf)const;

	// parameter
	const int itermax_ = 3;
	double sigma_color_ = 4;//4.0//0.5 spiral
	double lambda_attenuation_ = 1.0;
	double lambda_ = 1.0;//pre 1, post 16 
	const float unknown_val_ = -1.f;//-1
	const float float_min_val_ = 0.f;//1.0E-6f;
private:
	bool is_weight_ready = false;
	void Calculate_lut(const int& channels);
	std::vector<float> LUT;
	cv::Mat w_h_, w_v_;
};

inline void WFGS::Init(const int& channels){
	Calculate_lut(channels);
}

inline void WFGS::Calculate_lut(const int& channels) {
	const int num_levels = 256 * 256 * channels;
	LUT.resize(num_levels);
	const float inv_sigmacolor = 1.f / static_cast<float>(sigma_color_);
	const int scale = (channels == 1) ? 3 : 1;

	for (int i = 0; i < num_levels; ++i) {
		LUT[i] = exp(-sqrt(static_cast<float>(i * scale)) * inv_sigmacolor);//eq.2
	}
}

inline bool WFGS::Execute(
	const cv::Mat& src, 
	const cv::Mat& conf, 
	cv::Mat& dst, 
	cv::Mat& dst_conf
)const{
	CV_Assert(!src.empty() && src.depth() == CV_32F && src.channels() <= 4);

	// init
	cv::Mat output = src.clone();
	output.setTo(0.0f, output == unknown_val_);//unknownVal

	bool isCalcConf = false;
	if (!conf.empty()) {
		isCalcConf = true;
		dst_conf = conf.clone();
		output = output.mul(conf);
	}

	cv::Mat tmp, input;
	tmp.create(output.size(), output.type());

	auto lambda = lambda_;
	for (int iter = 0; iter < itermax_; ++iter) {
		input = output.clone();
		HorizontalPass(w_h_, input, tmp, lambda);
		VerticalPass(w_v_, tmp, output, lambda);
		lambda *= lambda_attenuation_;
	}

	if (isCalcConf) {
		lambda = lambda_;
		cv::Mat conf_input;
		for (int iter = 0; iter < itermax_; ++iter) {
			conf_input = dst_conf.clone();
			HorizontalPass(w_h_, conf_input, tmp, lambda);
			VerticalPass(w_v_, tmp, dst_conf, lambda);
			lambda *= lambda_attenuation_;
		}

		cv::Mat div = cv::Mat::ones(conf.size(), CV_32FC1);
		dst_conf.copyTo(div, dst_conf > float_min_val_);
		dst_conf.setTo(0.0f, dst_conf <= float_min_val_);

		output /= div;
		output.setTo(unknown_val_, dst_conf <= float_min_val_);//unknownVal
	}
	dst = output;

	return true;
}


#endif
