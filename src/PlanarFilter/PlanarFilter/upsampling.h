#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

class upsampling
{
public:
	upsampling() {

	};
	upsampling(const cv::Mat&K, cv::Rect&roi, const float& scale): scale_(scale){
		cx_ = K.at<float>(0, 0);
		cy_ = K.at<float>(1, 1);
		fx_ = K.at<float>(0, 2);
		fy_ = K.at<float>(1, 2);	

		K_new_ = K * scale;
		K_new_.at<float>(0, 0) = K_new_.at<float>(0, 0) - roi.x;
		K_new_.at<float>(1, 1) = K_new_.at<float>(1, 1) - roi.y;
		K_new_.at<float>(2, 2) = 1.f;

		roi_.width = roi.width * scale;
		roi_.height = roi.height * scale;
		roi_.x = roi.x * scale;
		roi_.y = roi.y * scale;
	};
	~upsampling() {};

	void run1(const cv::Mat& guide, const cv::Mat& xyz, cv::Mat& dense_depth, cv::Mat& confidence);
	void run2(const cv::Mat& guide, const cv::Mat& sparse_depth, const cv::Rect& roi, cv::Mat& dense_depth, cv::Mat& confidence);
	cv::Mat get_guideroi() { return guide_roi.clone(); };
	cv::Mat get_depthroi() { return depth_roi.clone(); };
	cv::Mat get_newintrinsic() { return K_new_.clone(); };
	void set_guide_intrinsic(cv::Mat& K);//3x3‘z’è
private:
	cv::Mat K_;
	cv::Mat K_new_;
	double fgs_lambda_ = 24;
	double fgs_sigma_color_ = 8;
	double fgs_lambda_attenuation_ = 0.25;
	int fgs_num_iter_ = 2;
	double planar_coeff_ = 1e-5;

	float conf_thresh_ = 0.4f;
	float cx_ = 9.693e+02;
	float cy_ = 5.542e+02;
	float fx_ = 8.057e+02;
	float fy_ = 6.131e+02;
	//float scale_ = 0.25f;
	float scale_ = 1.0f;
	cv::Mat guide_roi;
	cv::Mat depth_roi;
	cv::Rect roi_ = cv::Rect(60, 50, 350, 200);
};