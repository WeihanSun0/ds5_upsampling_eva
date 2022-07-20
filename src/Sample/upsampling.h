#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

class upsampling
{
public:
	// number is method for upsampling
	upsampling() {};

	~upsampling() {};
	// set SPAD parameters for xyz2depthmap
	void set_cam_paramters(float cx, float cy, float fx, float fy){
		cx_ = cx;
		cy_ = cy;
		fx_ = fx;
		fy_ = fy;
	}
	// upsampling paramter
	void set_upsampling_parameters(float fgs_lambda_flood, float fgs_sigma_flood, 
									float fgs_lambda_spot, float fgs_sigma_spot) 
	{ 
		fgs_lambda_flood_ = fgs_lambda_flood; 
		fgs_sigma_color_flood_ = fgs_sigma_flood; 
		fgs_lambda_spot_ = fgs_lambda_spot; 
		fgs_sigma_color_spot_ = fgs_sigma_spot; 
	};
	// main processing
	bool run(const cv::Mat& rgb, const cv::Mat& flood_pc, const cv::Mat& spot_pc, cv::Mat& dense, cv::Mat& conf);

	// convert depth map to point cloud
	void depth2pc(const cv::Mat& depth, cv::Mat& pc);
private:
	// FGS parameters
	double fgs_lambda_flood_ = 24;
	double fgs_sigma_color_flood_ = 8;
	double fgs_lambda_attenuation_ = 0.25;
	double fgs_lambda_spot_ = 700;
	double fgs_sigma_color_spot_ = 5;
	int fgs_num_iter_ = 2;

	float conf_thresh_ = 0.00f;
	// upsampling paramters
	int range_flood = 15;
	int range_spot = 50;
	// camera paramters
	float fx_ = 135.51;
	float fy_ = 135.51;
	float cx_ = 159.81;
	float cy_ = 120.41;
	float scale_ = 1.0f;
};

