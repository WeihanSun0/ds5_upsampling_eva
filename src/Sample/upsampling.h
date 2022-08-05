#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <list>

class upsampling
{
public:
	// number is method for upsampling
	upsampling();

	~upsampling() {};
	// set SPAD parameters for xyz2depthmap
	void set_cam_paramters(float cx, float cy, float fx, float fy){
		cx_ = cx;
		cy_ = cy;
		fx_ = fx;
		fy_ = fy;
	}
	// paramter setting
	void set_upsampling_parameters(double fgs_lambda_flood, double fgs_sigma_flood, 
									double fgs_lambda_spot, double fgs_sigma_spot,
									int fgs_num_iter_flood, int fgs_num_iter_spot); 
	void get_default_upsampling_parameters(double& fgs_lambda_flood, double& fgs_sigma_flood, 
										double& fgs_lambda_spot, double& fgs_sigma_spot, 
										int& fgs_num_iter_flood, int& fgs_num_iter_spot);
	void set_preprocessing_parameters(int edge_dilate_size, float edge_threshold, 
										int canny_low_threshold, int canny_high_threshold,
										int flood_range);

	void get_default_preprocessing_parameters(int& edge_dilate_size, float& edge_threshold,
										int& canny_low_threshold, int& canny_high_threshold, 
										int& flood_range);	
	// main processing
	bool run(const cv::Mat& rgb, const cv::Mat& flood_pc, const cv::Mat& spot_pc, cv::Mat& dense, cv::Mat& conf);
	// for show depthmap
	cv::Mat get_flood_depthMap() {return this->m_flood_dmap;};
	cv::Mat get_spot_depthMap() {return this->m_spot_dmap;};
	// convert depth map to point cloud
	void depth2pc(const cv::Mat& depth, cv::Mat& pc);
private:
	void clear();
	void initialization(cv::Mat& dense, cv::Mat& conf);
	void fgs_f(const cv::Mat & sparse, const cv::Mat& mask, const cv::Rect& roi, const float& lambda, 
					cv::Mat& dense, cv::Mat& conf);
	void spot_guide_proc(const cv::Mat& guide);
	void spot_depth_proc(const cv::Mat& pc_spot);
	void spot_preprocessing(const cv::Mat& guide, const cv::Mat& pc_spot);
	void flood_guide_proc2(const cv::Mat& guide);
	void flood_depth_proc(const cv::Mat& pc_flood);
	void flood_guide_proc(const cv::Mat& guide);
	void flood_preprocessing(const cv::Mat& guide, const cv::Mat& pc_flood);
	void run_flood(const cv::Mat& img_guide, const cv::Mat& pc_flood, cv::Mat& dense, cv::Mat& conf);
	void run_spot(const cv::Mat& img_guide, const cv::Mat& pc_spot, cv::Mat& dense, cv::Mat& conf);
private:
	const int guide_width = 960;
	const int guide_height = 540;
	const int grid_width = 80;
	const int grid_height = 60;
	int m_mode = 0; // 0: no processing, 1: only flood, 2: only spot, 3: both flood and spot
	// temperary data
	cv::Mat m_flood_mask; // 8UC1
	cv::Mat m_flood_range; // 8UC1
	cv::Mat m_spot_mask; // 8UC1
	cv::Mat m_spot_range; // 8UC1
	cv::Mat m_flood_dmap; // 32FC1 depth map resolution same as guide
	cv::Mat m_spot_dmap; // 32FC1 depth map resuolution same as guide 
	cv::Mat m_flood_grid; // 32FC1
	cv::Mat m_flood_edge; // 32FC1
	cv::Mat m_guide_edge; // 8UC1 0 or 255
	cv::Rect m_flood_roi;
	cv::Rect m_spot_roi;
	// FGS parameters
	double fgs_lambda_flood_ = 48; // 0.1~100
	double fgs_sigma_color_flood_ = 8; // 1~20
	double fgs_lambda_attenuation_ = 0.25;
	double fgs_lambda_spot_ = 700; // 1~1000
	double fgs_sigma_color_spot_ = 5; // 1~20
	int fgs_num_iter_flood = 1; //1~5
	int fgs_num_iter_spot = 3; //1

	// float conf_thresh_ = 0.00f; // 0~1
	// upsampling paramters
	int range_flood = 20; // 2~40
	// int range_spot = 50; // 
	// camera paramters
	float fx_ = 135.51;
	float fy_ = 135.51;
	float cx_ = 159.81;
	float cy_ = 120.41;
	float scale_ = 1.0f;
	// flood preprocessing paramters
	int m_guide_edge_dilate_size = 5; // dilate size of guide image edge
	float m_dist_thresh = 0.5; // (mm) max distance for preprocessing
	float m_depth_edge_thresh = 0.9; // threshold for depth edge
	int m_canny_low_thresh = 45;
	int m_canny_high_thresh = 160;
	float m_max_depth_edge_thresh = 3.0;
	// processing flag
	bool depth_edge_proc_on = true;
	bool guide_edge_proc_on = true;
	cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> m_fgs_filter;
};

