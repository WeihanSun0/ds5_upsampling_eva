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
	cv::Mat get_flood_depthMap();
	cv::Mat get_spot_depthMap();
	// convert depth map to point cloud
	void depth2pc(const cv::Mat& depth, cv::Mat& pc);
private:
	void clear();
	void fgs_f(const cv::Mat & guide, const cv::Mat & sparse, const cv::Mat& mask, 
					float fgs_lambda, float fgs_simga_color, float fgs_lambda_attenuation, 
					float fgs_num_iter, const cv::Rect& roi, 
					cv::Mat& dense, cv::Mat& conf);
	void initialization(cv::Mat& dense, cv::Mat& conf);
	void spot_preprocessing(const cv::Mat& guide, const cv::Mat& pc_spot);
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
	double fgs_lambda_flood_ = 24;
	double fgs_sigma_color_flood_ = 8;
	double fgs_lambda_attenuation_ = 0.25;
	double fgs_lambda_spot_ = 700;
	double fgs_sigma_color_spot_ = 5;
	int fgs_num_iter_ = 2;

	float conf_thresh_ = 0.00f;
	// upsampling paramters
	int range_flood = 20;
	int range_spot = 50;
	// camera paramters
	float fx_ = 135.51;
	float fy_ = 135.51;
	float cx_ = 159.81;
	float cy_ = 120.41;
	float scale_ = 1.0f;
	// flood preprocessing paramters
	int m_guide_edge_dilate_size = 10; // dilate size of guide image edge
	float m_dist_thresh = 0.5; // (mm) max distance for preprocessing
	float m_depth_edge_thresh = 0.05; // threshold for depth edge
};

