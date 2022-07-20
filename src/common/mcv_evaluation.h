/*
 * Copyright 2022 Sony Semiconductor Solutions Corporation.
 *
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Semiconductor
 * Solutions Corporation.
 * No part of this file may be copied, modified, sold, and distributed in any
 * form or by any means without prior explicit permission in writing from
 * Sony Semiconductor Solutions Corporation.
 *
 */
#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/quality.hpp>

class Evaluation
{
public:
	Evaluation();
	~Evaluation() {};

	enum EDGE {
		CANNY,
		LAPLACIAN,
		SOBEL,
		USER
	};	

	bool is_set_K_ = false;
	bool is_mask_ = true;

	double tau = 0.05;
	double tau_percent = 0.2;

	double gt_z_min_;
	double gt_z_max_;

	void set_gt(const cv::Mat& gt);
	void set_K(const cv::Mat& K);
	void set_edge(const cv::Mat& guide, int type = CANNY);
	void set_est(const cv::Mat& result);
	void set_gt_normal(const cv::Mat& normal);
	void set_normal_thresh_degree(double degree);

	void export_result() {};
	void colorize_depth() {};
	void exec();
	void mask(cv::Mat& map);
	void mask_all();
	void set_edge_from_normal();
	cv::Mat get_error_map(std::string& txt, int i=0);
	void dump(std::string strPath);
	void print();
	void file_output(const std::string& strFn);
	void fs_output(std::fstream& fs);

private:
	void calc_MAD();	   // MAD: Median Absolute Deviation 
	void calc_iRMSE();
	double calc_RMSE();	   // RMSE: Root Mean Square Eroor
	void calc_Edge_RMSE(); // (non)Edge RMSE: Root Mean Square Error
	void calc_PPE();	   // PPE: Point-to-Plane Error
	void calc_SD();	       // SD: Spatial Density

	// Scale - InDependent[mm]
	void calc_VP();	       // VP: Valid Pixel Ratio[%]
	void calc_MNS();	   // MNS: Mean Normal Similarity[��]
	void calc_MENS();      // MENS: Mean Edge Normal Similarity[��]
	//
	void calc_SSIM();
	void calc_BadPix();
	void calc_BadPixRatio();

	double calc_MAE();
	void calc_iMAE();


private:
	cv::Mat K_;

	cv::Mat mask_;
	cv::Mat gt_depth_;
	cv::Mat gt_cloud_mat_;
	cv::Mat gt_normal_;
	cv::Mat gt_normalmap_;
	cv::Mat i_gt_depth_;

	cv::Mat est_depth_;
	cv::Mat est_cloud_mat_;
	cv::Mat est_normal_;
	cv::Mat est_normalmap_;
	cv::Mat i_est_depth_;

	cv::Mat normal_edge_map_;
	cv::Mat eval_area_;
	cv::Mat edge_area_;


	// Scale Dependent[mm]
	// error map
	cv::Mat diff_;
	cv::Mat diff_abs_;
	cv::Mat diff_squared_;

	cv::Mat i_diff_;
	cv::Mat i_diff_abs_;
	cv::Mat i_diff_squared_;

	cv::Mat ppe_dotmap_;
	cv::Mat ns_map_;
	cv::Mat ssim_map_;
	cv::Mat badpix_map_;
	cv::Mat badratiopix_map_;

	cv::Mat sd_map_;
	cv::Mat edge_ns_map_;
	cv::Mat nonedge_ns_map_;
public:
	// evaluation value 
	double mad_;
	double rmse_;
	double irmse_;
	double edge_rmse_;
	double nonedge_rmse_;
	double ppe_;
	double sd_;
	double vp_;
	double mns_;
	double edge_mns_;
	double nonedge_mns_;
	double ssim_;

	double badpix_;
	double badratiopix_;

	double normal_similality_thresh_;

	double mae_;
	double imae_;
};

inline bool isnan_vec(const cv::Vec3d& vec) {
	return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
}