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
#include"mcv_evaluation.h"
#include"depth_evaluation.h"

Evaluation::Evaluation() {
	mad_ = std::numeric_limits<double>::quiet_NaN();
	rmse_ = std::numeric_limits<double>::quiet_NaN();
	irmse_ = std::numeric_limits<double>::quiet_NaN();
	edge_rmse_ = std::numeric_limits<double>::quiet_NaN();
	nonedge_rmse_ = std::numeric_limits<double>::quiet_NaN();
	ppe_ = std::numeric_limits<double>::quiet_NaN();
	sd_ = std::numeric_limits<double>::quiet_NaN();
	vp_ = std::numeric_limits<double>::quiet_NaN();
	mns_ = std::numeric_limits<double>::quiet_NaN();
	edge_mns_ = std::numeric_limits<double>::quiet_NaN();
	nonedge_mns_ = std::numeric_limits<double>::quiet_NaN();
	ssim_ = std::numeric_limits<double>::quiet_NaN();
	badpix_ = std::numeric_limits<double>::quiet_NaN();
	badratiopix_ = std::numeric_limits<double>::quiet_NaN();
	set_normal_thresh_degree(20);
}


// set ground truth 
void Evaluation::set_gt(const cv::Mat& gt) {
	gt_depth_ = gt.clone();
	gt_depth_.convertTo(gt_depth_, CV_64FC1);
	i_gt_depth_ = 1. / gt_depth_;
	cv::minMaxIdx(gt_depth_, &gt_z_min_, &gt_z_max_);
}

// set camera parameter cx, cy, fx, fy
void Evaluation::set_K(const cv::Mat& K) {
	K_ = K.clone();
	K_.convertTo(K_, CV_64FC1);
}


// set estimation object
void Evaluation::set_est(const cv::Mat& result) {
	est_depth_ = result.clone();
	est_depth_.convertTo(est_depth_, CV_64FC1);
	i_est_depth_ = 1. / est_depth_;
}

// set normal image
void Evaluation::set_gt_normal(const cv::Mat& normal)
{
	gt_normal_ = normal.clone();
	gt_normal_.convertTo(gt_normal_, CV_64FC3);
}


void Evaluation::set_normal_thresh_degree(double degree)
{
	double radian = degree * CV_PI / 180.0;
	normal_similality_thresh_ = std::cos(radian);
}

void Evaluation::set_edge(const cv::Mat& guide, int type) {
	// grayscale
	cv::Mat img = guide.clone();
	if (img.channels() == 3) {
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	}

	cv::Mat edge, img_s_x, img_s_y;
	switch (type){
	case CANNY:
		cv::Canny(img, edge, 125, 255);
		break;

	case LAPLACIAN:
		cv::Laplacian(img, edge, 3);
		cv::convertScaleAbs(edge, edge, 1, 0);
		cv::threshold(edge, edge, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		break;

	case SOBEL:
		cv::Sobel(img, img_s_x, CV_8UC1, 1, 0, 3);
		cv::Sobel(img, img_s_y, CV_8UC1, 0, 1, 3);
		edge = abs(img_s_x) + abs(img_s_y);
		cv::convertScaleAbs(edge, edge, 1, 0);
		cv::threshold(edge, edge, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		break;
	case USER:
		edge = img;
	default:
		break;
	}

	edge_area_ = edge;
}

void Evaluation::exec()
{
	cv::rgbd::depthTo3d(gt_depth_, K_, gt_cloud_mat_);
	cv::rgbd::depthTo3d(est_depth_, K_, est_cloud_mat_);

	diff_ = (gt_depth_ - est_depth_);
	diff_abs_ = cv::abs(diff_);
	diff_squared_ = diff_.mul(diff_);

	i_diff_ = (i_gt_depth_ - i_est_depth_);
	i_diff_abs_ = cv::abs(i_diff_);
	i_diff_squared_ = i_diff_.mul(i_diff_);


	// calculate normal of gt, est
	if(gt_normal_.empty()) calcNormal(gt_depth_, K_, gt_cloud_mat_, gt_normal_);
	calcNormal(est_depth_, K_, est_cloud_mat_, est_normal_);

	mask_ = cv::Mat::ones(gt_depth_.size(), CV_8U);
	mask_.setTo(0, est_depth_ == 0);
	mask_.setTo(0, gt_depth_ == 0);

	if (is_mask_) {
		mask(diff_);
		mask(diff_abs_);
		mask(diff_squared_);
	}
	// set_edge_from_normal();
	calc_MAE();
	calc_iMAE();
	calc_MAD();
	calc_RMSE();
	calc_iRMSE();
	calc_Edge_RMSE();
	calc_PPE();
	calc_SD();
	calc_VP();
	calc_MNS();
	calc_MENS();
	
	calc_SSIM();
	calc_BadPix();
	calc_BadPixRatio();
}

void Evaluation::mask(cv::Mat& map)
{
	map.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
}

void Evaluation::mask_all()
{
	diff_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	diff_abs_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	diff_squared_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	ppe_dotmap_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	ns_map_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	ssim_map_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	badpix_map_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
	badratiopix_map_.setTo(std::numeric_limits<double>::quiet_NaN(), mask_ == 0);
}

void Evaluation::set_edge_from_normal()
{
	normal_edge(gt_normal_, normal_edge_map_);
	edge_area_=cv::Mat::zeros(gt_normal_.size(),CV_8U);
	edge_area_.setTo(255, normal_edge_map_ < normal_similality_thresh_);
}


//---------------------------------------------------
// Scale Dependent[mm]
//---------------------------------------------------

// MAD: Median Absolute Deviation 
void Evaluation::calc_MAD() {
	if (is_mask_) mask(diff_abs_);
	mad_ = medianMat(diff_abs_);
}

// iRMSE : inverse Root Mean Square Eroor
void Evaluation::calc_iRMSE() {
	if (is_mask_) mask(i_diff_squared_);
	irmse_ = calcRootMeanValid(i_diff_squared_);
}

// RMSE : Root Mean Square Eroor
double Evaluation::calc_RMSE() {
	if (is_mask_) mask(diff_squared_);
	rmse_ = calcRootMeanValid(diff_squared_);
	return rmse_;
}

// Edge RMSE : Root Mean Square Error
void Evaluation::calc_Edge_RMSE() {
	if (edge_area_.empty()) {
		puts("edge empty");
		return;
	}
	// Edge RMSE
	cv::Mat edge_diff_squared_ = diff_squared_.clone();
	edge_diff_squared_.setTo(std::numeric_limits<double>::quiet_NaN(), edge_area_ == 0);
	if(is_mask_) mask(edge_diff_squared_);
	edge_rmse_ = calcRootMeanValid(edge_diff_squared_);

	// Non Edge RMSE
	cv::Mat nonedge_diff_squared_ = diff_squared_.clone();
	nonedge_diff_squared_.setTo(std::numeric_limits<double>::quiet_NaN(), edge_area_ != 0);
	mask(nonedge_diff_squared_);
	nonedge_rmse_ = calcRootMeanValid(nonedge_diff_squared_);
}

// PPE: Point-to-Plane Error
void Evaluation::calc_PPE() {
	cv::Mat diff_xyz = est_cloud_mat_ - gt_cloud_mat_;	
	cv::Mat dotmap = calcDotmap(diff_xyz, gt_normal_);

	ppe_dotmap_ = dotmap.mul(dotmap);
	if (is_mask_) mask(ppe_dotmap_);
	ppe_ = calcRootMeanValid(ppe_dotmap_);
};



// SD: Spatial Density
void Evaluation::calc_SD() {
	cv::Mat est_cloud_mat_pad_;
	cv::copyMakeBorder(est_cloud_mat_, est_cloud_mat_pad_, 1, 1, 1, 1, CV_HAL_BORDER_REFLECT);
	sd_map_=cv::Mat::zeros(est_depth_.size(), CV_64FC1);
	for (int y = 1; y < est_cloud_mat_pad_.rows - 1; ++y) {
		for (int x = 1; x < est_cloud_mat_pad_.cols - 1; ++x) {
			cv::Vec3d Vi = est_cloud_mat_pad_.at<cv::Vec3d>(y, x);
			if (isnan_vec(Vi)) continue;
			int count=0;
			double norm_sum = 0;
			for (int sx = -1; sx <= 1; ++sx) {
				for (int sy = -1; sy <= 1; ++sy) {
					int u = x + sx;
					int v = y + sy;
					cv::Vec3d Vj = est_cloud_mat_pad_.at<cv::Vec3d>(v, u);
					if (!isnan_vec(Vj)) {
						double norm = cv::norm(Vi - Vj);
						count++;
						norm_sum += norm;
					}

				}
			}
			double mean_norm = norm_sum / (double)count;
			sd_map_.at<double>(y-1, x-1) = mean_norm;
		}
	}
	sd_= calcNonNaNmean(sd_map_);
};	 

// VP: Valid Pixel Ratio[%]
void Evaluation::calc_VP() {
	vp_ = (double)cv::countNonZero(est_depth_) / (double)est_depth_.total();
};	 

// MNS: Mean Normal Similarity[��]
void Evaluation::calc_MNS() {
	ns_map_ = calcDotmap(est_normal_, gt_normal_);
	patch_nan_double(ns_map_);
	if (is_mask_) mask(ns_map_);
	mns_ = calcNonNaNmean(ns_map_);
};	 

// MENS: Mean Edge Normal Similarity[��]
void Evaluation::calc_MENS() {
	// Edge normal similarity
	edge_ns_map_ = ns_map_.clone();
	edge_ns_map_.setTo(std::numeric_limits<double>::quiet_NaN(), edge_area_ == 0);
	if (is_mask_) mask(edge_ns_map_);
	edge_mns_ = calcNonNaNmean(edge_ns_map_);

	// nonEdge normal similarity
	nonedge_ns_map_ = ns_map_.clone();
	nonedge_ns_map_.setTo(std::numeric_limits<double>::quiet_NaN(), edge_area_ != 0);
	if (is_mask_) mask(nonedge_ns_map_);
	nonedge_mns_ = calcNonNaNmean(nonedge_ns_map_);
}; 


void Evaluation::calc_SSIM()
{
	cv::Scalar res = cv::quality::QualitySSIM::compute(gt_depth_, est_depth_, ssim_map_);
	if (is_mask_) mask(ssim_map_);
	ssim_ = calcNonNaNmean(ssim_map_);
}


void Evaluation::calc_BadPix()
{
	badpix_map_ = cv::Mat::zeros(gt_depth_.size(), CV_8U);
	badpix_map_.setTo(255, diff_abs_ > tau);
	badpix_ = (double)cv::countNonZero(badpix_map_) / (double)badpix_map_.total();
}

void Evaluation::calc_BadPixRatio()
{
	badratiopix_map_ = cv::Mat::zeros(gt_depth_.size(), CV_8U);
	cv::Mat ratio_map;
	ratio_map =cv::abs(diff_ / gt_depth_);
	badratiopix_map_.setTo(255, ratio_map > tau_percent);
	badratiopix_ = (double)cv::countNonZero(badratiopix_map_) / (double)badratiopix_map_.total();
}


//
double Evaluation::calc_MAE() {
	if (is_mask_) mask(diff_abs_);
	mae_ = calcRootMeanValid(diff_abs_);
	return mae_;
}
void Evaluation::calc_iMAE() {
	if (is_mask_) mask(i_diff_abs_);
	imae_ = calcRootMeanValid(i_diff_abs_);
}

cv::Mat cm(cv::Mat& in, cv::InputArray mask=cv::noArray()) {
	cv::Mat mask_ = mask.getMat();
	cv::Mat normalize;
	cv::normalize(in, normalize,0,255,cv::NORM_MINMAX, CV_8UC1, mask_);
	cv::Mat dst;
	cv::applyColorMap(normalize, dst,cv::COLORMAP_TURBO);
	return dst;
}

cv::Mat cm_minmax(
	const cv::Mat& z,
	const double& min,
	const double& max,
	int type = cv::COLORMAP_TURBO
) {
	cv::Mat z_u8 = z - min;
	z_u8.convertTo(z_u8, CV_8U, 255. / (max - min));

	cv::Mat z_turbo;
	cv::Mat z_colormap;
	cv::applyColorMap(z_u8, z_colormap, type);
	return z_colormap;
}

void swap_coordinate_to_Blender(cv::Mat& normal) {
	// OpenCV���W�n�ɂ���
	for (int y = 0; y < normal.rows; ++y) {
		for (int x = 0; x < normal.cols; ++x) {
			double c = normal.at<cv::Vec3d>(y, x)[0];
			double b = normal.at<cv::Vec3d>(y, x)[1];
			double a = normal.at<cv::Vec3d>(y, x)[2];
			normal.at<cv::Vec3d>(y, x)[0] = -a;
			normal.at<cv::Vec3d>(y, x)[1] = -b;
			normal.at<cv::Vec3d>(y, x)[2] = c;
		}
	}
}

cv::Mat convert_N(cv::Mat& normal) {
	swap_coordinate_to_Blender(normal);
	cv::Mat N = ((normal + 1.0) / 2.0) * (double)(UINT16_MAX);
	N.convertTo(N, CV_16UC3);
	return N;
}
void Evaluation::dump(std::string strPath= "") {
	// cv::imwrite(strPath + "/" + "est_dpeth_org.tiff", est_depth_);
	cv::imwrite(strPath + "/" + "gt_depth.png", cm_minmax(gt_depth_,gt_z_min_,gt_z_max_));
	cv::imwrite(strPath + "/" + "est_depth_.png", cm_minmax(est_depth_, gt_z_min_, gt_z_max_));
	cv::imwrite(strPath + "/" + "gt_normal.png", convert_N(gt_normal_));
	cv::imwrite(strPath + "/" + "est_normal_.png", convert_N(est_normal_));

	cv::imwrite(strPath + "/" + "diff_.png", cm(diff_));
	cv::imwrite(strPath + "/" + "diff_abs_.png", cm(diff_abs_));
	cv::imwrite(strPath + "/" + "diff_squared_.png", cm(diff_squared_));
	cv::imwrite(strPath + "/" + "i_diff_.png", cm(i_diff_));
	cv::imwrite(strPath + "/" + "/" + "i_diff_abs_.png", cm(i_diff_abs_));
	cv::imwrite(strPath + "/" + "i_diff_squared_.png", cm(i_diff_squared_));
	cv::imwrite(strPath + "/" + "ppe_dotmap_.png", cm(ppe_dotmap_));
	cv::imwrite(strPath + "/" + "ns_map_.png", cm(ns_map_));
	cv::imwrite(strPath + "/" + "ssim_map_.png", cm(ssim_map_));
	cv::imwrite(strPath + "/" + "badpix_map_.png", cm(badpix_map_));
	cv::imwrite(strPath + "/" + "badratiopix_map_.png", cm(badratiopix_map_));
	cv::imwrite(strPath + "/" + "sd_map_.png", cm(sd_map_));
	cv::imwrite(strPath + "/" + "edge_ns_map_.png", cm(edge_ns_map_));
	cv::imwrite(strPath + "/" + "nonedge_ns_map_.png", cm(nonedge_ns_map_));
	cv::imwrite(strPath + "/" + "edge_area.png", cm(edge_area_));
}

void Evaluation::print()
{
	if (0) {
		std::cout << "mae";
		std::cout << "," << "imae";
		std::cout << "," << "mad";
		std::cout << "," << "rmse";
		std::cout << "," << "irmse";
		std::cout << "," << "edgermse";
		std::cout << "," << "nonedgermse";
		std::cout << "," << "ppe";
		std::cout << "," << "sd";
		std::cout << "," << "vp";
		std::cout << "," << "mns";
		std::cout << "," << "edgemns";
		std::cout << "," << "nonedgemns";
		std::cout << "," << "ssim";
		std::cout << "," << "badpix";
		std::cout << "," << "badratiopix";
		std::cout << "," << std::endl;
	}
	if (1) {
		std::cout << mae_;
		std::cout << "," << imae_;
		std::cout << "," << mad_;
		std::cout << "," << rmse_;
		std::cout << "," << irmse_;
		std::cout << "," << edge_rmse_;
		std::cout << "," << nonedge_rmse_;
		std::cout << "," << ppe_;
		std::cout << "," << sd_;
		std::cout << "," << vp_;
		std::cout << "," << mns_;
		std::cout << "," << edge_mns_;
		std::cout << "," << nonedge_mns_;
		std::cout << "," << ssim_;
		std::cout << "," << badpix_;
		std::cout << "," << badratiopix_;
		std::cout << "," << std::endl;
	}
	if (0) {
		std::cout << "---------------" << std::endl;
		std::cout << "MAE:" << mae_ << std::endl;
		std::cout << "iMAE:" << imae_ << std::endl;
		std::cout << "MAD:" << mad_ << std::endl;
		std::cout << "RMSE:" << rmse_ << std::endl;
		std::cout << "iRMSE:" << irmse_ << std::endl;
		std::cout << "Edge RMSE:" << edge_rmse_ << std::endl;
		std::cout << "nonEdge RMSE:" << nonedge_rmse_ << std::endl;
		std::cout << "PPE:" << ppe_ << std::endl;
		std::cout << "SD:" << sd_ << std::endl;
		std::cout << "VP:" << vp_ << std::endl;
		std::cout << "MNS:" << mns_ << std::endl;
		std::cout << "Edge MNS:" << edge_mns_ << std::endl;
		std::cout << "nonEdge MNS:" << nonedge_mns_ << std::endl;
		std::cout << "ssim:" << ssim_ << std::endl;
		std::cout << "badpix:" << badpix_ << std::endl;
		std::cout << "badpix(ratio):" << badratiopix_ << std::endl;
		std::cout << "---------------" << std::endl;
	}
}




void Evaluation::file_output(const std::string& strFn)
{
	std::fstream fs;
	fs.open(strFn, std::ios::out);

	if (0) {
		fs << "mae";
		fs << "," << "imae";
		fs << "," << "mad";
		fs << "," << "rmse";
		fs << "," << "irmse";
		fs << "," << "edgermse";
		fs << "," << "nonedgermse";
		fs << "," << "ppe";
		fs << "," << "sd";
		fs << "," << "vp";
		fs << "," << "mns";
		fs << "," << "edgemns";
		fs << "," << "nonedgemns";
		fs << "," << "ssim";
		fs << "," << "badpix";
		fs << "," << "badratiopix";
		fs << "," << std::endl;
	}
	if (0) {
		fs << mae_;
		fs << "," << imae_;
		fs << "," << mad_;
		fs << "," << rmse_;
		fs << "," << irmse_;
		fs << "," << edge_rmse_;
		fs << "," << nonedge_rmse_;
		fs << "," << ppe_;
		fs << "," << sd_;
		fs << "," << vp_;
		fs << "," << mns_;
		fs << "," << edge_mns_;
		fs << "," << nonedge_mns_;
		fs << "," << ssim_;
		fs << "," << badpix_;
		fs << "," << badratiopix_;
		fs << "," << std::endl;
	}
	if (1) {
		fs << "MAE:" << mae_ << std::endl;
		fs << "iMAE:" << imae_ << std::endl;
		fs << "MAD:" << mad_ << std::endl;
		fs << "RMSE:" << rmse_ << std::endl;
		fs << "iRMSE:" << irmse_ << std::endl;
		fs << "Edge RMSE:" << edge_rmse_ << std::endl;
		fs << "nonEdge RMSE:" << nonedge_rmse_ << std::endl;
		fs << "PPE:" << ppe_ << std::endl;
		fs << "SD:" << sd_ << std::endl;
		fs << "VP:" << vp_ << std::endl;
		fs << "MNS:" << mns_ << std::endl;
		fs << "Edge MNS:" << edge_mns_ << std::endl;
		fs << "nonEdge MNS:" << nonedge_mns_ << std::endl;
		fs << "ssim:" << ssim_ << std::endl;
		fs << "badpix:" << badpix_ << std::endl;
		fs << "badpix(ratio):" << badratiopix_ << std::endl;
	}
	fs.close();
}

cv::Mat Evaluation::get_error_map(std::string& txt, int i)
{
	if (i == 0) {txt = "diff"; return cm(diff_);}
	if (i == 1) {txt = "abs diff"; return cm(diff_abs_);}
	if (i == 2) { txt = "squared diff"; return cm(diff_squared_); }
	if (i == 3) { txt = "inverse diff"; return cm(i_diff_); }
	if (i == 4) { txt = "inverse diff abs"; return cm(i_diff_abs_); }
	if (i == 5) { txt = "inverse diff squared"; return  cm(i_diff_squared_); }
	if (i == 6) { txt = "point to plane error"; return cm(ppe_dotmap_); }
	if (i == 7) { txt = "normal similality"; return cm(ns_map_); }
	if (i ==  8) {txt = "ssim"; return cm(ssim_map_);}
	if (i ==  9) {txt = "badpix"; return cm(badpix_map_);}
	if (i == 10) {txt = "badpix[ratio]"; return cm(badratiopix_map_);}
	if (i == 11) { txt = "spartial density"; return cm(sd_map_); }
	if (i == 12) { txt = "edge noraml similality"; return cm(edge_ns_map_); }
	if (i == 13) { txt = "non edge normal similality"; return  cm(nonedge_ns_map_); }
}

void Evaluation::fs_output(std::fstream& fs)
{
	fs << "MAE:" << mae_ << std::endl;
	fs << "iMAE:" << imae_ << std::endl;
	fs << "MAD:" << mad_ << std::endl;
	fs << "RMSE:" << rmse_ << std::endl;
	fs << "iRMSE:" << irmse_ << std::endl;
	fs << "Edge RMSE:" << edge_rmse_ << std::endl;
	fs << "nonEdge RMSE:" << nonedge_rmse_ << std::endl;
	fs << "PPE:" << ppe_ << std::endl;
	fs << "SD:" << sd_ << std::endl;
	fs << "VP:" << vp_ << std::endl;
	fs << "MNS:" << mns_ << std::endl;
	fs << "Edge MNS:" << edge_mns_ << std::endl;
	fs << "nonEdge MNS:" << nonedge_mns_ << std::endl;
	fs << "ssim:" << ssim_ << std::endl;
	fs << "badpix:" << badpix_ << std::endl;
	fs << "badpix(ratio):" << badratiopix_ << std::endl;
}