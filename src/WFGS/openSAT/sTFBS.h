#pragma once

/**
* @file		sTFBS.h
* @brief	時間方向拡張とガイド追加を許容したFBS
* @author	yusuke moriuchi
* @date		2019/12/28
* @details	時間方向拡張はsparse data。追加のガイドはguide2を追加指定する。
*			BaseLib.FBSFpara1.h, BaseLib.FBSFpara1.cpp, BaseLib.FBSFimplSparseWithSV.hppから移植中
*/

/*  ********************************************************************************************
 *
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  ********************************************************************************************/

#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <boost/version.hpp>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>
typedef std::unordered_map<long long /* hash */, int /* vert id */>  mapId;


typedef struct sData_1frame {
	int size;										// datasize
	std::vector<int> x, y;							// x, y
	long long t;											// frame number
	int downscale;									// downscale ratio
	std::vector<float> d;							// depth
	std::vector<std::vector<unsigned char>> gcU8;	// guide(r, g, b)
	std::vector<std::vector<unsigned char>> gdU8;	// guideSub(d)
	std::vector<float> c;							// confidence;
	sData_1frame():size(0),t(0),downscale(0){}
}sData_1frame;



using PastData = boost::circular_buffer<sData_1frame>;

void ResizeData(sData_1frame& data, int size, int guideSize, int guideSubSize);

void set_data(
	sData_1frame& data,
	cv::InputArray depth_,
	cv::InputArray guide_,
	cv::InputArray guideSub_,
	cv::InputArray conf_,
	cv::InputArray mask_,
	const float& maskMinVal
);

class FBS {
public:
	FBS() : npixels_(0), npixels_sparse_(0),nvertices_(0), dim_(0), cols_(0),rows_(0), is_temporal_(false), SparseVertIdx_(0){}
	void execute(
		cv::InputArray guide,
		cv::InputArray src,
		cv::OutputArray dst,
		cv::InputArray confidence,
		cv::InputArray mask 
	);
	//static PastData DEFAULT_SDATA;	// 非一時的に静的インスタンスを宣言しデフォルト値を生成
	PastData pastdata_;

	long long t_ = 0;			     			 // frame number
	std::vector<double> sigma_color_ = { 8.0, 8.0, 8.0 };// sigma_luma, sigma_chroma x2 / sigma_B, G, R;
	void set_downscale(const int& ds) { downscale_ = ds; }
	void init(cv::InputArray reference_);

	void filter(
		cv::InputArray src,
		cv::InputArray confidence,
		cv::OutputArray dst,
		cv::InputArray mask_ = cv::noArray()
	);
	void set_sigma_spatial(const double& in) { sigma_spatial_ = in; }
	void set_lambda(const double& in) { lambda_ = in; }

	double get_sigma_spatial() { return sigma_spatial_; }

private:
	int downscale_ = 4;                                 // downscale ratio
	double sigma_spatial_ = 16.0;					// sigma_space
	const double sigma_temporal_ = 1.0;					// sigma_time
	const std::vector<double> sigma_guide_ = { 16.0 };	// sigma_depth edge
	double lambda_ = 8.0;							// lambda
	const int num_iter_ = 100;							// numbers of iter.
	const double max_tol_ = 1E-5;						// tolerance for opt.
	const int noNeedConvRGB2YUV_ = 1;					// if 1, you ignore RGB2YUV conv. and use RGB itself
	const bool skipSparseSplatIdx_ = false;

	int npixels_;		// image size (cols * rows)
	int npixels_sparse_;// sum of sparse points
	int nvertices_;		// 
	int dim_;			// 
	int cols_;			// 
	int rows_;			// 
	bool is_temporal_;	// FBSから追加

	Eigen::VectorXf m_, n_;

	std::vector<long long> hash_vec_;// 
	mapId hashed_coords_;			 // 

	// update_splat_idx
	mapId SparseHashedCoords_;
	int SparseVertIdx_;

	std::vector<int> splat_idx_s_;	           //
	std::vector<int> splat_idx_;		       // npixelsSparse(static) + npixels(active)
	std::vector<std::pair<int, int>> blur_idx_;// 

	// const_blur_mat_and_bistochastize
	Eigen::SparseMatrix<float, Eigen::ColMajor> blurs_;	// 
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dm_;	// 
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dn_;	// 


	struct bs_params {
		float lam = 128.0f;
		float A_diag_min = 1e-5f;
		float cg_tol = 1e-5f;
		int cg_maxiter = 25;
	};
	bs_params bs_param;

	// :::::::::::::::::::::::::::::::::::::
	// Util
	void splat(const Eigen::VectorXf& input, Eigen::VectorXf& dst)const;
	void blur(const Eigen::VectorXf& input, Eigen::VectorXf& dst)const;
	void diagonal(const Eigen::VectorXf& v, Eigen::SparseMatrix<float>& mat) const;
	void update_splat_idx(int& pix_idx, mapId& hashed_coords, int& vert_idx, const long long* coord);
	void hashed_coords_2_blur_idx(
		mapId& hashed_coords, 
		std::vector<std::pair<int, int>>& blur_idx
	) const;

	void bistochastize();
	void construct_Blur(mapId& hashed_coords);

	void slice(
		cv::OutputArray dst_,
		const std::vector<int>& splat_idx,
		const Eigen::VectorXf& y,
		const cv::InputArray mask_ = cv::noArray()
	)const;


	// :::::::::::::::::::::::::::::::::::::
	// Body
	void solve(
		const cv::Mat& target, 
		const cv::Mat& confidence, 
		const cv::Mat& mask, 
		cv::Mat& output
	)const;

};