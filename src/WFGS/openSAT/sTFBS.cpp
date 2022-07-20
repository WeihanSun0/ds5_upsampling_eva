/**
* @file		sTFBS.Util.cpp
* @brief	FBSの実装の内、utility関数群の定義ファイル
* @author	yusuke moriuchi
* @date		2019/12/28
* @details	BaseLib.FBSFpara1.h, BaseLib.FBSFpara1.cpp, BaseLib.FBSFimplSparseWithSV.hppから移植中
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
#include "sTFBS.h"
#include <opencv2/opencv.hpp>

void FBS::diagonal(const Eigen::VectorXf& v, Eigen::SparseMatrix<float>& mat) const
{
	mat = Eigen::SparseMatrix<float>(v.size(), v.size());
	for (int i = 0; i < int(v.size()); i++)
		mat.insert(i, i) = v(i);
}

void FBS::splat(const Eigen::VectorXf& input, Eigen::VectorXf& output) const
{
	output.setZero();
	for (int i = 0; i < int(this->splat_idx_.size()); i++)
		output(this->splat_idx_[i]) += input(i);
}

void FBS::update_splat_idx(int& pix_idx, mapId& hashed_coords, int& vert_idx, const long long* coord)
{
	// convert the coordinate to a hash value
	long long hash_coord = 0;
	for (int i = 0; i < dim_; ++i)
		hash_coord += coord[i] * hash_vec_[i];

	// pixels whom are alike will have the same hash value.
	// We only want to keep a unique list of hash values, therefore make sure we only insert
	// unique hash values.
	mapId::iterator it = hashed_coords.find(hash_coord);
	if (it == hashed_coords.end())
	{
		hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
		splat_idx_[pix_idx] = vert_idx;
		++vert_idx;
	}
	else
	{
		splat_idx_[pix_idx] = it->second;
	}//if else

	++pix_idx;
}

void FBS::blur(const Eigen::VectorXf& input, Eigen::VectorXf& output) const
{
	output.setZero();
	output = input * dim_ * 2;// 10;
	for (int i = 0; i < int(blur_idx_.size()); i++)
		output(blur_idx_[i].first) += input(blur_idx_[i].second);
}

void FBS::hashed_coords_2_blur_idx(mapId& hashed_coords, std::vector<std::pair<int, int>>& blur_idx) const
{
	//blur_idx.clear();
	//blur_idx.shrink_to_fit();
	for (int offset = -1; offset <= 1; ++offset) {
		if (offset == 0) continue;
		for (int i = 0; i < dim_; ++i)
		{
			long long offset_hash_coord = offset * hash_vec_[i];
			for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
			{
				long long neighb_coord = it->first + offset_hash_coord;
				mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
				if (it_neighb != hashed_coords.end())
				{
					blur_idx.push_back(std::pair<int, int>(it->second, it_neighb->second));
				}
			}
		}
	}
}

void FBS::slice(
	cv::OutputArray dst_,
	const std::vector<int>& splat_idx,
	const Eigen::VectorXf& y,
	const cv::InputArray mask_
)const{
	const cv::Mat mask = mask_.getMat();
	cv::Mat& output = dst_.getMatRef();
	const int tarSize = output.cols * output.rows;//mask入力時にnpixelsと異なる
	const float* pfm = (float*)mask.data;

	if (output.depth() == CV_32F) {
		float* pftar = (float*)output.data;
		if (mask.empty()) {
			for (int i = 0; i < tarSize; i++)
				pftar[i] = y(splat_idx[npixels_sparse_ + i]);
		}
		else {
			int c = 0;
			for (int i = 0; i < tarSize; i++) {
				if (pfm[i] > 0.0f) {
					pftar[i] = y(splat_idx[npixels_sparse_ + c]);
					c++;
				}
			}
		}
	}
	else if (output.depth() == CV_8U) {
		uchar* pftar = (uchar*)output.data;
		if (mask.empty()) {
			for (int i = 0; i < tarSize; i++)
				pftar[i] = cv::saturate_cast<uchar>(y(splat_idx[npixels_sparse_ + i]) * 255.0f);
		}
		else {
			int c = 0;
			for (int i = 0; i < tarSize; i++) {
				if (pfm[i] > 0.0f) {
					pftar[i] = cv::saturate_cast<uchar>(y(splat_idx[npixels_sparse_ + i]) * 255.0f);
					c++;
				}
			}
		}
	}
	else if (output.depth() == CV_16S) {
		int16_t* pftar = (int16_t*)output.data;
		if (mask.empty()) {
			for (int i = 0; i < tarSize; i++)
				pftar[i] = cv::saturate_cast<short>(y(splat_idx[npixels_sparse_ + i]) * 65535.0f - 32768.0f);
		}
		else {
			int c = 0;
			for (int i = 0; i < tarSize; i++) {
				if (pfm[i] > 0.0f) {
					pftar[i] = cv::saturate_cast<short>(y(splat_idx[npixels_sparse_ + i]) * 65535.0f - 32768.0f);
					c++;
				}
			}
		}
	}
	else if (output.depth() == CV_16U) {
		uint16_t* pftar = (uint16_t*)output.data;
		if (mask.empty()) {
			for (int i = 0; i < tarSize; i++)
				pftar[i] = cv::saturate_cast<ushort>(y(splat_idx[npixels_sparse_ + i]) * 65535.0f);
		}
		else {
			int c = 0;
			for (int i = 0; i < tarSize; i++) {
				if (pfm[i] > 0.0f) {
					pftar[i] = cv::saturate_cast<ushort>(y(splat_idx[npixels_sparse_ + i]) * 65535.0f);
					c++;
				}
			}
		}
	}
}


void FBS::init(cv::InputArray reference_){

	cv::Mat reference;
	if (reference_.channels() == 1 || noNeedConvRGB2YUV_) {
		reference = reference_.getMat();									// use colorspace of inputimage
	}
	else {
		cv::cvtColor(reference_, reference, cv::COLOR_BGR2YCrCb);	// convert colorspace RGB->YUV
	}

	cols_ = reference.cols;
	rows_ = reference.rows;
	npixels_ = cols_ * rows_;


	is_temporal_ = ((bool)pastdata_.size()) ? true : false;
	dim_ = 2 + reference.channels() + (int)is_temporal_;

	npixels_sparse_ = 0;
	for (int t = 0; t < (int)pastdata_.size(); t++) {
		npixels_sparse_ += pastdata_[t].size;
	}

	bs_param.lam = (float)lambda_;
	bs_param.cg_maxiter = num_iter_;
	bs_param.cg_tol = (float)max_tol_;

	hash_vec_.resize(dim_);
	for (int i = 0; i < dim_; ++i) {
		hash_vec_[i] = static_cast<long long>(std::pow(255, i));
	}

	hashed_coords_.clear();
	hashed_coords_.reserve(npixels_ + npixels_sparse_);

	// construct Splat(Slice) matrices
	splat_idx_.resize(npixels_ + npixels_sparse_);

	const int colorSize = (int)reference.channels();

	int pix_idx(0);
	int vert_idx(0);
	std::vector<long long> coord(dim_);
	// sparse
	if (skipSparseSplatIdx_) {// skip
		pix_idx = npixels_sparse_;
		vert_idx = SparseVertIdx_;
		hashed_coords_ = SparseHashedCoords_;
	}
	else {
		for (int t = 0; t < (int)pastdata_.size(); ++t) {
			//double sigma_spatial = sigma_spatial;//等倍倍率に合わせる

			for (int i = 0; i < (int)pastdata_[t].size; ++i) {
				coord[0] = int(pastdata_[t].x[i] / sigma_spatial_);
				coord[1] = int(pastdata_[t].y[i] / sigma_spatial_);
				for (int k = 0; k < colorSize; ++k) {
					coord[2 + k] = int(pastdata_[t].gcU8[k][i] / sigma_color_[k]);
				}
				if (is_temporal_) {
					coord[2 + colorSize + 0] = int(pastdata_[t].t / sigma_temporal_);
				}
				update_splat_idx(pix_idx, hashed_coords_, vert_idx, &coord[0]);
			}
		}
		SparseVertIdx_ = vert_idx;
		SparseHashedCoords_ = hashed_coords_;
	}

	// image
	const int downscale = (int)pastdata_.size() > 0 ? pastdata_[0].downscale : 1;
	//double sigma_spatial = param.sigma_spatial;//縮小倍率に合わせる
	const double sigma_spatial_new = sigma_spatial_ * downscale / downscale_;//等倍倍率に合わせる
	const unsigned char* pref = (const unsigned char*)reference.data;

	for (int y = 0; y < rows_; ++y) {
		for (int x = 0; x < cols_; ++x) {

			coord[0] = int(x / sigma_spatial_new);
			coord[1] = int(y / sigma_spatial_new);
			for (int k = 0; k < colorSize; ++k) {
				coord[2 + k] = int(pref[k] / sigma_color_[k]);
			}
			if (is_temporal_) {
				coord[2 + colorSize + 0] = int(t_ / sigma_temporal_);
			}
			update_splat_idx(pix_idx, hashed_coords_, vert_idx, &coord[0]);
			pref += colorSize;	// skip 1 or 3 bytes (y)
		}
	}
	nvertices_ = static_cast<int>(hashed_coords_.size());

	construct_Blur(hashed_coords_);
	bistochastize();

}//init

void FBS::filter(
	cv::InputArray src,
	cv::InputArray confidence,
	cv::OutputArray dst,
	cv::InputArray mask_)
{
	// init
	CV_Assert(!src.empty() && (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32F) && src.channels() <= 4);
	CV_Assert(!confidence.empty() && (confidence.depth() == CV_8U || confidence.depth() == CV_32F) && confidence.channels() == 1);

	std::vector<cv::Mat> src_channels;
	std::vector<cv::Mat> dst_channels;
	if (src.channels() == 1)
		src_channels.push_back(src.getMat());
	else
		split(src, src_channels);

	const cv::Mat conf = confidence.getMat();
	const cv::Mat mask = mask_.getMat();

	// solve ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	for (int i = 0; i < src.channels(); i++)
	{
		cv::Mat cur_res = src_channels[i].clone();

		solve(cur_res, conf, mask, cur_res);
		cur_res.convertTo(cur_res, src.type());
		dst_channels.push_back(cur_res);
	}

	// dst
	dst.create(src.size(), src_channels[0].type());
	if (src.channels() == 1)
	{
		cv::Mat& dstMat = dst.getMatRef();
		dstMat = dst_channels[0];
	}
	else
		merge(dst_channels, dst);

	CV_Assert(src.type() == dst.type() && src.size() == dst.size());

}//filterSparse3

void FBS::solve(
	const cv::Mat& target,
	const cv::Mat& confidence,
	const cv::Mat& mask,
	cv::Mat& output
)const{
	// the size is determined based on "nvertices"
	const int nvertices = this->nvertices_;
	Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices, nvertices);
	Eigen::VectorXf b(nvertices);
	Eigen::VectorXf y(nvertices);
	Eigen::VectorXf y0(nvertices);
	Eigen::VectorXf y1(nvertices);
	Eigen::VectorXf w_splat(nvertices);

	// the size is determined based on "npixels"
	Eigen::VectorXf x(npixels_ + npixels_sparse_);
	Eigen::VectorXf w = Eigen::VectorXf::Ones(npixels_ + npixels_sparse_);
	// w x h
	int tarSize = target.cols * target.rows;	// when we use mask, It different from "this-> npixels".

	// construct x
	{
		int ti = 0;
		if (target.depth() == CV_32F) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++) {
					x(ti + i) = pastdata_[t].d[i];
				}
				ti += pastdata_[t].size;
			}
			// image
			const float* pft = (float*)target.data;
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					x(npixels_sparse_ + i) = pft[i];
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						x(npixels_sparse_ + c) = pft[i];
						++c;
					}
				}
			}
		}
		else if (target.depth() == CV_8U) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++) {
					x(ti + i) = cv::saturate_cast<float>(pastdata_[t].d[i]) / 255.0f;
				}
				ti += pastdata_[t].size;
			}
			// image
			const uchar* pft = reinterpret_cast<const uchar*>(target.data);
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					x(npixels_sparse_ + i) = cv::saturate_cast<float>(pft[i]) / 255.0f;
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						x(npixels_sparse_ + i) = cv::saturate_cast<float>(pft[i]) / 255.0f;
						c++;
					}
				}
			}
		}
		else if (target.depth() == CV_16S) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++)
					x(ti + i) = (cv::saturate_cast<float>(pastdata_[t].d[i]) + 32768.0f) / 65535.0f;
				ti += pastdata_[t].size;
			}//t
			// image
			const int16_t* pft = reinterpret_cast<const int16_t*>(target.data);
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					x(npixels_sparse_ + i) = (cv::saturate_cast<float>(pft[i]) + 32768.0f) / 65535.0f;
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						x(npixels_sparse_ + i) = (cv::saturate_cast<float>(pft[i]) + 32768.0f) / 65535.0f;
						c++;
					}
				}
			}
		}
		else if (target.depth() == CV_16U) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++)
					x(ti + i) = cv::saturate_cast<float>(pastdata_[t].d[i]) / 65535.0f;
				ti += pastdata_[t].size;
			}
			// image
			const int16_t* pft = reinterpret_cast<const int16_t*>(target.data);
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					x(npixels_sparse_ + i) = cv::saturate_cast<float>(pft[i]) / 65535.0f;
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						x(npixels_sparse_ + i) = cv::saturate_cast<float>(pft[i]) / 65535.0f;
						c++;
					}
				}
			}
		}
	}

	// construct w
	{
		int ti = 0;
		if (confidence.empty()) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++) {
					w(ti + i) = pastdata_[t].c[i];
				}
				ti += pastdata_[t].size;
			}
		}
		else if (confidence.depth() == CV_32F) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++) {
					w(ti + i) = pastdata_[t].c[i];
				}
				ti += pastdata_[t].size;
			}
			// image
			const float* pfc = (float*)(confidence.data);
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					w(npixels_sparse_ + i) = pfc[i];
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						w(npixels_sparse_ + c) = pfc[i];
						c++;
					}
				}
			}
		}
		else if (confidence.depth() == CV_8U) {
			// sparse
			for (int t = 0; t < (int)pastdata_.size(); t++) {
				for (int i = 0; i < pastdata_[t].size; i++)
					w(ti + i) = cv::saturate_cast<float>(pastdata_[t].c[i]) / 255.0f;
				ti += pastdata_[t].size;
			}
			// image
			const uchar* pfc = reinterpret_cast<const uchar*>(confidence.data);
			if (mask.empty()) {
				for (int i = 0; i < tarSize; i++) {
					w(npixels_sparse_ + i) = cv::saturate_cast<float>(pfc[i]) / 255.0f;
				}
			}
			else {
				const float* pfm = (float*)mask.data;
				int c = 0;
				for (int i = 0; i < tarSize; i++) {
					if (pfm[i] > 0.0f) {
						w(npixels_sparse_ + i) = cv::saturate_cast<float>(pfc[i]) / 255.0f;
						c++;
					}
				}
			}
		}
	}

	// construct A
	splat(w, w_splat);
	diagonal(w_splat, A_data);
	A = bs_param.lam * (Dm_ - Dn_ * (blurs_ * Dn_)) + A_data;

	// construct b
	b.setZero();
	for (int i = 0; i < int(splat_idx_.size()); i++) {
		b(splat_idx_[i]) += x(i) * w(i);
	}
	// construct y0
	y0.setZero();
	for (int i = 0; i < int(splat_idx_.size()); i++) {
		y0(splat_idx_[i]) += x(i);
	}
	y1.setZero();
	for (int i = 0; i < int(splat_idx_.size()); i++) {
		y1(splat_idx_[i]) += 1.0f;
	}
	for (int i = 0; i < nvertices; i++) {
		y0(i) = y0(i) / y1(i);
	}


	// solve y = A-1b
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(A);
	cg.setMaxIterations(bs_param.cg_maxiter);
	cg.setTolerance(bs_param.cg_tol);
	//y = cg.solve(b);
	y = cg.solveWithGuess(b, y0);
	//std::cout << "#iterations:     " << cg.iterations() << std::endl;
	//std::cout << "estimated error: " << cg.error() << std::endl;

	//slice x<-St(A-1b)
	slice(output, splat_idx_, y, mask);
}

void FBS::construct_Blur(mapId& hashed_coords)
{
	if (blur_idx_.size() > 0) {
		blur_idx_.clear();//std::vector<std::pair<int, int>>().swap(this->blur_idx);
	}
	blur_idx_.reserve(nvertices_ * 4);
	Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices_);
	diagonal(ones_nvertices, blurs_);
	blurs_ *= (float)dim_ * 2.f;//10;

	// construct Blur matrices
	for (int offset = -1; offset <= 1; ++offset) {
		if (offset == 0) continue;
		for (int i = 0; i < dim_; ++i) {
			Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(blurs_.rows(), blurs_.cols());
			blur_temp.reserve(Eigen::VectorXi::Constant(nvertices_, 6));
			long long offset_hash_coord = offset * hash_vec_[i];

			for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it) {
				long long neighb_coord = it->first + offset_hash_coord;
				mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
				if (it_neighb != hashed_coords.end()) {
					blur_temp.insert(it->second, it_neighb->second) = 1.0f;
					blur_idx_.emplace_back(std::pair<int, int>(it->second, it_neighb->second));
				}
			}
			blurs_ += blur_temp;
		}
	}
	//std::cout << blur_idx_.size() << "," << nvertices_<<std::endl;
	blurs_.finalize();
}

void FBS::bistochastize() {
	// bistochastize
	int maxiter = 10;
	n_ = Eigen::VectorXf::Ones(nvertices_);
	m_ = Eigen::VectorXf::Zero(nvertices_);
	for (int i = 0; i < int(splat_idx_.size()); i++) {
		m_(splat_idx_[i]) += 1.0f;
	}
	Eigen::VectorXf bluredn(nvertices_);

	for (int i = 0; i < maxiter; i++) {
		blur(n_, bluredn);
		n_ = ((n_.array() * m_.array()).array() / bluredn.array()).array().sqrt();
	}//i
	blur(n_, bluredn);

	m_ = n_.array() * (bluredn).array();

	diagonal(m_, Dm_);
	diagonal(n_, Dn_);
}

// public:
void FBS::execute(
	cv::InputArray guide,
	cv::InputArray src,
	cv::OutputArray dst,
	cv::InputArray confidence,
	cv::InputArray mask = cv::noArray()
){
	init(guide);
	filter(src, confidence, dst, mask);
}
