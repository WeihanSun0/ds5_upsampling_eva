#include <cmath>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "FastBilateralSolver.h"
#include <unordered_map> //c++11

void callerror() {
	CV_Error(Error::StsBadSize,"Size of the filtered image must be equal to the size of the guide image");
}

using namespace cv;
namespace SAT {
	using mapId = std::unordered_map<long long /* hash */, int /* vert id */>;

	void fastBilateralSolver_sparse(
		InputArray guide,
		InputArray src,
		InputArray mask,
		InputArray confidence,
		cv::Mat_<float>& dst,
		double sigma_spatial,
		double sigma_luma,
		double sigma_chroma,
		double lambda,
		int num_iter,
		double max_tol
	){
		FastBilateralSolver tmp(guide, sigma_spatial, sigma_luma, sigma_chroma, lambda, num_iter, max_tol);
		cv::Mat dense_depth, tmp_result;
		tmp.filter(src, confidence, dense_depth);
		tmp.filter(mask, confidence, tmp_result);
		tmp_result.setTo(1.0f, tmp_result < FLT_EPSILON);
		dst=dense_depth / tmp_result;
	};
	void FastBilateralSolver::init(const cv::Mat& reference,double sigma_spatial,double sigma_luma,double sigma_chroma) {

		cols = reference.cols;
		rows = reference.rows;
		npixels = cols * rows
;
	
		mapId hashed_coords(npixels);
		splat_idx.resize(npixels);

		dim = reference.channels() == 1 ? 3 : 5;
		std::vector<long long> hash_vec(dim);
		for (int i = 0; i < dim; ++i)
			hash_vec[i] = static_cast<long long>(std::pow(255, i));

		//---------------------------------------------
		if (dim == 3){
			const unsigned char* pref = (const unsigned char*)reference.data;
			int vert_idx = 0;
			int pix_idx = 0;
			// construct Splat(Slice) matrices
			for (int y = 0; y < rows; ++y) {
				for (int x = 0; x < cols; ++x) {
					long long coord[3];
					coord[0] = int(x / sigma_spatial);
					coord[1] = int(y / sigma_spatial);
					coord[2] = int(pref[0] / sigma_luma);
					// convert the coordinate to a hash value
					long long hash_coord = 0;
					for (int i = 0; i < 3; ++i)
						hash_coord += coord[i] * hash_vec[i];
					// check hash values are unique or not
					mapId::iterator it = hashed_coords.find(hash_coord);
					if (it == hashed_coords.end()) {
						hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
						splat_idx[pix_idx] = vert_idx;
						++vert_idx;
					}
					else splat_idx[pix_idx] = it->second;
					pref += 1; // skip 1 bytes (y)
					++pix_idx;
				}
			}
		}
		else{
			cv::Mat reference_yuv;
			cv::cvtColor(reference, reference_yuv, COLOR_BGR2YCrCb);
			const unsigned char* pref = (const unsigned char*)reference_yuv.data;
			int vert_idx = 0;
			int pix_idx = 0;
			// construct Splat(Slice) matrices
			for (int y = 0; y < rows; ++y) {
				for (int x = 0; x < cols; ++x) {
					long long coord[5];
					coord[0] = int(x / sigma_spatial);
					coord[1] = int(y / sigma_spatial);
					coord[2] = int(pref[0] / sigma_luma);
					coord[3] = int(pref[1] / sigma_chroma);
					coord[4] = int(pref[2] / sigma_chroma);
					// convert the coordinate to a hash value
					long long hash_coord = 0;
					for (int i = 0; i < 5; ++i)
						hash_coord += coord[i] * hash_vec[i];
					// check hash values are unique or not
					mapId::iterator it = hashed_coords.find(hash_coord);
					if (it == hashed_coords.end()) {
						hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
						splat_idx[pix_idx] = vert_idx;
						++ vert_idx;
					}
					else splat_idx[pix_idx] = it->second;					
					pref += 3; // skip 3 bytes (y u v)
					++pix_idx;
				}
			}
		}
		//---------------------------------------------
		// construct Blur matrices
		construct_Blur(hashed_coords, hash_vec);
		bistochastize();
	}

	void FastBilateralSolver::Splat(const Eigen::VectorXf& input, Eigen::VectorXf& output){
		output.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++){
			output(splat_idx[i]) += input(i);
		}
	}

	void FastBilateralSolver::Blur(const Eigen::VectorXf& input, Eigen::VectorXf& output){
		output.setZero();
		output = input * 10;
		for (int i = 0; i < int(blur_idx.size()); i++){
			output(blur_idx[i].first) += input(blur_idx[i].second);
		}
	}

	void FastBilateralSolver::Slice(const Eigen::VectorXf& input, Eigen::VectorXf& output){
		output.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++){
			output(i) = input(splat_idx[i]);
		}
	}

	void FastBilateralSolver::filter(InputArray src, InputArray confidence, OutputArray dst) {
		CV_Assert(!src.empty() && src.channels() <= 4);
		CV_Assert(src.depth() == CV_8U||src.depth()==CV_16S||src.depth()==CV_16U||src.depth() == CV_32F);
		CV_Assert(!confidence.empty() && confidence.channels() == 1); 
		CV_Assert(confidence.depth() == CV_8U || confidence.depth() == CV_32F);			
		if (src.rows() != rows || src.cols() != cols){ callerror();return;}
		if (confidence.rows() != rows || confidence.cols() != cols) {callerror(); return;}

		std::vector<Mat> src_channels;
		std::vector<Mat> dst_channels;
		// input: single channel
		if (src.channels() == 1) {
			src_channels.push_back(src.getMat());
		}
		// input: multi channels
		else cv::split(src, src_channels);

		cv::Mat conf = confidence.getMat();

		for (int i = 0; i < src.channels(); i++) {
			Mat cur_res = src_channels[i].clone();
			solve(cur_res, conf, cur_res);
			cur_res.convertTo(cur_res, src.type());
			dst_channels.push_back(cur_res);
		}

		dst.create(src.size(), src_channels[0].type());
		if (src.channels() == 1) {
			cv::Mat& dstMat = dst.getMatRef();
			dstMat = dst_channels[0];
		}
		else
			cv::merge(dst_channels, dst);
		CV_Assert(src.type() == dst.type() && src.size() == dst.size());
	}

	void FastBilateralSolver::solve(const cv::Mat& target, const cv::Mat& confidence, cv::Mat& output){
		Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices, nvertices);
		Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices, nvertices);
		Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices, nvertices);
		Eigen::VectorXf b(nvertices);
		Eigen::VectorXf y(nvertices);
		Eigen::VectorXf y0(nvertices);
		Eigen::VectorXf y1(nvertices);
		Eigen::VectorXf w_splat(nvertices);

		Eigen::VectorXf x(npixels);
		Eigen::VectorXf w(npixels);

		normalize_cvMat2Eigen(target, x);
		normalize_cvMat2Eigen(confidence, w);

		//construct A
		Splat(w, w_splat);//Sc
		diagonal(w_splat, A_data);// diag(Sc)

		A = bs_param.lam * (Dm - Dn * (blurs * Dn)) + A_data;
		//construct b =S(ct)
		b.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++) {
			b(splat_idx[i]) += x(i) * w(i);
		}

		//construct guess for y
		y0.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++) {
			y0(splat_idx[i]) += x(i);
		}
		y1.setZero();
		for (int i = 0; i < int(splat_idx.size()); i++) {
			y1(splat_idx[i]) += 1.0f;
		}
		for (int i = 0; i < nvertices; ++i) {
			y0(i) = y0(i) / y1(i);
		}

		// solve Ay = b
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
		cg.compute(A);
		cg.setMaxIterations(bs_param.cg_maxiter);
		cg.setTolerance(bs_param.cg_tol);

		// y = cg.solve(b);
		y = cg.solveWithGuess(b, y0);
		//std::cout << "#iterations:     " << cg.iterations() << std::endl;
		//std::cout << "estimated error: " << cg.error() << std::endl;

		//slice
		bilateral2pixel(y, output);
	}

	void FastBilateralSolver::construct_Blur(mapId &hashed_coords, const std::vector<long long> &hash_vec) {
		// hash_coords -> blur
		nvertices = static_cast<int>(hashed_coords.size());
		// construct Blur matrices
		Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
		diagonal(ones_nvertices, blurs);
		blurs *= 10;//dim*2なのでは？？

		for (int offset = -1; offset <= 1; ++offset) {//ださい
			if (offset == 0) continue;
			for (int i = 0; i < dim; ++i) {
				Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
				blur_temp.reserve(Eigen::VectorXi::Constant(nvertices, 6));
				long long offset_hash_coord = offset * hash_vec[i];

				for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it) {
					long long neighb_coord = it->first + offset_hash_coord;
					mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
					if (it_neighb != hashed_coords.end()) {
						blur_temp.insert(it->second, it_neighb->second) = 1.0f;
						blur_idx.push_back(std::pair<int, int>(it->second, it_neighb->second));
					}
				}
				blurs += blur_temp;
			}
		}
		blurs.finalize();
	}

	void FastBilateralSolver::normalize_cvMat2Eigen(const cv::Mat& input, Eigen::VectorXf& output) {
		if (input.depth() == CV_16S) {
			const int16_t *pft = reinterpret_cast<const int16_t*>(input.data);
			for (int i = 0; i < npixels; i++) {
				output(i) = (cv::saturate_cast<float>(pft[i]) + 32768.0f) / 65535.0f;
			}
		}
		else if (input.depth() == CV_16U) {
			const uint16_t *pft = reinterpret_cast<const uint16_t*>(input.data);
			for (int i = 0; i < npixels; i++) {
				output(i) = cv::saturate_cast<float>(pft[i]) / 65535.0f;
			}
		}
		else if (input.depth() == CV_8U) {
			const uchar *pft = reinterpret_cast<const uchar*>(input.data);
			for (int i = 0; i < npixels; i++) {
				output(i) = cv::saturate_cast<float>(pft[i]) / 255.0f;
			}
		}
		else if (input.depth() == CV_32F) {
			const float *pft = reinterpret_cast<const float*>(input.data);
			for (int i = 0; i < npixels; i++) {
				output(i) = pft[i];
			}
		}
	}

	void FastBilateralSolver::bistochastize() {
		int maxiter = 10;
		n = Eigen::VectorXf::Ones(nvertices);
		m = Eigen::VectorXf::Zero(nvertices);
		for (int i = 0; i < int(splat_idx.size()); i++) {
			m(splat_idx[i]) += 1.0f;
		}
		Eigen::VectorXf bluredn(nvertices);
		for (int i = 0; i < maxiter; i++) {
			Blur(n, bluredn);
			n = ((n.array()*m.array()).array() / bluredn.array()).array().sqrt();
		}
		Blur(n, bluredn);
		m = n.array() * (bluredn).array();
		diagonal(m, Dm);
		diagonal(n, Dn);
	}

	void FastBilateralSolver::bilateral2pixel(Eigen::VectorXf& y, const cv::Mat output) {
		// この型チェックは入力 output =target.clone()が入力されている前提
		if (output.depth() == CV_16S) {
			int16_t *pftar = (int16_t*)output.data;
			for (int i = 0; i < int(splat_idx.size()); i++) {
				pftar[i] = cv::saturate_cast<short>(y(splat_idx[i]) * 65535.0f - 32768.0f);
			}
		}
		else if (output.depth() == CV_16U) {
			uint16_t *pftar = (uint16_t*)output.data;
			for (int i = 0; i < int(splat_idx.size()); i++) {
				pftar[i] = cv::saturate_cast<ushort>(y(splat_idx[i]) * 65535.0f);
			}
		}
		else if (output.depth() == CV_8U) {
			uchar *pftar = (uchar*)output.data;
			for (int i = 0; i < int(splat_idx.size()); i++) {
				pftar[i] = cv::saturate_cast<uchar>(y(splat_idx[i]) * 255.0f);
			}
		}
		else {
			float *pftar = (float*)(output.data);
			for (int i = 0; i < int(splat_idx.size()); i++) {
				pftar[i] = y(splat_idx[i]);
			}
		}
	};

}
