#include "HFBS.h"
#include<opencv2/core/eigen.hpp>
#include<omp.h>

bool HFBS::init(const cv::Mat& reference, double sigma_spatial, double sigma_luma)
{
	cols = reference.cols;
	rows = reference.rows;
	npixels = cols * rows;
	cv::Mat guide;
	if (reference.channels() == 3) {
		cv::cvtColor(reference, guide, cv::COLOR_BGR2GRAY);
	}
	else if (reference.channels() == 1) {
		guide = reference;
	}
	else {
		puts("error: input guide must be 1or3ch");
	}
	mapId hashed_coords(npixels);
	splat_idx.resize(npixels);

	std::vector<long long> hash_vec(dim);
	for (int i = 0; i < dim; ++i) {
		hash_vec[i] = static_cast<long long>(std::pow(255, i));
	}

	const unsigned char* pref = (const unsigned char*)guide.data;
	int vert_idx = 0;
	int pix_idx = 0;

	// construct Splat(Slice) matrices
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			long long coord[3];
			coord[0] = static_cast<int>(x / sigma_spatial);
			coord[1] = static_cast<int>(y / sigma_spatial);
			coord[2] = static_cast<int>(pref[0] / sigma_luma);

			// convert the coordinate to a hash value
			long long hash_coord = 0;
			for (int i = 0; i < 3; ++i) {
				hash_coord += coord[i] * hash_vec[i];
			}

			// check hash values are unique or not
			mapId::iterator it = hashed_coords.find(hash_coord);
			if (it == hashed_coords.end()) {
				hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
				splat_idx[pix_idx] = vert_idx;
				++vert_idx;//新規の頂点
			}
			else splat_idx[pix_idx] = it->second;
			pref += 1; // skip 1 bytes (y)
			++pix_idx;
		}
	}
	construct_Blur(hashed_coords, hash_vec);// blur_idx
	bistochastize(); // n, m1
	return false;
}

bool HFBS::construct_Blur(
	mapId& hashed_coords,
	const std::vector<long long>& hash_vec
) {
	nvertices = static_cast<int>(hashed_coords.size());
	if (blur_idx.size() > 0) {
		blur_idx.clear();
	}
	blur_idx.reserve(nvertices * 4);// 4倍未満になる
	int offset[2] = { -1,1 }; //

	for (auto& e : offset) {
		for (int i = 0; i < dim; ++i) {
			long long offset_hash_coord = e * hash_vec[i];

			for (auto& it : hashed_coords) {
				long long neighb_coord = it.first + offset_hash_coord;
				const auto it_neighb = hashed_coords.find(neighb_coord);
				if (it_neighb != hashed_coords.end()) {
					blur_idx.emplace_back(std::pair<int, int>(it.second, it_neighb->second));
				}
			}
		}
	}
	return false;
}


// (9) in paper
bool HFBS::bistochastize()
{
	Eigen::VectorXf m0(nvertices);
	Splat(Eigen::VectorXf::Ones(npixels), m0);// m0 = SI

	Eigen::VectorXf BI;
	Blur(Eigen::VectorXf::Ones(nvertices), BI);

	n = Eigen::sqrt(Eigen::VectorXf::Ones(nvertices).array() * m0.array() / (EPS + BI.array()));

	Eigen::VectorXf Bn;
	Blur(n, Bn);

	m1 = n.array() * Bn.array();
	return false;
}


bool HFBS::dense_solve_jacobi(
	const Eigen::VectorXf& pz_init,
	const Eigen::VectorXf& inv_A,
	Eigen::VectorXf& y
) {
	Eigen::VectorXf pzn = pz_init.array()* n.array();

	// Heavy ball method
	const double beta = 8.;//LOSS_SMOOTH_MULT
	Eigen::VectorXf coeff = beta * n.array() * inv_A.array();
	Eigen::VectorXf coeff_n = coeff.array() * n.array();

	Eigen::VectorXf Dpzn(nvertices);
	Diffuse(pzn, Dpzn);

	for (int i = 0; i < N_itr_HBM; ++i) {
		Diffuse(pzn.array() + Dpzn.array() * coeff_n.array(), Dpzn);
	}
	y = pz_init.array() + Dpzn.array() * coeff.array();;
	return false;
}


bool HFBS::HFBS_solve(const cv::Mat& target, const cv::Mat& confidence, cv::Mat& output)
{
	// 入力はCV_32Fだけ
	CV_Assert(target.depth() == CV_32F && confidence.depth() == CV_32F);

	Eigen::VectorXf x0(npixels);
	Eigen::VectorXf w0(npixels);
	Eigen::VectorXf y(nvertices);

	cv::cv2eigen(target.reshape(1, npixels), x0);
	cv::cv2eigen(confidence.reshape(1, npixels), w0);

	//normalize_cvMat2Eigen(target, x0);
	//normalize_cvMat2Eigen(target, w0);

	Eigen::VectorXf Sc(nvertices);
	Splat(w0, Sc);

	A_diag = Sc.array() + lambda * (m1.array() - 2.f * n.array().square());
	Eigen::VectorXf inv_A = A_diag.cwiseInverse();

	Eigen::VectorXf b(nvertices);
	Splat(w0.array() * x0.array(), b);

	//initial solution
	Eigen::VectorXf pz_init = b.array() * inv_A.array();
	dense_solve_jacobi(pz_init, inv_A, y);

	//slice (bilateral space -> pixel space)
	bilateral2pixel(y, output);
	return false;
}


// pixel space(1D) -> bilateral space(1D)
void HFBS::Splat(const Eigen::VectorXf& input, Eigen::VectorXf& dst)
{
	dst.setZero();
	const int N = splat_idx.size();
	for (int i = 0; i < N; i++) {
		dst(splat_idx[i]) += input(i);
	}
}

// bilateral space(1D) -> pixel space(1D)
void HFBS::Slice(const Eigen::VectorXf& input, Eigen::VectorXf& dst)
{
	dst.setZero();
	const int N = splat_idx.size();
	for (int i = 0 ; i < N; i++) {
		dst(i) = input(splat_idx[i]);
	}

}

//  Blur(y): By = 2*y + Dy
void HFBS::Blur(const Eigen::VectorXf& input, Eigen::VectorXf& dst)
{
	dst.setZero();
	dst = input * 2.f;
	for (std::pair<int, int> itr : blur_idx) {
		dst(itr.first) += input(itr.second);
	}
}

// Diffuse(y): Dy
void HFBS::Diffuse(const Eigen::VectorXf& input, Eigen::VectorXf& dst)
{
	dst.setZero();
	const int N = blur_idx.size();
	for (std::pair<int, int> itr : blur_idx) {
		dst(itr.first) += input(itr.second);
	}
}


void HFBS::filter(cv::InputArray src, cv::InputArray confidence, cv::OutputArray dst)
{
	CV_Assert(!src.empty() && src.channels() <= 4);
	CV_Assert(
		src.depth() == CV_8U || src.depth() == CV_16S ||
		src.depth() == CV_16U || src.depth() == CV_32F);
	CV_Assert(!confidence.empty() && confidence.channels() == 1);
	CV_Assert(confidence.depth() == CV_8U || confidence.depth() == CV_32F);
	if (src.rows() != rows || src.cols() != cols) { sizeerror(); return; }
	if (confidence.rows() != rows || confidence.cols() != cols) { sizeerror(); return; }

	std::vector<cv::Mat> src_channels;
	std::vector<cv::Mat> dst_channels;

	// input: single channel
	if (src.channels() == 1) {
		src_channels.push_back(src.getMat());
	}

	// input: multi channels
	else cv::split(src, src_channels);

	cv::Mat conf = confidence.getMat();

	for (int i = 0; i < src.channels(); i++) {
		cv::Mat cur_res = src_channels[i].clone();
		HFBS_solve(cur_res, conf, cur_res);		//<------- Main part of HFBS
		cur_res.convertTo(cur_res, src.type());
		dst_channels.push_back(cur_res);
	}

	dst.create(src.size(), src_channels[0].type());
	if (src.channels() == 1) {
		cv::Mat& dstMat = dst.getMatRef();
		dstMat = dst_channels[0];
	}
	else {
		cv::merge(dst_channels, dst);
	}
	CV_Assert(src.type() == dst.type() && src.size() == dst.size());
	return;
}



//-------------------------------------
//  Utility							   
//-------------------------------------

void HFBS::normalize_cvMat2Eigen(const cv::Mat& input, Eigen::VectorXf& output) {
	if (input.depth() == CV_16S) {
		const int16_t* pft = reinterpret_cast<const int16_t*>(input.data);
		for (int i = 0; i < npixels; i++) {
			output(i) = (cv::saturate_cast<float>(pft[i]) + 32768.0f) / 65535.0f;
		}
	}
	else if (input.depth() == CV_16U) {
		const uint16_t* pft = reinterpret_cast<const uint16_t*>(input.data);
		for (int i = 0; i < npixels; i++) {
			output(i) = cv::saturate_cast<float>(pft[i]) / 65535.0f;
		}
	}
	else if (input.depth() == CV_8U) {
		const uchar* pft = reinterpret_cast<const uchar*>(input.data);
		for (int i = 0; i < npixels; i++) {
			output(i) = cv::saturate_cast<float>(pft[i]) / 255.0f;
		}
	}
	else if (input.depth() == CV_32F) {
		const float* pft = reinterpret_cast<const float*>(input.data);
		for (int i = 0; i < npixels; i++) {
			output(i) = pft[i];
		}
	}
}


void HFBS::bilateral2pixel(Eigen::VectorXf& y, const cv::Mat output) {
	if (output.depth() == CV_16S) {
		int16_t* pftar = (int16_t*)output.data;
		for (int i = 0; i < int(splat_idx.size()); i++) {
			pftar[i] = cv::saturate_cast<short>(y(splat_idx[i]) * 65535.0f - 32768.0f);
		}
	}
	else if (output.depth() == CV_16U) {
		uint16_t* pftar = (uint16_t*)output.data;
		for (int i = 0; i < int(splat_idx.size()); i++) {
			pftar[i] = cv::saturate_cast<ushort>(y(splat_idx[i]) * 65535.0f);
		}
	}
	else if (output.depth() == CV_8U) {
		uchar* pftar = (uchar*)output.data;
		for (int i = 0; i < int(splat_idx.size()); i++) {
			pftar[i] = cv::saturate_cast<uchar>(y(splat_idx[i]) * 255.0f);
		}
	}
	else {
		float* pftar = (float*)(output.data);
		for (int i = 0; i < int(splat_idx.size()); i++) {
			pftar[i] = y(splat_idx[i]);
		}
	}
};