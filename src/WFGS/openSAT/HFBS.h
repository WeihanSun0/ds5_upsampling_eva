#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map> //c++11
#include <Eigen/Core>

using mapId = std::unordered_map<long long /* hash */, int /* vert id */>;
inline void sizeerror() {
	CV_Error(cv::Error::StsBadSize, "Size of the filtered image must be equal to the size of the guide image");
}

class HFBS
{
public:
	HFBS() {};
	~HFBS() {};

	void filter(cv::InputArray src, cv::InputArray confidence, cv::OutputArray dst);
	bool init(const cv::Mat& reference, double sigma_spatial, double sigma_luma);

private:
	int npixels;
	int nvertices;
	int cols;
	int rows;
	const int dim = 3;// HFBS  x, y, lum 3dim
	const int N_itr_HBM = 100;

	int stripe_sz;

	const float EPS = 1e-5;
	const double lambda = 16;
	//const double alpha = 1.0;

	std::vector<int> splat_idx;
	std::vector<std::pair<int, int> > blur_idx;

	Eigen::VectorXf n;//norm
	Eigen::VectorXf m1;
	Eigen::VectorXf A_diag;

	// utility
	void normalize_cvMat2Eigen(const cv::Mat& input, Eigen::VectorXf& output);
	void bilateral2pixel(Eigen::VectorXf& y, const cv::Mat output);

	bool construct_Blur(mapId& hashed_coords, const std::vector<long long>& hash_vec);
	bool bistochastize();

	bool dense_solve_jacobi(
		const Eigen::VectorXf& pz_init,
		const Eigen::VectorXf& inv_A,
		Eigen::VectorXf& y
	);

	// Splat, Blur, Slice (Eigen to Eigen)
	void Splat(const Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Slice(const Eigen::VectorXf& input, Eigen::VectorXf& dst);

	void Diffuse(const Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Blur(const Eigen::VectorXf& input, Eigen::VectorXf& dst);
	bool HFBS_solve(const cv::Mat& target, const cv::Mat& confidence, cv::Mat& output);
};
