#pragma once
/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */
#ifndef SAT_FBS
#define SAT_FBS
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <unordered_map> //c++11
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

using mapId = std::unordered_map<long long /* hash */, int /* vert id */>;
using namespace cv;

namespace SAT {


	/** @brief Interface for implementations of Fast Bilateral Solver.
		For more details about this solver see @cite BarronPoole2016 .
	*/
	class FastBilateralSolver {
	public:
		FastBilateralSolver(InputArray guide,double sigma_spatial,double sigma_luma,double sigma_chroma,double lambda,int num_iter,double max_tol){
			CV_Assert(guide.type() == CV_8UC1 || guide.type() == CV_8UC3);
			Mat gui = guide.getMat();
			bs_param.lam = lambda;
			bs_param.cg_maxiter = num_iter;
			bs_param.cg_tol = max_tol;
			init(gui, sigma_spatial, sigma_luma, sigma_chroma);
		};

		void filter(InputArray src, InputArray confidence, OutputArray dst);

	private:
		// calc Splat, Blur matrix
		void init(const cv::Mat& reference, double sigma_spatial, double sigma_luma, double sigma_chroma);

		// Splat, Blur, Slice (Eigen to Eigen)
		void Splat(const Eigen::VectorXf& input, Eigen::VectorXf& dst);
		void Blur(const Eigen::VectorXf& input, Eigen::VectorXf& dst);
		void Slice(const Eigen::VectorXf& input, Eigen::VectorXf& dst);

		// solve Ay = b
		void solve(const cv::Mat& src, const cv::Mat& confidence, cv::Mat& dst);

		// convert cv::Mat to Eigen::VectorXf
		void normalize_cvMat2Eigen(const cv::Mat & input, Eigen::VectorXf & output);
		void construct_Blur(mapId &hashed_coords, const std::vector<long long>& hash_vec);

		// bistochastize B to Dn, Dm
		void bistochastize();

		// convert bilateral space to pixel space
		void bilateral2pixel(Eigen::VectorXf & y,  const cv::Mat out);


		inline void diagonal(const Eigen::VectorXf& v, Eigen::SparseMatrix<float>& mat){
			mat = Eigen::SparseMatrix<float>(v.size(), v.size());
			for (int i = 0; i < int(v.size()); i++) {
				mat.insert(i, i) = v(i);
			}
		}

		int npixels;
		int nvertices;
		int dim;
		int cols;
		int rows;
		std::vector<int> splat_idx;
		std::vector<std::pair<int, int> > blur_idx;
		Eigen::VectorXf m;
		Eigen::VectorXf n;
		Eigen::SparseMatrix<float, Eigen::ColMajor> blurs;
		Eigen::SparseMatrix<float, Eigen::ColMajor> S;
		Eigen::SparseMatrix<float, Eigen::ColMajor> Dn;
		Eigen::SparseMatrix<float, Eigen::ColMajor> Dm;

		struct grid_params {
			float spatialSigma;
			float lumaSigma;
			float chromaSigma;
			grid_params():spatialSigma(8.0), lumaSigma(8.0),chromaSigma(8.0){}
		};

		struct bs_params {
			float lam;
			float A_diag_min;
			float cg_tol;
			int cg_maxiter;
			bs_params():lam(128.0) , A_diag_min(1e-5f), cg_tol(1e-5f), cg_maxiter(25){}
		};

		grid_params grid_param;
		bs_params bs_param;
	};


	/** @brief Simple one-line Fast Bilateral Solver filter call. If you have multiple images to filter with the same
	guide then use FastBilateralSolverFilter interface to avoid extra computations.
	@param guide image serving as guide for filtering. It should have 8-bit depth and either 1 or 3 channels.
	@param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point 32-bit depth and up to 4 channels.
	@param confidence confidence image with unsigned 8-bit or floating-point 32-bit confidence and 1 channel.
	@param dst destination image.
	@param sigma_spatial parameter, that is similar to spatial space sigma (bandwidth) in bilateralFilter.
	@param sigma_luma parameter, that is similar to luma space sigma (bandwidth) in bilateralFilter.
	@param sigma_chroma parameter, that is similar to chroma space sigma (bandwidth) in bilateralFilter.
	@param lambda smoothness strength parameter for solver.
	@param num_iter number of iterations used for solver, 25 is usually enough.
	@param max_tol convergence tolerance used for solver.
	For more details about the Fast Bilateral Solver parameters, see the original paper @cite BarronPoole2016.
	@note Confidence images with CV_8U depth are expected to in [0, 255] and CV_32F in [0, 1] range.
	*/
	void fastBilateralSolver_sparse(
		InputArray guide,
		InputArray src,
		InputArray mask,
		InputArray confidence,
		cv::Mat_<float>& dst,
		double sigma_spatial = 8,
		double sigma_luma = 8,
		double sigma_chroma = 8,
		double lambda = 128.0,
		int num_iter = 25,
		double max_tol = 1e-5
	);
}
#endif