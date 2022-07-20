#include <omp.h>
// #include "openSAT/WFGS.h"
#include "WFGS.h"

void HorizontalPass(const cv::Mat& w_h, const cv::Mat& src, cv::Mat& dst, const double& lambda) {
	const int W(src.cols);
	const int H(src.rows);

	// ?øΩ∆óÔøΩ?øΩ»ÇÔøΩ?øΩﬂïÔøΩ?øΩ?ªâ¬î\
	// for_each ?øΩg?øΩ?øΩ
#pragma omp parallel for
	for (int y = 0; y < H; ++y) {
		std::vector<double> u(W);// u : output(1ch)
		std::vector<double> c_tilde(W);
		std::vector<double> f_tilde(W);
		// forward (?øΩK?øΩE?øΩX?øΩÃèÔøΩ?øΩ?øΩ?øΩ@) caliculate c~
		const double fx0 = src.at<float>(y, 0);
		const double c0 = -w_h.at<float>(y, 0) * lambda;
		c_tilde[0] = c0 / (1.0 - c0);
		f_tilde[0] = fx0 / (1.0 - c0);

		for (int i = 1; i < W; ++i) {
			const double ai = -lambda * w_h.at<float>(y, i - 1);
			const double bi = 1.0 + lambda * (w_h.at<float>(y, i - 1) + w_h.at<float>(y, i));
			const double ci = -lambda * w_h.at<float>(y, i);
			const double fi = src.at<float>(y, i);

			c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai); //(Eq.8)
			f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai);
		}
		// backward (?øΩA?øΩ?øΩ?øΩÍéüÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩÃâÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩi?øΩ?øΩ?øΩÁè??øΩ…âÔøΩ?øΩ?øΩ)
		u[W - 1] = f_tilde[W - 1]; // ?øΩ≈âÔøΩ?øΩi?øΩ?øΩ 1*u=f ?øΩ?øΩ?øΩ?øΩu=f		

		for (int i = W - 2; i >= 0; --i) {
			u[i] = f_tilde[i] - c_tilde[i] * u[i + 1];//(Eq.9)
		}
		for (int i = 0; i < W; ++i) dst.at<float>(y, i) = u[i];//?øΩ?øΩ?øΩfor?øΩ∆ïÔøΩ?øΩ?øΩ?øΩ?øΩK?øΩv?øΩ»ÇÔøΩ?øΩ?øΩ?øΩ?øΩ«ÇÌÇ©?øΩ?øΩ‚Ç∑?øΩ?øΩ?øΩÃÇÔøΩ
	}
}

// To DO
// ?øΩA?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩA?øΩN?øΩZ?øΩX?øΩ?øΩ?øΩÈÇΩ?øΩﬂÇ…èc?øΩ?øΩ?øΩ]?øΩu
void VerticalPass(const cv::Mat& w_v, const cv::Mat& src, cv::Mat& dst, const double& lambda) {
	const int W(src.cols);
	const int H(src.rows);

	// ?øΩ∆óÔøΩ?øΩ»ÇÔøΩ?øΩﬂïÔøΩ?øΩ?ªâ¬î\
//#pragma omp parallel for
	for (int x = 0; x < W; ++x) {
		std::vector<double> u(H);// u : output(1ch)
		std::vector<double> c_tilde(H);
		std::vector<double> f_tilde(H);
		// forward (?øΩK?øΩE?øΩX?øΩÃèÔøΩ?øΩ?øΩ?øΩ@) caliculate c~
		const double fx0 = src.at<float>(0, x);
		const double c0 = -w_v.at<float>(0, x) * lambda;
		c_tilde[0] = c0 / (1.0 - c0);
		f_tilde[0] = fx0 / (1.0 - c0);

		for (int i = 1; i < H; ++i) {
			const double ai = -lambda * w_v.at<float>(i - 1, x);
			const double bi = 1.0 + lambda * (w_v.at<float>(i - 1, x) + w_v.at<float>(i, x));
			const double ci = -lambda * w_v.at<float>(i, x);
			const double fi = src.at<float>(i, x);

			c_tilde[i] = ci / (bi - c_tilde[i - 1] * ai); //(Eq.8)
			f_tilde[i] = (fi - f_tilde[i - 1] * ai) / (bi - c_tilde[i - 1] * ai);
		}
		// backward (?øΩA?øΩ?øΩ?øΩÍéüÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩ?øΩÃâÔøΩ?øΩ?øΩ?øΩ?øΩ?øΩi?øΩ?øΩ?øΩÁè??øΩ…âÔøΩ?øΩ?øΩ)
		u[H - 1] = f_tilde[H - 1]; // ?øΩ≈âÔøΩ?øΩi?øΩ?øΩ 1*u=f ?øΩ?øΩ?øΩ?øΩu=f		
		for (int i = H - 2; i >= 0; --i) {
			u[i] = f_tilde[i] - c_tilde[i] * u[i + 1];//(Eq.9)
		}

		for (int i = 0; i < H; ++i)
			dst.at<float>(i, x) = u[i];//?øΩ?øΩ?øΩfor?øΩ∆ïÔøΩ?øΩ?øΩ?øΩ?øΩK?øΩv?øΩ»ÇÔøΩ?øΩ?øΩ?øΩ?øΩ«ÇÌÇ©?øΩ?øΩ‚Ç∑?øΩ?øΩ?øΩÃÇÔøΩ
	}
}

bool WFGS::Calculate_weight(const cv::Mat& guide_) {
	CV_Assert(!guide_.empty());

	w_h_ = cv::Mat::zeros(guide_.size(), CV_32FC1);
	w_v_ = cv::Mat::zeros(guide_.size(), CV_32FC1);
	const int W = guide_.cols;
	const int H = guide_.rows;
	const int C = guide_.channels();

	// Horizontal Edge w_h
#pragma omp parallel for
	for (int y = 0; y < H; ++y) {
		const uchar* rgbData = guide_.ptr<uchar>(y);
		for (int x = 0; x < W - 1; ++x) {
			int index = 0;
			for (int k = 0; k < C; ++k) {
				auto color_diff = rgbData[x * C + k] - rgbData[(x + 1) * C + k];
				index += color_diff * color_diff;
				w_h_.at<float>(y, x) = LUT[index];
			}
		}
	}

	// Vertical Edge w_v
#pragma omp parallel for
	for (int y = 0; y < H - 1; ++y) {
		const uchar* rgbdata = guide_.ptr<uchar>(y);
		const uchar* rgb_nextdata = guide_.ptr<uchar>(y + 1);

		for (int x = 0; x < W; ++x) {
			int index = 0;

			for (int k = 0; k < C; ++k) {
				auto color_diff = rgbdata[x * C + k] - rgb_nextdata[x * C + k];
				index += color_diff * color_diff;
				w_v_.at<float>(y, x) = LUT[index];
			}
		}
	}
	is_weight_ready = true;
	return true;
}
