#include "upsampling.h"

//
// upsampling for DS4
//
inline void circle_fill_overlap(
	const cv::Mat& sparse,
	cv::Mat& out, 
	cv::Mat& white_mask, 
	cv::Mat& conf,
	const float& scale_factor,
	const int& r_circle
)
{
	//１回塗った部分の座標を保存しておく
	cv::Mat overlap_xy(out.size(), CV_32FC2); //x,y座標
	const int r_x_r = r_circle * r_circle;

	// x_anchor, y_anchor: 画像全体の x,y座標
	for (int y = 0; y < sparse.rows; ++y) {
		for (int x = 0; x < sparse.cols; ++x) {
			float z = sparse.at<float>(y, x);
			if (z > FLT_EPSILON) {
				const int xc = x * scale_factor;
				const int yc = y * scale_factor;

				// x, y: アンカーを中心とした半径 r_circle内の x,y座標
				for (int y = yc - r_circle; y < yc + r_circle; ++y) {
					for (int x = xc - r_circle; x < xc + r_circle; ++x) {
						if (x < 0 || y < 0 || x >= out.cols || y >= out.rows) continue;//領域例外処理
						float dx = x - xc;
						float dy = y - yc;
						float d_new = dx * dx + dy * dy;
						if (d_new < r_x_r) {
							if (out.at<float>(y, x) > FLT_EPSILON) {// すでに塗られている場合どちらが近いか距離を比較
								const float dx_old = overlap_xy.at<cv::Vec2f>(y, x)[0] - x;
								const float dy_old = overlap_xy.at<cv::Vec2f>(y, x)[1] - y;
								const float d_old = dx_old * dx_old + dy_old * dy_old;
								if (d_new < d_old) { //新しいやつのほうが近い場合更新する
									out.at<float>(y, x) = z;
									white_mask.at<float>(y, x) = 1.f;
									overlap_xy.at<cv::Vec2f>(y, x)[0] = xc;
									overlap_xy.at<cv::Vec2f>(y, x)[1] = yc;
									conf.at<float>(y, x) = d_new;
								}
							}
							else {// 塗られていなければ普通に塗る
								out.at<float>(y, x) = z;
								white_mask.at<float>(y, x) = 1.f;
								overlap_xy.at<cv::Vec2f>(y, x)[0] = xc;
								overlap_xy.at<cv::Vec2f>(y, x)[1] = yc;
								conf.at<float>(y, x) = d_new;
							}
						}
					}
				}

			}
		}
	}

	// 距離重みのつけ方要検討
	conf.forEach<float>([](float& p, const int* position) -> void {
		p = (std::isnan(p) || p < -FLT_EPSILON) ? 0.f : std::exp2f(-0.5 * p);
		});

}

bool upsampling(
	const cv::Mat& sparse,
	const cv::Mat& guide,
	const PARAM& param,
	cv::Mat& dense
) {
	const int W_orig = sparse.cols;
	const int H_orig = sparse.rows;

	const int W_small = W_orig * param.zoom;
	const int H_small = H_orig * param.zoom;
	cv::Mat dilate_small = cv::Mat::zeros(H_small, W_small, CV_32FC1);
	cv::Mat mask_small = cv::Mat::zeros(H_small, W_small, CV_32FC1);
	cv::Mat conf_small(H_small, W_small, CV_32FC1, cv::Scalar(-1));

	// dilation
	circle_fill_overlap(sparse, dilate_small, mask_small, conf_small, param.zoom, param.r_dilate);

	cv::Mat guide_cv_small;
	cv::resize(guide, guide_cv_small, cv::Size(W_small, H_small), cv::INTER_NEAREST);

	// FBS
	cv::Mat_<float> dense_FBS_small;
	SAT::fastBilateralSolver_sparse(
		guide_cv_small,
		dilate_small,
		mask_small,
		conf_small,
		dense_FBS_small,
		param.sigma_spatial,
		param.sigma_luma,
		param.sigma_chroma,
		param.lambda,
		param.num_iter
	);

	cv::resize(dense_FBS_small, dense, cv::Size(),(1./ param.zoom),(1. / param.zoom), cv::INTER_AREA);

	return true;
}
