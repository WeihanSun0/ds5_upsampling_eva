#pragma once
#ifndef  _DILATE
#define _DILATE
#include <opencv2/opencv.hpp>

inline void circle_fill_overlap(
	const cv::Mat& xyz_,
	cv::Mat& out,
	cv::Mat& white_mask,
	cv::Mat& conf,
	const float& scale_factor,
	const int& r_circle
) {
	//�P��h���������̍��W��ۑ����Ă���
	cv::Mat overlap_xy(out.size(), CV_32FC2); //x,y���W

	const int r_x_r = r_circle * r_circle;
	// x_anchor, y_anchor: �摜�S�̂� x,y���W
	for (int idx = 0; idx < xyz_.total(); ++idx) {
		int xc = xyz_.at<cv::Vec3f>(idx)[0] * scale_factor;
		int yc = xyz_.at<cv::Vec3f>(idx)[1] * scale_factor;
		float z = xyz_.at<cv::Vec3f>(idx)[2];
		if (z < FLT_EPSILON) continue; 

		// x, y: �A���J�[�𒆐S�Ƃ������a r_circle���� x,y���W
		for (int y = yc - r_circle; y < yc + r_circle; ++y) {
			for (int x = xc - r_circle; x < xc + r_circle; ++x) {
				if (x < 0 || y < 0 || x >= out.cols || y >= out.rows) continue;//�̈��O����
				float dx = static_cast<float>(x - xc);
				float dy = static_cast<float>(y - yc);
				float d_new = dx * dx + dy * dy;
				if (d_new < r_x_r) {
					if (out.at<float>(y, x) > FLT_EPSILON) {// ���łɓh���Ă���ꍇ�ǂ��炪�߂����������r
						const float dx_old = overlap_xy.at<cv::Vec2f>(y, x)[0] - x;
						const float dy_old = overlap_xy.at<cv::Vec2f>(y, x)[1] - y;
						const float d_old = dx_old * dx_old + dy_old * dy_old;
						if (d_new < d_old) { //�V������̂ق����߂��ꍇ�X�V����
							out.at<float>(y, x) = z;
							white_mask.at<float>(y, x) = 1.f;
							overlap_xy.at<cv::Vec2f>(y, x)[0] = static_cast<float>(xc);
							overlap_xy.at<cv::Vec2f>(y, x)[1] = static_cast<float>(yc);
							if (d_new == 0) {
								conf.at<float>(y, x) = FLT_EPSILON;
							}
							else {
								conf.at<float>(y, x) = 0;
							}
						}
					}
					else {// �h���Ă��Ȃ���Ε��ʂɓh��
						out.at<float>(y, x) = z;
						white_mask.at<float>(y, x) = 1.f;
						overlap_xy.at<cv::Vec2f>(y, x)[0] = xc;
						overlap_xy.at<cv::Vec2f>(y, x)[1] = yc;
						conf.at<float>(y, x) = d_new==0? FLT_EPSILON:d_new;
					}
				}
			}
		}
	}
	// �����d�݂̂����v����
	conf.forEach<float>([](float& p, const int* position) -> void {
		p = (std::isnan(p) || p < FLT_EPSILON) ? 0.f : std::exp2f(-0.2 * p);
		});

}

inline void circle_fill_overlap(
	const cv::Vec3f& xyz_,
	cv::Mat& out,
	cv::Mat& white_mask,
	cv::Mat& conf,
	cv::Mat& overlap_xy, //x,y���W
	const float& scale_factor,
	const int& r_circle
) {
	//�P��h���������̍��W��ۑ����Ă���

	const int r_x_r = r_circle * r_circle;
	// x_anchor, y_anchor: �摜�S�̂� x,y���W
	int xc = xyz_(0) * scale_factor;
	int yc = xyz_(1) * scale_factor;
	float z = xyz_(2);
	// x, y: �A���J�[�𒆐S�Ƃ������a r_circle���� x,y���W
	for (int y = yc - r_circle; y < yc + r_circle; ++y) {
		for (int x = xc - r_circle; x < xc + r_circle; ++x) {
			if (x < 0 || y < 0 || x >= out.cols || y >= out.rows) continue;//�̈��O����
			float dx = static_cast<float>(x - xc);
			float dy = static_cast<float>(y - yc);
			float d_new = dx * dx + dy * dy;
			if (d_new < r_x_r) {
				if (out.at<float>(y, x) > FLT_EPSILON) {
					// ���łɓh���Ă���ꍇ�ǂ��炪�߂����������r
					const float dx_old = overlap_xy.at<cv::Vec2f>(y, x)[0] - x;
					const float dy_old = overlap_xy.at<cv::Vec2f>(y, x)[1] - y;
					const float d_old = dx_old * dx_old + dy_old * dy_old;
					white_mask.at<float>(y, x)= white_mask.at<float>(y, x)/2.f;
					if (d_new < d_old) {
						//�V������̂ق����߂��ꍇ�X�V����
						out.at<float>(y, x) = z;
						white_mask.at<float>(y, x) = 1.f;
						overlap_xy.at<cv::Vec2f>(y, x)[0] = static_cast<float>(xc);
						overlap_xy.at<cv::Vec2f>(y, x)[1] = static_cast<float>(yc);

						if (d_new == 0) {
							conf.at<float>(y, x) = 1.f;
						}
						else {
							conf.at<float>(y, x) = std::exp2f(-0.05 * d_new);
						}
					}
				}
				else {// �h���Ă��Ȃ���Ε��ʂɓh��
					out.at<float>(y, x) = z;
					white_mask.at<float>(y, x) = 1.f;
					overlap_xy.at<cv::Vec2f>(y, x)[0] = xc;
					overlap_xy.at<cv::Vec2f>(y, x)[1] = yc;
					conf.at<float>(y, x) = d_new == 0 ? 1.f : std::exp2f(-0.05 * d_new);
				}
			}
		}
	}
}


inline void circledilate(cv::Mat& depth, cv::Mat& dilate, cv::Mat& conf) {

	dilate = depth.clone();
	cv::Mat out = cv::Mat::zeros(depth.size(), CV_32FC1);
	cv::Mat mask_ = cv::Mat::zeros(depth.size(), CV_32FC1);
	conf = cv::Mat::zeros(depth.size(), CV_32FC1);
	cv::Mat overlap_xy(out.size(), CV_32FC2); //x,y���W

	cv::Mat xyz_;
	cv::Mat z_;
	for (int y = 0; y < depth.rows; y++) {
		for (int x = 0; x < depth.cols; x++) {
			float z = depth.at<float>(y, x);
			if (z > 0) {
				xyz_.push_back(cv::Vec3f(x, y, z));
			}
		}
	}

	for (int i = 0; i < xyz_.total(); ++i) {
		float z = xyz_.at<cv::Vec3f>(i)[2];
		float factor = 1000.f / z;
		int r = 20 * factor;
		r = r < 2 ? 2 : r;
		r = r > 10 ? 10 : r;
		circle_fill_overlap(xyz_.at<cv::Vec3f>(i), dilate, mask_, conf, overlap_xy, 1, r);
	}
}

#endif