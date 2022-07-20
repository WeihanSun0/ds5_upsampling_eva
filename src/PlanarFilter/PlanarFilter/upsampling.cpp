#include "dilate.h"
#include "upsampling.h"
#include "planarfilter.h"
// #define SHOW_TIME 1

inline cv::Mat xyz2depthmap(
	const cv::Mat& xyz, const cv::Size& sz, const float& cx, const float& cy, const float& fx, const float& fy, float scale = 0.25f) {
	// m �� mm
	cv::Mat sparsedepth=cv::Mat::zeros(sz, CV_32FC1);
	const int width = sz.width;
	const int height = sz.height;

	if (xyz.type() == CV_32FC3) {
		for (int y = 0; y < xyz.rows; ++y) {
			for (int x = 0; x < xyz.cols; ++x) {
				float z = xyz.at<cv::Vec3f>(y,x)[2] * 1000.f;
				float u = ((xyz.at<cv::Vec3f>(y,x)[0] * 1000.f * fx / z) + cx) * scale;
				float v = ((xyz.at<cv::Vec3f>(y, x)[1] * 1000.f * fy / z) + cy) * scale;
				if (u > 0 && u < width) {
					if (v > 0 && v < height) {
						sparsedepth.at<float>(std::round(v), std::round(u)) = z;
					}
				}
			}
		}

	}
	else {
		int W = xyz.cols;
		int H = xyz.rows;
		if (W > H) {
			for (int i = 0; i < W; ++i) {
				float z = xyz.at<float>(2, i) * 1000.f;
				float u = ((xyz.at<float>(0, i) * 1000.f * fx / z) + cx) * scale;
				float v = ((xyz.at<float>(1, i) * 1000.f * fy / z) + cy) * scale;
				if (u > 0 && u < width) {
					if (v > 0 && v < height) {
						sparsedepth.at<float>(std::round(v), std::round(u)) = z;
					}
				}
			}
		}
		else {
			for (int i = 0; i < H; ++i) {
				float z = xyz.at<float>(i, 2) * 1000.f;
				float u = ((xyz.at<float>(i, 0) * 1000.f * fx / z) + cx) * scale;
				float v = ((xyz.at<float>(i, 1) * 1000.f * fy / z) + cy) * scale;
				if (u > 0 && u < width) {
					if (v > 0 && v < height) {
						sparsedepth.at<float>(std::round(v), std::round(u)) = z;
					}
				}
			}
		}
	}
	return sparsedepth;
}

inline cv::Mat stdimage_naive(const cv::Mat& in) {
	cv::Mat std_out = cv::Mat::zeros(in.size(), CV_32FC1);
	float tmp[10];
	for (int y = 1; y < std_out.rows - 1; ++y) {
		for (int x = 1; x < std_out.cols - 1; ++x) {
			tmp[0] = in.at<float>(y - 1, x - 1);
			tmp[1] = in.at<float>(y - 1, x);
			tmp[2] = in.at<float>(y - 1, x + 1);
			tmp[3] = in.at<float>(y, x - 1);
			tmp[4] = in.at<float>(y, x);//
			tmp[5] = in.at<float>(y, x + 1);
			tmp[6] = in.at<float>(y + 1, x - 1);
			tmp[7] = in.at<float>(y + 1, x);
			tmp[8] = in.at<float>(y + 1, x + 1);

			float count = 0.f;
			float sum = 0;
			for (int i = 0; i < 9; ++i) {
				if (tmp[i] > FLT_EPSILON) {
					count += 1.f;
					sum += tmp[i];
				}
			}
			if (count < FLT_EPSILON) {
				std_out.at<float>(y, x) = 0.f;
			}
			else {
				float mean = sum / count;
				float std = 0;
				for (int i = 0; i < 9; ++i) {
					std += (tmp[i] - mean) * (tmp[i] - mean);
				}
				float ans = std / 9.f;
				std_out.at<float>(y, x) = ans;
			}
		}
	}
	cv::sqrt(std_out, std_out);
	return std_out;
}

void upsampling::run2(const cv::Mat& guide, const cv::Mat& sparse_depth, const cv::Rect& roi, cv::Mat& dense_depth, cv::Mat& confidence) {

#ifdef SHOW_TIME
	std::chrono::system_clock::time_point start, end;
	double elapsed;
#endif
	roi_ = roi;
	// guide �̏k��
	cv::Mat guide_mini;
	guide.copyTo(guide_mini);
	//cv::resize(guide, guide_mini, cv::Size(), scale_, scale_);
	// depth map�̍쐬 (480,270)
	cv::Mat depth_mini;
	depth_mini = sparse_depth.clone();

	// roi crop
	guide_roi = guide_mini(roi_).clone();
	depth_roi = depth_mini(roi_).clone();

	// depth �� min/max
	cv::Mat mask_minmax = cv::Mat::zeros(depth_roi.size(), CV_8U);
	mask_minmax.setTo(1, depth_roi > 0);
	double min, max;
	cv::minMaxLoc(depth_roi, &min, &max, NULL, NULL, mask_minmax);

	// circle dilate
#ifdef SHOW_TIME
	start = std::chrono::system_clock::now();
#endif
	cv::Mat dilate, conf;
	circledilate(depth_roi, dilate, conf);
#ifdef SHOW_TIME
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "circledilate time = " << elapsed << " [ms]" << std::endl;
#endif

	// �g�債�k��
#ifdef SHOW_TIME
	start = std::chrono::system_clock::now();
#endif
	cv::Mat dilatemask, erodemask;
	cv::dilate(depth_roi, dilatemask, cv::Mat(), cv::Point(-1, -1), 8);
	cv::erode(dilatemask, erodemask, cv::Mat(), cv::Point(-1, -1), 5);

#ifdef SHOW_TIME
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "erode+erode time = " << elapsed << " [ms]" << std::endl;
#endif
	// mask ����
	cv::Mat mask = depth_roi.clone();
	mask.setTo(1, mask > 0);

	cv::Mat conf_raw = conf.clone();
	conf.setTo(0.f, conf < conf_thresh_);

	// �{��
#ifdef SHOW_TIME
	start = std::chrono::system_clock::now();
#endif
	cv::Mat dense_fgs, dense;
	auto filter = cv::ximgproc::createFastGlobalSmootherFilter(
		guide_roi, fgs_lambda_, fgs_sigma_color_, fgs_lambda_attenuation_, fgs_num_iter_);
#ifdef SHOW_TIME
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "FGS time = " << elapsed << " [ms]" << std::endl;
#endif

#ifdef SHOW_TIME
	start = std::chrono::system_clock::now();
#endif
	//planar_filter(erodemask, mask, conf, planar_coeff_, filter, dense_fgs, dense);
	planar_filter(sparse_depth, mask, conf, planar_coeff_, filter, dense_fgs, dense);
#ifdef SHOW_TIME
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "planar time = " << elapsed << " [ms]" << std::endl;
#endif
	// pose processing
	dense.setTo(0, dense < min);
	dense.setTo(0, dense > max);
	dense.setTo(0, erodemask == 0);

	// �㏈��
#ifdef SHOW_TIME
	start = std::chrono::system_clock::now();
#endif
	cv::Mat stb = stdimage_naive(dense);
#ifdef SHOW_TIME
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "stdimage_naive time = " << elapsed << " [ms]" << std::endl;
#endif

	cv::Mat tmp_mask = cv::Mat::zeros(stb.size(), CV_32FC1);
	tmp_mask.setTo(1, stb > 200);//mm
								
	// raw �M����߂�
	dense.setTo(0, tmp_mask == 1);
	dense.setTo(0, depth_roi != 0);
	dense += depth_roi;
	conf.setTo(0.4, conf == 0);
	conf.setTo(0.1, stb > 100);//mm
	conf.setTo(0, dense == 0);
	conf.setTo(1, depth_roi != 0);

	dense_depth = cv::Mat::zeros(sparse_depth.size(), sparse_depth.type());
	confidence = cv::Mat::zeros(sparse_depth.size(), sparse_depth.type());
	dense.copyTo(dense_depth);
	conf.copyTo(confidence);
	// dense_depth(roi_) = dense / 1000.f;// mm-> m
	// conf.copyTo(confidence(roi_));
	//confidence(roi_) = conf.clone();
}


void upsampling::run1(const cv::Mat& guide, const cv::Mat& xyz, cv::Mat& dense_depth, cv::Mat& confidence) {
	// guide �̏k��
	cv::Mat guide_mini;
	std::cout << guide.cols << "x" << guide.rows << "->" << scale_ << std::endl;
	cv::resize(guide, guide_mini, cv::Size(), scale_, scale_);
	// depth map�̍쐬 (480,270)
	cv::Mat depth_mini;
	depth_mini = xyz2depthmap(xyz, guide_mini.size(), cx_, cy_, fx_, fy_, scale_);

	// roi crop
	guide_roi = guide_mini(roi_).clone();
	depth_roi = depth_mini(roi_).clone();

	// depth �� min/max
	cv::Mat mask_minmax = cv::Mat::zeros(depth_roi.size(), CV_8U);
	mask_minmax.setTo(1, depth_roi > 0);
	double min, max;
	cv::minMaxLoc(depth_roi, &min, &max, NULL, NULL, mask_minmax);

	// circle dilate
	cv::Mat dilate, conf;
	circledilate(depth_roi, dilate, conf);

	// �g�債�k��
	cv::Mat dilatemask, erodemask;
	cv::dilate(depth_roi, dilatemask, cv::Mat(), cv::Point(-1, -1), 8);
	cv::erode(dilatemask, erodemask, cv::Mat(), cv::Point(-1, -1), 5);

	// mask ����
	cv::Mat mask = depth_roi.clone();
	mask.setTo(1, mask > 0);

	cv::Mat conf_raw = conf.clone();
	conf.setTo(0.f, conf < conf_thresh_);

	// �{��
	cv::Mat dense_fgs, dense;
	auto filter = cv::ximgproc::createFastGlobalSmootherFilter(
		guide_roi, fgs_lambda_, fgs_sigma_color_, fgs_lambda_attenuation_, fgs_num_iter_);

	planar_filter(erodemask, mask, conf, planar_coeff_, filter, dense_fgs, dense);

	// pose processing
	dense.setTo(0, dense < min);
	dense.setTo(0, dense > max);
	dense.setTo(0, erodemask == 0);

	// �㏈��
	cv::Mat stb = stdimage_naive(dense);
	cv::Mat tmp_mask = cv::Mat::zeros(stb.size(), CV_32FC1);
	tmp_mask.setTo(1, stb > 200);//mm
								
	// raw �M����߂�
	dense.setTo(0, tmp_mask == 1);
	dense.setTo(0, depth_roi != 0);
	dense += depth_roi;
	conf.setTo(0.4, conf == 0);
	conf.setTo(0.1, stb > 100);//mm
	conf.setTo(0, dense == 0);
	conf.setTo(1, depth_roi != 0);


	dense.copyTo(dense_depth);
	conf.copyTo(confidence);
	// dense_depth = dense / 1000.f;// mm-> m
	// confidence = conf;
}

void upsampling::set_guide_intrinsic(cv::Mat& K)
{

}
