#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief convert depthmap to color image
 * 
 * @param z : depthmap 
 * @param min : min value 
 * @param max : max value 
 * @param type 
 * @return cv::Mat : color image 
 */
cv::Mat z2colormap(
	const cv::Mat& z,
	const double& min,
	const double& max,
	int type = cv::COLORMAP_TURBO
) {
	cv::Mat z_u8 = z - min;
	z_u8.convertTo(z_u8, CV_8U, 255. / (max - min));

	cv::Mat z_turbo;
	cv::Mat z_colormap;
	cv::applyColorMap(z_u8, z_colormap, type);
	return z_colormap;
}

bool PlotContour(
	const cv::Mat& input, 
	cv::Mat& output, 
	cv::Mat& pseudp, 
	int step = 32, 
	double min = std::numeric_limits<double>::quiet_NaN(), 
	double max = std::numeric_limits<double>::quiet_NaN()
) {
	int step_width = 256 / step;
	//double min, max;
	cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
	mask.setTo(255, input > 0.f);

	if (isnan(min)||isnan(max)) {
		cv::minMaxLoc(input, &min, &max, 0, 0, mask);
	}
	cv::Mat normalize = (input - min) * 255 / (max - min);
	cv::Mat normalize_8u;
	normalize.convertTo(normalize_8u, CV_8U);

	cv::Mat stepimg = normalize_8u / step_width;
	stepimg = stepimg * step_width;

	cv::Mat contour = cv::Mat::zeros(stepimg.size(), CV_8UC1);

	for (int y = 0; y < (stepimg.rows - 1); ++y) {
		for (int x = 0; x < (stepimg.cols - 1); ++x) {
			if (stepimg.at<char>(y, x) == (stepimg.at<char>(y, x + 1) + step_width)
				|| stepimg.at<char>(y, x) == (stepimg.at<char>(y, x + 1) - step_width)) {
				contour.at<char>(y, x) = 255;
				contour.at<char>(y, x + 1) = 255;
			};
			if (stepimg.at<char>(y, x) == (stepimg.at<char>(y + 1, x) + step_width)
				|| stepimg.at<char>(y, x) == (stepimg.at<char>(y + 1, x) - step_width)) {
				contour.at<char>(y, x) = 255;
				contour.at<char>(y + 1, x) = 255;
			};
		}
	}
	if (pseudp.empty()) {
		cv::applyColorMap(normalize_8u, pseudp, cv::COLORMAP_TURBO);
	}
	output = pseudp.clone();
	output.setTo(255, contour);
	return true;
}

/**
 * @brief merge a number of images to one
 * 
 * @param vecImgs: list of images 
 * @param labels : label shown in lefttop corner of each images
 * @param size : layout of the grid of images 
 * @return cv::Mat : merged grid  
 */
cv::Mat mergeImages(const std::vector<cv::Mat>& vecImgs, const std::vector<std::string>& labels, const cv::Size& size)
{
	int num = vecImgs.size();
	int num_x = size.width;
	int num_y = size.height;
	int width = 0, height = 0;
	for(int i = 0; i < num; i++) {
		int cols = vecImgs[i].cols;
		int rows = vecImgs[i].rows;
		if (cols > width) 
			width = cols;
		if (rows > height) 
			height = rows;
	}
	int total_width =  width * num_x;
	int total_height = height * num_y;
	cv::Mat imgRes = cv::Mat::zeros(cv::Size(total_width, total_height), CV_8UC3);
	int count = 0;
	for (int r = 0; r < num_y; ++r) {
		for (int c = 0; c < num_x; ++c) {
			int id = r*num_x + c;
			int cols = vecImgs[id].cols;
			int rows = vecImgs[id].rows;
			cv::Rect roi = cv::Rect(width*c, height*r, cols, rows);
			cv::Mat tmp;
			if(vecImgs[id].channels() == 1) {
				cv::cvtColor(vecImgs[id], tmp, cv::COLOR_GRAY2BGR);
			} else {
				vecImgs[id].copyTo(tmp);
			}
			if (labels.size() > id && labels[id].compare("") != 0) {
				cv::putText(tmp, labels[id], cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
			}
			tmp.copyTo(imgRes(roi));
			count += 1;
			if (count >= num) // all images drawn
				return imgRes;
		}
	}
	return imgRes;
}

/**
 * @brief overlap 2 images
 * 
 * @param img1 
 * @param img2 
 * @param p : transparent rate 0.0~1.0 
 * @return cv::Mat: overlapped image 
 */
cv::Mat overlap(const cv::Mat& img1, const cv::Mat& img2, float p)
{
	cv::Mat input1, input2, overlapImg;
	if (img1.channels() == 1) {
		cv::cvtColor(img1, input1, cv::COLOR_GRAY2BGR);
	} else {
		img1.copyTo(input1);
	}
	if (img2.channels() == 1) {
		cv::cvtColor(img2, input2, cv::COLOR_GRAY2BGR);
	} else {
		img2.copyTo(input2);
	}
	cv::addWeighted(input1, p, input2, 1-p, 0, overlapImg);
	return  overlapImg;
}

/**
 * @brief mark sparse depth image to RGB image
 * 
 * @param imgRGB : background image 
 * @param sparseDepth : foreground color sparse depthmap 
 * @param mask : mask of sparse depthmap  
 * @param size : point size 
 * @return cv::Mat : marked image 
 */
cv::Mat markSparseDepth(const cv::Mat& imgRGB, const cv::Mat& sparseDepth, const cv::Mat& mask = cv::Mat(), int size = 3)
{
	cv::Mat img;
	imgRGB.copyTo(img);
	if (img.channels() == 1)
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	int width = imgRGB.cols;
	int height = imgRGB.rows;
	std::vector<std::pair<int, int>> vecPos;
	int tmp;
	mask.forEach<float>([&img, &sparseDepth, &size](float& pixel, const int pos[]) -> void {
		if (pixel == 1.0) {
			int x = pos[1];
			int y = pos[0];
			cv::circle(img, cv::Point(x, y), size, sparseDepth.at<cv::Vec3b>(y, x), -1);
		} 
	});
	return img;
}

