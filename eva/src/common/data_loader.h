/*
 * Copyright 2022 Sony Semiconductor Solutions Corporation.
 *
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Semiconductor
 * Solutions Corporation.
 * No part of this file may be copied, modified, sold, and distributed in any
 * form or by any means without prior explicit permission in writing from
 * Sony Semiconductor Solutions Corporation.
 *
 */
#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

//TL23 simulatorのdataloder
//

// a  b  c
// c -b -a
void swap_coordinate_to_OpenCV(cv::Mat& normal) {
	// OpenCV座標系にする
	for (int y = 0; y < normal.rows; ++y) {
		for (int x = 0; x < normal.cols; ++x) {
			double a = normal.at<cv::Vec3d>(y, x)[0];
			double b = normal.at<cv::Vec3d>(y, x)[1];
			double c = normal.at<cv::Vec3d>(y, x)[2];
			normal.at<cv::Vec3d>(y, x)[0] = c;
			normal.at<cv::Vec3d>(y, x)[1] = -b;
			normal.at<cv::Vec3d>(y, x)[2] = -a;
		}
	}
}

cv::Mat read_K(const std::string& fullpath) {
	std::ifstream file(fullpath);
	if (!file)return cv::Mat();

	std::string line;
	std::vector<double> K_vec;
	while (std::getline(file, line)) {
		std::stringstream ss{ line };
		std::string s;
		while (std::getline(ss, s, ' ')) {     // スペース（' '）で区切って，格納
			double data = std::stod(s);
			K_vec.push_back(data);
		}
	}
	cv::Mat K_vec_1d(K_vec);
	cv::Mat K=K_vec_1d.reshape(1, 3).clone();
	return K;
}

cv::Mat read_D(const std::string& fullpath) {
	cv::Mat D = cv::imread(fullpath, cv::IMREAD_UNCHANGED);
	cv::Mat D_split[3]; 
	cv::split(D, D_split);
	cv::Mat D1;
	D_split[0].convertTo(D1, CV_64FC1);

	cv::Mat depth = 1.0 - D1 / (double)(UINT16_MAX);
	depth *= 10.;
	return depth;
}

cv::Mat read_N(const std::string& fullpath) {
	cv::Mat N = cv::imread(fullpath, cv::IMREAD_UNCHANGED);
	N.convertTo(N, CV_64FC3);
	cv::Mat normal = N / static_cast<double>(UINT16_MAX);
	normal = normal * 2.0;
	normal = normal - 1.0;

	// OpenCVに合わせる
	swap_coordinate_to_OpenCV(normal);
	return normal;
}

