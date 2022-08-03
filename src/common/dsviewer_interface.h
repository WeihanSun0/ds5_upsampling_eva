#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include<opencv2/opencv.hpp>
/**
 * @brief Get the parameters of RGB camera
 * 
 * @param params : recived by call read_param function 
 * @param cx 
 * @param cy 
 * @param fx 
 * @param fy 
 * @return true : succeed
 * @return false : failed 
 */
bool get_rgb_params(std::map<std::string, float>& params, float& cx, float& cy, float& fx, float& fy)
{
    cx = params[std::string("RGB_CX")];
    cy = params[std::string("RGB_CY")];
    fx = params[std::string("RGB_FOCAL_LENGTH_X")];
    fy = params[std::string("RGB_FOCAL_LENGTH_Y")];
    if (cx == 0.0 || cy == 0.0 || fx == 0.0 || fy == 0.0) 
        return false;
    return true;
}

/**
 * @brief Read parameters from params.txt (viewer's paramter file)
 * 
 * @param strFile : parameter file name 
 * @param params : return parameters 
 * @return true : succeed 
 * @return false : failed
 */
bool read_param(const std::string& strFile, std::map<std::string, float>& params)
{
    std::fstream fs;
    fs.open(strFile.c_str(), std::ios::in);
    if (!fs.is_open())
        return false;
    std::string strline, strKey, strVal;
    
    while(std::getline(fs, strline)) {
        strline.erase(0, strline.find_first_not_of(" "));
        if (strcmp(strline.substr(0, 2).c_str(), "//") && !strline.empty()) {
            int pos = strline.find_first_of("\t");
            if (pos == -1)
                pos = strline.find_first_of(" ");
            strKey = strline.substr(0, pos);
            strVal = strline.substr(pos+1, strline.length()-pos-1); 
            strVal.erase(0, strVal.find_first_not_of("\t"));
            params[strKey] = atof(strVal.c_str());
        }
    }
    return true;
}

inline cv::Mat pc2detph(const cv::Mat& xyz, const cv::Size& sz, 
    const float& cx, const float& cy, const float& fx, const float& fy, float scale = 1.0f) 
{
	// m → mm
	cv::Mat sparsedepth=cv::Mat::zeros(sz, CV_32FC1);
	const int width = sz.width;
	const int height = sz.height;

	if (xyz.type() == CV_32FC3) {
		for (int y = 0; y < xyz.rows; ++y) {
			for (int x = 0; x < xyz.cols; ++x) {
				float z = xyz.at<cv::Vec3f>(y,x)[2];
				float uf = ((xyz.at<cv::Vec3f>(y, x)[0] * fx / z) + cx) * scale;
				float vf = ((xyz.at<cv::Vec3f>(y, x)[1] * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
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
				float z = xyz.at<float>(2, i);
				float uf = ((xyz.at<float>(0, i) * fx / z) + cx) * scale;
				float vf = ((xyz.at<float>(1, i) * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
					}
				}
			}
		}
		else {
			for (int i = 0; i < H; ++i) {
				float z = xyz.at<float>(i, 2);
				float uf = ((xyz.at<float>(i, 0) * fx / z) + cx) * scale;
				float vf = ((xyz.at<float>(i, 1) * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
					}
				}
			}
		}
	}
	return sparsedepth;
}


inline cv::Mat pc2detph1(const cv::Mat& xyz, const cv::Size& sz, 
    const float& cx, const float& cy, const float& fx, const float& fy, float scale = 1.0f) 
{
	// m → mm
	cv::Mat sparsedepth=cv::Mat::zeros(sz, CV_32FC1);
	const int width = sz.width;
	const int height = sz.height;

	if (xyz.type() == CV_32FC3) {
		for (int y = 0; y < xyz.rows; ++y) {
			for (int x = 0; x < xyz.cols; ++x) {
				float z = xyz.at<cv::Vec3f>(y,x)[2] * 1000.f;
				float uf = ((xyz.at<cv::Vec3f>(y, x)[0] * 1000.f * fx / z) + cx) * scale;
				float vf = ((xyz.at<cv::Vec3f>(y, x)[1] * 1000.f * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
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
				float uf = ((xyz.at<float>(0, i) * 1000.f * fx / z) + cx) * scale;
				float vf = ((xyz.at<float>(1, i) * 1000.f * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
					}
				}
			}
		}
		else {
			for (int i = 0; i < H; ++i) {
				float z = xyz.at<float>(i, 2) * 1000.f;
				float uf = ((xyz.at<float>(i, 0) * 1000.f * fx / z) + cx) * scale;
				float vf = ((xyz.at<float>(i, 1) * 1000.f * fy / z) + cy) * scale;
				int u = static_cast<int>(std::round(uf));
				int v = static_cast<int>(std::round(vf));
				if (uf > 0.0f && u < width) {
					if (vf > 0.0f && v < height) {
						sparsedepth.at<float>(v, u) = z;
					}
				}
			}
		}
	}
	return sparsedepth;
}

void save_depth2txt(const cv::Mat& img, const std::string& fn)
{
	std::fstream fs;
	fs.open(fn, std::ios::out);
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			float z = img.at<float>(j, i);
			fs << z << std::endl;
		}
	}
	fs.close();
}