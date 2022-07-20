/**
* @file		sTFBS.Data.cpp
* @brief	FBSの実装の内、sparse dataに関する関数群の定義ファイル
* @author	yusuke moriuchi
* @date		2019/12/28
* @details	BaseLib.FBSFpara1.h, BaseLib.FBSFpara1.cpp, BaseLib.FBSFimplSparseWithSV.hppから移植中
*/

/*  ********************************************************************************************
 *
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  ********************************************************************************************/
#include <opencv2/opencv.hpp>
#include "sTFBS.h"

void ResizeData(sData_1frame& data, int size, int guide_size, int guide_subsize)
{
	data.size = size;

	data.x.resize(size);
	data.y.resize(size);

	data.d.resize(size);
	data.c.resize(size);

	if (data.gcU8.size() <= 0)
		data.gcU8.resize(guide_size);
	for (int i = 0; i < guide_size; i++)
		data.gcU8[i].resize(size);

	if (data.gdU8.size() <= 0)
		data.gdU8.resize(guide_subsize);
	for (int i = 0; i < guide_subsize; i++)
		data.gdU8[i].resize(size);
}


void set_data(
	sData_1frame& data,
	const cv::InputArray depth_,
	const cv::InputArray guide_,
	const cv::InputArray guideSub_,
	const cv::InputArray conf_,
	const cv::InputArray mask_,
	const float& maskMinVal
) {
	const cv::Mat mask = mask_.getMat();
	const cv::Mat depth = depth_.getMat();
	const cv::Mat guide = guide_.getMat();
	const cv::Mat guideSub = guideSub_.getMat();
	const cv::Mat conf = conf_.getMat();

	int guideSize = 0;
	if (!guide.empty() && data.gcU8.empty()) {
		guideSize = guide.channels();
	}
	int guideSubSize = 0;
	if (!guideSub.empty() && data.gdU8.empty()) {
		guideSubSize = guideSub.channels();
	}

	const int w = depth.cols;
	const int h = depth.rows;

	if (mask.empty()) {
		ResizeData(data, w * h, guideSize, guideSubSize);
		// 実行されない
		for (int j = 0; j < h; j++) {
			const float* depthData = depth.ptr<float>(j);
			const uchar* guideData = guide.ptr<uchar>(j);
			const uchar* guideSubData = guideSub.ptr<uchar>(j);
			const float* confData = conf.ptr<float>(j);
			for (int i = 0; i < w; i++) {
				data.x[j * w + i] = i;
				data.y[j * w + i] = j;
				data.d[j * w + i] = (float)depthData[i];
				for (int k = 0; k < guideSize; k++)
					data.gcU8[k][j * w + i] = guideData[guide.channels() * i + k];
				for (int g = 0; g < guideSubSize; g++)
					data.gdU8[g][j * w + i] = guideSubData[guideSub.channels() * i + g];
				if (!conf.empty())
					data.c[j * w + i] = (float)confData[i];
			}
		}
	}
	else{
		ResizeData(data, cv::countNonZero(mask), guideSize, guideSubSize);

		int count = 0;
		for (int j = 0; j < h; ++j) {
			const float* maskData = mask.ptr<float>(j);
			const float* depthData = depth.ptr<float>(j);
			const uchar* guideData = guide.ptr<uchar>(j);
			const uchar* guideSubData = guideSub.ptr<uchar>(j);
			const float* confData = conf.ptr<float>(j);
			for (int i = 0; i < w; ++i) {
				if (maskMinVal < maskData[i]) {
					data.x[count] = i;
					data.y[count] = j;
					data.d[count] = (float)depthData[i];
					for (int k = 0; k < guideSize; k++)
						data.gcU8[k][count] = guideData[guide.channels() * i + k];
					for (int g = 0; g < guideSubSize; g++)
						data.gdU8[g][count] = guideSubData[guideSub.channels() * i + g];
					if (!conf.empty()) {
						data.c[count] = (float)confData[i];
					}
					++count;
				}
			}
		}
		//ResizeData(data, count, guideSize, guideSubSize);
	}
}
