#include "upsampling.h" 
#include <chrono>
#include <math.h>
// #define SHOW_TIME

/**
 * @brief convert point cloud to depthmap
 * 
 * @param xyz : point cloud 
 * @param sz : depth map
 * @param cx : camera parameter cx 
 * @param cy : camera parameter cy
 * @param fx : camera parameter fx 
 * @param fy : camera parameter fy 
 * @param scale : scale 
 * @return cv::Mat : depth map 
 */
inline cv::Mat xyz2depthmap(const cv::Mat& xyz, const cv::Size& sz, 
    const float& cx, const float& cy, const float& fx, const float& fy, float scale = 1.0f) 
{
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


void pre_processing(cv::Mat& guide, cv::Mat& sparse, cv::Mat& mask)
{
	//! TODO
} 

/**
 * @brief calculate confidence
 * 
 * @param sparse : input sparse 
 * @param mask : input mask 
 * @return cv::Mat 
 */
inline cv::Mat calc_confidence(const cv::Mat& sparse, const cv::Mat& mask)
{
	//! TODO
	cv::Mat conf;
	mask.copyTo(conf);
	return conf;
}

/**
 * @brief Fast Global Smoothing for sparse input 
 * 
 * @param guide : full resolution guide image 
 * @param sparse : sparse input
 * @param mask : mask for sparse 
 * @param fgs_lambda 
 * @param fgs_simga_color 
 * @param fgs_lambda_attenuation 
 * @param fgs_num_iter 
 * @param dense : dense result 
 * @param conf : confidence 
 */
void fgs(const cv::Mat & guide, const cv::Mat & sparse, const cv::Mat& mask, 
					float fgs_lambda, float fgs_simga_color, float fgs_lambda_attenuation, float fgs_num_iter,
					cv::Mat& dense, cv::Mat& conf)
{
	auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide, fgs_lambda, fgs_simga_color, fgs_lambda_attenuation, fgs_num_iter);
	cv::Mat matSparse, matMask;
	filter->filter(sparse, matSparse);
	filter->filter(mask, matMask);
	dense = matSparse / matMask;
}

/**
 * @brief Get the processing range of Upsampling
 * 
 * @param mask 
 * @param r 
 * @return cv::Mat 
 */
inline cv::Mat getUpsamplingRange(const cv::Mat& mask, int r)
{
	cv::Mat imgDilate;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(r, r));
	cv::dilate(mask, imgDilate, kernel);
	return imgDilate;
}

/**
 * @brief Upsampling main processing
 * 
 * @param img_guide : guide image 
 * @param pc_flood : flood point cloud 
 * @param pc_spot : spot point cloud 
 * @param dense : upsampling result dense depthmap 
 * @param conf : confidence map 
 * @return true 
 * @return false 
 */
bool upsampling::run(const cv::Mat& img_guide, const cv::Mat& pc_flood, const cv::Mat& pc_spot, 
						cv::Mat& dense, cv::Mat& conf)
{
#ifdef SHOW_TIME
	std::chrono::system_clock::time_point t_start, t_end;
	double elapsed;
#endif
	if (img_guide.empty()) // no guide
		return false;
	cv::Mat dmapFlood, dmapSpot, maskFlood, maskSpot; // input 
	cv::Mat rangeFlood, rangeSpot;
	cv::Mat denseFlood, denseSpot, confFlood, confSpot; // results
	if (!pc_flood.empty()) { // flood
		//pc to depthmap
#ifdef SHOW_TIME
		t_start = std::chrono::system_clock::now();
#endif
		// convert point cloud to depthmap
		dmapFlood = xyz2depthmap(pc_flood, img_guide.size(), this->cx_, this->cy_, this->fx_, this->fy_);
		// create mask image
		maskFlood = cv::Mat::zeros(dmapFlood.size(), dmapFlood.type());
		maskFlood.setTo(1.0, dmapFlood != 0.0); 
		// create range image
		rangeFlood = getUpsamplingRange(maskFlood, this->range_flood);
		// create conf image
		confFlood = calc_confidence(dmapFlood, rangeFlood);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "flood preprocessing time = " << elapsed << " [ms]" << std::endl;
#endif
		// preprocessing
		//! TODO
		// upsampling
#ifdef SHOW_TIME
		t_start = std::chrono::system_clock::now();
#endif
		fgs(img_guide, dmapFlood, maskFlood, 
			this->fgs_lambda_flood_, this->fgs_sigma_color_flood_, fgs_lambda_attenuation_, fgs_num_iter_, 
			denseFlood, confFlood);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "FGS flood processing time = " << elapsed << " [ms]" << std::endl;
#endif
	}
	if (!pc_spot.empty()) { // spot
#ifdef SHOW_TIME
		t_start = std::chrono::system_clock::now();
#endif
		// convert point cloud to depthmap
		dmapSpot = xyz2depthmap(pc_spot, img_guide.size(), this->cx_, this->cy_, this->fx_, this->fy_);
		// create mask image
		maskSpot = cv::Mat::zeros(dmapSpot.size(), dmapSpot.type());
		maskSpot.setTo(1.0, dmapSpot!= 0.0);
		// create range image
		rangeSpot = getUpsamplingRange(maskSpot, this->range_spot);
		// create conf image
		confSpot = calc_confidence(dmapSpot, rangeSpot);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "spot preprocessing time = " << elapsed << " [ms]" << std::endl;
#endif
		// preprocessing
#ifdef SHOW_TIME
		t_start = std::chrono::system_clock::now();
#endif
		// upsampling
		fgs(img_guide, dmapSpot, maskSpot, 
			this->fgs_lambda_spot_, this->fgs_sigma_color_spot_, fgs_lambda_attenuation_, fgs_num_iter_, 
			denseSpot, confSpot);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "FGS spot processing time = " << elapsed << " [ms]" << std::endl;
#endif
	}
	// post processing
#ifdef SHOW_TIME
        t_start = std::chrono::system_clock::now();
#endif
	if (!denseFlood.empty()) {
		denseFlood.setTo(std::nan(""), rangeFlood == 0.0);
		confFlood.setTo(std::nan(""), rangeFlood == 0.0);
	}
	if (!denseSpot.empty()) {
		denseSpot.setTo(std::nan(""), rangeSpot == 0.0);
		confSpot.setTo(std::nan(""), rangeSpot == 0.0);
	}
	if (!denseFlood.empty()) {
		if (!denseSpot.empty()) {
			//merge
			denseFlood.copyTo(dense);
			denseSpot.copyTo(dense, rangeFlood == 0.0);
			confFlood.copyTo(conf);
			confSpot.copyTo(conf, rangeFlood == 0.0);
			return true;
		} else { // only flood
			denseFlood.copyTo(dense);
			confFlood.copyTo(conf);
			return true;
		}
	} else if(!denseSpot.empty()) {
		denseSpot.copyTo(dense);
		confSpot.copyTo(conf);
		return true;
	} else {
		return false;
	} 
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "post processing time = " << elapsed << " [ms]" << std::endl;
#endif
}

/**
 * @brief convert depthmap to point cloud
 * 
 * @param depth : input depthmap 
 * @param pc : output point cloud 
 */
void upsampling::depth2pc(const cv::Mat& depth, cv::Mat& pc)
{
	pc.create(depth.size(), CV_32FC3);
	float fx = this->fx_;
	float fy = this->fy_;
	float cx = this->cx_;
	float cy = this->cy_;

	for (int y = 0; y < depth.rows; ++y) {
		for (int x = 0; x < depth.cols; ++x) {
			float z = depth.at<float>(y, x);
			pc.at<cv::Vec3f>(y, x)[0] = (x - cx) * z / fx;
			pc.at<cv::Vec3f>(y, x)[1] = (y - cy) * z / fy;
			pc.at<cv::Vec3f>(y, x)[2] = z;
		}
	}
}