#include "upsampling.h" 
#include <chrono>
#include <math.h>
// #define SHOW_TIME

upsampling::upsampling()
{
	this->m_flood_mask = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_range = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_mask = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_range = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_dmap = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_dmap = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_grid = cv::Mat::zeros(cv::Size(grid_width, grid_height), CV_32FC1);
}

void upsampling::clear()
{
	this->m_flood_dmap.setTo(0.0);
	this->m_flood_grid.setTo(0.0);
	this->m_flood_mask.setTo(0.0);
	this->m_flood_range.setTo(0.0);
	this->m_spot_dmap.setTo(0.0);
	this->m_spot_mask.setTo(0.0);
	this->m_spot_range.setTo(0.0);
}

void upsampling::initialization(cv::Mat& dense, cv::Mat& conf)
{
	this->clear();
	if(dense.empty())
		dense = cv::Mat::zeros(cv::Size(this->guide_width, this->guide_height), CV_32FC1);
	else
		dense.setTo(0);
	
	if(conf.empty())
		conf = cv::Mat::zeros(cv::Size(this->guide_width, this->guide_height), CV_32FC1);
	else
		conf.setTo(0);
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
void upsampling::fgs_f(const cv::Mat & guide, const cv::Mat & sparse, const cv::Mat& mask, 
					float fgs_lambda, float fgs_simga_color, float fgs_lambda_attenuation, 
					float fgs_num_iter, const cv::Rect& roi, 
					cv::Mat& dense, cv::Mat& conf)
{
	auto filter = cv::ximgproc::createFastGlobalSmootherFilter(guide(roi), fgs_lambda, fgs_simga_color, 
					fgs_lambda_attenuation, fgs_num_iter);
	cv::Mat matSparse, matMask;
	filter->filter(sparse(roi), matSparse);
	filter->filter(mask(roi), matMask);
	dense(roi) = matSparse / matMask;
	conf(roi) = matMask;
}



inline cv::Mat getSobelAbs(const cv::Mat& src, int tapsize_sobel)
{
	cv::Mat h_sobels, v_sobels;
	cv::Sobel(src, h_sobels, CV_32F, 1, 0, tapsize_sobel);
	cv::Sobel(src, v_sobels, CV_32F, 0, 1, tapsize_sobel);
	cv::Mat dst = 0.5 * (abs(h_sobels) + abs(v_sobels));
	return dst;
}

inline void mark_block(cv::Mat& img, int u, int v, int r)
{
	int start_u = u-r;
	int start_v = v-r;
	int end_u = u + r;
	int end_v = v + r;
	start_u = start_u >= 0 ? start_u : 0;
	start_v = start_v >= 0 ? start_v : 0;
	end_u = end_u < img.cols ? end_u : img.cols;
	end_v = end_v < img.rows ? end_v : img.rows;
	img(cv::Rect(start_u, start_v, end_u-start_u, end_v-start_v)) = 1.0;
}

void upsampling::flood_preprocessing(const cv::Mat& img_guide, const cv::Mat& pc_flood)
{
	// create flood grid 
	cv::Mat grid = this->m_flood_grid;
	pc_flood.forEach<cv::Vec3f>([&grid](cv::Vec3f &p, const int * pos)->void{
		int j = pos[0];
		int i = pos[1];
		grid.at<float>(j, i) = p[2];
	});
	// depth edge
	this->m_flood_edge = getSobelAbs(this->m_flood_grid, 1);
	// guide edge
	cv::Mat imgEdgeGuide;
	cv::Canny(img_guide, imgEdgeGuide, 100, 130, 3);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
		cv::Size(this->m_guide_edge_dilate_size, this->m_guide_edge_dilate_size));
	cv::dilate(imgEdgeGuide, this->m_guide_edge, kernel);
	// make depthmap
	int width = this->guide_width;
	int height = this->guide_height;
	float cx = this->cx_;
	float cy = this->cy_;
	float fx = this->fx_;
	float fy = this->fy_;

	int minX = this->guide_width, minY = this->guide_height, maxX = 0, maxY = 0;
	for (int j = 0; j < this->grid_height; ++j) {
		for (int i = 0; i < this->grid_width; ++i) {
			cv::Vec3f val = pc_flood.at<cv::Vec3f>(j, i);
			float z = val[2];
			float x = val[0];
			float y = val[1];
			float uf = (x * fx / z) + cx;
			float vf = (y * fy / z) + cy;
			int u = static_cast<int>(std::round(uf));
			int v = static_cast<int>(std::round(vf));
			if (u >= 0 && u < width && v >= 0 && v < height) {
				if (z <= this->m_dist_thresh) { // candidate
					if (this->m_flood_edge.at<float>(j,i) > this->m_depth_edge_thresh) { // edge point
						if (this->m_guide_edge.at<uchar>(v, u) != 0) { // edge point
							continue;
						}
					}
				}
				m_flood_dmap.at<float>(v, u) = z;
				m_flood_mask.at<float>(v, u) = 1.0;
				mark_block(m_flood_range, u, v, this->range_flood);
				if (u < minX) minX = u;
				if (u > maxX) maxX = u;
				if (v < minY) minY = v;
				if (v > maxY) maxY = v;
			}
		}
	}
	this->m_flood_roi.x = minX;
	this->m_flood_roi.y = minY;
	this->m_flood_roi.width = maxX - minX;
	this->m_flood_roi.height = maxY - minY;
}

void upsampling::spot_preprocessing(const cv::Mat& guide, const cv::Mat& pc_spot)
{
	cv::Mat dmap = this->m_spot_dmap;
	cv::Mat mask = this->m_spot_mask;
	cv::Mat range = this->m_spot_range;
	int r = this->range_spot;
	int width = this->guide_width;
	int height = this->guide_height;
	float cx = this->cx_;
	float cy = this->cy_;
	float fx = this->fx_;
	float fy = this->fy_;

	pc_spot.forEach<cv::Vec3f>([&dmap, &mask, &range, width, height, cx, cy, fx, fy, r]
								(cv::Vec3f& p, const int* pos) -> void{
		float z = p[2];
		float uf = (p[0] * fx / z) + cx;
		float vf = (p[1] * fy / z) + cy;
		int u = static_cast<int>(std::round(uf));
		int v = static_cast<int>(std::round(vf));
		if (u >= 0 && u < width && v >= 0 && v < height) {
			dmap.at<float>(v, u) = z;
			mask.at<float>(v, u) = 1.0;
			mark_block(range, u, v, r);
		}
	});	
}

cv::Mat upsampling::get_spot_depthMap()
{
	return this->m_spot_dmap;
}

cv::Mat upsampling::get_flood_depthMap()
{
	return this->m_flood_dmap;
}

void upsampling::run_flood(const cv::Mat& img_guide, const cv::Mat& pc_flood, cv::Mat& dense, cv::Mat& conf)
{
#ifdef SHOW_TIME
	std::chrono::system_clock::time_point t_start, t_end;
	double elapsed;
	t_start = std::chrono::system_clock::now();
#endif
	//pc to depthmap
	this->flood_preprocessing(img_guide, pc_flood);
#ifdef SHOW_TIME
	t_end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
	std::cout << "xyz2depthmap time = " << elapsed << " [us]" << std::endl;
#endif

	//! TODO preprocessing
	// upsampling
#ifdef SHOW_TIME
	t_start = std::chrono::system_clock::now();
#endif
	// cv::Rect roi(0, 0, this->guide_width, this->guide_height);
	this->fgs_f(img_guide, this->m_flood_dmap, this->m_flood_mask, 
		this->fgs_lambda_flood_, this->fgs_sigma_color_flood_, 
		this->fgs_lambda_attenuation_, this->fgs_num_iter_, this->m_flood_roi, 
		dense, conf);
#ifdef SHOW_TIME
	t_end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
	std::cout << "FGS processing time = " << elapsed << " [us]" << std::endl;
#endif
	dense.setTo(std::nan(""), this->m_flood_range == 0.0);
	conf.setTo(std::nan(""), this->m_flood_range == 0.0);
}

void upsampling::run_spot(const cv::Mat& img_guide, const cv::Mat& pc_spot, cv::Mat& dense, cv::Mat& conf)
{
#ifdef SHOW_TIME
	std::chrono::system_clock::time_point t_start, t_end;
	double elapsed;
	t_start = std::chrono::system_clock::now();
#endif
	this->spot_preprocessing(img_guide, pc_spot);

#ifdef SHOW_TIME
	t_end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	std::cout << "spot preprocessing time = " << elapsed << " [ms]" << std::endl;
#endif

#ifdef SHOW_TIME
	t_start = std::chrono::system_clock::now();
#endif
	cv::Rect roi(0, 0, this->guide_width, this->guide_height);
	// upsampling
	fgs_f(img_guide, m_spot_dmap, m_spot_mask, 
		this->fgs_lambda_spot_, this->fgs_sigma_color_spot_, 
		fgs_lambda_attenuation_, fgs_num_iter_, roi,
		dense, conf);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "FGS spot processing time = " << elapsed << " [ms]" << std::endl;
#endif

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
	this->initialization(dense, conf);
	// set mode
	m_mode = 0;
	if (!pc_flood.empty()) { // flood
		m_mode = 1;
	}
	if (!pc_spot.empty()) { // spot
		if(m_mode == 1)
			m_mode = 3;
		else
			m_mode = 2;
	}

	if (m_mode == 0) {
		dense.setTo(std::nan(""));
		conf.setTo(std::nan(""));
		return false;
	}
	if (m_mode == 1) {
		this->run_flood(img_guide, pc_flood, dense, conf);
		return true;
	}
	if (m_mode == 2) {
		this->run_spot(img_guide, pc_spot, dense, conf);
		return true;
	}
	if (m_mode == 3) {
		cv::Mat denseSpot = cv::Mat::zeros(dense.size(), dense.type());
		cv::Mat confSpot = cv::Mat::zeros(conf.size(), conf.type());
		this->run_flood(img_guide, pc_flood, dense, conf);
		this->run_spot(img_guide, pc_spot, denseSpot, confSpot);
		// merge
		denseSpot.copyTo(dense, this->m_flood_range == 0);
		confSpot.copyTo(conf, this->m_spot_range == 0);
	}
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