#include "upsampling.h" 
#include <chrono>
#include <thread>
#include <math.h>
// #define SHOW_TIME

/**
 * @brief Construct a new upsampling::upsampling object
 * 
 */
upsampling::upsampling()
{
	this->m_flood_mask = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_range = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_mask = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_range = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_dmap = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_spot_dmap = cv::Mat::zeros(cv::Size(guide_width, guide_height), CV_32FC1);
	this->m_flood_grid = cv::Mat::zeros(cv::Size(grid_width, grid_height), CV_32FC1);
	this->m_guide_edge = cv::Mat::ones(cv::Size(guide_width, guide_height), CV_32FC1);
}

/**
 * @brief set upsampling parameters
 * 
 * @param fgs_lambda_flood  
 * @param fgs_sigma_flood 
 * @param fgs_lambda_spot 
 * @param fgs_sigma_spot 
 * @param fgs_num_iter_flood 
 * @param fgs_num_iter_spot 
 */
void upsampling::set_upsampling_parameters(double fgs_lambda_flood, double fgs_sigma_flood, 
											double fgs_lambda_spot, double fgs_sigma_spot,
											int fgs_num_iter_flood, int fgs_num_iter_spot) 
{ 
	this->fgs_lambda_flood_ = fgs_lambda_flood; 
	this->fgs_sigma_color_flood_ = fgs_sigma_flood; 
	this->fgs_lambda_spot_ = fgs_lambda_spot; 
	this->fgs_sigma_color_spot_ = fgs_sigma_spot; 
	this->fgs_num_iter_flood = fgs_num_iter_flood;
	this->fgs_num_iter_spot = fgs_num_iter_spot;
	if (this->fgs_lambda_flood_ < 1) this->fgs_lambda_flood_ = 1;
	if (this->fgs_sigma_color_flood_ < 1) this->fgs_sigma_color_flood_ = 1;
	if (this->fgs_num_iter_flood < 1) this->fgs_num_iter_flood = 1;
	if (this->fgs_num_iter_flood > 5) this->fgs_num_iter_flood = 5;
	if (this->fgs_num_iter_spot < 1) this->fgs_num_iter_spot = 1;
	if (this->fgs_num_iter_spot > 5) this->fgs_num_iter_spot = 5;
};


/**
 * @brief get default upsampling parameters 
 * 
 * @param fgs_lambda_flood 
 * @param fgs_sigma_flood 
 * @param fgs_lambda_spot 
 * @param fgs_sigma_spot 
 * @param fgs_num_iter_flood 
 * @param fgs_num_iter_spot 
 */
void upsampling::get_default_upsampling_parameters(double& fgs_lambda_flood, double& fgs_sigma_flood, 
													double& fgs_lambda_spot, double& fgs_sigma_spot, 
													int& fgs_num_iter_flood, int& fgs_num_iter_spot)
{
	fgs_lambda_flood = this->fgs_lambda_flood_;
	fgs_sigma_flood = this->fgs_sigma_color_flood_;
	fgs_lambda_spot = this->fgs_lambda_spot_;
	fgs_sigma_spot = this->fgs_sigma_color_spot_;
	fgs_num_iter_flood = this->fgs_num_iter_flood;
	fgs_num_iter_spot = this->fgs_num_iter_spot;
}

/**
 * @brief set preprocessing parameters
 * 
 * @param edge_dilate_size : guide edge dilate size 
 * @param edge_threshold : depth edge threshold 
 * @param canny_low_threshold : canny filter low threshold 
 * @param canny_high_threshold : canny filter high threshold 
 * @param flood_range : valid range for flood upsampling
 */
void upsampling::set_preprocessing_parameters(int edge_dilate_size, float edge_threshold, 
											int canny_low_threshold, int canny_high_threshold,
											int flood_range)
{
	this->m_guide_edge_dilate_size = edge_dilate_size;
	this->m_depth_edge_thresh = edge_threshold;
	this->m_canny_low_thresh = canny_low_threshold;
	this->m_canny_high_thresh = canny_high_threshold;
	this->range_flood = flood_range;
	if (edge_threshold > this->m_max_depth_edge_thresh)
		this->depth_edge_proc_on = false;
	else
		this->depth_edge_proc_on = true;
	if (canny_low_threshold > canny_high_threshold)
		this->guide_edge_proc_on = false;
	else
		this->guide_edge_proc_on = true;
	// checkerror
	if (this->range_flood < 2) this->range_flood = 2;
	if (this->m_guide_edge_dilate_size < 1) this->m_guide_edge_dilate_size = 1;
	if (this->m_depth_edge_thresh <= 0) this->m_depth_edge_thresh = 0.01;
};

/**
 * @brief get default preprocessing parameters
 * 
 * @param edge_dilate_size 
 * @param edge_threshold 
 * @param canny_low_threshold 
 * @param canny_high_threshold 
 * @param flood_range 
 */
void upsampling::get_default_preprocessing_parameters(int& edge_dilate_size, float& edge_threshold,
								int& canny_low_threshold, int& canny_high_threshold, int& flood_range)
{
	edge_dilate_size = this->m_guide_edge_dilate_size; 
	edge_threshold = this->m_depth_edge_thresh ;
	canny_low_threshold = this->m_canny_low_thresh;
	canny_high_threshold = this->m_canny_high_thresh;
	flood_range = this->range_flood;
};


/**
 * @brief clear buffers
 * 
 */
void upsampling::clear()
{
	this->m_flood_dmap.setTo(0.0);
	this->m_flood_grid.setTo(0.0);
	this->m_flood_mask.setTo(0.0);
	this->m_flood_range.setTo(0.0);
	this->m_spot_dmap.setTo(0.0);
	this->m_spot_mask.setTo(0.0);
	this->m_spot_range.setTo(0.0);
	this->m_guide_edge.setTo(1.0);
	this->m_flood_roi = cv::Rect(0, 0, this->guide_width, this->guide_height);
	this->m_spot_roi = cv::Rect(0, 0, this->guide_width, this->guide_height);
}

/**
 * @brief initialization 
 * 
 * @param dense 
 * @param conf 
 */
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
 * @brief FGS filter processing
 * 
 * @param sparse: sparse depth 
 * @param mask: mask  
 * @param roi: ROI 
 * @param dense: output dense depth 
 * @param conf: output confidence 
 */
void upsampling::fgs_f(const cv::Mat & sparse, const cv::Mat& mask, const cv::Rect& roi, const float& lambda,
					cv::Mat& dense, cv::Mat& conf)
{
	cv::Mat matSparse, matMask;
	cv::Mat sparse_roi, mask_roi;
	this->m_fgs_filter->filter(sparse(roi), matSparse);
	this->m_fgs_filter->filter(mask(roi), matMask);
	dense(roi) = matSparse / matMask;
	conf(roi) = matMask * lambda * 10;
	conf.setTo(1.0, conf > 1.0);
}

/**
 * @brief Sobel Edge detection
 * 
 * @param src: source image 
 * @param tapsize_sobel: tap size 
 * @return cv::Mat: edge image 
 */
inline cv::Mat getSobelAbs(const cv::Mat& src, int tapsize_sobel)
{
	cv::Mat h_sobels, v_sobels;
	cv::Sobel(src, h_sobels, CV_32F, 1, 0, tapsize_sobel);
	cv::Sobel(src, v_sobels, CV_32F, 0, 1, tapsize_sobel);
	cv::Mat dst = (abs(h_sobels) + abs(v_sobels));
	return dst;
}

/**
 * @brief mark upsampling valid rect (square)
 * 
 * @param img: valid map 
 * @param u: x-axis position
 * @param v: y-axis position 
 * @param r: range 
 */
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

/**
 * @brief depth information processing for flood
 * 
 * @param pc_flood: input flood point cloud 
 */
void upsampling::flood_depth_proc(const cv::Mat& pc_flood)
{
	cv::Mat pc_flood_cpy;
	pc_flood.copyTo(pc_flood_cpy);
	if (this->depth_edge_proc_on) {
		// create flood grid 
		cv::Mat grid = this->m_flood_grid;
		pc_flood.forEach<cv::Vec3f>([&grid](cv::Vec3f &p, const int * pos)->void{
			int j = pos[0];
			int i = pos[1];
			grid.at<float>(j, i) = p[2];
		});
		// depth edge filtering
		this->m_flood_edge = getSobelAbs(this->m_flood_grid, 3);
		cv::Mat imgEdgeDepth = cv::Mat::zeros(this->m_flood_edge.size(), CV_32FC3);
		imgEdgeDepth.setTo((1.0, 1.0, 1.0), this->m_flood_edge <= this->m_depth_edge_thresh);
		cv::multiply(pc_flood_cpy, imgEdgeDepth, pc_flood_cpy);
	}
	// make depthmap
	int width = this->guide_width;
	int height = this->guide_height;
	float cx = this->cx_;
	float cy = this->cy_;
	float fx = this->fx_;
	float fy = this->fy_;
	int minX = this->guide_width, minY = this->guide_height, maxX = 0, maxY = 0;
	int r = this->range_flood;
	cv::Mat flood_dmap = this->m_flood_dmap;
	cv::Mat flood_mask = this->m_flood_mask;
	cv::Mat flood_range = this->m_flood_range;
	pc_flood_cpy.forEach<cv::Vec3f>([&flood_dmap, &flood_mask, &flood_range, &minX, &minY, &maxX, &maxY, 
									r, cx, cy, fx, fy, width, height]
							(cv::Vec3f& p, const int * pos) -> void {
		float z = p[2];
		if (std::isnan(z) || z == 0.0) {

		} else {
			float x = p[0];
			float y = p[1];
			float uf = (x * fx / z) + cx;
			float vf = (y * fy / z) + cy;
			int u = static_cast<int>(std::round(uf));
			int v = static_cast<int>(std::round(vf));
			if (u >= 0 && u < width && v >= 0 && v < height) {
				flood_dmap.at<float>(v, u) = z;
				flood_mask.at<float>(v, u) = 1.0;
				mark_block(flood_range, u, v, r);
				if (u < minX) minX = u;
				if (u > maxX) maxX = u;
				if (v < minY) minY = v;
				if (v > maxY) maxY = v;
			}	
		}
	});
#if 0 // skipped for create FGS Filter parallelly
	//* mark roi
	this->m_flood_roi.x = minX;
	this->m_flood_roi.y = minY;
	this->m_flood_roi.width = maxX - minX;
	this->m_flood_roi.height = maxY - minY;
#endif
}

/**
 * @brief create FGS filter
 * 
 * @param guide: guide image 
 */
void upsampling::flood_guide_proc2(const cv::Mat& guide)
{
	cv::Rect roi = this->m_flood_roi;
	this->m_fgs_filter = cv::ximgproc::createFastGlobalSmootherFilter(guide(roi), 
							this->fgs_lambda_flood_, this->fgs_sigma_color_flood_, 
							this->fgs_lambda_attenuation_, this->fgs_num_iter_flood);
}

/**
 * @brief guide image processing for flood
 * 
 * @param guide: guide image 
 */
void upsampling::flood_guide_proc(const cv::Mat& guide)
{
	if (this->guide_edge_proc_on) {
		// guide edge
		cv::Mat imgEdgeGuide, imgBoarder;
		cv::Canny(guide, imgEdgeGuide, this->m_canny_low_thresh, this->m_canny_high_thresh, 3);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
			cv::Size(this->m_guide_edge_dilate_size, this->m_guide_edge_dilate_size));
		cv::dilate(imgEdgeGuide, imgBoarder, kernel);
		this->m_guide_edge.setTo(0.0, imgBoarder == 255);
	}
}

/**
 * @brief preprocessing for flood
 * 
 * @param img_guide: input guide image
 * @param pc_flood: input flood pointcloud 
 */
void upsampling::flood_preprocessing(const cv::Mat& img_guide, const cv::Mat& pc_flood)
{
	std::thread th3([this, img_guide]()->void{
		this->flood_guide_proc2(img_guide); //create FGS filter
	});
	std::thread th1([this, pc_flood]()->void{
		this->flood_depth_proc(pc_flood); //depth edge processing and convert to depthmap
	});
	std::thread th2([this, img_guide]()->void{
		this->flood_guide_proc(img_guide); //guide image processing
	});
	th1.join();
	th2.join();
	th3.join();
	if (this->guide_edge_proc_on) {
		// filtering depthmap by guide edge
		cv::multiply(this->m_flood_dmap, this->m_guide_edge, this->m_flood_dmap);
	}
}

/**
 * @brief create FGS Filter for spot
 * 
 * @param guide: guide image 
 */
void upsampling::spot_guide_proc(const cv::Mat& guide)
{
	cv::Rect roi = this->m_spot_roi;
	this->m_fgs_filter = cv::ximgproc::createFastGlobalSmootherFilter(guide(roi), 
							this->fgs_lambda_spot_, this->fgs_sigma_color_spot_, 
							this->fgs_lambda_attenuation_, this->fgs_num_iter_spot);

}

/**
 * @brief depth processing for spot
 * 
 * @param pc_spot point cloud of spot 
 */
void upsampling::spot_depth_proc(const cv::Mat& pc_spot)
{
	cv::Mat dmap = this->m_spot_dmap;
	cv::Mat mask = this->m_spot_mask;
	int width = this->guide_width;
	int height = this->guide_height;
	float cx = this->cx_;
	float cy = this->cy_;
	float fx = this->fx_;
	float fy = this->fy_;

	pc_spot.forEach<cv::Vec3f>([&dmap, &mask, width, height, cx, cy, fx, fy]
								(cv::Vec3f& p, const int* pos) -> void{
		float z = p[2];
		float uf = (p[0] * fx / z) + cx;
		float vf = (p[1] * fy / z) + cy;
		int u = static_cast<int>(std::round(uf));
		int v = static_cast<int>(std::round(vf));
		if (u >= 0 && u < width && v >= 0 && v < height) {
			dmap.at<float>(v, u) = z;
			mask.at<float>(v, u) = 1.0;
		}
	});	
}

/**
 * @brief preprocessing for spot
 * 
 * @param guide: guide image 
 * @param pc_spot: point cloud of spot 
 */
void upsampling::spot_preprocessing(const cv::Mat& guide, const cv::Mat& pc_spot)
{
	std::thread th1([this, guide]()->void{
		this->spot_guide_proc(guide); //create FGS filter
	});
	std::thread th2([this, pc_spot]()->void{
		this->spot_depth_proc(pc_spot);
	});
	th1.join();
	th2.join();
}

/**
 * @brief full processing for flood 
 * 
 * @param img_guide: image guide 
 * @param pc_flood: point cloud of flood 
 * @param dense: output dense depthmap 
 * @param conf: output confidence 
 */
void upsampling::run_flood(const cv::Mat& img_guide, const cv::Mat& pc_flood, cv::Mat& dense, cv::Mat& conf)
{
#ifdef SHOW_TIME
	std::chrono::system_clock::time_point t_start, t_end;
	double elapsed;
	t_start = std::chrono::system_clock::now();
#endif
	// preprocessing
	this->flood_preprocessing(img_guide, pc_flood);
#ifdef SHOW_TIME
	t_end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
	std::cout << "xyz2depthmap time = " << elapsed << " [us]" << std::endl;
#endif

	// upsampling
#ifdef SHOW_TIME
	t_start = std::chrono::system_clock::now();
#endif
	// cv::Rect roi(0, 0, this->guide_width, this->guide_height);
	this->fgs_f(this->m_flood_dmap, this->m_flood_mask, this->m_flood_roi, this->fgs_lambda_flood_,
				dense, conf);
#ifdef SHOW_TIME
	t_end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
	std::cout << "FGS processing time = " << elapsed << " [us]" << std::endl;
#endif
	// fill invalid regions
	dense.setTo(std::nan(""), this->m_flood_range == 0.0);
	conf.setTo(std::nan(""), this->m_flood_range == 0.0);
}

/**
 * @brief full processing for spot
 * 
 * @param img_guide: image guide 
 * @param pc_spot: point cloud of spot 
 * @param dense: output dense depthmap 
 * @param conf: output confidence 
 */
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
	// upsampling
	fgs_f(this->m_spot_dmap, this->m_spot_mask, this->m_spot_roi, this->fgs_lambda_spot_, 
		dense, conf);
#ifdef SHOW_TIME
		t_end = std::chrono::system_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "FGS spot processing time = " << elapsed << " [ms]" << std::endl;
#endif
#if 0 // skipped for full region results
	// fill invalid regions
	dense.setTo(std::nan(""), this->m_spot_range == 0.0);
	conf.setTo(std::nan(""), this->m_spot_range == 0.0);
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

	if (m_mode == 0) { // invalid
		dense.setTo(std::nan(""));
		conf.setTo(std::nan(""));
		return false;
	}
	if (m_mode == 1) { // flood only
		this->run_flood(img_guide, pc_flood, dense, conf);
		return true;
	}
	if (m_mode == 2) { // spot only
		this->run_spot(img_guide, pc_spot, dense, conf);
		return true;
	}
	if (m_mode == 3) { // flood + spot
		cv::Mat denseSpot = cv::Mat::zeros(dense.size(), dense.type());
		cv::Mat confSpot = cv::Mat::zeros(conf.size(), conf.type());
		this->run_flood(img_guide, pc_flood, dense, conf);
		this->run_spot(img_guide, pc_spot, denseSpot, confSpot);
		// merge
		denseSpot.copyTo(dense, this->m_flood_range == 0);
		confSpot.copyTo(conf, this->m_spot_range == 0);
		return true;
	}
	return false;
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