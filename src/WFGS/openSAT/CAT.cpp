#include "PlarnarFilter.h"
// #include "openSAT/PlarnarFilter.h"
#include <omp.h>
#include "CAT.h"
// #include "openSAT/CAT.h"
#include <chrono>
#include <opencv2/ximgproc.hpp>

void InpaintErrVal(
	cv::Mat& src,
	const float& err_val,
	const float& min_val = 0.0f,
	const float& max_val = 1.0f
) {
	cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
	mask.setTo(1.0, src == err_val);
	if (cv::countNonZero(mask)) {
		cv::inpaint(src, mask, src, 0.0, cv::INPAINT_NS);
		src.setTo(min_val, src < min_val);
		src.setTo(max_val, src > max_val);
	}
}

// �f�v�X�̏C�� �G���[��f�̌�����
bool FillErrorDepth(cv::Mat& dst1, const float& depth_range_max) {
	cv::Mat laplacian, laplacian2;
	cv::Laplacian(dst1, laplacian, CV_32F, 1, 1.0, 0.0);
	cv::Laplacian(laplacian, laplacian2, CV_32F, 1, 1.0, 0.0);
	laplacian2 = abs(laplacian2.mul(laplacian));

	cv::Mat lmask = cv::Mat::zeros(laplacian.size(), CV_8UC1);

	const float magic = 8.0f;//�Ӗ�
	lmask.setTo(1.0, laplacian2 > magic * mean(laplacian2)[0]);
	if (cv::countNonZero(lmask)) {
		cv::inpaint(dst1, lmask, dst1, 0.0, cv::INPAINT_NS);
	}

	dst1.setTo(0.0f, dst1 < 0.0f);
	dst1.setTo(depth_range_max, dst1 > depth_range_max);// max clip
	return true;
}

bool Resize(
	const cv::Size& size,
	const cv::Mat& dst1,
	const cv::Mat& conf_depthedge,
	const cv::Mat& conf_diffopt,
	const bool is_depth_edge_,
	cv::Mat& dst1_up,
	cv::Mat& conf_depthedge_up,
	cv::Mat& conf_diffopt_up
) {
	cv::resize(dst1, dst1_up, size);
	if (!conf_diffopt.empty()) {
		cv::resize(conf_diffopt, conf_diffopt_up, size, cv::INTER_AREA);
	}
	else {
		conf_diffopt_up = cv::Mat::ones(size, CV_32FC1);
	}
	if (is_depth_edge_) {
		cv::resize(conf_depthedge, conf_depthedge_up, size, cv::INTER_AREA);
	}
	return true;
}

bool Blend(
	const cv::Mat& raw_depth,
	const cv::Mat& dense_depth,
	const cv::Mat& conf_raw_,
	const cv::Mat& conf_dense_,
	cv::Mat& conf_blend,
	cv::Mat& dst_blend,
	const float& cval_raw = 1.0f, //�Ӗ�
	const float& cval_opt = 1.0f
) {
	const cv::Mat raw_conf = conf_raw_ * cval_raw;
	const cv::Mat raw_conf_inv = 1.0f - raw_conf;

	conf_blend = raw_conf + raw_conf_inv.mul(conf_dense_ * cval_opt);
	dst_blend = raw_conf.mul(raw_depth) + raw_conf_inv.mul(dense_depth);
	return true;
}


bool CAT::PrepareFilters(const int& channel) {
	// WFGS 
	wfgs_pre_.Init(channel);
	wfgs_pre_.lambda_ = 1.f;
	wfgs_post_.Init(channel);
	wfgs_post_.lambda_ = 16.f;
	// TFBS
	fbs_.sigma_color_.resize(channel);
	// conf
	set_lutColorDiff(channel);
	set_lutOptDiff();
	return true;
}

void CAT::Charge(
	const cv::Mat& src_depth_,
	const cv::Mat& src_guide_,
	const cv::Mat& mask_,
	cv::Rect& rect,
	const cv::InputArray conf_ = cv::noArray()
) {
	if (!rect.empty()) {
		CV_Assert(
			(rect.x + rect.width) <= src_depth_.cols && (rect.y + rect.height) <= src_depth_.rows);
		sz_ = src_depth_.size();
		roi_ = rect;
	}
	// ������

	// �z�o�b�t�@�ւ̒ǉ�
	pb_.buf_size = MIN(pb_.buf_capa, pb_.buf_size + 1);

	// depth
	MatLayer src_depth;
	src_depth.orig = rect.empty()?
		src_depth_.clone() : src_depth_(rect).clone();
	pb_.src_depth.push_back(src_depth);
	pb_.raw_depth = src_depth.orig.clone();
	src_depth.orig.convertTo(pb_.src_depth_u8, CV_8UC1, 255.0 / conv_range_max_);

	MatLayer tmp2;
	tmp2.orig = rect.empty() ? 
		src_depth_.clone() : src_depth_(rect).clone();
	pb_.dst_depth.push_back(tmp2);

	// guide
	const cv::Mat src_guide = rect.empty() ? src_guide_: src_guide_(rect).clone();
	MatLayer src_guide_vec, src_guide_u8_vec;
	if (src_guide.depth() == CV_8U) {
		const auto type = CV_MAKETYPE(CV_32F, src_guide.channels());
		src_guide.convertTo(src_guide_vec.orig, type, 1.0 / 255.0);
		src_guide_u8_vec.orig = src_guide.clone();
	}
	else {
		const auto type = CV_MAKETYPE(CV_8U, src_guide.channels());
		src_guide_vec.orig = src_guide.clone();
		src_guide.convertTo(src_guide_u8_vec.orig, type, 255.0);
	}
	pb_.guide.push_back(src_guide_vec);
	pb_.guide_u8.push_back(src_guide_u8_vec);
	pb_.conf_guideedge.orig = is_guide_edge_ ?
		getDoGmagnitude(src_guide_vec.orig, guide_edge_blur_sigma_, guide_edge_amp_)
		: cv::Mat::ones(src_depth.orig.size(), CV_32FC1);

	pb_.mask.orig =rect.empty()? mask_: mask_(rect);
	const cv::Mat conf_total = pb_.conf_guideedge.orig.mul(pb_.mask.orig);
	if (!conf_.empty()) {
		const cv::Mat conf = rect.empty() ?
			conf_.getMat() : conf_.getMat()(rect);//change 0514
		pb_.raw_conf = conf;
		pb_.conf_total.orig = conf_total.mul(conf);
	}
	else {
		pb_.raw_conf = pb_.mask.orig.clone();
		pb_.conf_total.orig = conf_total.clone();
	}

	// sData(sparse���̓o�^) (maskVec���L�[�ɔ���)
	sData_1frame sdata_;
	sdata_.downscale = 1;
	sdata_.t = frame_counter_++;// 0,1,2,...

	set_data(
		sdata_,
		src_depth.orig,
		src_guide_u8_vec.orig,
		pb_.src_depth_u8,
		pb_.conf_total.orig,
		pb_.mask.orig,
		0.0f
	);
	fbs_.pastdata_.push_back(sdata_);

	// �O����
	CopyLastResult();
	CopyPastInput();
	CreateImgPyrWithConfMask();
}

template<typename T>
inline T gauss(const int& i, const float& sigma) {
	auto sigma2 = static_cast<T>(2.0) * std::pow(sigma, static_cast<T>(2.0));
	auto r = std::pow(i, static_cast<T>(2.0));
	return static_cast<T>(std::exp(-r / sigma2));
}


void CAT::set_lutColorDiff(const int& C) {
	const int N_lut = 256 * C;
	const float sigma = 2.f * static_cast<float>(C); //2.0: conf_time_guide_sigma_color
	std::unique_ptr<float[]> lut_tmp(new float[N_lut]);
	{
		for (int i = 0; i < N_lut; ++i) {
			lut_tmp[i] = gauss<float>(i,sigma);
		}
	}
	lut_color_diff_ = std::move(lut_tmp);
}

void CAT::set_lutOptDiff() {
	const double sigma = 8.0;//conf_depth_diff_sigma_color
	const int N_lut = 256;
	std::unique_ptr<float[]> lut_tmp(new float[N_lut]);
	{
		for (int i = 0; i < N_lut; ++i) {
			lut_tmp[i] = gauss<float>(i, sigma);
		}
	}
	lut_opt_diff_ = std::move(lut_tmp);
}


// fixme maxRange ����
cv::Mat CAT::CalcConfFromDiff(
	const cv::InputArray src1_,
	const cv::InputArray src2_
)const {
	const int maxRange = 255;
	cv::Mat diff_flt = cv::abs(src2_.getMat() - src1_.getMat())-0.5f;
	cv::Mat diff,dst;
	diff_flt.convertTo(diff, CV_8UC1);
	cv::Mat lut(1, 256, CV_32F, lut_opt_diff_.get());
	cv::LUT(diff, lut, dst);
	return dst;
}

bool CAT::CopyPastInput(){
	if (is_wfgs_with_past_data_) {
		Update_sData_srcDepth(fbs_.pastdata_);
	}
	else if (is_update_sdata_conf) {
		Update_sData(fbs_.pastdata_); //sparsedata�̐M���x�̂ݗ��Ƃ�
	}
	return true;
}

void CAT::Update_srcDepth_withPastOptDepth(
	const float expected_rate
) {
	const int t_now = (int)pb_.guide.size()-1;
	const int t_pre = t_now-1;
	const int W = pb_.src_depth[t_now].orig.cols;
	const int H = pb_.src_depth[t_now].orig.rows;
	const int C = pb_.guide_u8[t_now].orig.channels();

	const uchar* p_guide_now = pb_.guide_u8[t_now].orig.ptr<uchar>(0);
	const uchar* p_guide_past = pb_.guide_u8[t_pre].orig.ptr<uchar>(0);

	float* const p_depth_now = pb_.src_depth[t_now].orig.ptr<float>(0);
	const float* p_depth_past = pb_.dst_depth[t_pre].orig.ptr<float>(0);

	float* const p_conf_total = pb_.conf_total.orig.ptr<float>(0);
	const float* p_conf_guideedge = pb_.conf_guideedge.orig.ptr<float>(0);

	const float weightTime = conf_time_attenuation_;
	for (int j = 0; j < H; ++j) {
		for (int i = 0; i < W; ++i) {
			const int index = W * j + i;
			const int p3 = index * C;
			int d3 = abs(p_guide_now[p3] - p_guide_past[p3]);

			for (int dd = 1; dd < C; ++dd) {
				d3 += abs(p_guide_now[p3 + dd] - p_guide_past[p3 + dd]);
			}
			float tConf = weightTime * lut_color_diff_[d3]* p_conf_guideedge[index];
			tConf = MAX(conf_time_min_val_, tConf);

			if (p_conf_total[index] < tConf) {
				p_conf_total[index] = tConf * expected_rate;
				p_depth_now[index] = p_depth_past[index];
			}
		}
	}
	pb_.dst_depth[t_now].orig = pb_.src_depth[t_now].orig.clone();
}

// sDataIMU�̃f�[�^����tar��srcDepth���X�V
// ���̍ہA�t���O�ɉ����āAsDataIMU�̐M���x���X�V
void CAT::Update_sData_srcDepth(PastData& sData) 
{
	const int sz_orig = 0;
	const int t_now = (int)pb_.guide.size() - 1;//�z�o�b�t�@�̍Ō�̃f�[�^�ɑ΂��ď���
	std::vector<uchar*> gData(pb_.buf_size);
	for (int t = 0; t < pb_.buf_size; ++t) {
		gData[t] = pb_.guide_u8[t].orig.ptr<uchar>(0);
	}
	const int W = pb_.src_depth[t_now].orig.cols;
	const int channels = pb_.guide_u8[t_now].orig.channels();


	float* const dData = pb_.src_depth[t_now].orig.ptr<float>(0);
	float* const ctData = pb_.conf_total.orig.ptr<float>(0);
	float* const cgeData = pb_.conf_guideedge.orig.ptr<float>(0);

	for (int t = 0; t < t_now; ++t) {
		const float weightTime = powf(conf_time_attenuation_, (float)abs(t_now - t));
		// 1����������sparse�f�[�^�̃T�C�Y
		for (int i = 0; i < sData[t].size; ++i) {
			const int p1 = W * sData[t].y[i] + sData[t].x[i];
			const int p3 = p1 * channels;
			int d3 = abs(gData[t_now][p3] - gData[t][p3]);
			for (int dd = 1; dd < channels; ++dd) {
				// channel = 0 
				d3 += abs(gData[t_now][p3 + dd] - gData[t][p3 + dd]);
			}
			const float tConf = 
				weightTime * lut_color_diff_[d3] * sData[t].c[i] * cgeData[p1];

			if (is_update_sdata_conf) {
				sData[t].c[i] = tConf;	// sDataIMU�̐M���x���X�V
			}

			if (ctData[p1] < tConf) {
				ctData[p1] = tConf;
				dData[p1] = sData[t].d[i];
			}
		}
	}
	pb_.dst_depth[t_now].orig = pb_.src_depth[t_now].orig.clone();
}

//���g�p
void CAT::Update_sData(	PastData& sData) {

	const int t_now = (int)pb_.guide.size() - 1;
	std::vector<uchar*> gData((int)pb_.buf_size);
	for (int t = 0; t < (int)pb_.buf_size; ++t) {
		gData[t] = pb_.guide_u8[t].orig.ptr<uchar>(0);
	}

	const int W = pb_.src_depth[t_now].orig.cols;
	const int C = pb_.guide_u8[t_now].orig.channels();

	for (int t = 0; t < t_now; ++t) {
		const float weightTime = powf(conf_time_attenuation_, (float)abs(t_now - t));

		const float* cgeData = pb_.conf_guideedge.orig.ptr<float>(0);
		for (int i = 0; i < sData[t].size; ++i) {
			const int p1 = W * sData[t].y[i] + sData[t].x[i];
			const int p3 = p1 * C;
			int d3 = abs(gData[t_now][p3] - gData[t][p3]);

			for (int dd = 1; dd < C; ++dd) {
				d3 += abs(gData[t_now][p3 + dd] - gData[t][p3 + dd]);
			}
			float tConf =
				weightTime * lut_color_diff_[d3] * sData[t].c[i] * cgeData[p1];
			tConf = MAX(conf_time_min_val_, tConf);
			// update
			sData[t].c[i] = tConf;
		}
	}
}

void CAT::CreateImgPyrWithConfMask() {
	const auto t_now = pb_.guide.size() - 1;
	const double QE = 1.0E-5;
	const int scaleX = pyr_scale_;
	const int scaleY = pyr_scale_;

	const int W_orig = pb_.src_depth[t_now].orig.cols;
	const int H_orig = pb_.src_depth[t_now].orig.rows;
	const int W_small = (int)(W_orig / scaleX + QE);
	const int H_small = (int)(H_orig / scaleY + QE);
	const auto sz_small = cv::Size(W_small, H_small);
	const float rW = (float)W_orig / W_small;
	const float rH = (float)H_orig / H_small;

	// initialize
	pb_.mask.small = cv::Mat::zeros(sz_small, pb_.mask.orig.type());
	pb_.src_depth[t_now].small = cv::Mat(sz_small, pb_.src_depth[t_now].orig.type(), unknownval_);	
	pb_.conf_total.small = cv::Mat::zeros(sz_small, pb_.conf_total.orig.type());

	const float* p_depth_orig = pb_.src_depth[t_now].orig.ptr<float>(0);
	const float* p_conf_orig = pb_.conf_total.orig.ptr<float>(0);
	const float* p_mask_orig = pb_.mask.orig.ptr<float>(0);

	float* const p_depth_small = pb_.src_depth[t_now].small.ptr<float>(0);
	float* const p_conf_small = pb_.conf_total.small.ptr<float>(0);
	float* const p_mask_small = pb_.mask.small.ptr<float>(0);

	//const int C = pb_.src_depth[t_now].small.channels();//1

	// depth��conf �̏d�ݕt���k��
#pragma omp parallel for
	for (int j = 0; j < H_small; ++j) {
		for (int i = 0; i < W_small; ++i) {
			const int ind_small = W_small * j + i;
			const int sx = static_cast<int>(rW * i);
			const int ex = static_cast<int>(rW * (i + 1));
			const int sy = static_cast<int>(rH * j);
			const int ey = static_cast<int>(rH * (j + 1));

			int count(0);
			double sum_depth(0.);
			double sum_conf(0.);
			for (int y = sy; y < ey; ++y) {
				for (int x = sx; x < ex; ++x) {
					const int index = W_orig * y + x;
					if (p_conf_orig[index] > 0.f) {
						const float tmp_conf = p_conf_orig[index] + 1.0E-5f;
						sum_conf += static_cast<double>(tmp_conf);
						sum_depth += tmp_conf * p_depth_orig[index];
						if (p_mask_orig[index] > 0.0f) {
							p_mask_small[ind_small] = 1.0f;
						}
						++count;
					}
				}
			}
			if (count > 0) {
				p_conf_small[ind_small] = (float)(sum_conf / count);
				p_depth_small[ind_small] = (float)(sum_depth / sum_conf);
			}
		}
	}

	{
		// conf update
		double minConf, maxConf, minConf2, maxConf2;
		cv::minMaxLoc(pb_.conf_total.orig, &minConf, &maxConf);
		cv::minMaxLoc(pb_.conf_total.small, &minConf2, &maxConf2);
		const float mindiff = static_cast<float>(minConf - minConf2);
		const float maxratio = static_cast<float>(maxConf / maxConf2);
		pb_.conf_total.small = (pb_.conf_total.small + mindiff) * maxratio;
	}
	pb_.dst_depth[t_now].small = pb_.src_depth[t_now].small.clone();

	cv::resize(
		pb_.conf_guideedge.orig,
		pb_.conf_guideedge.small,
		sz_small, 0, 0, cv::INTER_AREA);

	// guide
	cv::resize(
		pb_.guide[t_now].orig, 
		pb_.guide[t_now].small, 
		sz_small, 0, 0, cv::INTER_AREA);

	if (pb_.guide[t_now].small.depth() == CV_8U) {
		pb_.guide_u8[t_now].small = pb_.guide[t_now].small.clone();
	}
	else {
		const auto type = CV_MAKETYPE(CV_8U, pb_.guide[t_now].small.channels());
		pb_.guide[t_now].small.convertTo(pb_.guide_u8[t_now].small, type, 255);
	}

}
void CAT::pbSetCapacity() {
	pb_.src_depth.set_capacity(pb_.buf_capa);
	pb_.dst_depth.set_capacity(pb_.buf_capa);
	pb_.guide.set_capacity(pb_.buf_capa);
	pb_.guide_u8.set_capacity(pb_.buf_capa);
	fbs_.pastdata_.set_capacity(pb_.buf_capa);
}
void CAT::pbPopFront() {
	pb_.src_depth.pop_front();
	pb_.dst_depth.pop_front();
	pb_.guide.pop_front();
	pb_.guide_u8.pop_front();
	fbs_.pastdata_.pop_front();
	pb_.buf_size -= 1;
}
void CAT::pbClear() {
	pb_.src_depth.clear();
	pb_.dst_depth.clear();
	pb_.guide.clear();
	pb_.guide_u8.clear();
	fbs_.pastdata_.clear();
	pb_.buf_size = 0;
}

bool CAT::CopyLastResult() {
	// skip if previous frame is empty
	if (!is_wfgs_with_pastoutput_ || (pb_.guide.size()) < 2) {
		return false;
	}
	const float Npix = static_cast<float>(pb_.mask.orig.cols * pb_.mask.orig.rows);
	const float estExpectedRate = static_cast<float>(cv::countNonZero(pb_.mask.orig)) / Npix;
	Update_srcDepth_withPastOptDepth(estExpectedRate);
	return true;
}

// �M���x�ݒ� FBS�O��̒l���r
bool CAT::set_confidence(
	const cv::Mat& dst0,
	const cv::Mat& dst1,
	cv::Mat& conf_diffOpt
)const {
	const float scale = 255.0f / depth_range_max_;
	conf_diffOpt = CalcConfFromDiff(dst0 * scale, dst1 * scale);
	const int ite_erodation = 0;
	if (ite_erodation)
		cv::erode(conf_diffOpt, conf_diffOpt, cv::Mat(), cv::Point(-1, -1), ite_erodation);
	return true;
}

bool CAT::WFGS_pre(
	const cv::Mat& src,
	const cv::Mat& guide,
	cv::Mat& conf,
	cv::Mat& dst0,
	cv::Mat& conf_fillRate
)const {
	if (conf_fillRate.empty()) {
		conf_fillRate = cv::Mat::ones(src.size(), CV_32FC1);
	}

	auto fgs = cv::ximgproc::createFastGlobalSmootherFilter(
		guide,
		wfgs_pre_.lambda_,//1.0
		wfgs_pre_.sigma_color_,
		wfgs_pre_.lambda_attenuation_
	);
	cv::Mat tmp, conf_tmp2;
	fgs->filter(src.mul(conf), dst0);
	fgs->filter(conf, conf_fillRate);	
	cv::Mat div = cv::Mat::ones(conf.size(), CV_32FC1);
	conf_fillRate.copyTo(div, conf_fillRate > 0.f);
	conf_fillRate.setTo(0.0f, conf_fillRate <= 0.f);
	dst0 /= div;
	
	//wfgs_pre_.Calculate_weight(guide);//���O�v�Z
	//wfgs_pre_.Execute(src, conf, dst0, conf_fillRate);
	InpaintErrVal(dst0, unknownval_, 0.0f, depth_range_max_);

	// calc filling rate map
	if (!conf_fillRate.empty()) {
		const float WarrantyRate = warranty_rate_ / pyr_scale_;
		conf_fillRate *= WarrantyRate * WarrantyRate;
		conf_fillRate += conf_time_min_val_;//(1.0E-5f) FBS�̍œK������nan����̂��߁@<< ���v�m�F
		conf_fillRate.setTo(1.0f, conf_fillRate > 1.0f);
	}

	return true;
}


void CAT::Plarnar_WFGS_pre(
	const cv::Mat& src,
	const cv::Mat& guide,
	cv::Mat& conf,
	cv::Mat& dst0,
	cv::Mat& conf_fillrate
)const {
	//plarnar_filter
	double min_d, max_d;
	cv::minMaxLoc(src, &min_d, &max_d, 0, 0, src > DBL_EPSILON);

	cv::Mat dst_fgs;
	cv::Mat mask = cv::Mat::zeros(src.size(), CV_32FC1);
	mask.setTo(1, src > 0);

	auto fgs = cv::ximgproc::createFastGlobalSmootherFilter(
		guide,
		wfgs_post_.lambda_,
		wfgs_post_.sigma_color_,
		wfgs_post_.lambda_attenuation_
	);

	conf_fillrate = conf.clone();

	const double eps = 1e-3;
	planar_filter(src, mask, conf, eps, fgs, dst_fgs, dst0);

	// avoid invalid outputs
	cv::patchNaNs(dst0, -1);//NaN ��-1
	dst0.setTo(min_d, dst0 < min_d);
	dst0.setTo(max_d, dst0 > max_d);
}




bool CAT::TFBS(
	const cv::Mat& guide,
	const cv::Mat& dst0,
	const cv::Mat& conf_fillrate,
	cv::Mat& dst1,
	cv::Mat& conf_diffopt
){
	// init parameter
	fbs_.t_ = fbs_.pastdata_.back().t;
	fbs_.set_downscale(pyr_scale_);
	const cv::Mat conf = conf_fillrate * conf_fbs_at_fgs_;
	fbs_.init(guide);
	fbs_.filter(dst0, conf, dst1);
	
	FillErrorDepth(dst1, depth_range_max_);	//onoff
	set_confidence(dst0, dst1, conf_diffopt);
	return true;
}

void CAT::HFBS_main(
	const cv::Mat& guide,
	const cv::Mat& dst0,
	const cv::Mat& conf_fillrate,
	cv::Mat& dst1,
	cv::Mat& conf_diffopt
)const {
	// init parameter
	cv::Mat conf = conf_fillrate * conf_fbs_at_fgs_;
	conf.setTo(0, conf < 0.2);
	cv::Mat dst1_hfbs;
	cv::Mat mask = cv::Mat::zeros(conf.size(), conf.type());
	mask.setTo(1, conf > 0);
	HFBS hfbs;
	hfbs.init(guide, 4, 8.0);
	hfbs.filter(dst0.mul(mask), conf.mul(mask), dst1);
	cv::Mat dev;
	hfbs.filter(mask, conf.mul(mask), dev);
	dst1 /= dev;

	// avoid invalid outputs
	double min_d, max_d;
	cv::minMaxLoc(dst0, &min_d, &max_d, 0, 0, dst0 > 0);
	dst1.setTo(min_d, dst1 < min_d);
	dst1.setTo(max_d, dst1 > max_d);
	cv::patchNaNs(dst1, -1);//NaN ��-1

	FillErrorDepth(dst1, depth_range_max_);	//onoff
	set_confidence(dst0, dst1, conf_diffopt);
	conf_diffopt.setTo(0, conf_diffopt < 0.2);

}

void CAT::Plarnar_HFBS_main(
	const cv::Mat& guide,
	const cv::Mat& dst0,
	const cv::Mat& conf_fillrate,
	cv::Mat& dst1,
	cv::Mat& conf_diffopt
)const {
	// init parameter
	cv::Mat conf = conf_fillrate * conf_fbs_at_fgs_;
	conf.setTo(0, conf < 0.2);
	cv::Mat dst1_hfbs;
	cv::Mat mask= cv::Mat::zeros(conf.size(),conf.type());
	mask.setTo(1, conf > 0);
	HFBS hfbs;
	hfbs.init(guide, 4, 8.0);
	planar_filter(dst0, mask, conf, 1e-3, &hfbs, dst1_hfbs, dst1);

	// avoid invalid outputs
	double min_d, max_d;
	cv::minMaxLoc(dst0, &min_d, &max_d, 0, 0, dst0 > 0);
	dst1.setTo(min_d, dst1 < min_d);
	dst1.setTo(max_d, dst1 > max_d);
	cv::patchNaNs(dst1, -1);//NaN ��-1

	FillErrorDepth(dst1, depth_range_max_);	//onoff
	set_confidence(dst0, dst1, conf_diffopt);
	conf_diffopt.setTo(0, conf_diffopt < 0.2);
}

void CAT::Plarnar_tFBS_main(
	const cv::Mat& guide,
	const cv::Mat& dst0,
	const cv::Mat& conf_fillrate,
	cv::Mat& dst1,
	cv::Mat& conf_diffopt
) {
	// init parameter
	fbs_.t_ = fbs_.pastdata_.back().t;
	fbs_.set_downscale(pyr_scale_);
	const cv::Mat conf = conf_fillrate * conf_fbs_at_fgs_;
	fbs_.init(guide);

	cv::Mat dst1_fbs, mask;

	planar_filter(dst0, mask, conf, 1e-3, &fbs_, dst1_fbs, dst1);
	double min_d, max_d;
	cv::minMaxLoc(dst0, &min_d, &max_d, 0, 0, dst0 > DBL_EPSILON);
	// avoid invalid outputs
	cv::patchNaNs(dst1, -1);//NaN ��-1
	dst1.setTo(min_d, dst1 < min_d);
	dst1.setTo(max_d, dst1 > max_d);

	FillErrorDepth(dst1, depth_range_max_);	//onoff
	set_confidence(dst0, dst1, conf_diffopt);

}

bool CAT::WFGS_post(
	const cv::Mat& dst1,
	const cv::Mat& conf_diffopt,
	cv::Mat& dst2,
	cv::Mat& conf_depthedge,
	cv::Mat& conf_result
)const {
	// conf depth edge
	if (is_depth_edge_) {
		const float scale_tmp = depth_edge_amp_ / depth_range_max_;
		conf_depthedge = getDoGmagnitude(dst1, depth_edge_blur_sigma_, scale_tmp);
	}

	// �g��
	cv::Mat dst1_up, conf_depthedge_up, conf_diffopt_up;
	Resize(
		pb_.raw_depth.size(),
		dst1,
		conf_depthedge,
		conf_diffopt,
		is_depth_edge_,
		dst1_up,
		conf_depthedge_up,
		conf_diffopt_up
	);
	conf_depthedge_up.setTo(0, conf_depthedge_up < 0.5);
	cv::Mat dst_blend, conf_blend;
	Blend(
		pb_.raw_depth,
		dst1_up,
		pb_.raw_conf,
		conf_diffopt_up,
		conf_blend,
		dst_blend,
		conf_fgs_at_raw_,
		conf_fgs_at_fbs_
	);
	cv::Mat conf = is_depth_edge_ ? conf_blend.mul(conf_depthedge_up) : conf_blend;
	conf.setTo(0, conf < 0.1);
	// run weighted FGS

	auto fgs = cv::ximgproc::createFastGlobalSmootherFilter(
		pb_.guide_u8.back().orig,
		wfgs_post_.lambda_,
		wfgs_post_.sigma_color_,
		wfgs_post_.lambda_attenuation_
	);
	cv::Mat conf_tmp = cv::Mat::ones(conf.size(),conf.type());
	fgs->filter(dst_blend.mul(conf), dst2);
	fgs->filter(conf, conf_result);
	conf_result.copyTo(conf_tmp, conf_result > 0.f);
	//conf_tmp.setTo(0.0f, conf_tmp <= 0.f);
	dst2 /= conf_tmp;
	return true;
}

bool CAT::Run(cv::Mat& dst_depth, cv::Mat& dst_conf){
	const cv::Mat src(pb_.dst_depth.back().small);
	const cv::Mat guide(pb_.guide_u8.back().small);
	cv::Mat conf(pb_.conf_total.small);
	conf.setTo(0, conf<0.1f);

	// pre processinng
	cv::Mat dst0, conf_fillrate;
	switch (pre_method){
	case ORI_WFGS: WFGS_pre(src, guide, conf, dst0, conf_fillrate); break;
	case PLA_WFGS: Plarnar_WFGS_pre(src, guide, conf, dst0, conf_fillrate); break;
	default: puts("pre: not supported"); break;
	}

	// main
	cv::Mat dst1, conf_diffopt;

	switch (main_method) {
	case ORI_TFBS: TFBS(guide, dst0, conf_fillrate, dst1, conf_diffopt); break;
	case ORI_HFBS: HFBS_main(guide, dst0, conf_fillrate, dst1, conf_diffopt); break;
	case PLA_HFBS: Plarnar_HFBS_main(guide, dst0, conf_fillrate, dst1, conf_diffopt); break;
	case PLA_TFBS: Plarnar_tFBS_main(guide, dst0, conf_fillrate, dst1, conf_diffopt); break;
	default: puts("main: not supported"); break;
	}
	pb_.dst_depth.back().small = dst1;// �o�^

	// post processing
	// ROI�Ή�


	if (roi_.empty()) {
		cv::Mat conf_depthedge;
		WFGS_post(dst1, conf_diffopt, dst_depth, conf_depthedge, dst_conf);
		pb_.dst_depth.back().orig = dst_depth.clone();// �o�^
	}
	else {
		if (dst_depth.empty()) dst_depth.create(sz_, dst1.type());
		if (dst_conf.empty()) dst_conf.create(sz_, dst1.type());

		cv::Mat dst2, conf_depthedge, conf;
		WFGS_post(dst1, conf_diffopt, dst2, conf_depthedge, conf);
		pb_.dst_depth.back().orig = dst_depth.clone();// �o�^

		cv::Mat dst2_roi(dst_depth, roi_);
		cv::Mat conf_roi(dst_conf, roi_);
		dst2.copyTo(dst2_roi);
		conf.copyTo(conf_roi);
	}
	return true;
}
