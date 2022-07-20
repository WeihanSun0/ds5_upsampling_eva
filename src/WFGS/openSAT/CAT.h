#pragma once
#define CV_SIMD128
// #include "openSAT/WFGS.h"
// #include "openSAT/sTFBS.h"
// #include "openSAT/DepthEdge.h"
// #include "openSAT/HFBS.h"
#include "WFGS.h"
#include "sTFBS.h"
#include "DepthEdge.h"
#include "HFBS.h"

using PastData = boost::circular_buffer<sData_1frame>;
struct MatLayer {
	cv::Mat orig;
	cv::Mat small;
};

enum FILTERING_METHODS : int {
	ORI_FGS       = 1,     //1
	ORI_WFGS      = 1 << 1,//2
	ORI_FBS       = 1 << 2,//4
	ORI_TFBS      = 1 << 3,//8
	ORI_HFBS      = 1 << 4,//16
	ORI_THFBS     = 1 << 5,//32
	// plarnar			   //
	PLA_FGS   = 1 << 6,	   //64
	PLA_WFGS  = 1 << 7,	   //128
	PLA_FBS   = 1 << 8,	   //256
	PLA_TFBS  = 1 << 9,	   //512
	PLA_HFBS  = 1 << 10,   //1024
	PLA_THFBS = 1 << 11,   //2048
};


class CAT {

public:
	CAT(const int& frameSize_ = 5) {
		pb_.buf_capa = frameSize_;
		pbSetCapacity();
		//p_tfbs_ = &fbs_;
	};
	virtual ~CAT() { pbClear(); };

	int pre_method = FILTERING_METHODS::ORI_WFGS;
	int main_method = FILTERING_METHODS::ORI_TFBS;

	// data
	struct PyrBuf {
		int buf_size = 0;
		int buf_capa = 5;
		const int pyr_layers = 2;
		boost::circular_buffer<MatLayer> src_depth;	// Depth *
		boost::circular_buffer<MatLayer> dst_depth;	// Depth *
		boost::circular_buffer<MatLayer> guide;		// RGB *
		boost::circular_buffer<MatLayer> guide_u8;	// RGB Uchar *
		//PastData pastdata;	// Depth, Guide(b,g,r), coord(x,y), ...
		cv::Mat raw_depth;
		cv::Mat raw_conf;
		cv::Mat src_depth_u8;
		MatLayer mask;
		MatLayer conf_guideedge;
		MatLayer conf_total;
	};
	PyrBuf pb_;
	// flag
	const bool is_guide_edge_          = true;
	const bool is_wfgs_with_pastoutput_= true; //�ߋ��œK�����ʂ̗��p
	const bool is_wfgs_with_past_data_ = true; //�ߋ��X�p�[�X�_�̗��p
	const bool is_update_sdata_conf    = false;//true �Ńn���O
	const bool is_depth_edge_          = true; //false�Ńn���O-->�C��
	const bool is_imu_correction_      = true; 

	bool PrepareFilters(const int& channel);

	void Charge(
		const cv::Mat& src_depth_, 
		const cv::Mat& src_guide_, 
		const cv::Mat& mask_, 
		cv::Rect& rect, 
		const cv::InputArray conf_
	);

	bool Run(cv::Mat& dst_depth, cv::Mat& dst_conf);

	bool Reset() {
		pbClear();
		return false;
	}

	// setter
	void set_depth_range_max(const float& in) { depth_range_max_ = in;}
	void set_wfgs_pre_param(const double& lambda_) {
		wfgs_pre_.lambda_ = lambda_;//1,4,8
	};
	void set_wfgs_post_param(const double& lambda_) {
		wfgs_post_.lambda_ = lambda_;//4,8,16
	};
	void set_tfbs_sigma_spatial(const double& in) { fbs_.set_sigma_spatial(in); }//8,16,32
	void set_tfbs_lambda(const double& in) { fbs_.set_lambda(in); }//1,8,32



private:
	cv::Mat conf_;
	// ROI
	cv::Size sz_;
	cv::Rect roi_;
	// filter
	WFGS wfgs_pre_;
	WFGS wfgs_post_;
	FBS fbs_;
	// general
	long long frame_counter_ = 0;             //FBS�p�̃J�E���^ ����؂����Ƃ��̃P�A�K�v
	float depth_range_max_ = 1.0;             //depth�̍ő�l�������
	const int pyr_scale_ = 4;                 //FBS����downscale_�ƍ��킹��
	const float unknownval_ = -1.f;		      
	const float warranty_rate_ = 9.0f;	      
	const float float_min_val_ = 1.0E-6f;     // conf�̉����l�B�����l���傫�Ȓl������ 
	const double conv_range_max_ = 10.0;      //guideU8�p
	// conf
	const double guide_edge_blur_sigma_ = 3.0;// blur
	const float guide_edge_amp_ = 2.0f;		  // amplitude
	const float conf_time_attenuation_ = 0.5f;//weightTime�ɑ��
	const float conf_time_min_val_ = 1.0E-5f;
	// guide edge
	const double depth_edge_blur_sigma_ = 0.f;// blur   (0.0 / dc.pyrScale)�ĂȂ��Ă���;
	const float depth_edge_amp_     = 1.5f;   // amplitude
	const float conf_fbs_at_fgs_ = 1.0f;      // = confVal_FGS_atFBS;			
	const float conf_fgs_at_raw_ = 1.0f;      // = 1.0f;
	const float conf_fgs_at_fbs_ = 1.0f;      // = 4.0f / (16.0f * 16.0f);

	std::unique_ptr<float[]> lut_color_diff_;
	std::unique_ptr<float[]> lut_opt_diff_;

	// Charge
	bool CopyLastResult();
	bool CopyPastInput(void);
	void CreateImgPyrWithConfMask();

	// data 
	void pbSetCapacity();
	void pbPopFront();
	void pbClear();
	void Discharge() { pbPopFront(); }

	//LUT
	void set_lutColorDiff(const int& C);
	void set_lutOptDiff();

	void Reinit(const int& frameSize_) {
		pb_.buf_capa = frameSize_;
		pbSetCapacity();
	}

	cv::Mat CalcConfFromDiff(
		const cv::InputArray srcDepth_, 
		const cv::InputArray srcGuide_
	)const;

	bool set_confidence(
		const cv::Mat& dst0,
		const cv::Mat& dst1,
		cv::Mat& conf_diffOpt)const;


	bool WFGS_pre(
		const cv::Mat& src,
		const cv::Mat& guide,
		cv::Mat& conf,
		cv::Mat& dst0,
		cv::Mat& conf_fillRate
	)const;

	void Plarnar_WFGS_pre(
		const cv::Mat& src,
		const cv::Mat& guide,
		cv::Mat& conf,
		cv::Mat& dst0,
		cv::Mat& conf_fillrate
	)const;


	bool TFBS(
		const cv::Mat& guide,
		const cv::Mat& dst0,
		const cv::Mat& conf_fillRate,
		cv::Mat& dst1,
		cv::Mat& conf_diffOpt
	);

	void HFBS_main(
		const cv::Mat& guide,
		const cv::Mat& dst0, 
		const cv::Mat& conf_fillrate,
		cv::Mat& dst1,
		cv::Mat& conf_diffopt
	)const;

	void Plarnar_HFBS_main(
		const cv::Mat& guide, 
		const cv::Mat& dst0, 
		const cv::Mat& conf_fillrate, 
		cv::Mat& dst1,
		cv::Mat& conf_diffopt
	)const;

	void Plarnar_tFBS_main(
		const cv::Mat& guide,
		const cv::Mat& dst0,
		const cv::Mat& conf_fillrate,
		cv::Mat& dst1,
		cv::Mat& conf_diffopt
	);


	bool WFGS_post(
		const cv::Mat& dst1,
		const cv::Mat& conf_diffopt,
		cv::Mat& dst2,
		cv::Mat& conf_depthedge,
		cv::Mat& conf_result
	)const;

	void Update_sData(PastData& sDataIMU);
	void Update_sData_srcDepth(PastData& sDataIMU);
	void Update_srcDepth_withPastOptDepth(const float expected_rate = 1.0f);

};