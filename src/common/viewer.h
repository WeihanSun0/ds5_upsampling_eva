#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

class myViz
{
public:
	myViz() :myWindow("test") {
		T_view.at<float>(2, 3) = -0.5;  // Z���W�̐ݒ�
		myWindow.setViewerPose(cv::Affine3f(T_view));
	};
	~myViz() {};
	cv::viz::Viz3d myWindow;

	void set(cv::Mat& pc, const cv::Mat& color, int ptsize = -1) {
		wClouds.push_back(cv::viz::WCloud(pc, color));
		ptsizes_.push_back(ptsize > 0 ? ptsize : ptsize_);
	};
	void set(cv::Mat& pc, cv::viz::Color color, int ptsize = -1) {
		wClouds.push_back(cv::viz::WCloud(pc, color));
		ptsizes_.push_back(ptsize > 0 ? ptsize : ptsize_);
	};
	void set(const std::string& fn, int ptsize = -1) {
		cv::Mat pc, pc_color;
		pc = cv::viz::readCloud(fn, pc_color);
		wClouds.push_back(cv::viz::WCloud(pc, pc_color));
		ptsizes_.push_back(ptsize > 0 ? ptsize : ptsize_);
	};

	void show(bool spinonce=false) {
		for (int i = 0; i < wClouds.size(); ++i) {
			myWindow.showWidget(std::to_string(i), wClouds[i]);
			myWindow.setRenderingProperty(std::to_string(i), cv::viz::POINT_SIZE, ptsizes_[i]);
		}
		for (int i = 0; i < arrows.size(); ++i) {
			myWindow.showWidget("arrow_" + std::to_string(i), arrows[i]);
		}

		myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(0.25));
		if (spinonce) {
			myWindow.spinOnce(1,true);//������ʍX�V
			wClouds.clear();
			arrows.clear();
		}
		else {
			myWindow.spin();
		}
	}

	void set_normal(const cv::Mat& pc, cv::Mat& normal, int step = 20, cv::viz::Color color = cv::viz::Color::amethyst()) {
		for (int y = 0; y < pc.rows; y += step) {
			for (int x = 0; x < pc.cols; x += step) {
				cv::Vec3d start = pc.at<cv::Vec3d>(y, x);
				cv::Vec3d end = start + normal.at<cv::Vec3d>(y, x) * 0.1;
				arrows.push_back(cv::viz::WArrow(start, end, 0.01f, color));
			}
		}
	}

	void set_image(const cv::Mat& image) {
		cv::Rect rect(0, 0, 320, 240);
		auto img = cv::viz::WImageOverlay::WImageOverlay(image, rect);
		myWindow.showWidget("img", img);
	}
	void set_image2(const cv::Mat& image) {
		cv::Rect rect(0, 240, 320, 240);
		auto img = cv::viz::WImageOverlay::WImageOverlay(image, rect);
		myWindow.showWidget("img2", img);
	}

	void set_text(std::string& text, const cv::Point& pos = cv::Point(330, 30)) {
		// pos �̌��_�͍���
		cv::viz::WText viztext(text, pos);
		myWindow.showWidget("text", viztext);
	};

	void set_text2(std::string& text, const cv::Point& pos = cv::Point(30, 250)) {
		// pos �̌��_�͍���
		cv::viz::WText viztext(text, pos);
		myWindow.showWidget("text_err", viztext);
	};

	void dumpImage(const cv::String& fn) {
		cv::imwrite(fn, myWindow.getScreenshot());
	};


private:
	std::vector<int> ptsizes_;
	int ptsize_ = 1;
	std::vector <cv::viz::WCloud> wClouds;
	std::vector<cv::viz::WArrow> arrows;
	cv::Mat T_view = cv::Mat::eye(4, 4, CV_32FC1);// �J�����̎p����ݒ�
};