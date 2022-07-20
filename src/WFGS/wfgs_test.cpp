#include <cstdio>
#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include "z2color.h" // draw color
#include "viewer.h" // visualize 3D
#include "data_loader.h" // use sim data
#include "mcv_evaluation.h" // error eva
#include <chrono>

// #define DUMP_IMAGES

// include SAT
#include "openSAT/CAT.h"
using namespace std;


int main(int argc, char * argv[])
{
    if (argc != 7) {
        cout << "USAGE: app guide_image flood_dmap gt_dmap camera_k" << endl;
        exit(0);
    }
    const string fn_guide = string(argv[1]);
    const string fn_sparse = string(argv[2]);
    const string fn_gtDepth = string(argv[3]);
    const string fn_camK = string(argv[4]);
    const string fn_gtNormal = string(argv[5]);
    const string fn_output_folder = string(argv[6]);

    // read data
    cv::Mat imgGray = cv::imread(fn_guide, cv::IMREAD_GRAYSCALE);
    cv::Mat imgFlood = cv::imread(fn_sparse, -1);
    cv::Mat imgGtDepth = read_D(fn_gtDepth);
    cv::Mat matCamK = read_K(fn_camK);
    cv::Mat imgGtNormal = read_N(fn_gtNormal);
    cv::Mat imgMask;
    imgMask = cv::Mat::zeros(imgFlood.size(), imgFlood.type());
    imgMask.setTo(1.0, imgFlood != 0.0);

    // evaluaton
    Evaluation eva;
    eva.set_gt(imgGtDepth);
    eva.set_K(matCamK);
    eva.set_gt_normal(imgGtNormal);
    eva.set_edge(imgGray, 2);
    // upsampling 
    int count = 0; // frame count
    int best_frame = 5; // frame num for best performance 
    CAT dc;
    dc.PrepareFilters(1);
    dc.set_depth_range_max(40000.0);
    cv::Rect roi; // process region, full resolution for only definition
    cv::Mat dense, conf;  // result
    // time 
    std::chrono::system_clock::time_point t_start, t_end;
    float duration;
    //output results
    fstream fs;
    fs.open(fn_output_folder + "/result.txt", ios::out);
    
    while(count <= best_frame) { // use the 5th result for best performance
        count++;
        // run wfgs
        t_start = std::chrono::system_clock::now();
        dc.Charge(imgFlood, imgGray, imgMask, roi, cv::noArray());
        dc.Run(dense, conf);
        t_end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
#if 0
        double minV, maxV;
        cv::minMaxLoc(dense, &minV, &maxV);
        cv::Mat imgShow = z2colormap(dense, minV, maxV);
        cv::imshow("dense", imgShow);
        cv::waitKey(0);
#endif
        if (count == best_frame) {
            // eva results
            eva.set_est(dense);
            eva.exec();
            // save results
            fs << "TIME[ms]:" << duration << endl;
            eva.fs_output(fs);
#ifdef DUMP_IMAGES
            eva.dump(fn_output_folder);
#endif
        }
    }
    fs.close();
    exit(0);
}