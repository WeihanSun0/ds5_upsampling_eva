#pragma once
#ifndef  _PLARNAR
#define _PLARNAR
#include <opencv2/opencv.hpp>


template <class T1>
auto try_filter_2arg(T1 fanctor, const cv::Mat& src, cv::Mat& dst)
-> decltype(fanctor->filter(src, dst), bool()) {
    fanctor->filter(src, dst);
    return true;
}

auto try_filter_2arg(...) -> bool { return false;}

template <class T1>
auto try_filter_3arg(T1 fanctor, const cv::Mat& src, const cv::Mat& conf, cv::Mat& dst)
-> decltype(fanctor->filter(src, conf, dst), bool()) {
    fanctor->filter(src, conf, dst);
    return true;
}


auto try_filter_3arg(...) -> bool {
    puts("error: fanctor must have filter(src,dst) or filter(src,conf,dst) ");
    CV_Assert(false);
    return false;
}




// guide の有無によらない
template <typename T>
cv::Mat FilterWithGuide(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& conf, T f) {
    cv::Mat dst, mask_result;
    if (conf.empty()) {
        if (mask.empty()) {
            puts("error: mask and conf are empty");
            CV_Assert(false);
        }
        if (try_filter_2arg(f, src.mul(mask), dst)) {
            try_filter_2arg(f, mask, mask_result);
        }
        else {
            puts("error: this fanctor must have filter(src,conf,dst) ");
            CV_Assert(false);
        }
    }
    else {
        if (mask.empty()) {
            try_filter_3arg(f, src, conf, dst);
            return dst;
        }
        else {
            if (try_filter_2arg(f, src.mul(mask), dst)) {
                try_filter_2arg(f, mask, mask_result);
            }
            else {
                if (try_filter_3arg(f, src.mul(mask), conf, dst)) {
                    try_filter_3arg(f, mask, conf, mask_result);
                }
            }
        }
    }
    dst /= mask_result;
    return dst;
}



cv::Mat solve_image_ldl3(
    const cv::Mat& A11, const cv::Mat& A12, const cv::Mat& A13,
    const cv::Mat& A22, const cv::Mat& A23, const cv::Mat& A33,
    const cv::Mat& b1, const cv::Mat& b2, const cv::Mat& b3
) {
    // 画素並列で処理が可能
    // An unrolled LDL solver for a 3x3 symmetric linear system.
    const cv::Mat d1 = A11.clone();
    const cv::Mat L12 = A12 / d1;

    const cv::Mat d2 = A22 - L12.mul(A12);
    const cv::Mat L13 = A13 / d1;
    const cv::Mat L23 = (A23 - L13.mul(A12)) / d2;

    const cv::Mat d3 = A33 - L13.mul(A13) - L23.mul(L23).mul(d2);
    const cv::Mat y1 = b1;
    const cv::Mat y2 = b2 - L12.mul(y1);
    const cv::Mat y3 = b3 - L13.mul(y1) - L23.mul(y2);
    const cv::Mat x3 = y3 / d3;//Zz
    //x2 = y2 / d2 - L23.mul(x3);// Zy:不要
    //x1 = y1 / d1 - L12.mul(x2) - L13.mul(x3);//Zx:不要  
    return x3;
}



template <typename T>
void planar_filter(
    const cv::Mat& src, // z
    const cv::Mat& mask,
    const cv::Mat& conf,
    const double& eps,
    T filter, //filter(.)
    cv::Mat& dst,
    cv::Mat& dst_plarnar
) {
    CV_Assert((eps < 1e19) && (eps > 1e-19));

    const double eps_square = eps * eps;//plarnar 度合いの調整 大きいほど平面度合い低い
    const int H = src.rows;
    const int W = src.cols;
    const double H_dbl(H);
    const double W_dbl(W);

    const double mean_wh = ((W_dbl - 1.) + (H_dbl - 1)) / 2.;
    const double xy_scale = 2. / mean_wh;

    const double margin_w = (W_dbl - 1.) / 2.;
    const double margin_h = (H_dbl - 1.) / 2.;

    // 前処理 
    cv::Mat x = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat y = cv::Mat::zeros(src.size(), CV_32F);

    // Scaling the x, y coords to be in ~[0, 1]
    x.forEach<float>([margin_h, xy_scale](float& p, const int position[2]) -> void {
        float val = static_cast<float>(position[0]);
        p = (val - margin_h) * xy_scale;
        });
    y.forEach<float>([margin_w, xy_scale](float& p, const int position[2]) -> void {
        float val = static_cast<float>(position[1]);
        p = (val - margin_w) * xy_scale;
        });
    const cv::Mat xx = x.mul(x);
    const cv::Mat xy = x.mul(y);
    const cv::Mat yy = y.mul(y);
    const cv::Mat I = cv::Mat::ones(H, W, CV_32F);

    cv::Mat F1, Fx, Fy, Fz, Fxx, Fxy, Fxz, Fyy, Fyz;

    F1 = FilterWithGuide(I, mask, conf, filter);
    // 下記SIMD化
    Fx = FilterWithGuide(x, mask, conf, filter);
    Fy = FilterWithGuide(y, mask, conf, filter);
    Fz = FilterWithGuide(src, mask, conf, filter);
    Fxx = FilterWithGuide(xx, mask, conf, filter);
    Fxy = FilterWithGuide(xy, mask, conf, filter);
    Fxz = FilterWithGuide(x.mul(src), mask, conf, filter);
    Fyy = FilterWithGuide(yy, mask, conf, filter);
    Fyz = FilterWithGuide(y.mul(src), mask, conf, filter);

    const cv::Mat A11 = F1.mul(xx) - 2.f * x.mul(Fx) + Fxx + eps_square;
    const cv::Mat A22 = F1.mul(yy) - 2.f * y.mul(Fy) + Fyy + eps_square;
    const cv::Mat A12 = F1.mul(xy) - x.mul(Fy) - y.mul(Fx) + Fxy;
    const cv::Mat A13 = F1.mul(x) - Fx;
    const cv::Mat A23 = F1.mul(y) - Fy;
    const cv::Mat A33 = F1;// +eps_square;<- bug in paper's math
    const cv::Mat b1 = Fz.mul(x) - Fxz;
    const cv::Mat b2 = Fz.mul(y) - Fyz;
    const cv::Mat b3 = Fz;

    dst = Fz.clone();
    dst_plarnar = solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3);
};
#endif // ! _PLARNAR