#ifndef __UTILS_H_
#define __UTILS_H_

#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


// dft.cpp
Mat fft2d(Mat src, int flags = DFT_COMPLEX_OUTPUT);
void fft2d(Mat src, Mat& dst, int flags = DFT_COMPLEX_OUTPUT);
Mat ifft2d(Mat src, int flags = DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
void ifft2d(Mat src, Mat& dst, int flags = DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);

Mat fftshift2d(Mat src);
void fftshift2d(Mat src, Mat& dst);
Mat ifftshift2d(Mat src);
void ifftshift2d(Mat src, Mat& dst);

// utils.cpp
Mat addSaltPepper(Mat src, float ps=0.0, float pp=0.0, float salt_val=255, float pepp_val=0);
Mat contraharmonic(Mat src, Size kernel_size, float Q);
Mat median(Mat src, Size kernel_size);
Mat adaptiveMedian(Mat src, Size kernel_size);
Mat add_turbulence(Mat src, float k);
Mat wienerFilter(Mat src, float K);
#endif