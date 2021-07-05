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
void swapPhase(Mat src1, Mat src2, Mat& dst1, Mat& dst2);
Mat freqFilter(Mat src, Mat kernel);
Mat freqBoxFilter(Mat src, Size kernel_size);
Mat freqGaussFilter(Mat src, Size kernel_size);
Mat NotchFilter(Mat src);
#endif