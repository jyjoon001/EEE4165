#ifndef __UTILS_H_
#define __UTILS_H_

#include <iostream>
#include "opencv2/opencv.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

// utils.cpp
Mat cvtBGR2HSI(Mat src);
Mat cvtHSI2BGR(Mat src);
Mat sobelFilter(Mat src, float direction);
#endif