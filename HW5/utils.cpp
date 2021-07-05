
#include <iostream>
#include "opencv2/opencv.hpp"
#include "utils.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;

Mat cvtBGR2HSI(Mat src) {
	Mat dst(src.size(), src.type());
		
	Mat B(src.size(), src.type());
	Mat G(src.size(), src.type());
	Mat R(src.size(), src.type());

	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			B.at<float>(x, y) = src.at<Vec3f>(x, y)[0];
			G.at<float>(x, y) = src.at<Vec3f>(x, y)[1];
			R.at<float>(x, y) = src.at<Vec3f>(x, y)[2];

			if ((R.at<float>(x, y) == G.at<float>(x, y) && G.at<float>(x, y) == B.at<float>(x, y))) {
				dst.at<Vec3f>(x, y)[0] = 0;
			}
			else {
				float red = R.at<float>(x, y);
				float green = G.at<float>(x, y);
				float blue = B.at<float>(x, y);
				float denom = sqrt((red - green) * (red - green) + (red - blue) * (green - blue));
				float numer = (red - green + red - blue) / 2;
				float theta = acos(numer / denom);
				theta = 180 * theta / M_PI;
				if (B.at<float>(x, y) <= G.at<float>(x, y)) {
					dst.at<Vec3f>(x, y)[0] = theta / 360;
				}
				else {
					dst.at<Vec3f>(x, y)[0] = (360 - theta) / 360;
				}
			}
			dst.at<Vec3f>(x, y)[1] = 1 - 3 * min(min(R.at<float>(x, y), G.at<float>(x, y)), B.at<float>(x, y)) / (R.at<float>(x, y) + G.at<float>(x, y) + B.at<float>(x, y));
			dst.at<Vec3f>(x, y)[2] = (R.at<float>(x, y) + G.at<float>(x, y) + B.at<float>(x, y)) / 3;
		}
	}

	return dst;
}

Mat cvtHSI2BGR(Mat src) {
	Mat dst(src.size(), src.type());

	Mat H(src.size(), src.type());
	Mat S(src.size(), src.type());
	Mat I(src.size(), src.type());
	float B;
	float G;
	float R;

	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			H.at<float>(x, y) = src.at<Vec3f>(x, y)[0];
			S.at<float>(x, y) = src.at<Vec3f>(x, y)[1];
			I.at<float>(x, y) = src.at<Vec3f>(x, y)[2];

			float hue = H.at<float>(x, y);
			float sat = S.at<float>(x, y);
			float inten = I.at<float>(x, y);

			if (0 <= hue && 180 * hue / M_PI <= 120) {
				R = inten * (1 + sat * cos(hue) / cos(M_PI / 3 - hue));
				B = inten * (1 - sat);
				G = 3 * inten - (R + B);
			}
			else if (120 < 180 * hue / M_PI && 180 * hue / M_PI <= 240) {
				G = inten * (1 + sat * cos(hue) / cos(M_PI / 3 - hue));
				R = inten * (1 - sat);
				B = 3 * inten - (R + B);
			}
			else if (240 < 180 * hue / M_PI && 180 * hue / M_PI <= 360) {
				B = inten * (1 + sat * cos(hue) / cos(M_PI / 3 - hue));
				G = inten * (1 - sat);
				R = 3 * inten - (R + B);
			}
			dst.at<Vec3f>(x, y)[0] = B;
			dst.at<Vec3f>(x, y)[1] = G;
			dst.at<Vec3f>(x, y)[2] = R;
		}
	}

	return dst;
}

Mat sobelFilter(Mat src, float direction) {
	Mat sobel_matrix = Mat::zeros(Size(3, 3), CV_32F);
	float* sobel = (float*)sobel_matrix.data;
	if (direction == 0) {	
		sobel[0] = -1;	sobel[1] = -2;	sobel[2] = -1;
		sobel[3] = 0;	sobel[4] = 0;	sobel[5] = 0;
		sobel[6] = 1;	sobel[7] = 2;	sobel[8] = 1;
	}
	else if (direction == 1) {	
		sobel[0] = -1;	sobel[1] = 0;	sobel[2] = 1;
		sobel[3] = -2;	sobel[4] = 0;	sobel[5] = 2;
		sobel[6] = -1;	sobel[7] = 0;	sobel[8] = 1;
	}

	Mat dst(src.size(), src.type());
	filter2D(src, dst, -1, sobel_matrix);
	return dst;
}
