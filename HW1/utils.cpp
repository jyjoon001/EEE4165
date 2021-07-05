//DIP_HW01
//201161482 ���ؿ� 

#include "opencv2/opencv.hpp"
#include <iostream>
#include <limits>
int INF = std::numeric_limits<int>::max();

using namespace std;
using namespace cv;

float interp2D(Mat src, float x, float y, string interp_type) {
	if (interp_type == "nearest") {
		if (x - (int)x < 0.5 && y - (int)y < 0.5) {
			return src.at<uchar>(y, x);
		}
		else if (x - (int)x < 0.5 && y - (int)y >= 0.5) {
			return src.at<uchar>(y + 1, x);
		}
		else if (x - (int)x >= 0.5 && y - (int)y < 0.5) {
			return src.at<uchar>(y, x + 1);
		}
		else if (x - (int)x >= 0.5 && y - (int)y >= 0.5) {
			return src.at<uchar>(y + 1, x + 1);
		}
	}

	else if (interp_type == "bilinear") {
		if (x - int(x) == 0 && y - int(y) == 0)
			return src.at<uchar>(y, x);
		else {
			return src.at<uchar>(int(y), int(x)) * (int(x) - x + 1) * (int(y) - y + 1) + src.at<uchar>(int(y) + 1, int(x)) * (int(x) - x + 1) * (y - (int)y)
				+ src.at<uchar>(int(y), int(x) + 1) * (x - (int)x) * (int(y) - y + 1) + +src.at<uchar>(int(y) + 1, int(x) + 1) * (x - (int)x) * (y - (int)y);
		}
	}
}

Mat warpAffine(Mat src, Mat affmat, string interp_type) {
	Mat original = Mat::zeros(Size(2, 2), CV_32F);
	Mat invorg = Mat::zeros(Size(2, 2), CV_32F);
	Mat implement = Mat::zeros(Size(1, 3), CV_32F);
	float x, y, x_min=INF, y_min=INF, x_max = -INF, y_max = -INF;
	float* orig = (float*)original.data;
	orig[0] = affmat.at<float>(0, 0);	orig[1] = affmat.at<float>(1, 0);
	orig[2] = affmat.at<float>(0, 1);	orig[3] = affmat.at<float>(1, 1);

	for (int x_prime = 0; x_prime < src.rows; x_prime++) {
		for (int y_prime = 0; y_prime < src.cols; y_prime++) {
			float* imp_ele = (float*)implement.data;
			imp_ele[0] = y_prime * affmat.at<float>(0, 0) + x_prime * affmat.at<float>(0, 1);
			imp_ele[1] = y_prime * affmat.at<float>(1, 0) + x_prime * affmat.at<float>(1, 1);
			imp_ele[2] = 1;
			if (imp_ele[0] < x_min)
				x_min = imp_ele[0];
			if (imp_ele[0] > x_max)
				x_max = imp_ele[0];
			if (imp_ele[1] < y_min)
				y_min = imp_ele[1];
			if (imp_ele[1] > y_max)
				y_max = imp_ele[1];
		}
	}
	Mat dst(y_max - y_min + affmat.at<float>(1, 2),
		x_max - x_min + affmat.at<float>(0, 2),
		src.type());
	for (int x_prime = 0; x_prime < dst.rows; x_prime++) {
		for (int y_prime = 0; y_prime < dst.cols; y_prime++) {
			invorg = original.inv();
			x = (y_prime + x_min - affmat.at<float>(0, 2)) * invorg.at<float>(1, 1) + (x_prime + y_min - affmat.at<float>(1, 2)) * invorg.at<float>(1, 0);
			y = (y_prime + x_min - affmat.at<float>(0, 2)) * invorg.at<float>(0, 1) + (x_prime + y_min - affmat.at<float>(1, 2)) * invorg.at<float>(0, 0);
			if (int(y) < 0 || int(x) < 0 || (int(x)+1) > src.cols - 1 || (int(y)+1) > src.rows - 1)
				continue;
			dst.at<uchar>(x_prime, y_prime) = interp2D(src, x,y, interp_type);
		}
	}
	return dst;
};


Mat rotate(Mat src, float radian, string interp_type) {	
	Mat rotation_matrix = Mat::zeros(Size(3, 3), CV_32F);
	float* rot = (float*) rotation_matrix.data;
	rot[0] = (float)cos(radian);	rot[1] = (float)-sin(radian);	rot[2] = 0.0;
	rot[3] = (float)sin(radian);	rot[4] = (float)cos(radian);	rot[5] = 0.0;
	rot[6] = 0.0;					rot[7] = 0.0;					rot[8] = 1.0;

	Mat dst_rotation = warpAffine(src, rotation_matrix, interp_type);
	return dst_rotation;
}

Mat translation(Mat src, float x, float y, string interp_type) {
	Mat translation_matrix = Mat::zeros(Size(3, 3), CV_32F);
	float* trans = (float*)translation_matrix.data;
	trans[0] = 1.0;	trans[1] = 0.0;	trans[2] = x;
	trans[3] = 0.0;	trans[4] = 1.0;	trans[5] = y;
	trans[6] = 0.0;	trans[7] = 0.0;	trans[8] = 1.0;

	Mat dst_translation = warpAffine(src, translation_matrix, interp_type);
	return dst_translation;
}
