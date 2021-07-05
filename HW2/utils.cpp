//DIP HW2
//20161482 박준용

#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

float* getHist(Mat src) {
	float* hist = (float*)calloc(256, sizeof(hist));
	for (int y = 0; y < src.cols; y++) {
		for (int x = 0; x < src.rows; x++) {
			hist[src.at<uchar>(x, y)]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		hist[i] = hist[i]/(src.cols * src.rows);
	}
	return hist;
}

Mat gammaTransform(Mat src, float gamma) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8U);
	for (int y = 0; y < dst.cols; y++) {
		for (int x = 0; x < dst.rows; x++) {
			dst.at<uchar>(x, y) = (int) (pow((float)src.at<uchar>(x, y) / 255, gamma) * 255);
		}
	}

	std::ofstream gammaTransform;
	float* src_hist = getHist(src);
	float* dst_hist = getHist(dst);

	gammaTransform.open("gammaTransform.csv");
	for (int i = 0; i < 256; i++) {
		gammaTransform << i << "," << src_hist[i] << ",,";
		gammaTransform << i << "," << dst_hist[i] << "\n";
	}
	gammaTransform.close();
	return dst;
}

Mat histEqualization(Mat src) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8U);
	float* LUT = (float*)calloc(256, sizeof(LUT));
	float cdf = NULL;

	for (int i = 0; i < 256; i++) {
		cdf = cdf + getHist(src)[i];
		LUT[i] = cdf * 255;
	}
	for (int y = 0; y < src.cols; y++) {
		for (int x = 0; x < src.rows; x++) {
			dst.at<uchar>(x, y) = round(LUT[src.at<uchar>(x, y)]);
		}
	}
	std::ofstream histEqualization;
	float* src_hist = getHist(src);
	float* dst_hist = getHist(dst);

	histEqualization.open("histEqualization.csv");
	for (int i = 0; i < 256; i++) {
		histEqualization << i << "," << src_hist[i] << ",,";
		histEqualization << i << "," << dst_hist[i] << "\n";
	}
	histEqualization.close();
	return dst;
}

Mat boxFilter(Size size) {
	Mat box = Mat::zeros(size.height, size.width, CV_32F);
	for (int y = 0; y < box.cols; y++) {
		for (int x = 0; x < box.rows; x++) {
			box.at<float>(x, y) = 1 / (float)(size.height * size.width);
		}
	}
	return box;
}

Mat imFilter(Mat src, Mat kernel, string pad_type) {
	Mat padded = Mat::zeros(src.rows + kernel.rows - 1, src.cols + kernel.cols - 1, CV_32F);
	int colsDiv = (kernel.cols - 1) / 2;
	int rowsDiv = (kernel.rows - 1) / 2;

	if (pad_type == "zero") {
		for (int y = 0; y < src.cols; y++) {
			for (int x = 0; x < src.rows; x++) {
				padded.at<float>(x + rowsDiv, y + colsDiv) = (float)src.at<uchar>(x, y);
			}
		}
	}
	else if (pad_type == "mirror") {
		for (int x = rowsDiv; x < padded.rows - rowsDiv; x++) {
			for (int y = colsDiv; y < padded.cols - colsDiv; y++) {
				if (padded.rows - rowsDiv <= x) {
					padded.at<float>(x, y) = padded.at<float>(2 * (padded.rows - 1 - rowsDiv - x, y));
				}
				else {
					padded.at<float>(x, y) = (float)src.at<uchar>(abs(x - rowsDiv), y - colsDiv);
				}
			}
			for (int y = 0; y < colsDiv; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, kernel.cols - 1 - y);
			}
			for (int y = padded.cols - colsDiv; y < padded.cols; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, 2 * (padded.cols - 1 - colsDiv) - y);
			}
		}
		for (int x = 0; x < rowsDiv; x++) {
			for (int y = colsDiv; y < padded.cols - colsDiv; y++) {
				if (padded.rows - rowsDiv <= x) {
					padded.at<float>(x, y) = padded.at<float>(2 * (padded.rows - 1 - rowsDiv - x, y));
				}
				else {
					padded.at<float>(x, y) = (float)src.at<uchar>(abs(x - rowsDiv), y - colsDiv);
				}
			}
			for (int y = 0; y < colsDiv; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, kernel.cols - 1 - y);
			}
			for (int y = padded.cols - colsDiv; y < padded.cols; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, 2 * (padded.cols - 1 - colsDiv) - y);
			}
		}
		for (int x = padded.rows - rowsDiv; x < padded.rows; x++) {
			for (int y = colsDiv; y < padded.cols - colsDiv; y++) {
				if (padded.rows - rowsDiv <= x) {
					padded.at<float>(x, y) = padded.at<float>(2 * (padded.rows - 1 - rowsDiv - x, y));
				}
				else {
					padded.at<float>(x, y) = (float)src.at<uchar>(abs(x - rowsDiv), y - colsDiv);
				}
			}
			for (int y = 0; y < colsDiv; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, kernel.cols - 1 - y);
			}
			for (int y = padded.cols - colsDiv; y < padded.cols; y++) {
				padded.at<float>(x, y) = padded.at<float>(x, 2 * (padded.cols - 1 - colsDiv) - y);
			}
		}
	}
	Mat dst(src.rows, src.cols, CV_32F);
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			float conv = 0;
			for (int x_kernel = 0; x_kernel < kernel.rows; x_kernel++) {
				for (int y_kernel = 0; y_kernel < kernel.cols; y_kernel++) {
					conv = conv + padded.at<float>(x + x_kernel, y + y_kernel) * kernel.at<float>(kernel.rows - 1 - x_kernel, kernel.cols - 1 - y_kernel);
				}
			}
			dst.at<float>(x, y) = conv / (kernel.rows * kernel.cols);
		}
	}
	return dst;
}
