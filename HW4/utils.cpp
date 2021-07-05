//DIP_HW03
//20161482 ¹ÚÁØ¿ë 

#include <iostream>
#include "opencv2/opencv.hpp"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
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
		hist[i] = hist[i] / (src.rows * src.cols);
	}
	return hist;
}

Mat addSaltPepper(Mat src, float ps, float pp, float salt_val, float pepp_val)
{
	Mat dst(src.size(), src.type());
	dst = src;
	dst.convertTo(dst, CV_32F); 
	int val_ps = ((src.rows * src.cols) * ps);
	int val_pp = ((src.rows * src.cols) * pp);
	srand((int)time(NULL));

	int	dst_row;
	int dst_col;
	for (int k = 0; k < val_ps; k++) {
		dst_row = rand() % src.rows;
		dst_col = rand() % src.cols;
		dst.at<float>(dst_row, dst_col) = salt_val;
	}

	for (int k = 0; k < val_pp; k++) {
		dst_row = rand() % src.rows;
		dst_col = rand() % src.cols;
		dst.at<float>(dst_row, dst_col) = pepp_val;
	}

	dst.convertTo(dst, CV_8U);

	std::ofstream saltPepperHist;
	float* dst_hist = getHist(dst);

	saltPepperHist.open("saltPepperHist.csv");
	for (int i = 0; i < 256; i++) {
		saltPepperHist << i << "," << dst_hist[i] << "\n";
	}
	saltPepperHist.close();

	return dst;
}

Mat contraharmonic(Mat src, Size kernel_size, float Q)
{
	Mat padded(src.rows + kernel_size.height - 1, src.cols + kernel_size.width - 1, CV_32F);
	
	float k_row1 = (kernel_size.height - 1) / 2;
	float k_col1 = (kernel_size.width - 1) / 2;
	for (int x_prime = 0; x_prime < padded.rows; x_prime++) {
		for (int y_prime = 0; y_prime < padded.cols; y_prime++) {
			if (x_prime < (int)k_row1 || y_prime < (int)k_col1 || x_prime > padded.rows - 1 - ((int)k_row1) || y_prime > padded.cols - 1 - ((int)k_col1)) {
				padded.at<float>(x_prime, y_prime) = 0;
			}
			else {
				padded.at<float>(x_prime, y_prime) = (float)src.at<uchar>(x_prime - ((int)k_row1), y_prime - ((int)k_col1));
			}
		}
	}
	Mat dst(src.size(), CV_32F);
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			float num = 0;
			float den = 0;
			for (int x_kernel = 0; x_kernel < kernel_size.height; x_kernel++) {
				for (int y_kernel = 0; y_kernel < kernel_size.width; y_kernel++) {
					num += pow((int)padded.at<float>(x + x_kernel, y + y_kernel), Q + 1.0);
					den += pow((int)padded.at<float>(x + x_kernel, y + y_kernel), Q);
				}
			}
			dst.at<float>(x, y) = num / den;
		}
	}
	dst.convertTo(dst, CV_8U);

	return dst;
}

Mat median(Mat src, Size kernel_size)
{
	Mat padded(src.rows + kernel_size.height - 1, src.cols + kernel_size.width - 1, src.type());

	float k_row1 = (kernel_size.height - 1) / 2;
	float k_col1 = (kernel_size.width - 1) / 2;
	for (int x_prime = 0; x_prime < padded.rows; x_prime++) {
		for (int y_prime = 0; y_prime < padded.cols; y_prime++) {
			if (x_prime < (int)k_row1 || y_prime < (int)k_col1 || x_prime > padded.rows - 1 - ((int)k_row1) || y_prime > padded.cols - 1 - ((int)k_col1)) {
				padded.at<uchar>(x_prime, y_prime) = 0;
			}
			else {
				padded.at<uchar>(x_prime, y_prime) = src.at<uchar>(x_prime - ((int)k_row1), y_prime - ((int)k_col1));
			}
		}
	}

	Mat dst(src.size(), src.type());
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			int* median_array = new int[kernel_size.height * kernel_size.width];
			int i = 0;
			for (int x_kernel = 0; x_kernel < kernel_size.height; x_kernel++) {
				for (int y_kernel = 0; y_kernel < kernel_size.width; y_kernel++) {
					median_array[i] = padded.at<uchar>(x + x_kernel, y + y_kernel);
					i = i + 1;
				}
			}
			sort(median_array, median_array + kernel_size.height * kernel_size.width);
			dst.at<uchar>(x, y) = median_array[(kernel_size.height * kernel_size.width) / 2];
			delete[] median_array;
		}
	}
	return dst;
}

Mat adaptiveMedian(Mat src, Size kernel_size)
{
	Size kernel = Size(1, 1);
	Mat dst(src.size(), src.type());
	Mat padded(src.rows + kernel_size.height - 1, src.cols + kernel_size.width - 1, src.type());
	
	float k_row1 = (kernel_size.height - 1) / 2;
	float k_col1 = (kernel_size.width - 1) / 2;
	for (int x_prime = 0; x_prime < padded.rows; x_prime++) {
		for (int y_prime = 0; y_prime < padded.cols; y_prime++) {
			if (x_prime < (int)k_row1 || y_prime < (int)k_col1 || x_prime > padded.rows - 1 - ((int)k_row1) || y_prime > padded.cols - 1 - ((int)k_col1)) {
				padded.at<uchar>(x_prime, y_prime) = 0;
			}
			else {
				padded.at<uchar>(x_prime, y_prime) = src.at<uchar>(x_prime - ((int)k_row1), y_prime - ((int)k_col1));
			}
		}
	}
for_loop:
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			unsigned char* median_array = (unsigned char*)malloc(kernel.height * kernel.width * sizeof(unsigned char));
			int i = 0;
			for (int x_kernel = 0; x_kernel < kernel.height; x_kernel++) {
				for (int y_kernel = 0; y_kernel < kernel.width; y_kernel++) {
					median_array[i] = padded.at<uchar>(x + (kernel_size.height - kernel.height) / 2 + x_kernel, y + (kernel_size.width - kernel.width) / 2 + y_kernel);
					i = i + 1;
				}
			}
			sort(median_array, median_array + kernel.height * kernel.width);
			int kernel_min = median_array[0];
			int kernel_max = median_array[kernel.height * kernel.width - 1];
			int kernel_med = median_array[(kernel.height * kernel.width) / 2];

			float k_row2 = kernel_size.height / 2;
			float k_col2 = kernel_size.width / 2;
			if (kernel_min < kernel_med && kernel_med < kernel_max) {
				if (kernel_min < padded.at<uchar>(x + k_row2, y + k_col2) && padded.at<uchar>(x + k_row2, y + k_col2) < kernel_max) {
					dst.at<uchar>(x, y) = padded.at<uchar>(x + k_row2, y + k_col2);
				}
				else {
					dst.at<uchar>(x, y) = kernel_med;
				}
			}
			else {
				if (kernel.height < kernel_size.height) {
					kernel += Size(2, 2);
					goto for_loop;
				}
				else {
					dst.at<uchar>(x, y) = kernel_med;
				}
			}
			free(median_array);
		}
	}
	return dst;
}

Mat add_turbulence(Mat src, float k)
{
	Mat src_fft;
	src_fft = fftshift2d(fft2d(src));

	Mat tmp[2];
	Mat src_magnitude;
	Mat src_phase;
	split(src_fft, tmp);
	magnitude(tmp[0], tmp[1], src_magnitude);
	phase(tmp[0], tmp[1], src_phase);

	Mat H(src.size(), CV_32FC2);
	for (int u = 0; u < src.rows; u++) {
		for (int v = 0; v < src.cols; v++) {
			H.at<Vec2f>(u, v)[0] = exp(-k * pow((u-floor(src.rows/2)) * (u - floor(src.rows / 2)) + (v - floor(src.cols / 2)) * (v - floor(src.cols / 2)), 5.0 / 6));
			H.at<Vec2f>(u, v)[1] = 0;
		}
	}

	Mat dst(src.size(), CV_32FC2);
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			dst.at<Vec2f>(x, y)[0] = src_fft.at<Vec2f>(x, y)[0] * H.at<Vec2f>(x, y)[0] - src_fft.at<Vec2f>(x, y)[1] * H.at<Vec2f>(x, y)[1];
			dst.at<Vec2f>(x, y)[1] = src_fft.at<Vec2f>(x, y)[1] * H.at<Vec2f>(x, y)[0] + src_fft.at<Vec2f>(x, y)[0] * H.at<Vec2f>(x, y)[1];
		}
	}

	dst = ifft2d(fftshift2d(dst));
	dst.convertTo(dst, CV_8U);
	return dst;
}

Mat wienerFilter(Mat src, float K)
{
	Mat src_fft;
	src_fft = fftshift2d(fft2d(src));

	Mat tmp[2];
	Mat src_magnitude;
	Mat src_phase;
	split(src_fft, tmp);
	magnitude(tmp[0], tmp[1], src_magnitude);
	phase(tmp[0], tmp[1], src_phase);

	Mat H(src.size(), CV_32FC2);
	for (int u = 0; u < src.rows; u++) {
		for (int v = 0; v < src.cols; v++) {
			H.at<Vec2f>(u, v)[0] = exp(-K * pow((u - floor(src.rows / 2)) * (u - floor(src.rows / 2)) + (v - floor(src.cols / 2)) * (v - floor(src.cols / 2)), 5.0 / 6));
			//H.at<Vec2f>(u, v)[0] = exp(-0.0025 * pow((u - floor(src.rows / 2)) * (u - floor(src.rows / 2)) + (v - floor(src.cols / 2)) * (v - floor(src.cols / 2)), 5.0 / 6));
			//H.at<Vec2f>(u, v)[0] = exp(-0.001 * pow((u - floor(src.rows / 2)) * (u - floor(src.rows / 2)) + (v - floor(src.cols / 2)) * (v - floor(src.cols / 2)), 5.0 / 6));
			//H.at<Vec2f>(u, v)[0] = exp(-0.00025 * pow((u - floor(src.rows / 2)) * (u - floor(src.rows / 2)) + (v - floor(src.cols / 2)) * (v - floor(src.cols / 2)), 5.0 / 6));
			H.at<Vec2f>(u, v)[1] = 0;
		}
	}

	Mat H_magnitude;
	Mat H_phase;
	split(H, tmp);
	magnitude(tmp[0], tmp[1], H_magnitude);
	phase(tmp[0], tmp[1], H_phase);

	Mat dst(src.size(), CV_32FC2);
	Mat dst_magnitude(src.size(), CV_32F);
	Mat dst_phase(src.size(), CV_32F);
	Mat dst_real(src.size(), CV_32F);
	Mat dst_imaginary(src.size(), CV_32F);

	for (int u = 0; u < src.rows; u++) {
		for (int v = 0; v < src.cols; v++) {
			dst_magnitude.at<float>(u, v) = (1 / H_magnitude.at<float>(u, v)) *
				(H_magnitude.at<float>(u, v) * H_magnitude.at<float>(u, v)) / (H_magnitude.at<float>(u, v) * H_magnitude.at<float>(u, v) + K) *
				src_magnitude.at<float>(u, v);
			dst_phase.at<float>(u, v) = src_phase.at<float>(u, v);
		}
	}

	polarToCart(dst_magnitude, dst_phase, dst_real, dst_imaginary);

	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			dst.at<Vec2f>(x, y)[0] = dst_real.at<float>(x, y);
			dst.at<Vec2f>(x, y)[1] = dst_imaginary.at<float>(x, y);
		}
	}

	dst = ifft2d(fftshift2d(dst));
	dst.convertTo(dst, CV_8U);

	return dst;
}
