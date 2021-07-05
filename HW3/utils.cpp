//DIP_HW03
//201161482 ¹ÚÁØ¿ë 

#include <iostream>
#include "opencv2/opencv.hpp"
#include "utils.h"
#include <math.h>

using namespace std;
using namespace cv;
float Sizes = 1000;
int INF = std::numeric_limits<int>::max();

Mat piecewise(Mat src1, Mat src2) {
	Mat result = Mat::zeros(src1.rows, src1.cols, CV_32F);

	for (int row = 0; row < src1.rows; row++) {
		for (int col = 0; col < src1.cols; col++) {
			result.at<float>(row, col) = src1.at<float>(row, col) * src2.at<float>(row, col);
		}
	}
	return result;
}

Mat merge(Mat planes[]) {
	Mat result = Mat::zeros(planes[0].rows, planes[0].cols, CV_32FC2);
	for (int row = 0; row < planes[0].rows; row++) {
		for (int col = 0; col < planes[0].cols; col++) {
			result.at<Vec2f>(row, col)[0] = planes[0].at<float>(row, col);
			result.at<Vec2f>(row, col)[1] = planes[1].at<float>(row, col);
		}
	}
	return result;
}

Mat freqFilter(Mat src, Mat kernel) {
	Mat src_padded(src.rows * 2, src.cols * 2, src.type());
	Mat dst(src.rows, src.cols, CV_32F, src.type());
	Mat dst_padded(src.rows * 2, src.cols * 2, CV_32FC2);

	if (src.size() != kernel.size()) {
		Mat kernel_padded(dst_padded.size(), CV_32F, src.type());
		for (int x = 0; x < kernel.rows; x++) {
			for (int y = 0; y < kernel.cols; y++) {
				kernel_padded.at<float>(x, y) = kernel.at<float>(x, y);
			}
		}
	}

	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			src_padded.at<uchar>(x, y) = src.at<uchar>(x, y);
		}
	}

	src_padded = fft2d(src_padded);
	src_padded = fftshift2d(src_padded);
	kernel = fftshift2d(kernel);

	Mat kernel_mag;
	Mat kernel_phase;
	Mat tmp[2];
	split(kernel, tmp);
	magnitude(tmp[0], tmp[1], kernel_mag);
	phase(tmp[0], tmp[1], kernel_phase);

	Mat src_padded_mag;
	Mat src_padded_phase;
	split(src_padded, tmp);
	magnitude(tmp[0], tmp[1], src_padded_mag);
	phase(tmp[0], tmp[1], src_padded_phase);

	Mat dst_padded_mag(dst_padded.size(), CV_32F);
	Mat dst_padded_phase(dst_padded.size(), CV_32F);
	Mat dst_padded_re(dst_padded.size(), CV_32F);
	Mat dst_padded_im(dst_padded.size(), CV_32F);

	for (int x = 0; x < dst_padded.rows; x++) {
		for (int y = 0; y < dst_padded.cols; y++) {
			dst_padded_mag.at<float>(x, y) = src_padded_mag.at<float>(x, y) * kernel_mag.at<float>(x, y);
			dst_padded_phase.at<float>(x, y) = src_padded_phase.at<float>(x, y);
		}
	}

	polarToCart(dst_padded_mag, dst_padded_phase, dst_padded_re, dst_padded_im);

	for (int x = 0; x < dst_padded.rows; x++) {
		for (int y = 0; y < dst_padded.cols; y++) {
			dst_padded.at<Vec2f>(x, y)[0] = dst_padded_re.at<float>(x, y);
			dst_padded.at<Vec2f>(x, y)[1] = dst_padded_im.at<float>(x, y);
		}
	}

	dst_padded = fftshift2d(dst_padded);
	dst_padded = ifft2d(dst_padded);

	for (int x = 0; x < dst.rows; x++) {
		for (int y = 0; y < dst.cols; y++) {
			dst.at<float>(x, y) = dst_padded.at<float>(x, y);
		}
	}

	return dst;
}

Mat freqBoxFilter(Mat src, Size kernel_size) {

	Mat src_padded(src.rows * 2, src.cols * 2, src.type());
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			src_padded.at<uchar>(x, y) = src.at<uchar>(x, y);
		}
	}

	Mat src1;
	src1 = fft2d(src_padded);

	fftshift2d(src1, src1);
	Mat temp[2];
	split(src1, temp);

	Mat src_mag;
	Mat src_phase;
	magnitude(temp[0], temp[1], src_mag);
	phase(temp[0], temp[1], src_phase);
	log(1 + src_mag, src_mag);
	normalize(src_mag, src_mag, 0, 1, NORM_MINMAX);
	log(1 + src_phase, src_phase);
	normalize(src_phase, src_phase, 0, 1, NORM_MINMAX);

	imshow("Linear_chirp_mag", src_mag);
	imshow("Linear_chirp_phase", src_phase);
	imwrite("Linear_chirp_mag.tif", src_mag);
	imwrite("Linear_chirp_phase.tif", src_phase);

	Mat padded(src.cols * 2, src.rows * 2, CV_32FC2);

	Mat kernel(kernel_size, CV_32F);
	for (int x = 0; x < kernel.rows; x++) {
		for (int y = 0; y < kernel.cols; y++) {
			kernel.at<float>(x, y) = 1 / pow(kernel.rows, 2);
		}
	}

	Mat result(padded.size(), CV_32F, src.type());
	for (int x = 0; x < kernel.rows; x++) {
		for (int y = 0; y < kernel.cols; y++) {
			result.at<float>(x, y) = kernel.at<float>(x, y);
		}
	}

	Mat dst;
	dst = fft2d(result);
	padded = freqFilter(src, dst);

	fftshift2d(dst, dst);
	Mat tmp[2];
	split(dst, tmp);

	Mat dst_mag;
	Mat dst_phase;
	magnitude(tmp[0], tmp[1], dst_mag);
	phase(tmp[0], tmp[1], dst_phase);
	log(1 + dst_mag, dst_mag);
	normalize(dst_mag, dst_mag, 0, 1, NORM_MINMAX);
	log(1 + dst_phase, dst_phase);
	normalize(dst_phase, dst_phase, 0, 1, NORM_MINMAX);

	if (kernel.rows == 13) {
		imshow("Box 13X13 kernel", dst_mag);
		imshow("Box 13X13 phase", dst_phase);
		imwrite("Box 13X13 kernel.tif", dst_mag);
		imwrite("Box 13X13 phase.tif", dst_phase);
	}if (kernel.rows == 21) {
		imshow("Box 21X21 kernel", dst_mag);
		imshow("Box 21X21 phase", dst_phase);
		imwrite("Box 21X21 kernel.tif", dst_mag);
		imwrite("Box 21X21 phase.tif", dst_phase);
	}
	return padded;
}


Mat freqGaussFilter(Mat src, Size kernel_size) {
	Mat padded(src.cols * 2, src.rows * 2, CV_32FC2, src.type());
	Mat kernel(kernel_size, CV_32F);
	float sigma = NULL;
	if (kernel.rows == 31) {
		sigma = 5;
	}if (kernel.rows == 55) {
		sigma = 9;
	}

	float k_row1 = (kernel_size.height - 1) / 2;
	float k_row2 = kernel_size.height / 2;
	float k_col1 = (kernel_size.width - 1) / 2;
	float k_col2 = kernel_size.width / 2;

	float N = NULL;
	for (int i = -k_row1; i <= k_row2; i++) {
		for (int j = -k_col1; j <= k_col2; j++) {
			N += exp(-((i * i + j * j) / (2 * pow(sigma, 2))));
		}
	}

	for (int x = -k_row1; x <= k_row2; x++) {
		for (int y = -k_col1; y <= k_col2; y++) {
			kernel.at<float>(x + k_row1, y + k_col1) = exp(-(float)(x * x + y * y) / (2 * pow(sigma, 2))) / N;
		}
	}

	Mat result(padded.size(), CV_32F, src.type());
	for (int x = 0; x < kernel.rows; x++) {
		for (int y = 0; y < kernel.cols; y++) {
			result.at<float>(x, y) = kernel.at<float>(x, y);
		}
	}

	Mat dst;
	dst = fft2d(result);
	padded = freqFilter(src, dst);

	fftshift2d(dst, dst);
	Mat tmp[2];
	split(dst, tmp);

	Mat dst_mag;
	Mat dst_phase;
	magnitude(tmp[0], tmp[1], dst_mag);
	phase(tmp[0], tmp[1], dst_phase);
	log(1 + dst_mag, dst_mag);
	normalize(dst_mag, dst_mag, 0, 1, NORM_MINMAX);
	log(1 + dst_phase, dst_phase);
	normalize(dst_phase, dst_phase, 0, 1, NORM_MINMAX);

	if (kernel.rows == 31) {
		imshow("Gauss 31X31 kernel", dst_mag);
		imshow("Gauss 31X31 phase", dst_phase);
		imwrite("Gauss 31X31 kernel.tif", dst_mag);
		imwrite("Gauss 31X31 phase.tif", dst_phase);
	}if (kernel.rows == 55) {
		imshow("Gauss 55X55 kernel", dst_mag);
		imshow("Gauss 55X55 phase", dst_phase);
		imwrite("Gauss 55X55 kernel.tif", dst_mag);
		imwrite("Gauss 55X55 phase.tif", dst_phase);
	}
	return padded;
}

Mat freqFilter_notchkernel(Mat src, Mat kernel) {
	Mat padded = Mat::zeros(Sizes, Sizes, CV_32F);

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			padded.at<float>(row, col) = src.at<uchar>(row, col) / 255.0;
		}
	}

	Mat plane[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat src_padded = merge(plane);
	src_padded = fft2d(src_padded);

	Mat src_plane[2] = {Mat::zeros(src_padded.size(), CV_32F), Mat::zeros(src_padded.size(), CV_32F)};
	Mat src_mag;
	Mat src_phase;

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			src_plane[0].at<float>(row, col) = src_padded.at<Vec2f>(row, col)[0];
			src_plane[1].at<float>(row, col) = src_padded.at<Vec2f>(row, col)[1];
		}
	}
	magnitude(src_plane[0], src_plane[1], src_mag);
	src_mag = fftshift2d(src_mag);

	phase(src_plane[0], src_plane[1], src_phase);
	src_phase = fftshift2d(src_phase);

	Mat merged = piecewise(src_mag, kernel);
	merged = ifftshift2d(merged);
	src_phase = ifftshift2d(src_phase);

	Mat dft_re = Mat::zeros(src_padded.size(), CV_32F);
	Mat dft_im = Mat::zeros(src_padded.size(), CV_32F);
	for (int row = 0; row < src_padded.rows; row++) {
		for (int col = 0; col < src_padded.cols; col++) {
			dft_re.at<float>(row, col) = merged.at<float>(row, col) * cos(src_phase.at<float>(row, col));
			dft_im.at<float>(row, col) = merged.at<float>(row, col) * sin(src_phase.at<float>(row, col));
		}
	}
	Mat src_merged[] = { dft_re, dft_im };
	Mat tmp = merge(src_merged);
	tmp = ifft2d(tmp, 3);

	Mat dst = Mat::zeros(src.rows, src.cols, CV_32F);
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			dst.at<float>(row, col) = tmp.at<Vec2f>(row, col)[0];
		}
	}
	return dst;
}

Mat NotchFilter(Mat src) {

	Mat Dpk[1000];
	Mat Dmk[1000];
	Mat Hpk[1000];
	Mat Hmk[1000];
	Mat fliter = Mat::zeros(Sizes, Sizes, CV_32F);

	double N = 0.25;
	int vk = 30;

	for (int i = 0; i < src.rows; i++) {
		Dpk[i] = Mat::zeros(Sizes, Sizes, CV_32F);;
		Dmk[i] = Mat::zeros(Sizes, Sizes, CV_32F);;
		Hpk[i] = Mat::zeros(Sizes, Sizes, CV_32F);;
		Hmk[i] = Mat::zeros(Sizes, Sizes, CV_32F);;
	}

	for (int i = 0; i < 1; i++) {
		int uk = 0;
		if (i == 0) {
			uk = 500;
		}

		for (int row = 0; row < src.rows; row++) {
			for (int col = 0; col < src.cols; col++) {
				int y1 = row - round(Dpk[i].rows / 2) - (uk - 1);
				int x1 = col - round(Dpk[i].cols / 2) - vk;

				int y2 = row - round(Dpk[i].rows / 2) + uk;
				int x2 = col - round(Dpk[i].cols / 2) + vk;

				float denom_1 = pow(sqrt(pow(-uk, 2) + pow(-vk, 2)) / sqrt(y1 * y1 + x1 * x1), N);
				float denom_2 = pow(sqrt(pow(+uk, 2) + pow(+vk, 2)) / sqrt(y2 * y2 + x2 * x2), N);

				if (y1 == 0 && x1 == 0)
					Hpk[i].at<float>(row, col) = 0;
				else
					Hpk[0].at<float>(row, col) = 1 / (1 + denom_1);

				if (y2 == 0 && x2 == 0)
					Hmk[i].at<float>(row, col) = 0;
				else
					Hmk[0].at<float>(row, col) = 1 / (1 + denom_2);
			}
		}
		if (i == 0) {
			fliter = piecewise(Hpk[0], Hmk[0]);
		}
		if (i >= 1) {
			fliter = piecewise(fliter, (piecewise(Hpk[i], Hmk[i])));
		}
	}

	Mat tmp[4];
	tmp[0] = Mat::zeros(Sizes, Sizes, CV_32F);
	tmp[1] = Mat::zeros(Sizes, Sizes, CV_32F);
	tmp[2] = Mat::zeros(Sizes, Sizes, CV_32F);
	tmp[3] = Mat::zeros(Sizes, Sizes, CV_32F);


	for (int i = 0; i < 5; i++) {
		for (int row = 950; row <= 999; row++) {
			for (int col = 515; col <= 500 + vk - 1; col++) {
				tmp[0].at<float>(row, col) = fliter.at<float>(row, col + 1);
			}
			for (int col = 500 + vk + 1; col <= 550; col++) {
				tmp[1].at<float>(row, col) = fliter.at<float>(row, col - 1);
			}
			for (int col = 515; col <= 500 + vk - 1; col++) {
				fliter.at<float>(row, col) = tmp[0].at<float>(row, col);
			}
			for (int col = 500 + vk + 1; col <= 550; col++) {
				fliter.at<float>(row, col) = tmp[1].at<float>(row, col);
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		for (int row = 0; row < 50; row++) {
			for (int col = 450; col <= 500 - vk - 1; col++) {
				tmp[2].at<float>(row, col) = fliter.at<float>(row, col + 1);
			}
			for (int col = 500 - vk + 1; col <= 485; col++) {
				tmp[3].at<float>(row, col) = fliter.at<float>(row, col - 1);
			}
			for (int col = 450; col <= 500 - vk - 1; col++) {
				fliter.at<float>(row, col) = tmp[2].at<float>(row, col);
			}
			for (int col = 500 - vk + 1; col <= 485; col++) {
				fliter.at<float>(row, col) = tmp[3].at<float>(row, col);
			}
		}
	}

	for (int row = 0; row < src.rows; row++) {
		for (int col = 450; col <= 485; col++) {
			fliter.at<float>(row, col) = fliter.at<float>(0, col);
		}
		for (int col = 515; col <= 550; col++) {
			fliter.at<float>(row, col) = fliter.at<float>(999, col);
		}
	}

	Mat dst = freqFilter_notchkernel(src, fliter);
	float size = -INF;
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			if (dst.at<float>(row, col) > size)
				size = dst.at<float>(row, col);
		}
	}

	Mat result = Mat::zeros(dst.rows, dst.cols, CV_8U);
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			result.at<uchar>(row, col) = (dst.at<float>(row, col) / size) * 255;
		}
	}
	imshow("filtering_image.tif", result);
	return result;
}


void swapPhase(Mat src1, Mat src2, Mat& dst1, Mat& dst2) {
	Mat src1_padded(src1.rows * 2, src1.cols * 2, src1.type());
	Mat src2_padded(src2.rows * 2, src2.cols * 2, src2.type());

	for (int x = 0; x < src1.rows; x++) {
		for (int y = 0; y < src1.cols; y++) {
			src1_padded.at<uchar>(x, y) = src1.at<uchar>(x, y);
		}
	}
	for (int x = 0; x < src2.rows; x++) {
		for (int y = 0; y < src2.cols; y++) {
			src2_padded.at<uchar>(x, y) = src2.at<uchar>(x, y);
		}
	}

	Mat dft1;
	Mat dft2;

	dft1 = fft2d(src1_padded);
	dft2 = fft2d(src2_padded);

	dft1 = fftshift2d(dft1);
	dft2 = fftshift2d(dft2);

	Mat dft1_re(src1_padded.rows, src1_padded.cols, CV_32F);
	Mat dft1_im(src1_padded.rows, src1_padded.cols, CV_32F);
	Mat src1_mag;
	Mat src1_phase;

	Mat dft2_re(src2_padded.rows, src2_padded.cols, CV_32F);
	Mat dft2_im(src2_padded.rows, src2_padded.cols, CV_32F);
	Mat src2_mag;
	Mat src2_phase;

	Mat tmp[2];

	split(dft1, tmp);
	magnitude(tmp[0], tmp[1], src1_mag);
	phase(tmp[0], tmp[1], src1_phase);

	split(dft2, tmp);
	magnitude(tmp[0], tmp[1], src2_mag);
	phase(tmp[0], tmp[1], src2_phase);

	polarToCart(src1_mag, src2_phase, dft1_re, dft1_im);
	polarToCart(src2_mag, src1_phase, dft2_re, dft2_im);

	for (int x = 0; x < src1_padded.rows; x++) {
		for (int y = 0; y < src1_padded.cols; y++) {
			dft1.at<Vec2f>(x, y)[0] = dft1_re.at<float>(x, y);
			dft1.at<Vec2f>(x, y)[1] = dft1_im.at<float>(x, y);
		}
	}

	for (int x = 0; x < src2_padded.rows; x++) {
		for (int y = 0; y < src2_padded.cols; y++) {
			dft2.at<Vec2f>(x, y)[0] = dft2_re.at<float>(x, y);
			dft2.at<Vec2f>(x, y)[1] = dft2_im.at<float>(x, y);
		}
	}

	Mat dst1_padded(src1_padded.rows, src1_padded.cols, CV_32F);
	Mat dst2_padded(src2_padded.rows, src2_padded.cols, CV_32F);

	dft1 = fftshift2d(dft1);
	dft2 = fftshift2d(dft2);

	dst1_padded = ifft2d(dft1);
	dst2_padded = ifft2d(dft2);

	dst1_padded.convertTo(dst1_padded, CV_8U);
	dst2_padded.convertTo(dst2_padded, CV_8U);

	Mat dst1_out(src1.rows, src1.cols, src1.type());
	Mat dst2_out(src2.rows, src2.cols, src2.type());

	for (int x = 0; x < src1.rows; x++) {
		for (int y = 0; y < src1.cols; y++) {
			dst1_out.at<uchar>(x, y) = dst1_padded.at<uchar>(x, y);
		}
	}

	for (int x = 0; x < src2.rows; x++) {
		for (int y = 0; y < src2.cols; y++) {
			dst2_out.at<uchar>(x, y) = dst2_padded.at<uchar>(x, y);
		}
	}
	dst1 = dst1_out;
	dst2 = dst2_out;

	log(1 + src1_mag, src1_mag);
	normalize(src1_mag, src1_mag, 0, 1, NORM_MINMAX);
	log(1 + src1_phase, src1_phase);
	normalize(src1_phase, src1_phase, 0, 1, NORM_MINMAX);

	log(1 + src2_mag, src2_mag);
	normalize(src2_mag, src2_mag, 0, 1, NORM_MINMAX);
	log(1 + src2_phase, src2_phase);
	normalize(src2_phase, src2_phase, 0, 1, NORM_MINMAX);

	imshow("Rectangle.tif_mag", src1_mag);
	imshow("building.tif_mag", src2_mag);
	imshow("Rectangle.tif_phase", src1_phase);
	imshow("building.tif_phase", src2_phase);
	imwrite("Rectangle.tif_mag.tif", src1_mag);
	imwrite("building.tif_mag.tif", src2_mag);
	imwrite("Rectangle.tif_phase.tif", src1_phase);
	imwrite("building.tif_phase.tif", src2_phase);
}
