# include "utils.h"

Mat fft2d(Mat src, int flags) {
	if (src.type() == CV_8UC1)
		src = Mat_<float>(src);
	Mat dst;
	dft(src, dst, flags);

	return dst;
}

void fft2d(Mat src, Mat& dst, int flags) {
	dst = fft2d(src, flags);
}

Mat ifft2d(Mat src, int flags) {
	Mat dst;
	idft(src, dst, flags);

	return dst;
}

void ifft2d(Mat src, Mat& dst, int flags) {
	dst = ifft2d(src, flags);
}

Mat fftshift2d(Mat src, int height, int width) {
	int src_type = src.type();
	if (src_type == CV_8U)
		src.convertTo(src, CV_32F);

	int H = src.rows;
	int W = src.cols;
	int C = src.channels();

	int cx = floor(height / 2);
	int cy = floor(width / 2);
	
	Mat dst = Mat::zeros(src.size(), src.type());
	float* src_data = (float*)src.data;
	float* dst_data = (float*)dst.data;
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++)
			for (int c = 0; c < C; c++)
				dst_data[W * C * ((cx + i) % H) + C * ((cy + j) % W) + c] = src_data[W * C * i + C * j + c];

	if (src_type == CV_8U)
		dst.convertTo(dst, CV_8U);
	return dst;
}

void fftshift2d(Mat src, Mat& dst, int height, int width) {
	dst = fftshift2d(src, height, width);
}

Mat fftshift2d(Mat src) {
	return fftshift2d(src, src.rows + 1, src.cols + 1);
}

void fftshift2d(Mat src, Mat& dst) {
	dst = fftshift2d(src);
}

Mat ifftshift2d(Mat src) {
	return fftshift2d(src, src.rows, src.cols);
}

void ifftshift2d(Mat src, Mat& dst){
	dst = ifftshift2d(src);
}
