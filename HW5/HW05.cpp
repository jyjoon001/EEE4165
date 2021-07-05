#include "utils.h"

int main(int argc, char* argv[])
{

	Mat src = imread("../images/lenna.tif", IMREAD_COLOR);
	if (src.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}
	Mat src_float = Mat_<Vec3f>(src);
	src_float = src_float / 255;

	Mat dst = cvtBGR2HSI(src_float);
	Mat hsi[3];
	split(dst, hsi);

	imshow("lenna", src);
	imshow("H", hsi[0]);
	imwrite("H.tif", hsi[0]);
	imshow("S", hsi[1]);
	imwrite("S.tif", hsi[1]);
	imshow("I", hsi[2]);
	imwrite("I.tif", hsi[2]);

	Mat bgr[3];
	split(src_float, bgr);
	bgr[0] = abs(sobelFilter(bgr[0], 0)) + abs(sobelFilter(bgr[0], 1));
	bgr[1] = abs(sobelFilter(bgr[1], 0)) + abs(sobelFilter(bgr[1], 1));
	bgr[2] = abs(sobelFilter(bgr[2], 0)) + abs(sobelFilter(bgr[2], 1));

	Mat src_hsi = cvtBGR2HSI(src_float);
	split(src_hsi, hsi);
	hsi[2] = abs(sobelFilter(hsi[2], 0)) + abs(sobelFilter(hsi[2], 1));

	Mat dst_rgb;
	merge(bgr, 3, dst_rgb);
	dst_rgb = Mat_<Vec3b>(dst_rgb * 255);

	Mat sobel_hsi;
	merge(hsi, 3, sobel_hsi);
	Mat dst_hsi = cvtHSI2BGR(sobel_hsi);
	dst_hsi = Mat_<Vec3b>(dst_hsi * 255);

	imshow("Sobel in B", bgr[0]);
	imshow("Sobel in G", bgr[1]);
	imshow("Sobel in R", bgr[2]);
	imshow("Sobel in RGB", dst_rgb);
	imshow("Sobel in HSI", dst_hsi);
	imwrite("Sobel in B.tif", bgr[0]);
	imwrite("Sobel in G.tif", bgr[1]);
	imwrite("Sobel in R.tif", bgr[2]);
	imwrite("Sobel in RGB.tif", dst_rgb);
	imwrite("Sobel in HSI.tif", dst_hsi);

	waitKey(0);
	destroyAllWindows();
	return 0;
}