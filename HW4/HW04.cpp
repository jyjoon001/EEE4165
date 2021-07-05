#include "utils.h"

int main(int argc, char* argv[])
{
	
	// 구현 1 //
	Mat src1 = imread("../images/Pattern.tif", IMREAD_GRAYSCALE);
	if (src1.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}
	
	imshow("original", src1);
	Mat noise = addSaltPepper(src1, 0.1, 0.0);
	imshow("Salt_noise_01", noise);
	imwrite("Salt_noise_01.tif", noise);

	noise = addSaltPepper(src1, 0.5, 0.0);
	imshow("Salt_noise_05", noise);
	imwrite("Salt_noise_05.tif", noise);

	noise = addSaltPepper(src1, 0.0, 0.1);
	imshow("Pepper_noise_01", noise);
	imwrite("Pepper_noise_01.tif", noise);

	noise = addSaltPepper(src1, 0.0, 0.5);
	imshow("Pepper_noise_05", noise);
	imwrite("Pepper_noise_05.tif", noise);

	noise = addSaltPepper(src1, 0.05, 0.05);
	imshow("Salt & Pepper_noise_005", noise);
	imwrite("Salt & Pepper_noise_005.tif", noise);

	noise = addSaltPepper(src1, 0.25, 0.25);
	imshow("Salt & Pepper_noise_025", noise);
	imwrite("Salt & Pepper_noise_025.tif", noise);
	
	// Ckt_board 폴더내에 있는 6개 파일을 바꿔가면서 해주세요. //
	
	Mat src2 = imread("../images/Ckt_board/Ckt_board_salt&pepper_0.3.tif", IMREAD_GRAYSCALE);
	if (src2.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}

	Mat dst_contra_1_5 = contraharmonic(src2, Size(3, 3), 1.5);
	Mat dst_contra_0 = contraharmonic(src2, Size(3, 3), 0.0);
	Mat dst_contra_minus_1_5 = contraharmonic(src2, Size(3, 3), -1.5);
	Mat dst_meidan_3 = median(src2, Size(3, 3));
	Mat dst_meidan_5 = median(src2, Size(5, 5));
	Mat dst_adamed_7 = adaptiveMedian(src2, Size(7, 7));
	Mat dst_adamed_11 = adaptiveMedian(src2, Size(11, 11));

	imshow("Contraharmonic mean 1.5", dst_contra_1_5);
	imshow("Contraharmonic mean 0", dst_contra_0);
	imshow("Contraharmonic mean -1.5", dst_contra_minus_1_5);
	imshow("Median 3", dst_meidan_3);
	imshow("Median 3", dst_meidan_5);
	imshow("Adaptive median 7", dst_adamed_7);
	imshow("Adaptive median 7", dst_adamed_11);

	imwrite("Contraharmonic mean 1.5.tif", dst_contra_1_5);
	imwrite("Contraharmonic mean 0.0.tif", dst_contra_0);
	imwrite("Contraharmonic mean -1.5.tif", dst_contra_minus_1_5);
	imwrite("Median 3.tif", dst_meidan_3);
	imwrite("Median 5.tif", dst_meidan_5);
	imwrite("Adaptive median 7.tif", dst_adamed_7);
	imwrite("Adaptive median 11.tif", dst_adamed_11);

	Mat src3 = imread("../images/sogang.tif", IMREAD_GRAYSCALE);
	if (src3.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}

	Mat add_turbulence_0025 = add_turbulence(src3, 0.0025);
	Mat add_turbulence_001 = add_turbulence(src3, 0.001);
	Mat add_turbulence_00025 = add_turbulence(src3, 0.00025);
	imshow("Sogang_0025",add_turbulence_0025);
	imwrite("Sogang_0025.tif", add_turbulence_0025);
	imshow("Sogang_001", add_turbulence_001);
	imwrite("Sogang_001.tif", add_turbulence_001);
	imshow("Sogang_00025", add_turbulence_00025);
	imwrite("Sogang_00025.tif", add_turbulence_00025);
	
	// K값을 계속 바꾸면서 하세요. //
	Mat Wiener_filter_image = wienerFilter(add_turbulence_0025, 1e-6);
	imshow("Wiener_filter_image_0025", Wiener_filter_image);
	imwrite("Wiener_Sogang_0025.tif", Wiener_filter_image);
	float PSNR;
	int s = 0;
	int MSE = 0;

	for (int x = 0; x < src3.rows; x++) {
		for (int y = 0; y < src3.cols; y++) {
			if (s < src3.at<uchar>(x, y)) s = src3.at<uchar>(x, y);
			MSE = MSE + pow(src3.at<uchar>(x, y) - Wiener_filter_image.at<uchar>(x, y), 2);
		}
	}

	MSE = MSE / (src3.rows * src3.cols);

	PSNR = 10 * log(pow(s, 2) / MSE) / log(10);
	printf("PSNR for src(0.0025) : %f\n", PSNR);

	Wiener_filter_image = wienerFilter(add_turbulence_001, 1e-6);
	imshow("Wiener_filter_image_001", Wiener_filter_image);
	imwrite("Wiener_Sogang_001.tif", Wiener_filter_image);
	int s2 = 0;
	int MSE2 = 0;

	for (int x = 0; x < src3.rows; x++) {
		for (int y = 0; y < src3.cols; y++) {
			if (s2=0 < src3.at<uchar>(x, y)) s2 = src3.at<uchar>(x, y);
			MSE2 = MSE2 + pow(src3.at<uchar>(x, y) - Wiener_filter_image.at<uchar>(x, y), 2);
		}
	}
	MSE2 = MSE2 / (src3.rows * src3.cols);

	PSNR = 10 * log(pow(s2, 2) / MSE2) / log(10);
	printf("PSNR for src(0.001) : %f\n", PSNR);

	Wiener_filter_image = wienerFilter(add_turbulence_00025, 1e-6);
	imshow("Wiener_filter_image_00025", Wiener_filter_image);
	imwrite("Wiener_Sogang_00025.tif", Wiener_filter_image);
	int s3 = 0;
	int MSE3 = 0;

	for (int x = 0; x < src3.rows; x++) {
		for (int y = 0; y < src3.cols; y++) {
			if (s3 = 0 < src3.at<uchar>(x, y)) s3 = src3.at<uchar>(x, y);
			MSE3 = MSE3 + pow(src3.at<uchar>(x, y) - Wiener_filter_image.at<uchar>(x, y), 2);
		}
	}
	MSE3 = MSE3 / (src3.rows * src3.cols);

	PSNR = 10 * log(pow(s3, 2) / MSE3) / log(10);
	printf("PSNR for src(0.00025) : %f\n", PSNR);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
