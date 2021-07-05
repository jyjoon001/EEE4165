#include "utils.h"

int main(int argc, char* argv[])
{
	// 구현 1 //

	Mat src1 = imread("../images/Linear_chirp.tif", IMREAD_GRAYSCALE);
	if (src1.empty()) {
		cout << "Input images do not exist." << endl;
		exit(-1);	
	}
	imshow("Original", src1);
	Mat dst = freqBoxFilter(src1, Size(13, 13));
	dst.convertTo(dst, CV_8U);
	imshow("Box filtering 13X13", dst);
	imwrite("Box filtering 13X13.tif", dst);
	dst = freqBoxFilter(src1, Size(21, 21));
	dst.convertTo(dst, CV_8U);
	imshow("Box filtering 21X21", dst);
	imwrite("Box filtering 21X21.tif", dst);
	dst = freqGaussFilter(src1, Size(31, 31));
	dst.convertTo(dst, CV_8U);
	imshow("Gaussian filtering 31X31", dst);
	imwrite("Gaussian filtering 31X31.tif", dst);
	dst = freqGaussFilter(src1, Size(55, 55));
	dst.convertTo(dst, CV_8U);
	imshow("Gaussian filtering 55X55", dst);
	imwrite("Gaussian filtering 55X55.tif", dst);

	// 구현 2 //
	Mat src2 = imread("../images/Astronaut.tif", IMREAD_GRAYSCALE);
	if (src2.empty()) {
		cout << "Input images do not exist." << endl;
		exit(-1);	
	}
	imshow("Astronaut", src2);
	dst = NotchFilter(src2);
	imwrite("filtering_image.tif", dst);

	// 구현 3 //
	Mat src3 = imread("../images/Rectangle.tif", IMREAD_GRAYSCALE);
	Mat src4 = imread("../images/building.tif", IMREAD_GRAYSCALE);
	if (src3.empty() || src4.empty()) {
		cout << "Input images do not exist." << endl;
		exit(-1);
	}
	Mat dst1, dst2;
	swapPhase(src3, src4, dst1, dst2);

	imshow("Magnitude(Retangle) + Phase(building)", dst1);
	imshow("Magnitude(building) + Phase(Retangle)", dst2);
	imwrite("Magnitude(Retangle) + Phase(building).tif", dst1);
	imwrite("Magnitude(building) + Phase(Retangle).tif", dst2);


	waitKey(0);
	destroyAllWindows();
	return 0;
}