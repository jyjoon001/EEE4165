#include "utils.h"

int main(int argc, char* argv[])
{
	Mat src1 = imread("../images/spine.tif", IMREAD_GRAYSCALE);
	Mat src2 = imread("../images/pollen.tif", IMREAD_GRAYSCALE);
	Mat src3 = imread("../images/test_pattern.tif", IMREAD_GRAYSCALE);
	if (src1.empty() || src2.empty() || src3.empty()) {
		cout << "Input images do not exist." << endl;
		exit(-1);
	}
	/*
	//구현 1
	imshow("spine", src1);
	Mat dst1 = gammaTransform(src1, 0.6);
	Mat dst2 = gammaTransform(src1, 0.4);
	Mat dst3 = gammaTransform(src1, 0.3);
	imshow("spine_gamma_06", dst1);
	imwrite("spine_gamma_06.tif", dst1);
	imshow("spine_gamma_04", dst2);
	imwrite("spine_gamma_04.tif", dst2);
	imshow("spine_gamma_03", dst3);
	imwrite("spine_gamma_03.tif", dst3);

	//구현 2
	imshow("pollen", src2);
	Mat dst4 = histEqualization(src2);
	imshow("Hist_Equalization_pollen", dst4);
	imwrite("Hist_Equalization_pollen.tif", dst4);
	*/
	//구현 3
	imshow("test_patten", src3);
	Mat box_3 = boxFilter(Size(3, 3));
	Mat box_11 = boxFilter(Size(11, 11));
	Mat box_25 = boxFilter(Size(29, 29));
	
	// zero padding
	Mat dst5 = imFilter(src3, box_3, "zero");
	imshow("test_patten_zero_3", dst5);
	imwrite("test_patten_zero_3.tif", dst5);
	Mat dst6 = imFilter(src3, box_11, "zero");
	imshow("test_patten_zero_11", dst6);
	imwrite("test_patten_zero_11.tif", dst6);
	Mat dst7 = imFilter(src3, box_25, "zero");
	imshow("test_patten_zero_25", dst7);
	imwrite("test_patten_zero_25.tif", dst7);
	
	// mirror padding
	Mat dst8 = imFilter(src3, box_3, "mirror");
	imshow("test_patten_mirror_3", dst8);
	imwrite("test_patten_mirror_3.tif", dst8);
	Mat dst9 = imFilter(src3, box_11, "mirror");
	imshow("test_patten_mirror_11", dst9);
	imwrite("test_patten_mirror_11.tif", dst9);
	Mat dst10 = imFilter(src3, box_25, "mirror");
	imshow("test_patten_mirror_25", dst10);
	imwrite("test_patten_mirror_25.tif", dst10);
	
	waitKey(0);
	destroyAllWindows();
	return 0;
}
