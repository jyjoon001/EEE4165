#include "utils.h"

int main(int argc, char* argv[])
{
	//구현 1//

	Mat src = imread("../images/cameraman.tif", IMREAD_ANYCOLOR);
	if (src.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}
	int H = src.rows;
	int W = src.cols;

	string flags_1 = "nearest";
	string flags_2 = "bilinear";

	Mat affine_A = Mat::zeros(Size(3, 3), CV_32F);
	float* aff_data_A = (float*)affine_A.data;
	aff_data_A[0] = 2.0;	aff_data_A[1] = 0.0;	aff_data_A[2] = 0.0;
	aff_data_A[3] = 0.0;	aff_data_A[4] = 2.0;	aff_data_A[5] = 0.0;
	aff_data_A[6] = 0.0;	aff_data_A[7] = 0.0;	aff_data_A[8] = 1.0;
	Mat affine_B = Mat::zeros(Size(3, 3), CV_32F);
	float* aff_data_B = (float*)affine_B.data;
	aff_data_B[0] = 1.0;	aff_data_B[1] = 0.5;	aff_data_B[2] = 0.0;
	aff_data_B[3] = 0.5;	aff_data_B[4] = 1.0;	aff_data_B[5] = 0.0;
	aff_data_B[6] = 0.0;	aff_data_B[7] = 0.0;	aff_data_B[8] = 1.0;

	imshow("Origin_1", src);
	//a) 
	Mat dst = warpAffine(src, affine_A, flags_1);
	Mat dst2 = warpAffine(dst, affine_B, flags_1);
	imshow("implement1_a", dst2);
	imwrite("implement1_a.tif", dst2);
	//b)
	Mat affine_C = affine_B * affine_A;
	Mat dst3 = warpAffine(src, affine_C, flags_1);
	imshow("implement1_b", dst3);
	imwrite("implement1_b.tif", dst3);
	//c)
	Mat affine_C_p = affine_A * affine_B;
	Mat dst4 = warpAffine(src, affine_C_p, flags_1);
	imshow("implement1_c", dst4);
	imwrite("implement1_c.tif", dst4);


	//구현 2//

	Mat src_2 = imread("../images/Letter_T.tif", IMREAD_ANYCOLOR);
	if (src_2.empty()) {
		cout << "Input image does not exist." << endl;
		exit(-1);
	}
	//a) 
	//nearest
	imshow("Origin_2", src_2);
	Mat dst_5 = rotate(src_2, 0.523598, flags_1);
	Mat dst_6 = translation(dst_5, 20, 0, flags_1);
	imshow("implement2_a_nearest", dst_6);
	imwrite("implement2_a_nearest.tif", dst_6);
	//bilinear
	Mat dst_7 = rotate(src_2, 0.523598, flags_2);
	Mat dst_8 = translation(dst_7, 20, 0, flags_2);
	imshow("implement2_a_bilinear", dst_8);
	imwrite("implement2_a_bilinear.tif", dst_8);
	//b) 
	//nearest
	Mat dst_9 = translation(src_2, 20, 0, flags_1);
	Mat dst_10 = rotate(dst_9, 0.523598, flags_1);
	imshow("implement2_b_nearest", dst_10);
	imwrite("implement2_b_nearest.tif", dst_10);
	//bilinear
	Mat dst_11 = translation(src_2, 20, 0, flags_2);
	Mat dst_12 = rotate(dst_11, 0.523598, flags_2);
	imshow("implement2_b_bilinear", dst_12);
	imwrite("implement2_b_bilinear.tif", dst_12);
	
	waitKey(0);
	destroyAllWindows();

	return 0;
}