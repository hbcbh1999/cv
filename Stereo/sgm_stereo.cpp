#include "opencv2/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "SGM.h"
#include <vector>


int main(int argc, char *argv[]){



	cv::Mat imageLeft, imageRight, imageLeftLast;
	cv::Mat grayLeft, grayRight, grayLeftLast;
	imageLeft=cv::imread("/home/sanyu/spsstereo/data_stereo_flow_2012/training/colored_0/000013_10.png",CV_LOAD_IMAGE_COLOR);
	imageRight=cv::imread("/home/sanyu/spsstereo/data_stereo_flow_2012/training/colored_1/000013_10.png",CV_LOAD_IMAGE_COLOR);
	imageLeftLast=cv::imread("/home/sanyu/spsstereo/data_stereo_flow_2012/training/colored_0/000013_10.png",CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(imageLeft,grayLeft,CV_BGR2GRAY);
	cv::cvtColor(imageRight,grayRight,CV_BGR2GRAY);
	cv::cvtColor(imageLeftLast,grayLeftLast,CV_BGR2GRAY);
	const Scalar colorBlue(225.0, 0.0, 0.0, 0.0);
	const Scalar colorRed(0.0, 0.0, 225.0, 0.0);
	const Scalar colorOrange(0.0, 69.0, 225.0, 0.0);
	const Scalar colorYellow(0.0, 255.0, 225.0, 0.0);

	const int PENALTY1 = 400; //400 stereo
	const int PENALTY2 = 6000; //6600 stereo
	const int winRadius = 2;  //2 stereo


//-- Compute the stereo part
	cv::Mat disparity(grayLeft.rows, grayLeft.cols, CV_8UC1);
	SGMStereo sgmstereo(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius);
	sgmstereo.runSGM(disparity);
	imwrite("../disparity.jpg", disparity);
	imshow("disparity", disparity);
	sgmstereo.writeDerivative();


	waitKey(0);
	
	return 0;

}





