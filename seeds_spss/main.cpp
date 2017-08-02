#include "SEED/seeds2.h"
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include <string>
#include <vector>
#include <png++/png.hpp>
#include "spsstereo/SPSStereo.h"
#include "spsstereo/defParameter.h"
#include "opencv2/imgproc/imgproc.hpp"

void makeSegmentBoundaryImage(const png::image<png::rgb_pixel>& inputImage,
							  const png::image<png::gray_pixel_16>& segmentImage,
							  std::vector< std::vector<int> >& boundaryLabels,
							  png::image<png::rgb_pixel>& segmentBoundaryImage);
void writeDisparityPlaneFile(const std::vector< std::vector<double> >& disparityPlaneParameters, const std::string outputDisparityPlaneFilename);
void writeBoundaryLabelFile(const std::vector< std::vector<int> >& boundaryLabels, const std::string outputBoundaryLabelFilename);


int main(int argc, char* argv[])
{
	int NR_SUPERPIXELS = 200;

	/*SEED: preprosessing superpixels*/
	char* leftImageFilename;
	char* rightImageFilename;

	if(argc > 1)
  	{
    		leftImageFilename = argv[1];   // color image
		rightImageFilename = argv[2];
  	}
  	else
    	{
      		printf("Error : no filename given as input");
      		printf("Usage : %s image_name [number_of_superpixels]\n",argv[0]);
      		return -1;
    	}


  	IplImage* imgSEED = cvLoadImage(leftImageFilename);
  	if ((!imgSEED))
    	{
      		printf("Error while opening %s\n", leftImageFilename);
      		return -1;
    	}


   	int NR_BINS = 5; // Number of bins in each histogram channel

	std::cout<<"width: "<<imgSEED->width<<std::endl;
	std::cout<<"height: "<<imgSEED->height<<std::endl;

   	printf("Generating SEEDS with %d superpixels\n", NR_SUPERPIXELS);
   	SEEDS seeds(imgSEED, NR_BINS, NR_SUPERPIXELS);
   	seeds.runSEED(imgSEED, leftImageFilename);

	UINT* labelSEED = new UINT[imgSEED->width * imgSEED->height];
	seeds.readLabels(labelSEED);
	int superpixelTotal = 0;
	superpixelTotal = seeds.nr_Superpixels();

	/*Slanted-plane: regularize superpixels */

    	std::string leftImage_ = argv[1];
    	std::string rightImage_ = argv[2];

	png::image<png::rgb_pixel> leftImage(leftImage_);
	png::image<png::rgb_pixel> rightImage(rightImage_);

	std::string outputBaseFilename = leftImage_;
	size_t slashPosition = outputBaseFilename.rfind('/');
	if (slashPosition != std::string::npos) outputBaseFilename.erase(0, slashPosition+1);
	size_t dotPosition = outputBaseFilename.rfind('.');
	if (dotPosition != std::string::npos) outputBaseFilename.erase(dotPosition);
	std::string tempOutputDisparityImageFilename = outputBaseFilename + ".png";

	SPSStereo sps;
    	sps.setIterationTotal(outerIterationTotal, innerIterationTotal);
    	sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
    	sps.setInlierThreshold(lambda_d);
    	sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);
    
	png::image<png::gray_pixel_16> segmentImage;
	png::image<png::gray_pixel_16> disparityImage;
	std::vector< std::vector<double> > disparityPlaneParameters;
	std::vector< std::vector<int> > boundaryLabels;
	sps.compute(superpixelTotal, labelSEED,leftImage, rightImage, segmentImage, disparityImage, disparityPlaneParameters, boundaryLabels, tempOutputDisparityImageFilename);


	png::image<png::rgb_pixel> segmentBoundaryImage;
	makeSegmentBoundaryImage(leftImage, segmentImage, boundaryLabels, segmentBoundaryImage);
	

	
	std::string outputDisparityImageFilename = outputBaseFilename + "_left_disparity.png";
	std::string outputSegmentImageFilename = outputBaseFilename + "_segment.png";
	std::string outputBoundaryImageFilename = outputBaseFilename + "_boundary.png";
	std::string outputDisparityPlaneFilename = outputBaseFilename + "_plane.txt";
	std::string outputBoundaryLabelFilename = outputBaseFilename + "_label.txt";


	disparityImage.write(outputDisparityImageFilename);
	segmentImage.write(outputSegmentImageFilename);
	segmentBoundaryImage.write(outputBoundaryImageFilename);
	writeDisparityPlaneFile(disparityPlaneParameters, outputDisparityPlaneFilename);
	writeBoundaryLabelFile(boundaryLabels, outputBoundaryLabelFilename);


	
   return 0;
}


void makeSegmentBoundaryImage(const png::image<png::rgb_pixel>& inputImage,
							  const png::image<png::gray_pixel_16>& segmentImage,
							  std::vector< std::vector<int> >& boundaryLabels,
							  png::image<png::rgb_pixel>& segmentBoundaryImage)
{
	int width = static_cast<int>(inputImage.get_width());
	int height = static_cast<int>(inputImage.get_height());
	int boundaryTotal = static_cast<int>(boundaryLabels.size());

	segmentBoundaryImage.resize(width, height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			segmentBoundaryImage.set_pixel(x, y, inputImage.get_pixel(x, y));
		}
	}

	int boundaryWidth = 2;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int pixelLabelIndex = segmentImage.get_pixel(x, y);

			if (segmentImage.get_pixel(x + 1, y) != pixelLabelIndex) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) segmentBoundaryImage.set_pixel(x - w, y, png::rgb_pixel(128, 128, 128));
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) segmentBoundaryImage.set_pixel(x + w, y, png::rgb_pixel(128, 128, 128));
				}
			}
			if (segmentImage.get_pixel(x, y + 1) != pixelLabelIndex) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) segmentBoundaryImage.set_pixel(x, y - w, png::rgb_pixel(128, 128, 128));
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y + w < height) segmentBoundaryImage.set_pixel(x, y + w, png::rgb_pixel(128, 128, 128));
				}
			}
		}
	}

	boundaryWidth = 7;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int pixelLabelIndex = segmentImage.get_pixel(x, y);

			if (segmentImage.get_pixel(x + 1, y) != pixelLabelIndex) {
				png::rgb_pixel negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex && boundaryLabels[boundaryIndex][1] == segmentImage.get_pixel(x + 1, y))
						|| (boundaryLabels[boundaryIndex][0] == segmentImage.get_pixel(x + 1, y) && boundaryLabels[boundaryIndex][1] == pixelLabelIndex))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}
				if (boundaryLabels[pixelBoundaryIndex][2] == 3) continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {
					negativeSideColor.red = 0;  negativeSideColor.green = 225;  negativeSideColor.blue = 0;
					positiveSideColor.red = 0;  positiveSideColor.green = 225;  positiveSideColor.blue = 0;
				} else if (pixelLabelIndex == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					negativeSideColor.red = 225;  negativeSideColor.green = 0;  negativeSideColor.blue = 0;
					positiveSideColor.red = 0;  positiveSideColor.green = 0;  positiveSideColor.blue = 225;
				} else {
					negativeSideColor.red = 0;  negativeSideColor.green = 0;  negativeSideColor.blue = 225;
					positiveSideColor.red = 225;  positiveSideColor.green = 0;  positiveSideColor.blue = 0;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) segmentBoundaryImage.set_pixel(x - w, y, negativeSideColor);
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) segmentBoundaryImage.set_pixel(x + w, y, positiveSideColor);
				}
			}
			if (segmentImage.get_pixel(x, y + 1) != pixelLabelIndex) {
				png::rgb_pixel negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex && boundaryLabels[boundaryIndex][1] == segmentImage.get_pixel(x, y + 1))
						|| (boundaryLabels[boundaryIndex][0] == segmentImage.get_pixel(x, y + 1) && boundaryLabels[boundaryIndex][1] == pixelLabelIndex))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}
				if (boundaryLabels[pixelBoundaryIndex][2] == 3) continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {
					negativeSideColor.red = 0;  negativeSideColor.green = 225;  negativeSideColor.blue = 0;
					positiveSideColor.red = 0;  positiveSideColor.green = 225;  positiveSideColor.blue = 0;
				} else if (pixelLabelIndex == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					negativeSideColor.red = 225;  negativeSideColor.green = 0;  negativeSideColor.blue = 0;
					positiveSideColor.red = 0;  positiveSideColor.green = 0;  positiveSideColor.blue = 225;
				} else {
					negativeSideColor.red = 0;  negativeSideColor.green = 0;  negativeSideColor.blue = 225;
					positiveSideColor.red = 225;  positiveSideColor.green = 0;  positiveSideColor.blue = 0;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) segmentBoundaryImage.set_pixel(x, y - w, negativeSideColor);
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y+ w < height) segmentBoundaryImage.set_pixel(x, y + w, positiveSideColor);
				}
			}
		}
	}
}

void writeDisparityPlaneFile(const std::vector< std::vector<double> >& disparityPlaneParameters, const std::string outputDisparityPlaneFilename) {
	std::ofstream outputFileStream(outputDisparityPlaneFilename.c_str(), std::ios_base::out);
	if (outputFileStream.fail()) {
		std::cerr << "error: can't open file (" << outputDisparityPlaneFilename << ")" << std::endl;
		exit(0);
	}

	int segmentTotal = static_cast<int>(disparityPlaneParameters.size());
	for (int segmentIndex = 0; segmentIndex < segmentTotal; ++segmentIndex) {
		outputFileStream << disparityPlaneParameters[segmentIndex][0] << " ";
		outputFileStream << disparityPlaneParameters[segmentIndex][1] << " ";
		outputFileStream << disparityPlaneParameters[segmentIndex][2] << std::endl;
	}

	outputFileStream.close();
}

void writeBoundaryLabelFile(const std::vector< std::vector<int> >& boundaryLabels, const std::string outputBoundaryLabelFilename) {
	std::ofstream outputFileStream(outputBoundaryLabelFilename.c_str(), std::ios_base::out);
	if (outputFileStream.fail()) {
		std::cerr << "error: can't open output file (" << outputBoundaryLabelFilename << ")" << std::endl;
		exit(1);
	}

	int boundaryTotal = static_cast<int>(boundaryLabels.size());
	for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
		outputFileStream << boundaryLabels[boundaryIndex][0] << " ";
		outputFileStream << boundaryLabels[boundaryIndex][1] << " ";
		outputFileStream << boundaryLabels[boundaryIndex][2] << std::endl;
	}
	outputFileStream.close();
}
