#include "SGM.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "opencv2/photo.hpp"
SGM::SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_):
imgLeftLast(imgLeftLast_), imgLeft(imgLeft_), imgRight(imgRight_), PENALTY1(PENALTY1_), PENALTY2(PENALTY2_), winRadius(winRadius_)
{
	this->WIDTH  = imgLeft.cols;
	this->HEIGHT = imgLeft.rows;
	censusImageLeft	 = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageRight = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	cost = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	costRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	directCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
	accumulatedCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
std::cout<<"=========================================================="<<std::endl;
};

void SGM::writeDerivative(){}

void SGM::computeCensus(const cv::Mat &image, cv::Mat &censusImg){


	for (int y = winRadius + 1; y < HEIGHT - winRadius - 1; ++y) {
		for (int x = winRadius; x < WIDTH - winRadius; ++x) {
			unsigned char centerValue = image.at<uchar>(y,x);

			int censusCode = 0;
			for (int neiY = -winRadius - 1; neiY <= winRadius + 1; ++neiY) {
				for (int neiX = -winRadius; neiX <= winRadius; ++neiX) {
					censusCode = censusCode << 1;
					if (image.at<uchar>(y + neiY, x + neiX) >= centerValue) censusCode += 1;		
				}
			}
			
			censusImg.at<uchar>(y,x) = static_cast<unsigned char>(censusCode);
		}
	}
}


int SGM::computeHammingDist(const uchar left, const uchar right){

	int var = static_cast<int>(left ^ right);
	int count = 0;

	while(var){
		var = var & (var - 1);
		count++;
	}
	return count;
}




void SGM::sumOverAllCost(){

	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			accumulatedCost.at<SGM::VecDf>(y,x) += directCost.at<SGM::VecDf>(y,x);
			
		}
	}
}



void SGM::createDisparity(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float imax = std::numeric_limits<float>::max();
			int min_index = 0;
			SGM::VecDf vec = accumulatedCost.at<SGM::VecDf>(y,x);

			for(int d = 0; d < DISP_RANGE; d++){
				if(vec[d] < imax ){ imax = vec[d]; min_index = d;}
			}
			disparity.at<uchar>(y,x) = static_cast<uchar>(DIS_FACTOR*min_index);
		}
	}

}

void SGM::setPenalty(const int penalty_1, const int penalty_2){
	PENALTY1 = penalty_1;
	PENALTY2 = penalty_2;
}

void SGM::postProcess(cv::Mat &disparity){}

void SGM::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl){}	

void SGM::resetDirAccumulatedCost(){}

void SGMFlow::resetDirAccumulatedCost(){

for (int x = 0 ; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			for(int d = 0; d < DISP_RANGE; d++){
            			directCost.at<SGM::VecDf>(y,x)[d] = 0.0;
			}
        	}
      	}


}


void SGM::runSGM(cv::Mat &disparity){

	std::cout<<"compute Census: ";
	computeCensus(imgLeft , censusImageLeft);
	computeCensus(imgLeftLast , censusImageLeftLast);
	computeCensus(imgRight, censusImageRight);
	std::cout<<"done"<<std::endl;
	std::cout<<"compute derivative: ";
	computeDerivative();
	std::cout<<"done"<<std::endl;
	std::cout<<"compute pixel-wise cost: ";
	computeCost();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;


	aggregation<0,-1>(cost);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	cv::Mat disparityLeft(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat disparityTemp(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityTemp);
	fastNlMeansDenoising(disparityTemp, disparityLeft); 


for(int y = 0; y < HEIGHT ; y++){
	for(int x = 0; x < WIDTH; x++){
		for(int d = 0; d < DISP_RANGE; d++){
			accumulatedCost.at<SGM::VecDf>(y,x)[d]=0.f;
			directCost.at<SGM::VecDf>(y,x)[d]=0.f;
		}
	}
}


	std::cout<<"compute costRight"<<std::endl;
	computeCostRight();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();

//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,-1>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();
//std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();

	cv::Mat disparityRight(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityTemp);
	fastNlMeansDenoising(disparityTemp, disparityRight);
	consistencyCheck(disparityLeft, disparityRight, disparity, 0);


	

}

void SGM::computeCost(){}

void SGM::computeCostRight(){}

SGM::~SGM(){
	censusImageRight.release();
	censusImageLeft.release();
	censusImageLeftLast.release();	
	cost.release();
	costRight.release();
	directCost.release();
	accumulatedCost.release();
}

void SGM::computeDerivative(){}

SGM::VecDf SGM::addPenalty(SGM::VecDf const& priorL,SGM::VecDf &localCost, float path_intensity_gradient ) {

	SGM::VecDf currL;
	float maxVal;

  	for ( int d = 0; d < DISP_RANGE; d++ ) {
    		float e_smooth = std::numeric_limits<float>::max();		
    		for ( int d_p = 0; d_p < DISP_RANGE; d_p++ ) {
      			if ( d_p - d == 0 ) {
        			// No penality
        			//e_smooth = std::min(e_smooth,priorL[d_p]);
				e_smooth = std::min(e_smooth,priorL[d]);
      			} else if ( abs(d_p - d) == 1 ) {
        			// Small penality
				e_smooth = std::min(e_smooth, priorL[d_p] + (PENALTY1));
      			} else {
        			// Large penality
				//maxVal=static_cast<float>(std::max((float)PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
        			//maxVal=std::max(PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2);				
				//e_smooth = std::min(e_smooth, (priorL[d_p] + maxVal));
				e_smooth = std::min(e_smooth, priorL[d_p] + PENALTY2);

      			}
    		}
    	currL[d] = localCost[d] + e_smooth;
  	}

	double minVal;
	cv::minMaxLoc(priorL, &minVal);

  	// Normalize by subtracting min of priorL cost
	for(int i = 0; i < DISP_RANGE; i++){
		currL[i] -= static_cast<float>(minVal);
	}

	return currL;
}

template <int DIRX, int DIRY>
void SGM::aggregation(cv::Mat cost) {

	if ( DIRX  == -1  && DIRY == 0) {
	std::cout<<"DIRECTION:(-1, 0) called,"<<std::endl;
      		// RIGHT MOST EDGE
      		for (int y = 0; y < HEIGHT; y++ ) {
			directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
      		}
      		for (int x = WIDTH - 2; x >= 0; x-- ) {
        		for ( int y = 0 ; y < HEIGHT ; y++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}

    	}	 

    	// Walk along the edges in a clockwise fashion
    	if ( DIRX == 1  && DIRY == 0) {
	std::cout<<"DIRECTION:( 1, 0) called,"<<std::endl;
      		// Process every pixel along this edge
     		for (int y = 0 ; y < HEIGHT ; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
     		for (int x = 1 ; x < WIDTH; x++ ) {
      			for ( int y = 0; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 0 && DIRY == 1) {
	std::cout<<"DIRECTION:( 0, 1) called,"<<std::endl;
     		//TOP MOST EDGE	
      		for (int x = 0; x < WIDTH ; x++ ) {
			directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
			
      		}
      		for (int y = 1 ; y < HEIGHT ; y++ ) {   
			for ( int x = 0; x < WIDTH; x++ ) {
          			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		} 
   	} 
	
	if ( DIRX == 0 && DIRY == -1) {
	std::cout<<"DIRECTION:( 0,-1) called,"<<std::endl;
      		// BOTTOM MOST EDGE
     		 for ( int x = 0 ; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
      		}
     		for ( int y = HEIGHT - 2; y >= 0; --y ) {
        		for ( int x = 0; x < WIDTH; x++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
    	}

	if ( DIRX == 1  && DIRY == 1) {
	std::cout<<"DIRECTION:( 1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -1) {
	std::cout<<"DIRECTION:( 1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 2; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == 1) {
	std::cout<<"DIRECTION:(-1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -1) {
	std::cout<<"DIRECTION:(-1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 2; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == 1) {
	std::cout<<"DIRECTION:( 2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == 2) {
	std::cout<<"DIRECTION:( 1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == 1) {
	std::cout<<"DIRECTION:(-2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}


	if ( DIRX == -1  && DIRY == 2) {
	std::cout<<"DIRECTION:(-1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == -1) {
	std::cout<<"DIRECTION:( 2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -2) {
	std::cout<<"DIRECTION:( 1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == -1) {
	std::cout<<"DIRECTION:(-2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -2) {
	std::cout<<"DIRECTION:(-1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}
}


void SGMStereo::computeDerivative(){
	cv::Mat gradx(HEIGHT, WIDTH, CV_32FC1);
	cv::Sobel(imgLeft, gradx, CV_32FC1,1,0);

	float sobelCapValue_ = 15;
	
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradx.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			derivativeStereoLeft.at<float>(y,x) = sobelValue;
		}
	}

	
	cv::Sobel(imgRight, gradx, CV_32FC1,1,0);
	
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = 1*gradx.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			derivativeStereoRight.at<float>(y, WIDTH - x -1) = sobelValue;
		}
	}
	


	gradx.release();
}



void SGMStereo::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl){

	std::vector<Point2i> occFalse;
	cv::Mat disparityWoConsistency(HEIGHT, WIDTH, CV_8UC1);

	//based on left
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			unsigned short disparityLeftValue =  static_cast<unsigned short>(disparityLeft.at<uchar>(y,x));
			unsigned short disparityRightValue =  static_cast<unsigned short>(disparityRight.at<uchar>(y,x - disparityLeftValue));	
			disparity.at<uchar>(y,x) = static_cast<uchar>(disparityLeftValue);
			disparityWoConsistency.at<uchar>(y,x) = static_cast<uchar>(disparityLeftValue * visualDisparity);
			if(abs(disparityRightValue - disparityLeftValue) > disparityThreshold){
				disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
				occFalse.push_back(Point2i(x,y));
			}		
		}
	}

	std::vector<Point2i> occRefine;
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = DISP_RANGE; x < WIDTH - winRadius - DISP_RANGE; x++){
			unsigned short top 	= static_cast<unsigned short>(disparity.at<uchar>(y+1,x)) == Dinvd ? 1 : 0;
			unsigned short bottom 	= static_cast<unsigned short>(disparity.at<uchar>(y-1,x)) == Dinvd ? 1 : 0;
			unsigned short left 	= static_cast<unsigned short>(disparity.at<uchar>(y,x-1)) == Dinvd ? 1 : 0;
			unsigned short right	= static_cast<unsigned short>(disparity.at<uchar>(y,x+1)) == Dinvd ? 1 : 0;
			unsigned short self	= static_cast<unsigned short>(disparity.at<uchar>(y,x  )) != Dinvd ? 1 : 0;
			if(((top + bottom + left + right) >= 3) && (self == 1)){
				occRefine.push_back(Point2i(x,y));
				occFalse.push_back(Point2i(x,y));
			}

		}
	}

	for(int i = 0; i < occRefine.size(); i++){
		int x = occRefine[i].x;
		int y = occRefine[i].y;
		disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
	}


 	if(interpl){
		const int len = occFalse.size();

		std::vector<int> newDisparity(len);

		for(int i = 0; i < len; i++){
			std::vector<int> neiborInfo;
			bool occlusion;
			int x = occFalse[i].x;
			int y = occFalse[i].y;


			{
				int xx = x + 1;
				int dirx = 0;
				int dirxDisparityPx = 0;
				while((dirx <= 10) && (xx <= WIDTH - winRadius - DISP_RANGE)){
					if(disparity.at<uchar>(y,xx) == static_cast<uchar>(Dinvd)) {xx++;continue;}
					dirxDisparityPx += static_cast<int>(disparity.at<uchar>(y,xx)); xx++; dirx++;
				}
				if(dirx != 0){neiborInfo.push_back(round(dirxDisparityPx/(float)dirx));}
			}

			{
				int xx = x - 1;
				int dirx = 0;
				int dirxDisparityNx = 0;
				while((dirx <= 10) && (xx >= winRadius + DISP_RANGE)){
					if(disparity.at<uchar>(y,xx) == static_cast<uchar>(Dinvd)) {xx--;continue;}
					dirxDisparityNx += static_cast<int>(disparity.at<uchar>(y,xx)); xx--; dirx++;
				}
				if(dirx != 0){neiborInfo.push_back(round(dirxDisparityNx/(float)dirx));}
			}
	
			if(neiborInfo.size() == 2){ occlusion = fabs((neiborInfo[0]-neiborInfo[1])/fmin(neiborInfo[0], neiborInfo[1])) > 0.2 ? true : false;}
			else{occlusion = false;}

			{
				int yy = y + 1;
				int diry = 0;
				int dirxDisparityPy = 0;
				while((diry < 1) && (yy < HEIGHT - winRadius)){
					if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy++;continue;}
					dirxDisparityPy = static_cast<int>(disparity.at<uchar>(yy,x)); yy++; diry++;
				}
				if(diry != 0){neiborInfo.push_back(round(dirxDisparityPy/(float)diry));}
			}

			{
				int yy = y - 1;
				int diry = 0;
				int dirxDisparityNy = 0;
				while((diry < 1) && (yy >= winRadius)){
					if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy--;continue;}
					dirxDisparityNy = static_cast<int>(disparity.at<uchar>(yy,x)); yy--; diry++;
				}
				if(diry != 0){neiborInfo.push_back(round(dirxDisparityNy/(float)diry));}

			}

			{
				int dirxy = 0;
				int yy = y + 1;
				int xx = x - 1;
				int dirxDisparityNxPy = 0;
				while((dirxy < 1) && (yy < HEIGHT - winRadius) && (xx >= winRadius + DISP_RANGE)){
					if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx--;continue;}
					dirxDisparityNxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx--; dirxy++;
				}
				if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxPy/(float)dirxy));}
			}

			{
				int dirxy = 0;
				int yy = y + 1;
				int xx = x + 1;
				int dirxDisparityPxPy = 0;
				while((dirxy < 1) && (yy < HEIGHT - winRadius) && (xx < WIDTH - winRadius - DISP_RANGE)){
					if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx++;continue;}
					dirxDisparityPxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx++; dirxy++;
				}
			if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxPy/(float)dirxy));}
				}
		
			{
				int dirxy = 0;
				int yy = y - 1;
				int xx = x + 1;
				int dirxDisparityPxNy = 0;
				while((dirxy < 1) && (yy >= winRadius) && (xx < WIDTH - winRadius - DISP_RANGE)){
					if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx++;continue;}
					dirxDisparityPxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx++; dirxy++;
				}
				if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxNy/(float)dirxy));}
			}
		
			{
				int dirxy = 0;
				int yy = y - 1;
				int xx = x - 1;
				int dirxDisparityNxNy = 0;
				while((dirxy < 1) && (yy >= winRadius) && (xx >= winRadius + DISP_RANGE)){
					if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx--;continue;}
					dirxDisparityNxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx--; dirxy++;
				}
				if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxNy/(float)dirxy));}
			}


			std::sort(neiborInfo.begin(), neiborInfo.end());

		
			int secLow = neiborInfo[1];
			int median = neiborInfo[floor(neiborInfo.size()/2.f)];
		

			unsigned short newValue = 0;	
			if(occlusion == true){
				newValue = secLow;			
			}else{
				newValue = median;	
			}
	
			newDisparity[i] = newValue ;

		}	

		for(int i = 0; i < len; i++){
			int x = occFalse[i].x;
			int y = occFalse[i].y;
			disparity.at<uchar>(y,x) = static_cast<uchar>(newDisparity[i]);

		}	
	}

}


void SGMStereo::computeCostRight(){

for(int y = winRadius; y < HEIGHT - winRadius; y++){
	//for(int x = winRadius; x < WIDTH - winRadius - DISP_RANGE; x++){
	for(int x = winRadius; x < WIDTH - DISP_RANGE - 1; x++){
		for(int d = 0; d < DISP_RANGE; d++){
				
			for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
				for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
					costRight.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoRight.at<float>(neiY, neiX)- 
										derivativeStereoLeft.at<float>(neiY, neiX + d))
							+ CENSUS_W * computeHammingDist(censusImageRight.at<uchar>(neiY, neiX), censusImageLeft.at<uchar>(neiY, neiX + d));
					}
					
				}
			}

		}
	}

//for WIDTH - winRadius - DISP_RANGE -> WIDTH - winRadius
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x =  WIDTH - DISP_RANGE - 1; x < WIDTH - winRadius; x++){
			int end = WIDTH - winRadius - 1 - x;
			for(int d = 0; d < end; d++){
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						costRight.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoRight.at<float>(neiY, neiX)- 
										derivativeStereoLeft.at<float>(neiY, neiX + d))
							+ CENSUS_W * computeHammingDist(censusImageRight.at<uchar>(neiY, neiX), censusImageLeft.at<uchar>(neiY, neiX + d));

					}
				}
			}
			float val = costRight.at<SGM::VecDf>(y,x)[end - 1];
			for(int d = end; d < DISP_RANGE; d++){
				costRight.at<SGM::VecDf>(y,x)[d] = val;
			}
		}
	}

}



void SGMStereo::calcHalfPixelRight() {

	for(int y = 1; y < HEIGHT; ++y){
		derivativeStereoRight.at<float>(y,0) = 15;
		for (int x = 0; x < WIDTH; ++x) {
			float centerValue = derivativeStereoRight.at<float>(y,x);
			float leftHalfValue = x > 0 ? (centerValue + derivativeStereoRight.at<float>(y,x-1))/2 : centerValue;
			float rightHalfValue = x < WIDTH - 1 ? (centerValue + derivativeStereoRight.at<float>(y,x+1))/2 : centerValue;
			float minValue = std::min(leftHalfValue, rightHalfValue);
			minValue = std::min(minValue, centerValue);
			float maxValue = std::max(leftHalfValue, rightHalfValue);
			maxValue = std::max(maxValue, centerValue);

			halfPixelRightMin.at<float>(y,x)=minValue;
			halfPixelRightMax.at<float>(y,x) = maxValue;
		}
	}
}

void SGMStereo::computeCost(){
	
//pixel intensity matching

	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = DISP_RANGE; x < WIDTH - winRadius; x++){
			for(int d = 0; d < DISP_RANGE; d++){				
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						cost.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
										    derivativeStereoRight.at<float>(neiY, neiX - d))
							+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), censusImageRight.at<uchar>(neiY, neiX - d));
					}
					
				}
			}

		}
	}

	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < DISP_RANGE; x++){
			for(int d = 0; d < x; d++){
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						cost.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
										    derivativeStereoRight.at<float>(neiY, neiX - d))
							+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), censusImageRight.at<uchar>(neiY, neiX - d));
						
					}
				}
			}
			float val = cost.at<SGM::VecDf>(y,x)[x - 1];
			for(int d = x; d < DISP_RANGE; d++){
				cost.at<SGM::VecDf>(y,x)[d] = val;
			}
		}
	}

}

SGMStereo::SGMStereo(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
		derivativeStereoLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeStereoRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		halfPixelRightMin = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		halfPixelRightMax = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
	}

SGMStereo::~SGMStereo(){
	derivativeStereoLeft.release();
	derivativeStereoRight.release();
}

void SGMStereo::postProcess(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < DISP_RANGE; x++){
			disparity.at<uchar>(y,x)=static_cast<uchar>(0);
		}
	}

}

void SGMStereo::writeDerivative(){
	imwrite("../derivativeStereoRight.jpg",derivativeStereoRight);
	imwrite("../derivativeStereoLeft.jpg",derivativeStereoLeft);
}


