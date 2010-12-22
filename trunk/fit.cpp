/**

Contributions by Ismail Ari

Copyright (c) 2008-2010 by Yao Wei <njustyw@gmail.com>, all rights reserved.
*/

#include <vector>
#include <string>
#include <iostream>
#include <conio.h>

#include "asmfitting.h"
#include "vjfacedetect.h"
#include "video_camera.h"
#include "find_classes.h"
#include "funcs.h"

using namespace std;

#define N_SHAPES_FOR_FILTERING 3
#define MAX_FRAMES_UNDER_THRESHOLD 15
#define NORMALIZE_POSE_PARAMS 0
#define CAM_WIDTH 640
#define CAM_HEIGHT 480
#define FRAME_TO_START_DECISION 50

#define PRINT_TIME_TICKS 0
#define PRINT_FEATURES 0
#define PRINT_FEATURE_SCALES 0
#define UP 2490368
#define DOWN 2621440
#define RIGHT 2555904
#define LEFT 2424832

const char* TRACKER_WINDOW_NAME = "Facexpress";
const char* PROCESSED_WINDOW_NAME = "Processed Results";

IplImage *tmpCimg1=0; // Temp. color images
IplImage *tmpGimg1=0, *tmpGimg2=0; // Temp. grayscale images
IplImage *tmpSimg1 = 0, *tmpSimg2 = 0; // Temp. signed images
IplImage *tmpFimg1 = 0, *tmpFimg2 = 0; // Temp float images
IplImage *sobel1=0, *sobel2=0, *sobelF1=0, *sobelF2=0;

cv::Mat features, featureScales, expressions;
//const int N_FEATURES = 7;		//find_classes'�n i�inde oldu�u i�in comment out

CvFont font;
int showTrackerGui = 1, showProcessedGui = 0, showRegionsOnGui = 0, showFeatures = 0, showShape = 0;


void setup_tracker(IplImage* sampleCimg) {
	if(showTrackerGui) cvNamedWindow(TRACKER_WINDOW_NAME,1);
	if(showProcessedGui) cvNamedWindow(PROCESSED_WINDOW_NAME, 1);

	tmpCimg1 = cvCreateImage(cvGetSize(sampleCimg), sampleCimg->depth, 3);
	tmpGimg1 = cvCreateImage(cvGetSize(sampleCimg), sampleCimg->depth, 1);
	tmpGimg2 = cvCreateImage(cvGetSize(sampleCimg), sampleCimg->depth, 1);
	tmpSimg1 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_16S, 1);
	tmpSimg2 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_16S, 1);

	sobel1 = cvCreateImage(cvGetSize(sampleCimg), sampleCimg->depth, 1);
	sobel2 = cvCreateImage(cvGetSize(sampleCimg), sampleCimg->depth, 1);
	sobelF1 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_32F, 1);
	sobelF2 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_32F, 1);

	tmpFimg1 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_32F, 1);
	tmpFimg2 = cvCreateImage(cvGetSize(sampleCimg), IPL_DEPTH_32F, 1);

	features = cv::Mat(1,N_FEATURES, CV_32FC1);
	featureScales = cv::Mat(1,N_FEATURES, CV_32FC1);
	cvZero(&(CvMat)featureScales);
}

void exit_tracker() {
	cvReleaseImage(&tmpCimg1);
	cvReleaseImage(&tmpGimg1);
	cvReleaseImage(&tmpGimg2);
	cvReleaseImage(&tmpSimg1);
	cvReleaseImage(&tmpSimg2);
	cvReleaseImage(&tmpFimg1);
	cvReleaseImage(&tmpFimg2);
	cvReleaseImage(&sobel1);
	cvReleaseImage(&sobel2);
	cvReleaseImage(&sobelF1);
	cvReleaseImage(&sobelF2);
	cvDestroyAllWindows();
}

void extract_features_and_display(IplImage* img, asm_shape shape){
	int iFeature = 0;
	CvScalar red = cvScalar(0,0,255);
	CvScalar green = cvScalar(0,255,0);
	CvScalar blue = cvScalar(255,0,0);
	CvScalar gray = cvScalar(150,150,150);
	CvScalar lightgray = cvScalar(150,150,150);
	CvScalar white = cvScalar(255,255,255);

	// cvCvtColor(img, tmpGimg1, CV_RGB2GRAY);
	cvCvtColor(img, tmpCimg1, CV_RGB2HSV);
	cvSplit(tmpCimg1, 0, 0, tmpGimg1, 0);
	// cvSmooth(tmpGimg1, tmpGimg1, CV_GAUSSIAN, 3, 3); // comment out to make it scale independent
	cvSobel(tmpGimg1, tmpSimg1, 1, 0, 3);
	cvSobel(tmpGimg1, tmpSimg2, 0, 1, 3);
	cvConvertScaleAbs(tmpSimg1, sobel1); // Sobel on x direction
	cvConvertScaleAbs(tmpSimg2, sobel2); // Sobel on y direction
	cvConvertScale(sobel1, sobelF1, 1/255.);
	cvConvertScale(sobel2, sobelF2, 1/255.);

	//cvFlip(sobelF2, NULL, 1);
	//cvShowImage(PROCESSED_WINDOW_NAME, sobelF2);

	int nPoints[1];
	CvPoint **faceComponent;
	faceComponent = (CvPoint **) cvAlloc (sizeof (CvPoint *));
	faceComponent[0] = (CvPoint *) cvAlloc (sizeof (CvPoint) * 10); // max 10 points per component


	// Draw all points
	for(int i = 0; i < shape.NPoints(); i++)
		if(showRegionsOnGui)
			cvCircle(img, cvPoint((int)shape[i].x, (int)shape[i].y), 0, lightgray);

	// Draw eyebrows
	CvPoint eyeBrowMiddle1 = cvPoint((int)(shape[55].x+shape[63].x)/2, (int)(shape[55].y+shape[63].y)/2);
	CvPoint eyeBrowMiddle2 = cvPoint((int)(shape[69].x+shape[75].x)/2, (int)(shape[69].y+shape[75].y)/2);
	if(showRegionsOnGui) {
		cvLine(img, cvPoint((int)(shape[53].x+shape[65].x)/2, (int)(shape[53].y+shape[65].y)/2), eyeBrowMiddle1, red, 2);
		cvLine(img, eyeBrowMiddle1, cvPoint((int)(shape[58].x+shape[60].x)/2, (int)(shape[58].y+shape[60].y)/2), red, 2);
		cvLine(img, cvPoint((int)(shape[66].x+shape[78].x)/2, (int)(shape[66].y+shape[78].y)/2), eyeBrowMiddle2, red, 2);
		cvLine(img, eyeBrowMiddle2, cvPoint((int)(shape[71].x+shape[73].x)/2, (int)(shape[71].y+shape[73].y)/2), red, 2);
	}
	
	// Draw eye
	CvPoint eyeMiddle1 = cvPoint((int)(shape[3].x+shape[9].x)/2, (int)(shape[3].y+shape[9].y)/2);
	CvPoint eyeMiddle2 = cvPoint((int)(shape[29].x+shape[35].x)/2, (int)(shape[29].y+shape[35].y)/2);
	if(showRegionsOnGui){ 
		cvCircle(img, eyeMiddle1, 2, lightgray, 1);
		cvCircle(img, eyeMiddle2, 2, lightgray, 1);
	}

	double eyeBrowDist1 = (cvSqrt((double)(eyeMiddle1.x-eyeBrowMiddle1.x)*(eyeMiddle1.x-eyeBrowMiddle1.x)+ 
			(eyeMiddle1.y-eyeBrowMiddle1.y)*(eyeMiddle1.y-eyeBrowMiddle1.y))
		/ shape.GetWidth());
	double eyeBrowDist2 = (cvSqrt((double)(eyeMiddle2.x-eyeBrowMiddle2.x)*(eyeMiddle2.x-eyeBrowMiddle2.x)+ 
			(eyeMiddle2.y-eyeBrowMiddle2.y)*(eyeMiddle2.y-eyeBrowMiddle2.y))
		/ shape.GetWidth());
	features.at<float>(0,iFeature) = (float)((eyeBrowDist1+eyeBrowDist2)/2);
	++iFeature;

	// Forehead
	nPoints[0] = 4;
	faceComponent[0][0] = cvPoint((int)(6*shape[63].x-shape[101].x)/5, (int)(6*shape[63].y-shape[101].y)/5);
	faceComponent[0][1] = cvPoint((int)(6*shape[69].x-shape[105].x)/5, (int)(6*shape[69].y-shape[105].y)/5);
	faceComponent[0][2] = cvPoint((int)(21*shape[69].x-shape[105].x)/20, (int)(21*shape[69].y-shape[105].y)/20);
	faceComponent[0][3] = cvPoint((int)(21*shape[63].x-shape[101].x)/20, (int)(21*shape[63].y-shape[101].y)/20);
	if(showRegionsOnGui)
		cvPolyLine(img, faceComponent, nPoints, 1, 1, blue);

	cvSetZero(tmpFimg1);
	cvSetZero(tmpFimg2);
	cvFillPoly(tmpFimg1, faceComponent, nPoints, 1, cvScalar(1,1,1));

	cvMultiplyAcc(tmpFimg1, sobelF2, tmpFimg2);
	CvScalar area = cvSum(tmpFimg1);
	CvScalar sum = cvSum(tmpFimg2);
	features.at<float>(0,iFeature) = (float)(sum.val[0]/area.val[0]);
	++iFeature;

	// Forehead Middle
	nPoints[0] = 4;
	faceComponent[0][0] = cvPoint((int)(29*shape[59].x+shape[102].x)/30, (int)(29*shape[59].y+shape[102].y)/30);
	faceComponent[0][1] = cvPoint((int)(29*shape[79].x+shape[104].x)/30, (int)(29*shape[79].y+shape[104].y)/30);
	faceComponent[0][2] = cvPoint((int)(11*shape[79].x-shape[104].x)/10, (int)(11*shape[79].y-shape[104].y)/10);
	faceComponent[0][3] = cvPoint((int)(11*shape[59].x-shape[102].x)/10, (int)(11*shape[59].y-shape[102].y)/10);
	if(showRegionsOnGui)
		cvPolyLine(img, faceComponent, nPoints, 1, 1, green);

	cvSetZero(tmpFimg1);
	cvSetZero(tmpFimg2);
	cvFillPoly(tmpFimg1, faceComponent, nPoints, 1, cvScalar(1,1,1));

	cvMultiplyAcc(tmpFimg1, sobelF1, tmpFimg2);
	area = cvSum(tmpFimg1);
	sum = cvSum(tmpFimg2);
	features.at<float>(0,iFeature) = (float)(sum.val[0]/area.val[0]);
	++iFeature;

	// Cheek 1
	nPoints[0] = 4;
	faceComponent[0][0] = cvPoint((int)(shape[80].x+4*shape[100].x)/5, (int)(shape[80].y+4*shape[100].y)/5);
	faceComponent[0][1] = cvPoint((int)(shape[111].x+9*shape[96].x)/10, (int)(shape[111].y+9*shape[96].y)/10);
	faceComponent[0][2] = cvPoint((int)(9*shape[111].x+shape[96].x)/10, (int)((9*shape[111].y+shape[96].y)/10 + shape.GetWidth()/20));
	faceComponent[0][3] = cvPoint((int)(4*shape[80].x+shape[100].x)/5, (int)(4*shape[80].y+shape[100].y)/5);
	if(showRegionsOnGui)
		cvPolyLine(img, faceComponent, nPoints, 1, 1, blue);

	cvSetZero(tmpFimg1);
	cvSetZero(tmpFimg2);
	cvFillPoly(tmpFimg1, faceComponent, nPoints, 1, cvScalar(1,1,1));
	//cvSmooth(sobelF1, sobelF1, CV_GAUSSIAN, 5, 5);
	//cvSmooth(sobelF2, sobelF2, CV_GAUSSIAN, 5, 5);
	cvMultiplyAcc(tmpFimg1, sobelF1, tmpFimg2);
	cvMultiplyAcc(tmpFimg1, sobelF2, tmpFimg2);
	area = cvSum(tmpFimg1);
	sum = cvSum(tmpFimg2);
	features.at<float>(0,iFeature) = (float)(sum.val[0]/area.val[0]);
	++iFeature;

	// Cheek 2
	faceComponent[0][0] = cvPoint((int)(shape[88].x+4*shape[106].x)/5, (int)(shape[88].y+4*shape[106].y)/5);
	faceComponent[0][1] = cvPoint((int)(shape[115].x+9*shape[110].x)/10, (int)(shape[115].y+9*shape[110].y)/10);
	faceComponent[0][2] = cvPoint((int)(9*shape[115].x+shape[110].x)/10, (int)((9*shape[115].y+shape[110].y)/10 + shape.GetWidth()/20));
	faceComponent[0][3] = cvPoint((int)(4*shape[88].x+shape[106].x)/5, (int)(4*shape[88].y+shape[106].y)/5);
	if(showRegionsOnGui)
		cvPolyLine(img, faceComponent, nPoints, 1, 1, blue);
	
	cvSetZero(tmpFimg1);
	cvSetZero(tmpFimg2);
	cvFillPoly(tmpFimg1, faceComponent, nPoints, 1, cvScalar(1,1,1));
	cvMultiplyAcc(tmpFimg1, sobelF1, tmpFimg2);
	cvMultiplyAcc(tmpFimg1, sobelF2, tmpFimg2);
	area = cvSum(tmpFimg1);
	sum = cvSum(tmpFimg2);
	features.at<float>(0,iFeature) = (float)(sum.val[0]/area.val[0]);
	++iFeature;

	// Chin
	/*nPoints[0] = 5;
	faceComponent[0][0] = cvPoint((int)shape[102].x, (int)shape[102].y);
	faceComponent[0][1] = cvPoint((int)shape[104].x, (int)shape[104].y);
	faceComponent[0][2] = cvPoint((int)shape[90].x, (int)shape[90].y);
	faceComponent[0][3] = cvPoint((int)shape[92].x, (int)shape[92].y);
	faceComponent[0][4] = cvPoint((int)shape[94].x, (int)shape[94].y);
	cvPolyLine(img, faceComponent, nPoints, 1, 1, blue);*/
	
	// Chin is not robust
	/*cvSetZero(tmpFimg1);
	cvSetZero(tmpFimg2);
	cvFillPoly(tmpFimg1, faceComponent, nPoints, 1, cvScalar(1,1,1));
	cvMultiplyAcc(tmpFimg1, sobelF1, tmpFimg2);
	cvMultiplyAcc(tmpFimg1, sobelF2, tmpFimg2);
	area = cvSum(tmpFimg1);
	sum = cvSum(tmpFimg2);
	features.at<float>(0,iFeature) = (float)(sum.val[0]/area.val[0]);
	++iFeature;*/

	CvPoint lipHeightL = cvPoint((int)shape[92].x, (int)shape[92].y);
	CvPoint lipHeightH = cvPoint((int)shape[84].x, (int)shape[84].y);
	if(showRegionsOnGui)
		cvLine(img, lipHeightL, lipHeightH, red, 2);
	double lipDistH = (cvSqrt((shape[92].x-shape[84].x)*(shape[92].x-shape[84].x)+ 
			(shape[92].y-shape[84].y)*(shape[92].y-shape[84].y))
		/ shape.GetWidth());
	features.at<float>(0,iFeature) = (float)lipDistH;
	++iFeature;

	CvPoint lipR = cvPoint((int)shape[80].x, (int)shape[80].y);
	CvPoint lipL = cvPoint((int)shape[88].x, (int)shape[88].y);
	if(showRegionsOnGui)
		cvLine(img, lipR, lipL, red, 2);
	double lipDistW = (cvSqrt((shape[80].x-shape[88].x)*(shape[80].x-shape[88].x)+ 
			(shape[80].y-shape[88].y)*(shape[80].y-shape[88].y))
		/ shape.GetWidth());
	features.at<float>(0,iFeature) = (float)lipDistW;
	++iFeature;

	cvFree(faceComponent);
	//cvShowImage(TRACKER_WINDOW_NAME, img);
}


void initialize(IplImage* img, int iFrame){
	if(iFrame >= 5) {
		for(int iFeature = 0; iFeature < N_FEATURES; iFeature++){
			featureScales.at<float>(0,iFeature) += features.at<float>(0,iFeature) / (FRAME_TO_START_DECISION-5);
		}
	}
	if(showProcessedGui) {	//kald�r�labilir sanki
		cvFlip(tmpGimg1, NULL, 1);
		cvShowImage(PROCESSED_WINDOW_NAME, tmpGimg1);
	}
	if(showTrackerGui) {
		CvScalar expColor = cvScalar(0,0,0);
		cvFlip(img, NULL, 1);

		if(iFrame%5 != 0)
			cvPutText(img, "Initializing... Do neutral expression please.", cvPoint(5, 12), &font, expColor);		
		
		cvShowImage(TRACKER_WINDOW_NAME, img);
	}
}

void normalizeFeatures(){
	for(int iFeature = 0; iFeature < N_FEATURES; iFeature++)
		featureScales.at<float>(0,iFeature) = 1 / featureScales.at<float>(0,iFeature);
}

void track(IplImage* img){
	for(int iFeature = 0; iFeature < N_FEATURES; iFeature++){
		features.at<float>(0,iFeature) = features.at<float>(0,iFeature) * featureScales.at<float>(0,iFeature);
	}

	if(PRINT_FEATURES) {		//kal�d�r�labilir sanki
		for(int i=0; i<N_FEATURES; i++)
			printf("%3.2f ", features.at<float>(0,i));
		printf("\n");
	}
	if(PRINT_FEATURE_SCALES) {	//kal�d�r�labilir sanki
		for(int i=0; i<N_FEATURES; i++)
			printf("%3.2f ", featureScales.at<float>(0,i));
		printf("\n");
	}

	if(showProcessedGui) {	//kald�r�labilir sanki
		cvFlip(sobelF1, NULL, 1);
		cvShowImage(PROCESSED_WINDOW_NAME, sobelF1);
		cvFlip(sobelF2, NULL, 1);
		cvShowImage("ada", sobelF2);
	}

	if(showTrackerGui) {
		CvScalar expColor = cvScalar(0,0,0);
		cvFlip(img, NULL, 1);

		int start=12, step=15, current;
			
		current = start;
		for(int i = 0; i < N_EXPRESSIONS; i++, current+=step) {
			cvPutText(img, EXP_NAMES[i], cvPoint(5, current), &font, expColor);
		}
		
		expressions = get_class_weights(features);
			
		current = start - 3; 
		for(int i = 0; i < N_EXPRESSIONS; i++, current+=step) {	//expressionlar�n yo�unlu�u g�stermek i�in
			cvLine(img, cvPoint(80, current), cvPoint((int)(80+expressions.at<double>(0,i)*50), current), expColor, 2);
		}

		current += step + step;
		if(showFeatures == 1){	//featurelar�n normalize de�erlerini bast�rmak i�in
			for(int i=0; i<N_FEATURES; i++){
				current += step;
				char buf[4];
				sprintf(buf, "%.2f", features.at<float>(0,i));
				cvPutText(img, buf, cvPoint(5, current), &font, expColor);
			}
		}
		
		cvShowImage(TRACKER_WINDOW_NAME, img);	//videoyu(resmi) g�ster
	}

}

int write_features(string filename, int j, int numFeats, IplImage* img){
	

	CvScalar expColor = cvScalar(0,0,0);
	cvFlip(img, NULL, 1);
	
	if(j%100 == 0){
		ofstream file(filename, ios::app);
		if(!file.is_open()){
			cout << "error opening file: " << filename << endl;
			exit(1);
		}
		
		cvPutText(img, "Features Taken:", cvPoint(5, 12), &font, expColor);
		
		for(int i=0; i<N_FEATURES; i++){
			char buf[4];
			sprintf(buf, "%.2f", features.at<float>(0,i) * featureScales.at<float>(0,i));
			printf("%3.2f, ", features.at<float>(0,i) * featureScales.at<float>(0,i));
			file << buf << " ";
		}
		file << endl;
		printf("\n");

		cvShowImage("taken photo", img);
		numFeats++;

		file.close();
	}
	if(j%100 != 0){
		cvPutText(img, "Current Expression: ", cvPoint(5, 12), &font, expColor);
		cvPutText(img, EXP_NAMES[numFeats/N_SAMPLES], cvPoint(5, 27), &font, expColor);
		cvPutText(img, "Features will be taken when the counter is 0.", cvPoint(5, 42), &font, expColor);
		char buf[12];
		sprintf(buf, "Counter: %2d", 100-(j%100));
		cvPutText(img, buf, cvPoint(5, 57), &font, expColor);
	}
	
	cvShowImage(TRACKER_WINDOW_NAME, img);
	return numFeats;
}


/*bool menu(IplImage* img, char *filename){
	bool create;
	CvScalar expColor = cvScalar(0,0,0);
	while(1){
		img = read_from_camera();
		cvFlip(img, NULL, 1);
		cvPutText(img, "If you would like to CREATE a new expression class press \"c\".", cvPoint(5, 12), &font, expColor);
		cvPutText(img, "If you would like to READ expression class from file press \"r\".", cvPoint(5, 27), &font, expColor);
		cvShowImage(TRACKER_WINDOW_NAME, img);

		char key = cvWaitKey(20);
		if(key == 99 || key == 67){ //8gerial 13enter 27esc 99c 67C 114r 82R
			create = true;
			break;
		}
		else if(key == 114 || key == 82){
			create = false;
			break;
		}
	}
	while(1){
		img = read_from_camera();
		cvFlip(img, NULL, 1);
		if(create) cvPutText(img, "Please enter the name of the file you want to create", cvPoint(5, 12), &font, expColor);
		else cvPutText(img, "Please enter the name of the file you want to read from", cvPoint(5, 12), &font, expColor);
		
		cvShowImage(TRACKER_WINDOW_NAME, img);
		char key = cvWaitKey(20);
		if(key!=-1){
			break;
		}
	}
	if(create == true){
		//img = read_from_camera();
		//cvFlip(img, NULL, 1);
		//cvShowImage(TRACKER_WINDOW_NAME, img);
	}
	else{
	
	}
	return create;
}	*/








int select(int selection, int range){
	int *deneme = new int[2];
	deneme[0] = cvWaitKey();
	if(deneme[0] == UP){
		selection = (selection+range-1)%range;
	}
	else if(deneme[0] == DOWN){
		selection = (selection+range+1)%range;
	}
	else if(deneme[0] == 13){
		selection += range;
	}
	return selection;
}

bool menu(IplImage* img, char *filename){
	bool create;
	CvScalar expColor = cvScalar(0,0,0);
	CvScalar mark = cvScalar(255,255,0);
	char key;
	IplImage* print = cvCreateImage( cvSize(img->width, img->height ), img->depth, img->nChannels );
	int selection, range, step;

	selection = 0;
	range = 2;
	step = 15;
	do{
		cvCopy(img, print);
		cvRectangle(print, cvPoint(0, 8+selection*step), cvPoint(319, 5+selection*step), mark, 12);
		cvPutText(print, "Read Expression Class From File", cvPoint(5, 12), &font, expColor);
		cvPutText(print, "Create New Expression Class", cvPoint(5, 27), &font, expColor);
		cvShowImage(TRACKER_WINDOW_NAME, print);
		selection = select(selection, range);
		if(selection>=range){
			selection -= range;
			if(selection == 0) create=false;
			else create=true;
			break;
		}
	}while(1);


	
	cvCopy(img, print);
	if(create){
		cvPutText(print, "Enter filename to create a new file.", cvPoint(5, 12), &font, expColor);
	}
	else{
		cvPutText(print, "Please select the file.", cvPoint(5, 12), &font, expColor);
	}
	cvRectangle(print, cvPoint(20, 12), cvPoint(50, 12), mark, 5);

	cvShowImage(TRACKER_WINDOW_NAME, print);
	
	do{
		key = cvWaitKey();
		if(create == true){
		}
	}while(1);

	return create;
}	



int main(int argc, char *argv[])
{
	asmfitting fit_asm;
	char* model_name = NULL;
	//char* shape_model_name = NULL;
	char* cascade_name= NULL;
	char* filename = NULL;
	char* shape_output_filename = NULL;
	char* pose_output_filename = NULL;
	char* features_output_filename = NULL;
	int use_camera = 0;
	int image_or_video = -1;
	int i;
	int n_iteration = 30;
	int maxComponents = 10;

	if(1 == argc)	usage_fit();
	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-') usage_fit();
		if(++i > argc) usage_fit();
		switch(argv[i-1][1])
		{
		case 'm':
			model_name = argv[i];
			break;
		case 's':
			//shape_model_name = argv[i];
			break;
		case 'h':
			cascade_name = argv[i];
			break;
		case 'i':
			if(image_or_video >= 0 || use_camera)
			{
				fprintf(stderr, "only process image/video/camera once\n");
				usage_fit();
			}
			filename = argv[i];
			image_or_video = 'i';
			break;
		case 'v':
			if(image_or_video >= 0 || use_camera)
			{
				fprintf(stderr, "only process image/video/camera once\n");
				usage_fit();
			}
			filename = argv[i];
			image_or_video = 'v';
			break;
		case 'c':
			if(image_or_video >= 0)
			{
				fprintf(stderr, "only process image/video/camera once\n");
				usage_fit();
			}
			use_camera = 1;
			i--;		//bu eklendi(bug d�zelsin diye)
			break;
		case 'H':
			usage_fit();
			break;
		case 'n':
			n_iteration = atoi(argv[i]);
			break;
		case 'S':
			shape_output_filename = argv[i];	//break???
		case 'P':
			pose_output_filename = argv[i];		//break???
		case 'F':
			features_output_filename = argv[i];
			break;
		case 'g':
			showTrackerGui = atoi(argv[i]);
			break;
		case 'e':
			showProcessedGui = atoi(argv[i]);
			break;
		case 'r':
			showRegionsOnGui = atoi(argv[i]);
			break;
		case 'f':
			showFeatures = atoi(argv[i]);
			break;
		case 't':
			showShape = atoi(argv[i]);
			break;
		case 'x':
			maxComponents = atoi(argv[i]);
			break;
		default:
			fprintf(stderr, "unknown options\n");
			usage_fit();
		}
	}

	if(fit_asm.Read(model_name) == false)
		return -1;
	
	if(init_detect_cascade(cascade_name) == false)
		return -1;

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, CV_AA);

	asm_shape meanShape = fit_asm.GetModel()->GetMeanShape();
	//CvMat* sampleMatPtr;
	//sampleMatPtr = (CvMat*)cvLoad(shape_model_name);
	//cv::Mat sampleMat(sampleMatPtr);
	int nPoints = meanShape.NPoints();
	//int nShapes = sampleMat.rows;
	//cv::PCA pca( sampleMat, cv::Mat(), CV_PCA_DATA_AS_ROW, maxComponents);


	// case 1: process image, we can process multi-person image alignment
	// also you can process single face alignment by coding like this
	// asm_shape detshape, shape;	
	// bool flag = face_detect.DetectCentralFace(detshape, image);
	// if(flag) asm_common::InitShapeFromDetBox(shape, detshape, 
	//		fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	// fit_asm.Fitting(shape, image, n_iteration);
	// shape.Write(stdout); //print result
	// for(int l = 0; l < shape.NPoints(); l++)
	//		printf("(%g, %g) ", shape[i].x, shape[i].y);
	if(image_or_video == 'i')
	{
		IplImage * image = cvLoadImage(filename, 1);
		if(image == 0)
		{
			fprintf(stderr, "Can not Open image %s\n", filename);
			exit(0);
		}

		double t = (double)cvGetTickCount();
		int nFaces;
		asm_shape *shapes = NULL, *detshapes = NULL;
		
		// step 1: detect face
		bool flag =detect_all_faces(&detshapes, nFaces, image);
		
		// step 2: initialize shape from detect box
		if(flag)
		{
			shapes = new asm_shape[nFaces];
			for(int i = 0; i < nFaces; i++)
			{
				InitShapeFromDetBox(shapes[i], detshapes[i], fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
			}
		}
		else 
		{
			fprintf(stderr, "This image does not contain any faces!\n");
			exit(0);
		}
		
		// step 3: image alignment fitting
		fit_asm.Fitting2(shapes, nFaces, image, n_iteration);
		
		t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
		if(PRINT_TIME_TICKS)
			printf("Time spent: %.2f millisec\n", t);
			
		// step 4: draw and show result in GUI
		for(int i = 0; i < nFaces; i++)
		{
			fit_asm.Draw(image, shapes[i]);
			char* shape_output_filename = filename;			//sadece 1 ki�i i�in kaydediyor bu �ekilde, �ok ki�i i�in istersek annotator kodunu de�i�tirmemiz de gerekir.
			shape_output_filename[strlen(shape_output_filename)-3]='p';
			shape_output_filename[strlen(shape_output_filename)-2]='t';
			shape_output_filename[strlen(shape_output_filename)-1]='s';
			save_shape(shapes[i], shape_output_filename);
			if(shape_output_filename != NULL) {
				save_shape(shapes[i], shape_output_filename);
			}
		}
		// Skipping pose parameters computation
		
		if(showTrackerGui) {
			cvNamedWindow("Fitting", 1);
			cvShowImage("Fitting", image);
			cvWaitKey(0);			
			cvReleaseImage(&image);		
		}

		// step 5: free resource
		delete[] shapes;
		free_shape_memeory(&detshapes);
	}

	// case 2: process video, here we assume that the video contains only one face,
	// if not, we process with the most central face
	else if(image_or_video == 'v')
	{
		int frame_count;
		asm_shape shape, detshape;
		bool flagFace = false, flagShape = false;
		int countFramesUnderThreshold = 0;
		IplImage* image; 
		/* NOTE: the image must not be released, it will be dellocated automatically
		by the class asm_cam_or_avi*/
		int j, key;
		FILE *fpShape, *fpPose, *fpFeatures;
	
		frame_count = open_video(filename);
		if(frame_count == -1)	return false;

		if(shape_output_filename != NULL) {
			fopen_s(&fpShape, shape_output_filename, "w");
			
			if (fpShape == NULL) {
				fprintf(stderr, "Can't open output file %s!\n", shape_output_filename);
				exit(1);
			}
		}

		if(pose_output_filename != NULL) {
			fopen_s(&fpPose, pose_output_filename, "w");
			
			if (fpPose == NULL) {
				fprintf(stderr, "Can't open output file %s!\n", shape_output_filename);
				exit(1);
			}
		}

		if(features_output_filename != NULL) {
			fopen_s(&fpFeatures, features_output_filename, "w");
			
			if (fpFeatures == NULL) {
				fprintf(stderr, "Can't open output file %s!\n", features_output_filename);
				exit(1);
			}
		}

		asm_shape shapes[N_SHAPES_FOR_FILTERING]; // Will be used for median filtering
		asm_shape shapeCopy, shapeAligned;
		//cv::Mat shapeParams, poseParams;
		//shapeParams.create(1, nPoints*2, sampleMat.type());
		//poseParams.create(1, maxComponents, sampleMat.type());

		for(j = 0; j < frame_count; j ++)
		{
			double t = (double)cvGetTickCount();
			if(PRINT_TIME_TICKS)
				printf("Tracking frame %04i: ", j);
			
			image = read_from_video(j);
			if(j == 0)
				setup_tracker(image);
			
			if(flagShape == false)
			{
				//Firstly, we detect face by using Viola_jones haarlike-detector
				flagFace = detect_one_face(detshape, image);
				
				//Secondly, we initialize shape from the detected box
				if(flagFace) 
				{	
					InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
				}
				else goto show;
			}
			
			//Thirdly, we do image alignment 
			flagShape = fit_asm.ASMSeqSearch(shape, image, j, true, n_iteration);
			shapeCopy = shape;
			if(j==0)
				for(int k=0; k < N_SHAPES_FOR_FILTERING; k++)
					shapes[k] = shapeCopy;
			else {
				for(int k=0; k < N_SHAPES_FOR_FILTERING-1; k++)
					shapes[k] = shapes[k+1];
				shapes[N_SHAPES_FOR_FILTERING-1] = shapeCopy;
			}
			shapeCopy = get_weighted_mean(shapes, N_SHAPES_FOR_FILTERING);

			//If success, we draw and show its result
			if(flagShape && showShape){ 
				fit_asm.Draw(image, shapeCopy);
			}
			if(!flagShape) { // We let the old tracker result to be seen for a few frames
				if(countFramesUnderThreshold == 0)
					flagShape = false;
				else
					flagShape = true;
				countFramesUnderThreshold = (countFramesUnderThreshold + 1) % MAX_FRAMES_UNDER_THRESHOLD;
			}

			// Compute Pose Parameters
			shapeAligned = shapeCopy;
			shapeAligned.AlignTo(meanShape);
			for(int k=0; k < shapeAligned.NPoints(); k++) {
//				shapeParams.at<float>(0,2*k) = shapeAligned[k].x;
//				shapeParams.at<float>(0,2*k+1) = shapeAligned[k].y;
			}
			//pca.project(shapeParams, poseParams);
			//pca.backProject(poseParams, shapeParams);

			for(int k=0; k < shape.NPoints(); k++) {
//				shapeAligned[k].x = shapeParams.at<float>(0,2*k);
//				shapeAligned[k].y = shapeParams.at<float>(0,2*k+1);
			}
			
			if (NORMALIZE_POSE_PARAMS)
				for(int k=0; k < maxComponents; k++) {
//					poseParams.at<float>(0,k) /= sqrt(pca.eigenvalues.at<float>(0,k));
				}

			extract_features_and_display(image, shapeCopy);
			if(j < FRAME_TO_START_DECISION) initialize(image, j);
			if(j == FRAME_TO_START_DECISION) normalizeFeatures();
			if(j >= FRAME_TO_START_DECISION) track(image);
//			if(shape_output_filename != NULL) write_vector(shapeParams, fpShape);
//			if(pose_output_filename != NULL) write_vector(poseParams, fpPose);
			if(features_output_filename != NULL) write_vector(features, fpFeatures);


			// fit_asm.Draw(image, shapeCopy);

show:
			// fit_asm.Draw(edges, shape);

			key = cv::waitKey(20);
			if(key > 0)
				break;
			
			t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
			if(PRINT_TIME_TICKS)
				printf("Time spent: %.2f millisec\n", t);
		}

		if(shape_output_filename != NULL) fclose(fpShape);
		if(pose_output_filename != NULL) fclose(fpPose);
		if(features_output_filename != NULL) fclose(fpFeatures);
		close_video();
	}

	// case 3: process camera
	else if(use_camera)
	{
		asm_shape shape, detshape;
		bool flagFace = false, flagShape = false;
		int countFramesUnderThreshold = 0;
		IplImage* image;  
		int j = 0, key;
				
		if(open_camera(0, CAM_WIDTH, CAM_HEIGHT) == false)
			return -1;
		
		asm_shape shapes[N_SHAPES_FOR_FILTERING]; // Will be used for median filtering		//NERDEEEE???
		asm_shape shapeCopy, shapeAligned;
		//cv::Mat shapeParams, poseParams;
		//shapeParams.create(1, nPoints*2, sampleMat.type());
		//poseParams.create(1, maxComponents, sampleMat.type());

		bool create = false;
		char * filenameArr = "classes\\deneme.txt";
		int numFeats = 0;
		if(CAM_WIDTH==640) image = cvLoadImage("images\\facexpressL.png");
		else image = cvLoadImage("images\\facexpress.png");
		//create = menu(image, filenameArr);
		string filename = "classes\\ismail.txt";
		cout << create << endl << filename << endl;
		while(1)
		{
			double t = (double)cvGetTickCount();	//DURSUN MU???
			if(PRINT_TIME_TICKS)
				printf("Tracking frame %04i: ", j);

			// NOTE: when the parameter is set 1, we can read from camera
			image = read_from_camera();
			if(j == 0)
				setup_tracker(image);
			
			if(flagShape == false)
			{
				//Firstly, we detect face by using Viola_jones haarlike-detector
				flagFace = detect_one_face(detshape, image);
				
				//Secondly, we initialize shape from the detected box
				if(flagFace)
					InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
				else 
					goto show2;
			}
			
			//Thirdly, we do image alignment 
			flagShape = fit_asm.ASMSeqSearch(shape, image, j, true, n_iteration);
			shapeCopy = shape;
			if(j==0)
				for(int k=0; k < N_SHAPES_FOR_FILTERING; k++)
					shapes[k] = shapeCopy;
			else {
				for(int k=0; k < N_SHAPES_FOR_FILTERING-1; k++)
					shapes[k] = shapes[k+1];
				shapes[N_SHAPES_FOR_FILTERING-1] = shapeCopy;
			}
			shapeCopy = get_weighted_mean(shapes, N_SHAPES_FOR_FILTERING);	//son frame'in etkisi fazla, ilk frame'in etkisi az

			//If success, we draw and show its result
			if(flagShape && showShape){ 
				fit_asm.Draw(image, shapeCopy);
			}
			if(!flagShape) {		//nap�yoruz anlamad�m false ya da true yapmak neyi de�i�tiriyo, d�ng� ba��nda de�eri de�i�iyo zaten
				if(countFramesUnderThreshold == 0)
					flagShape = false;
				else
					flagShape = true;
				countFramesUnderThreshold = (countFramesUnderThreshold + 1) % MAX_FRAMES_UNDER_THRESHOLD;
			}
			// Compute Pose Parameters
			shapeAligned = shapeCopy;
			shapeAligned.AlignTo(meanShape);	//niye align ediyoruz zaten align edilmi� de�il mi?
			for(int k=0; k < shapeAligned.NPoints(); k++) {
//				shapeParams.at<float>(0,2*k) = shapeAligned[k].x;
//				shapeParams.at<float>(0,2*k+1) = shapeAligned[k].y;
			}
			//pca.project(shapeParams, poseParams);
			//pca.backProject(poseParams, shapeParams);

			for(int k=0; k < shape.NPoints(); k++) {
//				shapeAligned[k].x = shapeParams.at<float>(0,2*k);
//				shapeAligned[k].y = shapeParams.at<float>(0,2*k+1);
			}
			
			if (NORMALIZE_POSE_PARAMS)
				for(int k=0; k < maxComponents; k++) {
					//poseParams.at<float>(0,k) /= sqrt(pca.eigenvalues.at<float>(0,k));
				}
				
			// fit_asm.Draw(image, shapeCopy);
			extract_features_and_display(image, shapeCopy);
			if(j < FRAME_TO_START_DECISION)			{	initialize(image, j);																	}
			else if(j == FRAME_TO_START_DECISION)	{	normalizeFeatures(); if(create) config(filename); else read_config_from_file(filename); } 
			else if(numFeats < N_POINTS && create)	{	numFeats = write_features(filename, j, numFeats, image);								}
			else if(numFeats == N_POINTS && create) {	read_config_from_file(filename); numFeats++;											}
			else									{	track(image);																			}

show2:
			// fit_asm.Draw(edges, shape);

			key = cvWaitKey(20);           // wait 20 ms
			if(key == 27)
				break;
			if(key == 's') {			//KALDIRALIM MI?
				for(int i=0; i<N_FEATURES; i++)
					printf("%3.2f, ", features.at<float>(0,i));
				printf("\n");
			}

			j++;
			t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
			if(PRINT_TIME_TICKS)
				printf("Time spent: %.2f millisec\n", t);

		}
		close_camera();
	}

	exit_tracker();

    return 0;
}



