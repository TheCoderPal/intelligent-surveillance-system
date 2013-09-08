#include <opencv2/opencv.hpp>
#include "surveillance.h"

using namespace std;

int main() {
	//VideoCapture cap("D:/Video For Backgroud/mv2_002.avi");
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		printf("error!\n");
		return -1;
	}
	/*
	CvCapture* capture = NULL;
	IplImage* _frame = NULL;

	capture = cvCaptureFromCAM(0);
	*/

	Mat frame;
	Mat print_screen;
	Mat foreground;
	Mat background;
	Mat marked;

	int reset = 0;

	//while (_frame = cvQueryFrame(capture), _frame != 0) {
	//	frame = Mat(_frame, false);
	while (cap.read(frame)) {

		runSurveillance(frame, background, foreground, print_screen, marked, reset);
		reset = 0;

		imshow("frame", frame);
		imshow("background", background);
		imshow("foreground", foreground);
		imshow("print_screen", print_screen);
		imshow("marked", marked);

		int key = cvWaitKey(30);
		switch (key) {
		case 's':
		case 'S':
			reset = 1;
			break;
		case 'q':
		case 'Q':
			//cvReleaseImage(&_frame);
			exit(0);
			break;
		default:
			break;
		}
	}
	return 0;
}
