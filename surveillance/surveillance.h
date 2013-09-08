#ifndef _SURVEILLANCE_H
#define _SURVEILLANCE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cstdio>
#include <ctime>

#if (defined WIN32 || defined _WIN32 || defined WINCE)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define LABEL_PEDESTRIAN	1
#define LABEL_VEHICLE		2
#define LABEL_OTHER			0
#define MAXSIZE 480

using namespace cv;

struct Object {
	vector<Point> gray, white;
	Point point_bottom_left, point_top_right;

	Object() {
		point_bottom_left = Point(32767, 32767);
		point_top_right = Point(0, 0);
	}
};

#ifdef _WIN32
const Point dirs[8] = {Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(0, -1), Point(-1, -1), Point(0, -1), Point(1, -1)};
#else
const Point dirs[8] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {0, -1}, {-1, -1}, {0, -1}, {1, -1}};
#endif

void runSurveillance(const Mat &original_image, Mat &background_image, Mat &foreground_image, Mat &print_screen_image, Mat &marked_image, int reset = 0);
int getObjects(const Mat &original_foreground, const Mat &foreground, vector<Object> &object_list);
int classifyObjects(const Mat &frame, const vector<Object> &object_list, int* &object_label);
void printScreen(const Mat &frame);

#endif