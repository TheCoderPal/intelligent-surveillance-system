#ifndef _SURVEILLANCE_H
#define _SURVEILLANCE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <list>
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

using namespace std;
using namespace cv;

const int state_num = 6;
const int measure_num = 4;

struct Object {
	vector<Point> gray, white;
	Point point_bottom_left, point_top_right;

	Object() {
		point_bottom_left = Point(32767, 32767);
		point_top_right = Point(0, 0);
	}
	Object(const int &cx, const int &cy, const int &width, const int &height) {
		Point point_bottom_left(cx - width / 2, cy - height / 2);
		Point point_top_right(cx + width / 2, cy + height / 2);
	}
	float compare(const Object &obj) {
		Point overlap_bottom_left;
		Point overlap_top_right;
		overlap_bottom_left.x = MAX(point_bottom_left.x, obj.point_bottom_left.x);
		overlap_bottom_left.y = MAX(point_bottom_left.y, obj.point_bottom_left.y);
		overlap_top_right.x = MIN(point_top_right.x, obj.point_top_right.x);
		overlap_top_right.y = MIN(point_top_right.y, obj.point_top_right.y);

		int a = (point_top_right.x - point_bottom_left.x) * (point_top_right.y - point_bottom_left.y);
		int b = (obj.point_top_right.x - obj.point_bottom_left.x) * (obj.point_top_right.y - obj.point_bottom_left.y);
		int c = (overlap_top_right.x - overlap_bottom_left.x) - (overlap_top_right.y - overlap_bottom_left.y);

		return (float)c / (a + b - c);
	}
};

struct TrackedObject : public Object {
	KalmanFilter kalman_filter;
	Mat measurement;
	Mat state;

	int color_int;
	Scalar color_scalar;

	TrackedObject(const Object &obj) : 
		Object(obj),
		kalman_filter(state_num, measure_num, 0),
		measurement(measure_num, 1, CV_32F) {
			state = Mat::zeros(state_num, 1, CV_32F);
			state.at<float>(0) = (float)(obj.point_bottom_left.x + obj.point_top_right.x) / 2;
			state.at<float>(1) = (float)(obj.point_bottom_left.y + obj.point_top_right.y) / 2;
			state.at<float>(4) = (float)obj.point_top_right.x - obj.point_bottom_left.x + 1;
			state.at<float>(5) = (float)obj.point_top_right.y - obj.point_bottom_left.y + 1;

			kalman_filter.transitionMatrix = *(Mat_<float>(6, 6) <<
				1, 0, 1, 0, 0, 0,
				0, 1, 0, 1, 0, 0,
				0, 0, 1, 0, 0, 0,
				0, 0, 0, 1, 0, 0,
				0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 1);

			setIdentity(kalman_filter.measurementMatrix);
			setIdentity(kalman_filter.processNoiseCov, Scalar::all(1e-5));
			setIdentity(kalman_filter.measurementNoiseCov, Scalar::all(1e-1));
			setIdentity(kalman_filter.errorCovPost, Scalar::all(1));

			kalman_filter.statePost = state;

			color_scalar = Scalar((unsigned char)rand()%256, (unsigned char)rand()%256, (unsigned char)rand()%256);
			color_int = ((int)color_scalar[0] << 16) + ((int)color_scalar[1] << 8) + (int)color_scalar[2];
	}

	void correct(const Object &obj) {
		measurement.at<float>(0) = (float)(obj.point_bottom_left.x + obj.point_top_right.x) / 2;
		measurement.at<float>(1) = (float)(obj.point_bottom_left.y + obj.point_top_right.y) / 2;
		measurement.at<float>(2) = (float)obj.point_top_right.x - obj.point_bottom_left.x;
		measurement.at<float>(3) = (float)obj.point_top_right.y - obj.point_bottom_left.y;
	}
};

#if (defined WIN32 || defined _WIN32 || defined WINCE)
const Point dirs[8] = {Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(0, -1), Point(-1, -1), Point(0, -1), Point(1, -1)};
#else
const Point dirs[8] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {0, -1}, {-1, -1}, {0, -1}, {1, -1}};
#endif

void runSurveillance(const Mat &original_image, Mat &background_image, Mat &foreground_image, Mat &print_screen_image, Mat &marked_image, list<TrackedObject> &tracked_object_list, int reset, int track_start);
int getObjects(const Mat &original_foreground, const Mat &foreground, vector<Object> &object_list);
int classifyObjects(const Mat &frame, const vector<Object> &object_list, int* &object_label);
int trackObjects(const Mat &frame, const vector<Object> &object_list, list <TrackedObject> &tracked_object_list);
void printScreen(const Mat &frame);

#endif