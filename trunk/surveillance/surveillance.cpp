#include "surveillance.h"

BackgroundSubtractorMOG2 mog;
int gAlarm = 0;
int timestamp_last = clock();
int timestamp_now;
int first_time = 1;
int tracking = 0;

void runSurveillance(const Mat &original_image, Mat &background_image, Mat &foreground_image, Mat &print_screen_image, Mat &marked_image, list<TrackedObject> &tracked_object_list, int reset = 0, int track_start = 0) {
	srand(time(NULL));
	Mat temp_image;
	Mat resized_image;

	if (original_image.cols > MAXSIZE || original_image.rows > MAXSIZE) {
		double ratio = original_image.cols > original_image.rows ? (double)MAXSIZE / original_image.cols : (double)MAXSIZE / original_image.rows;
		resize(original_image, resized_image, Size(0, 0), ratio, ratio);
	}
	marked_image = resized_image.clone();

	if (reset) {
		mog = BackgroundSubtractorMOG2();
		tracking = 0;
	}

	mog(resized_image, foreground_image, 5e-4);
	mog.getBackgroundImage(background_image);

	erode(foreground_image, temp_image, Mat());
	dilate(temp_image, temp_image, Mat());

	gAlarm = 0;
	vector<Object> object_list;
	getObjects(foreground_image, temp_image, object_list);
	//int* object_label = new int[object_list.size()];
	//gAlarm = classifyObjects(resized_image, object_list, object_label);

	gAlarm = trackObjects(resized_image, object_list, tracked_object_list);

	for (list<TrackedObject>::iterator i = tracked_object_list.begin(); i != tracked_object_list.end(); i++) {
		rectangle(marked_image, (*i).point_bottom_left, (*i).point_top_right, (*i).color_scalar);
	}

/*	Scalar rectangle_color;
	for (int i = 0; i < object_list.size(); i++) {
		switch (object_label[i]) {
		case LABEL_PEDESTRIAN:
			rectangle_color = Scalar(0, 255, 0);
			break;
		case LABEL_VEHICLE:
			rectangle_color = Scalar(255, 0, 0);
			break;
		default:
			rectangle_color = Scalar(255, 255, 255);
			break;
		}
		rectangle(marked_image, object_list[i].point_bottom_left, object_list[i].point_top_right, rectangle_color);
	}
*/
	//delete []object_label;

	timestamp_now = clock();
	if (!gAlarm)
		timestamp_last = timestamp_now;
	else {
		if (timestamp_now - timestamp_last >= 2000) {
			printScreen(original_image);
			print_screen_image = resized_image.clone();
			timestamp_last = timestamp_now;
		}
	}

	if (first_time || reset) {
		print_screen_image = resized_image.clone();
		first_time = 0;
	}

}

int getObjects(const Mat &original_foreground, const Mat &foreground, vector<Object> &object_list) {
	int num_objects = 0;
	int num_rows = foreground.rows;
	int num_cols = foreground.cols;
	bool* vst = new bool[num_rows * num_cols];
	memset(vst, 0, num_rows * num_cols);
	for (int i = 0; i < num_cols; i++)
		for (int j = 0; j < num_rows; j++)
			if (!vst[j * num_cols + i] && foreground.at<uchar>(j, i) > 130) {
				Object new_object;
				queue<Point> point_queue;
				point_queue.push(Point(i, j));
				vst[j * num_cols + i] = true;
				while (!point_queue.empty()) {
					Point point_now = point_queue.front();
					point_queue.pop();
					for (int d = 0; d < 8; d++) {
						Point point_next = point_now + dirs[d];
						if (point_next.x < 0 || point_next.x >= num_cols) continue;
						if (point_next.y < 0 || point_next.y >= num_rows) continue;
						if (vst[point_next.y * num_cols + point_next.x]) continue;
						vst[point_next.y * num_cols + point_next.x] = true;
						uchar gray_value = original_foreground.at<uchar>(point_next.y, point_next.x);
						if (gray_value > 0) {
							if (gray_value < 130)
								new_object.gray.push_back(point_next);
							else {
								new_object.white.push_back(point_next);
								new_object.point_bottom_left.x = MIN(new_object.point_bottom_left.x, point_next.x);
								new_object.point_bottom_left.y = MIN(new_object.point_bottom_left.y, point_next.y);
								new_object.point_top_right.x = MAX(new_object.point_top_right.x, point_next.x);
								new_object.point_top_right.y = MAX(new_object.point_top_right.y, point_next.y);
							}
							point_queue.push(point_next);
						}
					}
				}
				object_list.push_back(new_object);
				num_objects++;
			}
	delete []vst;
	return num_objects;
}

int trackObjects(const Mat &frame, const vector<Object> &object_list, list<TrackedObject> &tracked_object_list) {
	int gAlarm = 0;
	int *mark = new int[object_list.size()];
	memset(mark, 255, sizeof(int) * object_list.size());

	for (list<TrackedObject>::iterator i = tracked_object_list.begin(); i != tracked_object_list.end();) {
		double max_similarity = 0;
		int l = -1;

		Mat prediction = (*i).kalman_filter.predict();
		int cx = cvRound(prediction.at<float>(0));
		int cy = cvRound(prediction.at<float>(1));
		float vx = prediction.at<float>(2);
		float vy = prediction.at<float>(3);
		int width = cvRound(prediction.at<float>(4));
		int height = cvRound(prediction.at<float>(5));

		Object obj_prediction(cx, cy, width, height);

		for (int j = 0; j < (int)object_list.size(); j++) {
			float similarity = obj_prediction.compare(object_list[j]);
			if (similarity > 0.5 && similarity > max_similarity) {
				max_similarity = similarity;
				l = j;
			}
		}

		if (l == -1) {
			i = tracked_object_list.erase(i);
		}
		else {
			mark[l] = (*i).color_int;
			(*i).correct(object_list[l]);
			if ((float)((*i).point_top_right.y - (*i).point_bottom_left.y) / ((*i).point_top_right.x - (*i).point_bottom_left.x))
				gAlarm = 1;
			i++;
		}
	}

	for (int i = 0; i < (int)object_list.size(); i++)
		if (mark[i] < 0) {
			tracked_object_list.push_back(object_list[i]);
		}
	delete []mark;

	return gAlarm;
}

int classifyObjects(const Mat &frame, const vector<Object> &object_list, int* &object_label) {
	memset(object_label, LABEL_OTHER, sizeof(int) * object_list.size());
	int gAlarm = 0;
	for (int i = 0; i < (int)object_list.size(); i++) {
		const Object &temp_object = object_list[i];
		/*if (temp_object.white.size() < 50) {
			object_label[i] = LABEL_OTHER;
			continue;
		}*/
		if (temp_object.point_bottom_left.x < 3 || temp_object.point_bottom_left.y < 3
			|| temp_object.point_top_right.x > frame.cols - 3 || temp_object.point_top_right.y > frame.rows - 3) {
			object_label[i] = LABEL_OTHER;
			continue;
		}
		int object_height = temp_object.point_top_right.y - temp_object.point_bottom_left.y;
		int object_width = temp_object.point_top_right.x - temp_object.point_bottom_left.x;
		if ((double)object_height / object_width > 2) object_label[i] = LABEL_PEDESTRIAN;
		if ((double)object_height / object_width < 1) object_label[i] = LABEL_VEHICLE;
		if (object_label[i] == LABEL_PEDESTRIAN) gAlarm = 1;
	}
	return gAlarm;
}

void printScreen(const Mat &frame) {
#ifdef _WIN32
	_mkdir("PrtSc");
#else
	mkdir("PrtSc", 0777);
#endif
	time_t t = time(0); 
	char pic_name[255]; 
	strftime(pic_name, sizeof(pic_name), "PrtSc/%y%m%d%H%M%S.jpg",localtime(&t)); 
	imwrite(pic_name, frame);
}