#ifndef TEST0508_H
#define TEST0508_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<thread>
#include<math.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
#include <string> 
#include <direct.h>
#include <iostream>
#include "ippe.h"
using namespace cv;
using namespace dnn;
using namespace std;
class test
{
public:
	
	test();
	~test();
	void readImage();
	Mat letterbox_image(Mat& im, int w, int h);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

	void postprocessSaveConfidence(Mat& resizFrame, Mat& origFrame, const vector<Mat>& outs, vector<Point3f>& objectDetectionImage, vector<double>& objectDetectionConfidence, vector<Rect>& objectDetectionRect, double HighWidthHeightRatio, double LowWidthHeightRatio, double confThreshold, double nmsThreshold, int yoloWidth, int yoloHeight);

	void ObjectDetectionUpdateWithConfidence(vector<vector<Point3f>>& ObjectDetectionContinueMaxlenFrame, vector<vector<double>>& MaxlenFrameObjectDetectionConfidence, vector<Point3f>& newObjectDetection, vector<double>& newObjectDetectionConfidence, vector<vector<int>>& Track, int& trackCount, int maxlen);
	vector<String> getOutputsNames(const Net& net);
	
	void redNumberdetectionFilter();
	void objectDetectionGetTrackAndDrawTrack(Mat& src, vector<vector<Point3f>>& ObjectDetectionContinueMaxlenFrame, vector<vector<int>>Track, vector<vector<int>>& TrackNowSatifyLen, int SatifyLen);
	void DrawobjectDetectionTrack(Mat& src, vector<vector<Point3f>>& ObjectDetectionContinueMaxlenFrame, vector<vector<int>>& TrackNowSatifyLen, int SatifyLen);
	void distanceMatch(vector<Point3f > oldObjectDetection, vector<Point3f>newObjectDetection, vector<Point>& oldInNewdx, int x_Space, int y_Space);
	void get_offsets(vector<vector<Point3f>>saveObjectDetection, vector<int>& offsetAccumulate);
	void objectDetectionGetTrack(vector<vector<int>>Track, vector<vector<int>>& TrackNowSatifyLen, int SatifyLen);
	void redNumberdetectionFilterFromMovie();
	//结算定位部分
	double deg(double x);
	double rad(double x);
	void calGPSsaveConfidence(Mat& src, vector<vector<Point3f>>& ObjectDetectionContinueMaxlenFrame, vector<vector<double>>& MaxlenFrameObjectDetectionConfidence, vector<vector<int>>TrackNowSatifyLen, vector<Point3d>MaxlenFrameGPS, vector<Point3d>MaxlenFrameANGLE, vector<Point3d>& ObjectDetectionFilterFineGPS, vector<Point3d>& ObjectDetectionFilterExtraFeature);
	void locationEveryTarget(Point3d ObjectDetectionFilterCorrect, Point3d & ObjectDetectionGPS, double yaw, double lat, double lon, double height, double yitaX, double yitaY, double& brngImg, string & brngImgInformation);

	void computerThatLonLat(double lon, double lat, double brng, double dist, double& AfterLon, double& AfterLat);
	void GetIPPEAttitude(Mat srcImg, Rect src, Mat& rvec1, Mat& rvec2, Mat& tvec1, Mat& tvec2);
};
#endif


