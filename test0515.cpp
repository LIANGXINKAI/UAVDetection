#include "test0515.h"
#include "ippe.h"
#include<conio.h>
#include<time.h>
String g_classesFile = "C:\\Resource\\target.names";
String g_modelConfiguration = "C:\\Resource\\tiny.cfg";
String g_modelWeights = "C:\\Resource\\tiny_10000.weights";
String g_video = "C:\\Resource\\1.mp4";
typedef Vec<int16_t, 2> Vec2myi;
//String g_video = "rtsp://admin:admin@192.168.2.108:80/cam/realmonitor?channel=1&subtype=0&proto=Private3";
struct ImageData {
	int timeElipse;
	int ImageNum;
	Mat srcOrig;
};
struct aircraft_data {
	int num;
	double lon;
	double lat;
	int object_property;
	double cc;
	int appear_counts;
};
vector<aircraft_data> outputDetectionResult;

deque<ImageData> testImage;
void test::readImage() {
	// Open  a camera stream.
	VideoCapture cap;
	cap.open(g_video, CAP_FFMPEG); // camera id --liang
	Mat Image;
	int SampleSize = 10;
	if (!cap.isOpened()) {
		cout << "No camera is opened !!!" << endl;
		cv::waitKey(3000);
		return;
	}
	int resultImageNum = 0;
	bool detectionOpen = true;
	while ((detectionOpen) && (cap.isOpened())) {
		auto start = getTickCount();
		resultImageNum = resultImageNum + 1;
		cap >> Image;
		if (Image.empty()) {
			cout << "No PICTURES Done processing !!!" << endl;
			cv::waitKey(3000);
			break;
		}
		imshow("srcImg", Image);
		waitKey(1);
		cout << "时间： " << start << "ms" << endl;

		ImageData ImageDataTmp;
		ImageDataTmp.ImageNum = resultImageNum;
		ImageDataTmp.timeElipse = start;
		ImageDataTmp.srcOrig = Image.clone();
		if ((resultImageNum%SampleSize==0)&& (testImage.size()<1000))
		 testImage.push_back(ImageDataTmp);
	}
	return;
}
//全局变量
test::test()
{
}


test::~test()
{
}
double test::deg(double x) {
	return x * 180 / 3.1415926;
}
double test::rad(double d) {
	return d * 3.1415926 / 180.0;
}
template<typename T>
inline void ListRemoveAtIdx(vector<T>& list, size_t idx)
{
	if (idx < list.size())
	{
		list[idx] = list.back();
	}
	list.pop_back();
}
Mat test::letterbox_image(Mat& origFrame, int yoloWidth, int yoloHeight)
{
	//resize后明确其位置
	int new_w = origFrame.cols;
	int new_h = origFrame.rows;
	float resizeScale;
	//在保证图像宽高比不变的情况下,计算放缩后的宽高
	if (((float)yoloWidth / origFrame.cols) < ((float)yoloHeight / origFrame.rows)) {
		//这个说明高度比例大于宽度比例,所以new_h要重新设置
		new_w = yoloWidth;
		new_h = (origFrame.rows * yoloWidth) / origFrame.cols;
		resizeScale = float(origFrame.cols) / new_w;
	}
	else {
		new_h = yoloHeight;
		new_w = (origFrame.cols * yoloHeight) / origFrame.rows;
		resizeScale = float(origFrame.cols) / new_w;
	}

	Mat resized;
	resize(origFrame, resized, Size(new_w, new_h), INTER_CUBIC);
	Mat boxed = Mat::Mat(yoloWidth, yoloHeight, CV_8UC3, Scalar(128, 128, 128));
	resized.copyTo(boxed(Rect((yoloWidth - new_w) / 2, (yoloHeight - new_h) / 2, new_w, new_h)));
	return boxed; //返回的图像尺寸为需要的(w,h)
}
void test::postprocessSaveConfidence(Mat & resizFrame, Mat & origFrame, const vector<Mat> & outs, vector<Point3f> & objectDetectionImage, vector<double> & objectDetectionConfidence, vector<Rect> & objectDetectionRect, double HighWidthHeightRatio, double LowWidthHeightRatio, double confThreshold, double nmsThreshold, int yoloWidth, int yoloHeight)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	//resize后明确其位置
	int new_w = origFrame.cols;
	int new_h = origFrame.rows;
	float resizeScale;
	//在保证图像宽高比不变的情况下,计算放缩后的宽高
	if (((float)yoloWidth / origFrame.cols) < ((float)yoloHeight / origFrame.rows)) {
		//这个说明高度比例大于宽度比例,所以new_h要重新设置
		new_w = yoloWidth;
		new_h = (origFrame.rows * yoloWidth) / origFrame.cols;
		resizeScale = float(origFrame.cols) / new_w;
	}
	else {
		new_h = yoloHeight;
		new_w = (origFrame.cols * yoloHeight) / origFrame.rows;
		resizeScale = float(origFrame.cols) / new_w;
	}

	int colsTmp = resizeScale * yoloWidth;
	int rowsTmp = resizeScale * yoloHeight;
	int resizeZeroPointX = (yoloWidth - 1) / 2;
	int resizeZeroPointY = (yoloHeight - 1) / 2;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.8)
			{
				int centerX = (int)(data[0] * resizFrame.cols);
				int centerY = (int)(data[1] * resizFrame.rows);
				int width = (int)(data[2] * resizFrame.cols);
				int height = (int)(data[3] * resizFrame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	//0519:只保留图像中完整目标！
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		int x_tmp = resizeScale * (box.x - resizeZeroPointX) + float(origFrame.cols) / 2;
		int y_tmp = resizeScale * (box.y - resizeZeroPointY) + float(origFrame.rows) / 2;
		int width_tmp = box.width * resizeScale;
		int height_tmp = box.height * resizeScale;
		Rect boxResize = Rect(x_tmp, y_tmp, width_tmp, height_tmp);

		//	0519:判断边界的条件！
		//由于输出的边框没法固定，因此不好判断其边界的阈值，暂时不加这一块内容
		int ImgSelfsize = 140;
		int xMin_Space = 0-(x_tmp + width_tmp/2  - ImgSelfsize);
		int yMin_Space = 0-(y_tmp + height_tmp/2 - ImgSelfsize);
		int xMax_Space = x_tmp + width_tmp / 2 + ImgSelfsize - origFrame.cols;
		int yMax_Space = y_tmp + height_tmp / 2 + ImgSelfsize - origFrame.rows;
		int borderCompare = 100;
		if ((double(width_tmp) / height_tmp > LowWidthHeightRatio) && (double(width_tmp) / height_tmp < HighWidthHeightRatio))
		{
			if ((xMin_Space > borderCompare) || (xMax_Space > borderCompare) )
				continue;
			drawPred(classIds[idx], confidences[idx], boxes[idx].x, boxes[idx].y, boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height, resizFrame);
			drawPred(classIds[idx], confidences[idx], x_tmp, y_tmp, x_tmp + width_tmp, y_tmp + height_tmp, origFrame);
			objectDetectionImage.push_back(Point3f(x_tmp + width_tmp / 2, y_tmp + height_tmp / 2, classIds[idx]));
			objectDetectionRect.push_back(boxResize);
			objectDetectionConfidence.push_back(confidences[idx]);

		}
	}
}

void test::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat & frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);

	label = to_string(classId) + ":" + label;
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}
vector<String> test::getOutputsNames(const Net & net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
void test::ObjectDetectionUpdateWithConfidence(vector<vector<Point3f>> & ObjectDetectionContinueMaxlenFrame, vector<vector<double>> & MaxlenFrameObjectDetectionConfidence, vector<Point3f> & newObjectDetection, vector<double> & newObjectDetectionConfidence, vector<vector<int>> & Track, int& trackCount, int maxlen)
{
	//cout << "开始更新track数据" << endl;
	int removePointSize = ObjectDetectionContinueMaxlenFrame[0].size();
	ObjectDetectionContinueMaxlenFrame.erase(ObjectDetectionContinueMaxlenFrame.begin());
	ObjectDetectionContinueMaxlenFrame.push_back(newObjectDetection);
	MaxlenFrameObjectDetectionConfidence.erase(MaxlenFrameObjectDetectionConfidence.begin());
	MaxlenFrameObjectDetectionConfidence.push_back(newObjectDetectionConfidence);
	bool debug = false;
	if (debug) {
		cout << "ObjectDetectionContinueMaxlenFrame校正之后结果:" << endl;
		for (auto i : ObjectDetectionContinueMaxlenFrame) {
			cout << "ObjectDetectionContinueMaxlenFrame校正之后每个大小:" << i.size() << endl;
			for (auto j : i) {
				cout << "ObjectDetectionContinueMaxlenFrame校正之后结果 " << j << " " << endl;
			}
		}
	}
	for (auto& i : Track)
	{
		i.erase(i.begin() + 2);
		for (int k = 2; k < i.size(); k++) {
			i[k] = i[k] - removePointSize;
			if (i[k] < -1)
				i[k] = -1;
		}
		i.push_back(-1);
	}
	if (debug) {
		cout << "Track 校正之后结果:" << endl;
		for (auto i : Track) {
			for (auto j : i) {
				cout << "Track 校正之后结果 " << j << " " << endl;
			}
		}
	}
	vector<int>offsetAccumulate;
	get_offsets(ObjectDetectionContinueMaxlenFrame, offsetAccumulate);

	if (debug) {
		cout << "offsetAccumulate:" << endl;
		for (auto i : offsetAccumulate) {
			cout << "offsetAccumulate结果 " << i << " " << endl;
		}
	}
	// 根据过程中的x,y方向上的冗余
	vector<Point> oldInNewdx;
	distanceMatch(ObjectDetectionContinueMaxlenFrame[ObjectDetectionContinueMaxlenFrame.size() - 2], newObjectDetection, oldInNewdx, 50, 200);
	vector<Point> secondOldInNewdx;
	distanceMatch(ObjectDetectionContinueMaxlenFrame[ObjectDetectionContinueMaxlenFrame.size() - 3], newObjectDetection, secondOldInNewdx, 50, 400);
	vector<Point> thirdOldInNewdx;
	distanceMatch(ObjectDetectionContinueMaxlenFrame[ObjectDetectionContinueMaxlenFrame.size() - 4], newObjectDetection, thirdOldInNewdx, 50, 600);
	for (auto idx : oldInNewdx) {
		int id1 = int(idx.x) + offsetAccumulate[offsetAccumulate.size() - 2];
		int id2 = int(idx.y) + offsetAccumulate[offsetAccumulate.size() - 1];
		for (auto& lastIdx : Track)
		{
			if (id1 == lastIdx[lastIdx.size() - 2])
			{

				lastIdx[lastIdx.size() - 1] = id2;
				break;
			}
		}
	}

	for (auto idx : secondOldInNewdx) {
		int id1 = int(idx.x) + offsetAccumulate[offsetAccumulate.size() - 3];
		int id2 = int(idx.y) + offsetAccumulate[offsetAccumulate.size() - 1];
		for (auto& lastIdx : Track)
		{
			if ((id1 == lastIdx[lastIdx.size() - 3]) && (lastIdx[lastIdx.size() - 1] == -1))
			{
				lastIdx[lastIdx.size() - 1] = id2;
				break;
			}
		}
	}

	for (auto idx : thirdOldInNewdx) {
		int id1 = int(idx.x) + offsetAccumulate[offsetAccumulate.size() - 4];
		int id2 = int(idx.y) + offsetAccumulate[offsetAccumulate.size() - 1];
		for (auto& lastIdx : Track)
		{
			if ((id1 == lastIdx[lastIdx.size() - 4]) && (lastIdx[lastIdx.size() - 1] == -1))
			{
				lastIdx[lastIdx.size() - 1] = id2;
				break;
			}
		}
	}



	vector<int>newPointIdx;
	for (auto size = 0; size < newObjectDetection.size(); size++) {
		int matchTheOld = 0;
		for (auto idx : oldInNewdx) {
			if (size == idx.y) {
				matchTheOld = 1;
				break;
			}
		}
		if (matchTheOld < 1)
		{
			newPointIdx.push_back(size + offsetAccumulate[offsetAccumulate.size() - 1]);
		}
	}
	//将没有跟踪上的对应目标提交到跟踪序列中
	vector<vector<int>>new_tracks;
	for (auto i = 0; i < newPointIdx.size(); i++) {
		int new_trackslen = maxlen + 2;
		vector<int>new_tracksTmp(new_trackslen, -1);
		new_tracksTmp[0] = i + trackCount;
		new_tracksTmp[new_tracksTmp.size() - 1] = newPointIdx[i];
		new_tracks.push_back(new_tracksTmp);
	}
	Track.insert(Track.end(), new_tracks.begin(), new_tracks.end());
	trackCount = trackCount + new_tracks.size();

	for (auto i = 0; i < Track.size();) {
		bool Trackremove = true;
		// 从第三个数开始进行
		for (auto j = Track[i].begin() + 2; j < Track[i].end(); j++) {
			if (*j >= 0)
			{
				Trackremove = false;
				break;
			}
		}
		if (Trackremove)
		{
			Track.erase(Track.begin() + i);
		}
		if (!Trackremove)
		{
			i++;
		}
	}
}

void test::distanceMatch(vector<Point3f > oldObjectDetection, vector<Point3f>newObjectDetection, vector<Point> & oldInNewdx, int x_Space, int y_Space) {
	vector<int> old2newObjectDetection;
	bool debug = 0;
	for (auto i = 0; i < oldObjectDetection.size(); i++)
	{
		double mindistance = 2000000;
		int mindistanceIdx_oneOldinWhichNew = -1;
		for (auto j = 0; j < newObjectDetection.size(); j++)
		{
			if (oldObjectDetection[i].z == newObjectDetection[j].z)
			{
				//后续加无人机运动的阈值轨迹
				if ((abs(oldObjectDetection[i].x - newObjectDetection[j].x) < x_Space)  && (abs(oldObjectDetection[i].y - newObjectDetection[j].y) < y_Space)) {
					double mindistanceTmp = sqrt(pow((oldObjectDetection[i].x - newObjectDetection[j].x), 2) + pow((oldObjectDetection[i].y - newObjectDetection[j].y), 2));
					if (mindistanceTmp < mindistance)
					{
						mindistance = mindistanceTmp;
						mindistanceIdx_oneOldinWhichNew = j;
					}
				}

			}
		}
		old2newObjectDetection.push_back(mindistanceIdx_oneOldinWhichNew);
	}
	//cout << "old2newObjectDetection" << endl;
	for (auto i = 0; i < old2newObjectDetection.size(); i++)
	{

		//cout << "old2newObjectDetection:" << i << endl;
		if (old2newObjectDetection[i] != -1) {
			double mindistance = 2000000;
			int mindistanceIdx_oneNewinWhichOld = -1;
			for (auto j = 0; j < oldObjectDetection.size(); j++)
			{
				//cout << "oldObjectDetection:" << i << endl;
				if (newObjectDetection[old2newObjectDetection[i]].z == oldObjectDetection[j].z)
				{
					if ((abs(oldObjectDetection[j].x - newObjectDetection[old2newObjectDetection[i]].x) < x_Space)  && (abs(oldObjectDetection[j].y - newObjectDetection[old2newObjectDetection[i]].y) < y_Space)) {
						double mindistanceTmp = sqrt(pow((oldObjectDetection[j].x - newObjectDetection[old2newObjectDetection[i]].x), 2) + pow((oldObjectDetection[j].y - newObjectDetection[old2newObjectDetection[i]].y), 2));
						if (mindistanceTmp < mindistance)
						{
							mindistance = mindistanceTmp;
							mindistanceIdx_oneNewinWhichOld = j;
						}
					}
				}
			}
			if (mindistanceIdx_oneNewinWhichOld == i)
				oldInNewdx.push_back(Point(mindistanceIdx_oneNewinWhichOld, old2newObjectDetection[i]));
		}
	}
}
void test::get_offsets(vector<vector<Point3f>>saveObjectDetection, vector<int> & offsetAccumulate)
{
	vector<int>offsets;
	for (int i = 0; i < saveObjectDetection.size(); i++)
	{
		offsets.push_back(saveObjectDetection[i].size());
	}
	bool debug = 0;
	if (debug) {
		cout << "offsets:" << endl;
		for (auto i : offsets) {
			cout << "offsets结果 " << i << " " << endl;
		}
	}
	offsetAccumulate.push_back(0);
	for (auto it = offsets.begin() + 1; it != offsets.end(); it++)
	{
		offsetAccumulate.push_back(std::accumulate(offsets.begin(), it, 0));
	}

	if (debug) {
		cout << "offsetAccumulate:" << endl;
		for (auto i : offsetAccumulate) {
			cout << "offsetAccumulate结果 " << i << " " << endl;
		}
	}

}
void test::redNumberdetectionFilter()
{

	std::cout << "开始！！！" << endl;
	// Initialize the parameters
	float confThreshold = 0.85; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;  // Width of network's input image
	int inpHeight = 416; // Height of network's input image
	//0520长宽比的阈值设置修改
	double HighWidthHeightRatio = 5;
	double LowWidthHeightRatio = 0.2;
	vector<string> classes;


	// Load names of classes
	string classesFile = g_classesFile; //absolute path--liang
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);


	// Give the configuration and weight files for the model
	String modelConfiguration = g_modelConfiguration; //absolute path --liang
	String modelWeights = g_modelWeights; //absolute path --liang


	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);


	//save result
	int resultImageNum = 0;
	int processSampling = 1;//真正用的时候用1!!!!!
	int resultSampling = 1;
	vector<Point3f> ObjectDetectionTmp;
	string resultBackName = "_0507Result.jpg";

	vector<Point3d> ObjectDetectionAllGPS;
	vector<Point3d> ObjectDetectionAllExtraFeature;

	//创建文件夹
	string command;
	command = "mkdir -p c:\\testDetection\\GPS";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\detection";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\TrackResult";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\output";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\outputALL";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\TIMEGPS";
	system(command.c_str());
	//0520
	command = "mkdir -p c:\\testDetection\\Crop";
	system(command.c_str());



	//// Open  a camera stream.
	//VideoCapture cap;
	////cap.set(CAP_PROP_BUFFERSIZE, 1);
	//cap.open(g_video,CAP_FFMPEG); // camera id --liang

	Mat Image;

	//mid variable
	Mat ImageWaitDetection, blob;

	//mid Track variable
	int maxlen = 10;//保留maxlen帧的识别结果

	vector<vector<Point3f>> MaxlenFrameObjectDetection;//初始化保存检测识别目标的位置
	vector<vector<double>> MaxlenFrameObjectDetectionConfidence;//初始化保存检测识别目标的置信度

	vector<Point3d>MaxlenFrameGPS;
	vector<Point3d>MaxlenFrameANGLE;
	int trackCount = 0;
	vector<Rect>oldObjectDetectionRect;
	vector<vector<int>>Track;//前两列分别为track的序号与track的分数，后面为依次帧的点序号

	////detection loop;
	//if (!cap.isOpened()) {
	//	cout << "No camera is opened !!!" << endl;
	//	_cprintf("No PICTURES Done processing !!!");
	//	waitKey(3000);
	//	return;
	//}
	//AfxBeginThread(this->readImage, NULL);
	bool detectionOpen = true;
	while (detectionOpen) {
		if (testImage.size() < 1)
			continue;
		ImageData tmp = testImage[0];
		Image = tmp.srcOrig.clone();
		imshow("test", Image);
		waitKey(1);
		resultImageNum = tmp.ImageNum;
		testImage.pop_front();
		auto start=tmp.timeElipse; 

		int delay = 200;//ms
		auto now = start;
		
		if (resultImageNum % processSampling != 0)
			continue;
		
		//经纬度数据和欧拉角数据 
		//与int可以相互转化么？
		//string writeOnlyResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
		//imwrite(writeOnlyResultImage, Image);
		//continue;
		//外界纳入量
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//获得当前时间
		//和数组做比对 找最近的时间节点的数
		//增加图片延时量
		string timeGpsTxt = "c:\\testDetection\\TIMEGPS\\timeGps" + to_string(resultImageNum) + ".txt";
		ofstream timeGpsTxtPose(timeGpsTxt);
		int min = 9999999;
		//AlgorithmData b;
		//b.la = 0;
		//auto t =now-delay;//真正拍照时间为当前时间-时延
		//deque<AlgorithmData> v1(g_dataList);
		//for (size_t i = 0; i < v1.size(); i++)
		//{
		//	auto a = v1[i];
		//	timeGpsTxtPose <<std::setprecision(15)<<"当前计时："<<now<<",时延："<<delay<<",时间:"<<t<< ",时间窗时间t:" << a.t<<",经度:"<<a.lo<<",纬度："<<a.la<<endl;
		//	CString mm1;
		//	mm1.Format(_T("当前的时间窗差值：%d数据长度%d,最新数据为航向%f，roll为%f,pitch为%f,lo：%f,la：%f,高度：%d,时间窗最小值：%d"), abs((int)(a.t - t)),v1.size(), a.direction, a.roll, a.pitch, a.lo, a.la, a.height, min);
		//	//_cprintf("\n%S\n", mm1);
		//	if (abs((int)(a.t - t)) < min) {
		//		b = a;
		//		min = abs((int)(a.t - t));
		//	}
		//	else {
		//		_cprintf("\na.t:%d,t:%d\n",a.t,t);
		//	}
		//}
		//timeGpsTxtPose << std::setprecision(15)<<"最终匹配："<< "当前计时：" << now << ",时延：" << delay << ",时间:" << t << ",时间窗时间t:" << b.t << ",经度:" << b.lo << ",纬度：" << b.la << endl;
		//double lat = b.la;
		//double lon = b.lo;
		//double height = (double)b.height / 100;

		//double roll = b.roll;
		//double yaw = b.direction;
		//double pitch = b.pitch;

		double lat = 0;
		double lon = 0;
		double height = 28;

		double roll = 0;
		double yaw = 0;
		double pitch = 0;


		ImageWaitDetection = Image.clone();
		Mat ImageWaitDetectionResize = letterbox_image(ImageWaitDetection, inpWidth, inpHeight);
		// Create a 4D blob from a frame.

		blobFromImage(ImageWaitDetectionResize, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		//Sets the input to the network
		net.setInput(blob);

		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		vector<Point3f>objectDetectionImage;//直接将输出赋予objectDetectionImage变量，z存类别
		vector<double>objectDetectionConfidence;//保存识别结果的置信度

		vector<Rect>objectDetectionRect;
		postprocessSaveConfidence(ImageWaitDetectionResize, ImageWaitDetection, outs, objectDetectionImage, objectDetectionConfidence, objectDetectionRect, HighWidthHeightRatio, LowWidthHeightRatio, confThreshold, nmsThreshold, inpWidth, inpHeight);
		string trackPoseImgResultTxt = "c:\\testDetection\\GPS\\resultTrackPose_" + to_string(resultImageNum) + ".txt";

		ofstream OutFiletrackPose(trackPoseImgResultTxt);

		//05.13增加基于已知尺寸大小的位姿估计
		Mat srcAttitude = Image.clone();
		for (auto i : objectDetectionRect)
		{
			Mat rvec1, rvec2, tvec1, tvec2;
			GetIPPEAttitude(srcAttitude, i, rvec1, rvec2, tvec1, tvec2);
			OutFiletrackPose << "位置: " << endl;
			OutFiletrackPose << tvec1 << endl;
			OutFiletrackPose << "姿态: " << endl;
			OutFiletrackPose << rvec1 << endl;
		}
		Mat detectedFrameResult;
		ImageWaitDetection.convertTo(detectedFrameResult, CV_8U);
		//没有识别结果的依然输出,且在图像上明显标记
		if (objectDetectionImage.size() < 1) {
			std::cout << "当前帧没有找到目标" << endl;
			string resultTxt = "NO Object Detection";
			string resultTimeTxt = to_string(now)+"ms";

			putText(detectedFrameResult, resultTxt, Point(detectedFrameResult.rows / 2, detectedFrameResult.cols / 2), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);
			putText(detectedFrameResult, resultTimeTxt, Point(detectedFrameResult.rows / 10, detectedFrameResult.cols / 10), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);
			string writeResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
			imwrite(writeResultImage, detectedFrameResult);
			continue;
		}
		Scalar drawcolor;
		for (auto i : objectDetectionImage)
		{
			//std::cout << "objectDetectionImage：" << i << endl;
			if (i.z == 0)
				drawcolor = Scalar(255, 0, 0);
			if (i.z == 1)
				drawcolor = Scalar(0, 255, 0);
			if (i.z == 2)
				drawcolor = Scalar(0, 255, 255);
			string objectDetectionImagelabel = format("%d ", i.z);
			circle(detectedFrameResult, Point(i.x, i.y), 5, drawcolor, -1);
		}
		namedWindow("detectedFrameResult", WINDOW_NORMAL);
		imshow("detectedFrameResult", detectedFrameResult);
		waitKey(1);
		if (resultImageNum % resultSampling == 0)
		{
			string writeResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
			string resultTimeTxt = to_string(now) + "ms";
			putText(detectedFrameResult, resultTimeTxt, Point(detectedFrameResult.rows / 10, detectedFrameResult.cols / 10), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);
			imwrite(writeResultImage, detectedFrameResult);
		}

		Mat cropImage = Image.clone();

		if (objectDetectionImage.size() < 1) {
			std::cout << "没有找到目标" << endl;
			continue;
		}
		//显示识别出来的区域内容
		for (auto i = 0; i < objectDetectionRect.size(); i++)
		{
			int tmpsize = std::min(objectDetectionRect[i].width, objectDetectionRect[i].height);
			int x_tmp = max(0, int(objectDetectionRect[i].x+ objectDetectionRect[i].width/2- tmpsize/2));
			int y_tmp = max(0, int(objectDetectionRect[i].y+ objectDetectionRect[i].height / 2 - tmpsize / 2));

			int width_tmp = std::min(tmpsize, Image.cols - tmpsize);
			int height_tmp = std::min(tmpsize, Image.rows - tmpsize);
			Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
			Mat croptmp = cropImage(tmp);
			string cropname = "c:\\testDetection\\Crop\\Crop_" + to_string(resultImageNum) + "__" + to_string(i) + resultBackName;
			imwrite(cropname, croptmp);
			cout << "width_tmp: " << width_tmp << endl;
			cout << "height_tmp: " << height_tmp << endl;

			imshow("分割出来的目标图像", croptmp);
			waitKey(1);
		}

		//开始迭代更新,输出更新后的结果
		// 如果之前没有数值对Track相关数据进行初始化
		bool debug = 0; //debug 部分是调试track的过程
		if (objectDetectionImage.size() >= 1)
		{
			if (MaxlenFrameObjectDetection.size() < 1)
			{
				for (auto i = 0; i < maxlen; i++) {
					vector<Point3f>MaxlenFrame_tmp;
					vector<double>MaxlenFrameConfidence_tmp;

					MaxlenFrameObjectDetection.push_back(MaxlenFrame_tmp);
					MaxlenFrameObjectDetectionConfidence.push_back(MaxlenFrameConfidence_tmp);
				}
				MaxlenFrameObjectDetection.erase(MaxlenFrameObjectDetection.begin());
				MaxlenFrameObjectDetection.push_back(objectDetectionImage);
				MaxlenFrameObjectDetectionConfidence.erase(MaxlenFrameObjectDetectionConfidence.begin());
				MaxlenFrameObjectDetectionConfidence.push_back(objectDetectionConfidence);

				if (debug) {
					cout << "MaxlenFrameObjectDetection:" << endl;
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i) {
							cout << "识别结果 " << j << " " << endl;
						}
					}
				}
				vector<vector<int>>new_tracks;
				for (auto i = 0; i < objectDetectionImage.size(); i++) {
					int perTrackLen = maxlen + 2;
					vector<int>new_tracksTmp(perTrackLen, -1);
					new_tracksTmp[0] = i;
					new_tracksTmp[new_tracksTmp.size() - 1] = i;
					new_tracks.push_back(new_tracksTmp);
				}
				if (debug) {
					cout << "new_tracks:" << endl;
					for (auto i : new_tracks) {
						for (auto j : i) {
							cout << "新跟踪的结果 " << j << " " << endl;
						}
					}
				}
				for (auto i = 0; i < maxlen; i++) {
					MaxlenFrameGPS.push_back(Point3d(-1, -1, -1));
					MaxlenFrameANGLE.push_back(Point3d(-1, -1, -1));

				}
				MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
				MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
				MaxlenFrameANGLE.erase(MaxlenFrameANGLE.begin());
				MaxlenFrameANGLE.push_back(Point3d(pitch, yaw, roll));
				Track.insert(Track.end(), new_tracks.begin(), new_tracks.end());
				trackCount = new_tracks.size();

				//保存相关数据
				//保存相关数据
				string trackImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "Track.txt";
				string detectionImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Detection.txt";
				string AngleImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Angle.txt";
				string GPSImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_GPS.txt";

				bool SaveTXT = 1;
				ofstream OutFileTrack(trackImgResultTxt);
				ofstream OutFileDetection(detectionImgResultTxt);
				ofstream OutFileAngle(AngleImgResultTxt);
				ofstream OutFileGPS(GPSImgResultTxt);



				OutFileGPS << std::setprecision(10);
				OutFileAngle << std::setprecision(10);
				OutFileDetection << std::setprecision(10);
				OutFileTrack << std::setprecision(10);


				if (SaveTXT) {
					for (auto i : MaxlenFrameGPS)
						OutFileGPS << i << "\n";

					for (auto i : MaxlenFrameANGLE)
						OutFileAngle << i << "\n";
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i)
						{
							OutFileDetection << j << " ";
						}
						OutFileDetection << "\n";
					}
					for (auto i : Track) {
						for (auto j : i)
						{
							OutFileTrack << j << " ";
						}
						OutFileTrack << "\n";
					}

				}
				continue;
			}
			else {
				ObjectDetectionUpdateWithConfidence(MaxlenFrameObjectDetection, MaxlenFrameObjectDetectionConfidence, objectDetectionImage, objectDetectionConfidence, Track, trackCount, maxlen);

				if (debug) {
					cout << "Tracks:" << endl;
					for (auto i : Track) {
						cout << "第几个Tracks" << endl;
						for (auto j : i) {
							cout << "Tracks新跟踪的结果 " << j << " " << endl;
						}
					}
				}
				// GPS保存多帧对应的数值
				MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
				MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
				MaxlenFrameANGLE.erase(MaxlenFrameANGLE.begin());
				MaxlenFrameANGLE.push_back(Point3d(pitch, yaw, roll));

				
				string trackImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "Track.txt";
				string detectionImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Detection.txt";
				string AngleImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Angle.txt";
				string GPSImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_GPS.txt";

				bool SaveTXT = 1;
				ofstream OutFileTrack(trackImgResultTxt);
				ofstream OutFileDetection(detectionImgResultTxt);
				ofstream OutFileAngle(AngleImgResultTxt);
				ofstream OutFileGPS(GPSImgResultTxt);

				OutFileGPS << std::setprecision(10);
				OutFileAngle << std::setprecision(10);
				OutFileDetection << std::setprecision(10);
				OutFileTrack << std::setprecision(10);


				if (SaveTXT) {
					for (auto i : MaxlenFrameGPS)
						OutFileGPS << i << "\n";

					for (auto i : MaxlenFrameANGLE)
						OutFileAngle << i << "\n";
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i)
						{
							OutFileDetection << j << " ";
						}
						OutFileDetection << "\n";
					}
					for (auto i : Track) {
						for (auto j : i)
						{
							OutFileTrack << j << " ";
						}
						OutFileTrack << "\n";
					}

				}
			}
		}



		vector<vector<int>>TrackNowSatifyLen;
		int SatifyLen = 4;
		//objectDetectionGetTrack(Track, TrackNowSatifyLen, SatifyLen);

		vector<Point3d> ObjectDetectionFilterFineGPS;
		vector<Point3d> ObjectDetectionFilterExtraFeature;


		Mat trackImg = Image.clone();
		objectDetectionGetTrackAndDrawTrack(trackImg, MaxlenFrameObjectDetection, Track, TrackNowSatifyLen, SatifyLen);
		string ResultTrackname = "c:\\testDetection\\TrackResult\\resultTrack_" + to_string(resultImageNum) + resultBackName;
		imwrite(ResultTrackname, trackImg);

		//利用GPS数据进行位置估计

		if (TrackNowSatifyLen.size() < 1)
		{
			std::cout << "当前帧没有足够跟踪结果" << endl;
			continue;
		}
		Mat trackImgResult = Image.clone();
		//输出结果 经纬度 目标类别 置信度 出现次数 
		calGPSsaveConfidence(trackImgResult, MaxlenFrameObjectDetection, MaxlenFrameObjectDetectionConfidence, TrackNowSatifyLen, MaxlenFrameGPS, MaxlenFrameANGLE, ObjectDetectionFilterFineGPS, ObjectDetectionFilterExtraFeature);
		string trackImgResultImgName = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + resultBackName;

		string GPS_outputImgResultTxt = "c:\\testDetection\\output\\resultTrackGPS_" + to_string(resultImageNum) + "_GPS.txt";
		string Feature_outputImgResultTxt = "c:\\testDetection\\output\\resultTrackGPS_" + to_string(resultImageNum) + "_Feature.txt";

		bool SaveTXT = 1;
		ofstream OutFileGPS_output(GPS_outputImgResultTxt);
		ofstream OutFileGPS_Featureoutput(Feature_outputImgResultTxt);
		OutFileGPS_output << std::setprecision(10);
		OutFileGPS_Featureoutput << std::setprecision(10);

		for (auto i : ObjectDetectionFilterFineGPS)
			OutFileGPS_output << i << "\n";
		for (auto i : ObjectDetectionFilterExtraFeature)
			OutFileGPS_Featureoutput << i << "\n";
		imwrite(trackImgResultImgName, trackImgResult);

		if (ObjectDetectionAllGPS.size() < 1) {
			ObjectDetectionAllGPS.insert(ObjectDetectionAllGPS.end(), ObjectDetectionFilterFineGPS.begin(), ObjectDetectionFilterFineGPS.end());
			ObjectDetectionAllExtraFeature.insert(ObjectDetectionAllExtraFeature.end(), ObjectDetectionFilterExtraFeature.begin(), ObjectDetectionFilterExtraFeature.end());
			continue;
		}
		if (ObjectDetectionAllGPS.size() > 0)
		{
			for (auto i = 0; i < ObjectDetectionFilterExtraFeature.size(); i++) {
				bool detectionFound = false;
				for (auto j = 0; j < ObjectDetectionAllExtraFeature.size(); j++) {
					if (ObjectDetectionFilterExtraFeature[i].z == ObjectDetectionAllExtraFeature[j].z)
					{
						ObjectDetectionAllGPS[j].x = (ObjectDetectionAllGPS[j].x + ObjectDetectionFilterFineGPS[i].x) / 2;
						ObjectDetectionAllGPS[j].y = (ObjectDetectionAllGPS[j].y + ObjectDetectionFilterFineGPS[i].y) / 2;
						ObjectDetectionAllExtraFeature[j].x = ObjectDetectionFilterExtraFeature[i].x;
						ObjectDetectionAllExtraFeature[j].y = (ObjectDetectionAllExtraFeature[j].y + ObjectDetectionFilterExtraFeature[i].y) / 2;
						detectionFound = true;
						continue;
					}
				}
				if (!detectionFound) {
					cout << "ObjectDetectionAllExtraFeature add " << ObjectDetectionFilterExtraFeature[i] << endl;
					ObjectDetectionAllExtraFeature.push_back(ObjectDetectionFilterExtraFeature[i]);
					ObjectDetectionAllGPS.push_back(ObjectDetectionFilterFineGPS[i]);
				}
			}

			string AllGPSOutputTxt = "c:\\testDetection\\outputALL\\AllGPSOutput" + to_string(resultImageNum) + ".txt";
			string AllFeatureOutputTxt = "c:\\testDetection\\outputALL\\AllFeatureOutput" + to_string(resultImageNum) + ".txt";
			ofstream OutFileAllGPSOutput(AllGPSOutputTxt);
			ofstream OutFileAllFeature(AllFeatureOutputTxt);
			OutFileAllGPSOutput << std::setprecision(10);
			OutFileAllFeature << std::setprecision(10);

			for (auto i : ObjectDetectionAllGPS)
				OutFileAllGPSOutput << i << "\n";
			for (auto i : ObjectDetectionAllExtraFeature)
				OutFileAllFeature << i << "\n";
		}
		outputDetectionResult.clear();
		vector<aircraft_data>().swap(outputDetectionResult);
		for (auto i = 0; i < ObjectDetectionAllGPS.size(); i++) {
			aircraft_data aircraft_dataTmp;
			aircraft_dataTmp.lon = ObjectDetectionAllGPS[i].x;
			aircraft_dataTmp.lat = ObjectDetectionAllGPS[i].y;
			aircraft_dataTmp.object_property = ObjectDetectionAllGPS[i].z;
			aircraft_dataTmp.appear_counts = ObjectDetectionAllExtraFeature[i].x;
			aircraft_dataTmp.cc = ObjectDetectionAllExtraFeature[i].y;
			aircraft_dataTmp.num = ObjectDetectionAllExtraFeature[i].z;
			outputDetectionResult.push_back(aircraft_dataTmp);
		}


		//利用hist相关性滤除一部分结果  后续是否可以借鉴svm呢或许可以！！！
		//Mat similarityCompareImg = Image.clone();

		//for (auto i = 0; i < newObjectDetectionRect.size(); i++)
		//{
		//	double best_dSimilarity = -1;
		//	int best_label = -1;
		//	int x_tmp = max(0, newObjectDetectionRect[i].x);
		//	int y_tmp = max(0, newObjectDetectionRect[i].y);
		//	int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
		//	int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);

		//	Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
		//	Mat first = similarityCompareImg(tmp);
		//	for (auto j = 0; j < object_label.size(); j++)
		//	{

		//		Mat second = object_label[j].clone();
		//		imshow("first", first);
		//		imshow("second", second);
		//		waitKey(1);
		//		double dSimilarity = 0;
		//		compareHistColor(first, second, dSimilarity);
		//		if (dSimilarity > best_dSimilarity)
		//			best_label = j;
		//	}
		//	if ((best_label == newObjectDetection[i].z) && (best_dSimilarity > 0.9))
		//		objectDetectionImageCorrect.push_back(newObjectDetection[i]);
		//}
	}
}
void test::redNumberdetectionFilterFromMovie()
{

	std::cout << "开始！！！" << endl;
	// Initialize the parameters
	float confThreshold = 0.65; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;  // Width of network's input image
	int inpHeight = 416; // Height of network's input image
	double HighWidthHeightRatio = 5;
	double LowWidthHeightRatio = 0.2;
	vector<string> classes;


	// Load names of classes
	string classesFile = g_classesFile; //absolute path--liang
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);


	// Give the configuration and weight files for the model
	String modelConfiguration = g_modelConfiguration; //absolute path --liang
	String modelWeights = g_modelWeights; //absolute path --liang


										  // Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);


	//save result
	int resultImageNum = 0;
	int processSampling = 5;//真正用的时候用1!!!!!
	int resultSampling = 1;
	vector<Point3f> ObjectDetectionTmp;
	string resultBackName = "_0507Result.jpg";

	vector<Point3d> ObjectDetectionAllGPS;
	vector<Point3d> ObjectDetectionAllExtraFeature;

	//创建文件夹
	string command;
	command = "mkdir -p c:\\testDetection\\GPS";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\detection";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\TrackResult";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\output";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\outputALL";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\TIMEGPS";
	system(command.c_str());
	command = "mkdir -p c:\\testDetection\\Crop";
	system(command.c_str());


	//// Open  a camera stream.
	VideoCapture cap;
	//cap.set(CAP_PROP_BUFFERSIZE, 1);
	cap.open(g_video,CAP_FFMPEG); // camera id --liang

	Mat Image;

	//mid variable
	Mat ImageWaitDetection, blob;

	//mid Track variable
	int maxlen = 10;//保留maxlen帧的识别结果

	vector<vector<Point3f>> MaxlenFrameObjectDetection;//初始化保存检测识别目标的位置
	vector<vector<double>> MaxlenFrameObjectDetectionConfidence;//初始化保存检测识别目标的置信度

	vector<Point3d>MaxlenFrameGPS;
	vector<Point3d>MaxlenFrameANGLE;
	int trackCount = 0;
	vector<Rect>oldObjectDetectionRect;
	vector<vector<int>>Track;//前两列分别为track的序号与track的分数，后面为依次帧的点序号

							 ////detection loop;
							 //if (!cap.isOpened()) {
							 //	cout << "No camera is opened !!!" << endl;
							 //	_cprintf("No PICTURES Done processing !!!");
							 //	waitKey(3000);
							 //	return;
							 //}
							 //AfxBeginThread(this->readImage, NULL);
	bool detectionOpen = true;
	while ((detectionOpen) &&cap.isOpened()){
		cap >> Image;
		if (Image.empty()) {

			cout << "没有图像" << endl;
			return;
		}
		imshow("test", Image);
		waitKey(1);
		resultImageNum++;
		Mat udistImg; 
		//setup camera matrix:
		/*
		004参数
		Camera Intrinsics
		IntrinsicMatrix: [3×3 double]
		FocalLength: [1.7851e+03 1.7799e+03]
		PrincipalPoint: [941.6524 546.5905]
		Skew: 0

		Lens Distortion
		RadialDistortion: [-0.1740 0.0739 0.0710]
		TangentialDistortion: [0 0]
		*/
		Mat cameraMatrix(3, 3, CV_64FC1);
		cameraMatrix.setTo(0);
		cameraMatrix.at<double>(0, 0) = 1818.1278;
		cameraMatrix.at<double>(1, 1) = 1814.64;
		cameraMatrix.at<double>(0, 2) = 973.2279;
		cameraMatrix.at<double>(1, 2) = 565.1970;
		cameraMatrix.at<double>(2, 2) = 1;

		//setup distortion vector:
		cv::Mat distCoeffs(5, 1, CV_64FC1);
		distCoeffs.setTo(0); //There's no distortion in this demo.
		distCoeffs.at<double>(0) = -0.1789;
		distCoeffs.at<double>(1) = 0.1380;
		distCoeffs.at<double>(2) = 0.0275;
		distCoeffs.at<double>(3) = 4.06e-4;
		distCoeffs.at<double>(4) = 0.003;

		Mat view, rview, map1, map2;
		Size imageSize;
		imageSize = Image.size();
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
			getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
			imageSize, CV_32FC1, map1, map2);

		remap(Image, udistImg, map1, map2, INTER_LINEAR);
		Image = udistImg.clone();
		//undistort(Image, udistImg, cameraMatrix, distCoeffs);
		namedWindow("校正前",WINDOW_NORMAL);
		namedWindow("校正后", WINDOW_NORMAL);
		imshow("校正前", Image);
		imshow("校正后", udistImg);
		cout << "udistImg: "<<udistImg.size() << endl;
		waitKey(1);
		//testImage.pop_front();
		auto start = clock();

		int delay = 200;//ms
		auto now = start;

		if (resultImageNum % processSampling != 0)
			continue;

		//经纬度数据和欧拉角数据 
		//与int可以相互转化么？
		//string writeOnlyResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
		//imwrite(writeOnlyResultImage, Image);
		//continue;
		//外界纳入量
		//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		//获得当前时间
		//和数组做比对 找最近的时间节点的数
		//增加图片延时量
		//string timeGpsTxt = "c:\\testDetection\\TIMEGPS\\timeGps" + to_string(resultImageNum) + ".txt";
		//ofstream timeGpsTxtPose(timeGpsTxt);
		int min = 9999999;
		//AlgorithmData b;
		//b.la = 0;
		//auto t =now-delay;//真正拍照时间为当前时间-时延
		//deque<AlgorithmData> v1(g_dataList);
		//for (size_t i = 0; i < v1.size(); i++)
		//{
		//	auto a = v1[i];
		//	timeGpsTxtPose <<std::setprecision(15)<<"当前计时："<<now<<",时延："<<delay<<",时间:"<<t<< ",时间窗时间t:" << a.t<<",经度:"<<a.lo<<",纬度："<<a.la<<endl;
		//	CString mm1;
		//	mm1.Format(_T("当前的时间窗差值：%d数据长度%d,最新数据为航向%f，roll为%f,pitch为%f,lo：%f,la：%f,高度：%d,时间窗最小值：%d"), abs((int)(a.t - t)),v1.size(), a.direction, a.roll, a.pitch, a.lo, a.la, a.height, min);
		//	//_cprintf("\n%S\n", mm1);
		//	if (abs((int)(a.t - t)) < min) {
		//		b = a;
		//		min = abs((int)(a.t - t));
		//	}
		//	else {
		//		_cprintf("\na.t:%d,t:%d\n",a.t,t);
		//	}
		//}
		//timeGpsTxtPose << std::setprecision(15)<<"最终匹配："<< "当前计时：" << now << ",时延：" << delay << ",时间:" << t << ",时间窗时间t:" << b.t << ",经度:" << b.lo << ",纬度：" << b.la << endl;
		//double lat = b.la;
		//double lon = b.lo;
		//double height = (double)b.height / 100;

		//double roll = b.roll;
		//double yaw = b.direction;
		//double pitch = b.pitch;

		double lat = 0;
		double lon = 0;
		double height = 28;

		double roll = 0;
		double yaw = 0;
		double pitch = 0;


		ImageWaitDetection = Image.clone();
		Mat ImageWaitDetectionResize = letterbox_image(ImageWaitDetection, inpWidth, inpHeight);
		// Create a 4D blob from a frame.

		blobFromImage(ImageWaitDetectionResize, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		//Sets the input to the network
		net.setInput(blob);

		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		vector<Point3f>objectDetectionImage;//直接将输出赋予objectDetectionImage变量，z存类别
		vector<double>objectDetectionConfidence;//保存识别结果的置信度

		vector<Rect>objectDetectionRect;
		postprocessSaveConfidence(ImageWaitDetectionResize, ImageWaitDetection, outs, objectDetectionImage, objectDetectionConfidence, objectDetectionRect, HighWidthHeightRatio, LowWidthHeightRatio, confThreshold, nmsThreshold, inpWidth, inpHeight);

		Mat detectedFrameResult;
		ImageWaitDetection.convertTo(detectedFrameResult, CV_8U);
		Mat detectedResizeFrameResult;
		ImageWaitDetectionResize.convertTo(detectedResizeFrameResult, CV_8U);
		//没有识别结果的依然输出,且在图像上明显标记
		if (objectDetectionImage.size() < 1) {
			std::cout << "当前帧没有找到目标" << endl;
			string resultTxt = "NO Object Detection";
			string resultTimeTxt = to_string(now) + "ms";

			putText(detectedFrameResult, resultTxt, Point(detectedFrameResult.rows / 2, detectedFrameResult.cols / 2), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);

			putText(detectedFrameResult, resultTimeTxt, Point(detectedFrameResult.rows / 10, detectedFrameResult.cols / 10), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);
			string writeResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
			string writeResizeResultImage = "c:\\testDetection\\detection\\Resize_" + to_string(resultImageNum) + resultBackName;
			imwrite(writeResultImage, detectedFrameResult);
			//imwrite(writeResizeResultImage, detectedResizeFrameResult);

			continue;
		}



		string trackPoseImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Pose.txt";

		ofstream OutFiletrackPose(trackPoseImgResultTxt);

		//05.13增加基于已知尺寸大小的位姿估计
		Mat srcAttitude = Image.clone();
		for (auto i : objectDetectionRect)
		{
			Mat rvec1, rvec2, tvec1, tvec2;
			GetIPPEAttitude(srcAttitude, i, rvec1, rvec2, tvec1, tvec2);
			OutFiletrackPose << "位置: " << endl;
			OutFiletrackPose << tvec1 << endl;
			OutFiletrackPose << "姿态: " << endl;
			OutFiletrackPose << rvec1 << endl;
		}

		//rect的信息
		string RectResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Rect.txt";

		ofstream OutFileRect(RectResultTxt);

		//05.19增加基于识别RECT的值
		for (auto i : objectDetectionRect)
		{
			OutFileRect << i.x<<" "<<i.y<<" "<<i.width<<" "<<i.height<< endl;
		}
		Scalar drawcolor;
		Mat udistImgDetection = udistImg.clone();
		for (auto i : objectDetectionImage)
		{
			//std::cout << "objectDetectionImage：" << i << endl;
			if (i.z == 0)
				drawcolor = Scalar(255, 0, 0);
			if (i.z == 1)
				drawcolor = Scalar(0, 255, 0);
			if (i.z == 2)
				drawcolor = Scalar(0, 255, 255);
			int remapX = map1.at<float>(int(i.y), int(i.x));
			int remapY = map2.at<float>(int(i.y), int(i.x));

			string objectDetectionImagelabel = format("%d ", i.z);
			circle(detectedFrameResult, Point(i.x, i.y), 5, drawcolor, -1);
			//circle(udistImgDetection, Point(remapX, remapY), 5, drawcolor, -1);

		}
		namedWindow("detectedFrameResult", WINDOW_NORMAL);
		imshow("detectedFrameResult", detectedFrameResult);

		//namedWindow("udistImgDetection", WINDOW_NORMAL);
		//imshow("udistImgDetection", udistImgDetection);
		waitKey(1);
		if (resultImageNum % resultSampling == 0)
		{
			string writeResultImage = "c:\\testDetection\\detection\\" + to_string(resultImageNum) + resultBackName;
			string resultTimeTxt = to_string(now) + "ms";
			putText(detectedFrameResult, resultTimeTxt, Point(detectedFrameResult.rows / 10, detectedFrameResult.cols / 10), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 5);
			imwrite(writeResultImage, detectedFrameResult);
			//string writeResizeResultImage = "c:\\testDetection\\detection\\Resize_" + to_string(resultImageNum) + resultBackName;
			//imwrite(writeResizeResultImage, detectedResizeFrameResult);
		}

		Mat cropImage = Image.clone();

		if (objectDetectionImage.size() < 1) {
			std::cout << "没有找到目标" << endl;
			continue;
		}
		//显示识别出来的区域内容
		//0519:防止超出边界
		for (auto i = 0; i < objectDetectionRect.size(); i++)
		{
			int tmpsize = std::min(objectDetectionRect[i].width, objectDetectionRect[i].height);
			int x_tmp = max(0, int(objectDetectionRect[i].x + objectDetectionRect[i].width / 2 - tmpsize / 2));
			int y_tmp = max(0, int(objectDetectionRect[i].y + objectDetectionRect[i].height / 2 - tmpsize / 2));

			int width_tmp = std::min(tmpsize, Image.cols - x_tmp);
			int height_tmp = std::min(tmpsize, Image.rows - y_tmp);
			Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
			Mat croptmp = cropImage(tmp);
			string cropname = "c:\\testDetection\\Crop\\Crop_" + to_string(resultImageNum) + "__" + to_string(i) + resultBackName;
			imwrite(cropname, croptmp);
			//imshow("分割出来的目标图像", croptmp);
			//waitKey(1);
		}

		//开始迭代更新,输出更新后的结果
		// 如果之前没有数值对Track相关数据进行初始化
		bool debug = 0; //debug 部分是调试track的过程
		if (objectDetectionImage.size() >= 1)
		{
			if (MaxlenFrameObjectDetection.size() < 1)
			{
				for (auto i = 0; i < maxlen; i++) {
					vector<Point3f>MaxlenFrame_tmp;
					vector<double>MaxlenFrameConfidence_tmp;

					MaxlenFrameObjectDetection.push_back(MaxlenFrame_tmp);
					MaxlenFrameObjectDetectionConfidence.push_back(MaxlenFrameConfidence_tmp);
				}
				MaxlenFrameObjectDetection.erase(MaxlenFrameObjectDetection.begin());
				MaxlenFrameObjectDetection.push_back(objectDetectionImage);
				MaxlenFrameObjectDetectionConfidence.erase(MaxlenFrameObjectDetectionConfidence.begin());
				MaxlenFrameObjectDetectionConfidence.push_back(objectDetectionConfidence);

				if (debug) {
					cout << "MaxlenFrameObjectDetection:" << endl;
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i) {
							cout << "识别结果 " << j << " " << endl;
						}
					}
				}
				vector<vector<int>>new_tracks;
				for (auto i = 0; i < objectDetectionImage.size(); i++) {
					int perTrackLen = maxlen + 2;
					vector<int>new_tracksTmp(perTrackLen, -1);
					new_tracksTmp[0] = i;
					new_tracksTmp[new_tracksTmp.size() - 1] = i;
					new_tracks.push_back(new_tracksTmp);
				}
				if (debug) {
					cout << "new_tracks:" << endl;
					for (auto i : new_tracks) {
						for (auto j : i) {
							cout << "新跟踪的结果 " << j << " " << endl;
						}
					}
				}
				for (auto i = 0; i < maxlen; i++) {
					MaxlenFrameGPS.push_back(Point3d(-1, -1, -1));
					MaxlenFrameANGLE.push_back(Point3d(-1, -1, -1));

				}
				MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
				MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
				MaxlenFrameANGLE.erase(MaxlenFrameANGLE.begin());
				MaxlenFrameANGLE.push_back(Point3d(pitch, yaw, roll));
				Track.insert(Track.end(), new_tracks.begin(), new_tracks.end());
				trackCount = new_tracks.size();

				//保存相关数据
				string trackImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "Track.txt";
				string detectionImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Detection.txt";
				string AngleImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Angle.txt";
				string GPSImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_GPS.txt";

				bool SaveTXT = 1;
				ofstream OutFileTrack(trackImgResultTxt);
				ofstream OutFileDetection(detectionImgResultTxt);
				ofstream OutFileAngle(AngleImgResultTxt);
				ofstream OutFileGPS(GPSImgResultTxt);



				OutFileGPS << std::setprecision(10);
				OutFileAngle << std::setprecision(10);
				OutFileDetection << std::setprecision(10);
				OutFileTrack << std::setprecision(10);


				if (SaveTXT) {
					for (auto i : MaxlenFrameGPS)
						OutFileGPS << i << "\n";

					for (auto i : MaxlenFrameANGLE)
						OutFileAngle << i << "\n";
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i)
						{
							OutFileDetection << j << " ";
						}
						OutFileDetection << "\n";
					}
					for (auto i : Track) {
						for (auto j : i)
						{
							OutFileTrack << j << " ";
						}
						OutFileTrack << "\n";
					}

				}
				continue;
			}
			else {
				ObjectDetectionUpdateWithConfidence(MaxlenFrameObjectDetection, MaxlenFrameObjectDetectionConfidence, objectDetectionImage, objectDetectionConfidence, Track, trackCount, maxlen);

				if (debug) {
					cout << "Tracks:" << endl;
					for (auto i : Track) {
						cout << "第几个Tracks" << endl;
						for (auto j : i) {
							cout << "Tracks新跟踪的结果 " << j << " " << endl;
						}
					}
				}
				// GPS保存多帧对应的数值
				MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
				MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
				MaxlenFrameANGLE.erase(MaxlenFrameANGLE.begin());
				MaxlenFrameANGLE.push_back(Point3d(pitch, yaw, roll));

				string trackImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "Track.txt";
				string detectionImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Detection.txt";
				string AngleImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_Angle.txt";
				string GPSImgResultTxt = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + "_GPS.txt";

				bool SaveTXT = 1;
				ofstream OutFileTrack(trackImgResultTxt);
				ofstream OutFileDetection(detectionImgResultTxt);
				ofstream OutFileAngle(AngleImgResultTxt);
				ofstream OutFileGPS(GPSImgResultTxt);

				OutFileGPS << std::setprecision(10);
				OutFileAngle << std::setprecision(10);
				OutFileDetection << std::setprecision(10);
				OutFileTrack << std::setprecision(10);


				if (SaveTXT) {
					for (auto i : MaxlenFrameGPS)
						OutFileGPS << i << "\n";

					for (auto i : MaxlenFrameANGLE)
						OutFileAngle << i << "\n";
					for (auto i : MaxlenFrameObjectDetection) {
						for (auto j : i)
						{
							OutFileDetection << j << " ";
						}
						OutFileDetection << "\n";
					}
					for (auto i : Track) {
						for (auto j : i)
						{
							OutFileTrack << j << " ";
						}
						OutFileTrack << "\n";
					}

				}
			}
		}



		vector<vector<int>>TrackNowSatifyLen;
		int SatifyLen = 4;
		//objectDetectionGetTrack(Track, TrackNowSatifyLen, SatifyLen);

		vector<Point3d> ObjectDetectionFilterFineGPS;
		vector<Point3d> ObjectDetectionFilterExtraFeature;


		Mat trackImg = Image.clone();
		objectDetectionGetTrackAndDrawTrack(trackImg, MaxlenFrameObjectDetection, Track, TrackNowSatifyLen, SatifyLen);
		string ResultTrackname = "c:\\testDetection\\TrackResult\\resultTrack_" + to_string(resultImageNum) + resultBackName;
		imwrite(ResultTrackname, trackImg);

		//利用GPS数据进行位置估计

		if (TrackNowSatifyLen.size() < 1)
		{
			std::cout << "当前帧没有足够跟踪结果" << endl;
			continue;
		}
		Mat trackImgResult = Image.clone();
		//输出结果 经纬度 目标类别 置信度 出现次数 
		calGPSsaveConfidence(trackImgResult, MaxlenFrameObjectDetection, MaxlenFrameObjectDetectionConfidence, TrackNowSatifyLen, MaxlenFrameGPS, MaxlenFrameANGLE, ObjectDetectionFilterFineGPS, ObjectDetectionFilterExtraFeature);
		string trackImgResultImgName = "c:\\testDetection\\GPS\\resultTrack_" + to_string(resultImageNum) + resultBackName;

		string GPS_outputImgResultTxt = "c:\\testDetection\\output\\resultTrackGPS_" + to_string(resultImageNum) + "_GPS.txt";
		string Feature_outputImgResultTxt = "c:\\testDetection\\output\\resultTrackGPS_" + to_string(resultImageNum) + "_Feature.txt";

		bool SaveTXT = 1;
		ofstream OutFileGPS_output(GPS_outputImgResultTxt);
		ofstream OutFileGPS_Featureoutput(Feature_outputImgResultTxt);
		OutFileGPS_output << std::setprecision(10);
		OutFileGPS_Featureoutput << std::setprecision(10);

		for (auto i : ObjectDetectionFilterFineGPS)
			OutFileGPS_output << i << "\n";
		for (auto i : ObjectDetectionFilterExtraFeature)
			OutFileGPS_Featureoutput << i << "\n";
		imwrite(trackImgResultImgName, trackImgResult);

		if (ObjectDetectionAllGPS.size() < 1) {
			ObjectDetectionAllGPS.insert(ObjectDetectionAllGPS.end(), ObjectDetectionFilterFineGPS.begin(), ObjectDetectionFilterFineGPS.end());
			ObjectDetectionAllExtraFeature.insert(ObjectDetectionAllExtraFeature.end(), ObjectDetectionFilterExtraFeature.begin(), ObjectDetectionFilterExtraFeature.end());
			continue;
		}
		if (ObjectDetectionAllGPS.size() > 0)
		{
			for (auto i = 0; i < ObjectDetectionFilterExtraFeature.size(); i++) {
				bool detectionFound = false;
				for (auto j = 0; j < ObjectDetectionAllExtraFeature.size(); j++) {
					if (ObjectDetectionFilterExtraFeature[i].z == ObjectDetectionAllExtraFeature[j].z)
					{
						ObjectDetectionAllGPS[j].x = (ObjectDetectionAllGPS[j].x + ObjectDetectionFilterFineGPS[i].x) / 2;
						ObjectDetectionAllGPS[j].y = (ObjectDetectionAllGPS[j].y + ObjectDetectionFilterFineGPS[i].y) / 2;
						ObjectDetectionAllExtraFeature[j].x = ObjectDetectionFilterExtraFeature[i].x;
						ObjectDetectionAllExtraFeature[j].y = (ObjectDetectionAllExtraFeature[j].y + ObjectDetectionFilterExtraFeature[i].y) / 2;
						detectionFound = true;
						continue;
					}
				}
				if (!detectionFound) {
					cout << "ObjectDetectionAllExtraFeature add " << ObjectDetectionFilterExtraFeature[i] << endl;
					ObjectDetectionAllExtraFeature.push_back(ObjectDetectionFilterExtraFeature[i]);
					ObjectDetectionAllGPS.push_back(ObjectDetectionFilterFineGPS[i]);
				}
			}

			string AllGPSOutputTxt = "c:\\testDetection\\outputALL\\AllGPSOutput" + to_string(resultImageNum) + ".txt";
			string AllFeatureOutputTxt = "c:\\testDetection\\outputALL\\AllFeatureOutput" + to_string(resultImageNum) + ".txt";
			ofstream OutFileAllGPSOutput(AllGPSOutputTxt);
			ofstream OutFileAllFeature(AllFeatureOutputTxt);
			OutFileAllGPSOutput << std::setprecision(10);
			OutFileAllFeature << std::setprecision(10);

			for (auto i : ObjectDetectionAllGPS)
				OutFileAllGPSOutput << i << "\n";
			for (auto i : ObjectDetectionAllExtraFeature)
				OutFileAllFeature << i << "\n";
		}
		outputDetectionResult.clear();
		vector<aircraft_data>().swap(outputDetectionResult);
		for (auto i = 0; i < ObjectDetectionAllGPS.size(); i++) {
			aircraft_data aircraft_dataTmp;
			aircraft_dataTmp.lon = ObjectDetectionAllGPS[i].x;
			aircraft_dataTmp.lat = ObjectDetectionAllGPS[i].y;
			aircraft_dataTmp.object_property = ObjectDetectionAllGPS[i].z;
			aircraft_dataTmp.appear_counts = ObjectDetectionAllExtraFeature[i].x;
			aircraft_dataTmp.cc = ObjectDetectionAllExtraFeature[i].y;
			aircraft_dataTmp.num = ObjectDetectionAllExtraFeature[i].z;
			outputDetectionResult.push_back(aircraft_dataTmp);
		}


		//利用hist相关性滤除一部分结果  后续是否可以借鉴svm呢或许可以！！！
		//Mat similarityCompareImg = Image.clone();

		//for (auto i = 0; i < newObjectDetectionRect.size(); i++)
		//{
		//	double best_dSimilarity = -1;
		//	int best_label = -1;
		//	int x_tmp = max(0, newObjectDetectionRect[i].x);
		//	int y_tmp = max(0, newObjectDetectionRect[i].y);
		//	int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
		//	int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);

		//	Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
		//	Mat first = similarityCompareImg(tmp);
		//	for (auto j = 0; j < object_label.size(); j++)
		//	{

		//		Mat second = object_label[j].clone();
		//		imshow("first", first);
		//		imshow("second", second);
		//		waitKey(1);
		//		double dSimilarity = 0;
		//		compareHistColor(first, second, dSimilarity);
		//		if (dSimilarity > best_dSimilarity)
		//			best_label = j;
		//	}
		//	if ((best_label == newObjectDetection[i].z) && (best_dSimilarity > 0.9))
		//		objectDetectionImageCorrect.push_back(newObjectDetection[i]);
		//}
	}
}

void test::calGPSsaveConfidence(Mat & src, vector<vector<Point3f>> & ObjectDetectionContinueMaxlenFrame, vector<vector<double>> & MaxlenFrameObjectDetectionConfidence, vector<vector<int>>TrackNowSatifyLen, vector<Point3d>MaxlenFrameGPS, vector<Point3d>MaxlenFrameANGLE, vector<Point3d> & ObjectDetectionFilterFineGPS, vector<Point3d> & ObjectDetectionFilterExtraFeature) {
	vector<int>offsetAccumulate;
	get_offsets(ObjectDetectionContinueMaxlenFrame, offsetAccumulate);
	double yitaX = 57;
	double yitaY = 32;
	for (auto trackEvery : TrackNowSatifyLen) {
		double GPSX = 0, GPSY = 0;
		double classId = -1;
		Point TrackCirclePoint;
		double brngimg = -1;
		string brngimgInfor;
		int occurTimes = 0;
		double detectionConfidence = 0;
		Scalar drawcolorPoint;
		for (auto i = 0; i < ObjectDetectionContinueMaxlenFrame.size(); i++) {
			if (trackEvery[i + 2] == -1)
				continue;
			int offset1 = offsetAccumulate[i];
			occurTimes++;
			int idx1 = int(trackEvery[i + 2] - offset1);
			Point3d pt1 = ObjectDetectionContinueMaxlenFrame[i][idx1];
			double confidenceTmp = MaxlenFrameObjectDetectionConfidence[i][idx1];
			Point circlePt1 = Point(int(pt1.x), int(pt1.y));
			TrackCirclePoint = circlePt1;
			Point3d pt1GPStmp;
			classId = pt1.z;
			if (classId == 0)
				drawcolorPoint = Scalar(255, 0, 0);
			if (classId == 1)
				drawcolorPoint = Scalar(0, 255, 0);
			if (classId == 2)
				drawcolorPoint = Scalar(0, 255, 255);
			locationEveryTarget(pt1, pt1GPStmp, MaxlenFrameANGLE[i].y, MaxlenFrameGPS[i].x, MaxlenFrameGPS[i].y, MaxlenFrameGPS[i].z, yitaX, yitaY, brngimg, brngimgInfor);
			GPSX = GPSX + pt1GPStmp.x;
			GPSY = GPSY + pt1GPStmp.y;
			detectionConfidence = detectionConfidence + confidenceTmp;
		}
		double GPSxAverage = GPSX / occurTimes;
		double GPSyAverage = GPSY / occurTimes;
		double detectionConfidenceAverage = detectionConfidence / occurTimes;
		double trackID = trackEvery[0];//输出跟踪点的具体序号
		ObjectDetectionFilterFineGPS.push_back(Point3d(GPSxAverage, GPSyAverage, classId));
		ObjectDetectionFilterExtraFeature.push_back(Point3d(occurTimes, detectionConfidenceAverage, trackID));
		string resultTxt = to_string(GPSxAverage) + " " + to_string(GPSyAverage) + " " + to_string(classId) + " " + to_string(brngimg);
		string resultFeatureTxt = "occurTimes: " + to_string(occurTimes) + " detectionConfidenceAverage: " + to_string(detectionConfidenceAverage);
		string trackPointTxt = "trackPoint ID: " + to_string(trackID);
		cout << "GPS位置" << resultTxt << endl;
		cout << "resultTxt" << resultTxt << endl;
		circle(src, Point(960, 540), 5, Scalar(0, 0, 0), -1);
		circle(src, TrackCirclePoint, 5, drawcolorPoint, -1);
		putText(src, resultTxt, TrackCirclePoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
		putText(src, resultFeatureTxt, Point(TrackCirclePoint.x, TrackCirclePoint.y + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
		putText(src, trackPointTxt, Point(TrackCirclePoint.x, TrackCirclePoint.y + 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
		putText(src, brngimgInfor, Point(TrackCirclePoint.x, TrackCirclePoint.y + 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 3);
		imshow("calGPS", src);
		waitKey(1);
	}
}

void test::objectDetectionGetTrack(vector<vector<int>>Track, vector<vector<int>> & TrackNowSatifyLen, int SatifyLen)
{
	for (auto i : Track)
	{
		int lenTrackTMP = 0;
		if (i[i.size() - 1] > -1) {
			for (auto j : i) {
				if (j > -1)
				{
					lenTrackTMP = lenTrackTMP + 1;
				}
			}
		}
		if (lenTrackTMP > SatifyLen)
			TrackNowSatifyLen.push_back(i);
	}
}
void test::objectDetectionGetTrackAndDrawTrack(Mat & src, vector<vector<Point3f>> & ObjectDetectionContinueMaxlenFrame, vector<vector<int>>Track, vector<vector<int>> & TrackNowSatifyLen, int SatifyLen)
{
	vector<int>offsetAccumulate;
	get_offsets(ObjectDetectionContinueMaxlenFrame, offsetAccumulate);
	cv::RNG rng(12345);
	for (auto trackEvery : Track)
	{
		int lenTrackTMP = 0;
		if (trackEvery[trackEvery.size() - 1] > -1) {
			for (auto j : trackEvery) {
				if (j > -1)
				{
					lenTrackTMP = lenTrackTMP + 1;
				}
			}
		}
		if (lenTrackTMP > SatifyLen) {
			TrackNowSatifyLen.push_back(trackEvery);
			//画跟踪的结果！
			Scalar linecolor = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			Scalar pointcolor = Scalar(0, 255, 255);
			for (auto i = 0; i < ObjectDetectionContinueMaxlenFrame.size() - 1; i++) {
				if ((trackEvery[i + 2] == -1) || (trackEvery[i + 3] == -1))
					continue;
				int offset1 = offsetAccumulate[i];
				int offset2 = offsetAccumulate[i + 1];
				int idx1 = int(trackEvery[i + 2] - offset1);
				int idx2 = int(trackEvery[i + 3] - offset2);
				Point3d pt1 = ObjectDetectionContinueMaxlenFrame[i][idx1];
				Point3d pt2 = ObjectDetectionContinueMaxlenFrame[i + 1][idx2];
				Point circlePt1 = Point(int(pt1.x), int(pt1.y));
				Point circlePt2 = Point(int(pt2.x), int(pt2.y));
				circle(src, circlePt1, 10, pointcolor, -1);
				circle(src, circlePt2, 10, pointcolor, -1);
				line(src, circlePt1, circlePt2, linecolor, 20);
				namedWindow("track img", WINDOW_NORMAL);
				imshow("track img", src);
			}
			waitKey(1);
		}

	}
}
void test::DrawobjectDetectionTrack(Mat & src, vector<vector<Point3f>> & ObjectDetectionContinueMaxlenFrame, vector<vector<int>> & TrackNowSatifyLen, int SatifyLen) {

	vector<int>offsetAccumulate;
	get_offsets(ObjectDetectionContinueMaxlenFrame, offsetAccumulate);
	cv::RNG rng(12345);
	for (auto trackEvery : TrackNowSatifyLen) {
		Scalar linecolor = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Scalar pointcolor = Scalar(0, 255, 255);
		for (auto i = 0; i < ObjectDetectionContinueMaxlenFrame.size() - 1; i++) {
			if ((trackEvery[i + 2] == -1) || (trackEvery[i + 3] == -1))
				continue;
			int offset1 = offsetAccumulate[i];
			int offset2 = offsetAccumulate[i + 1];
			int idx1 = int(trackEvery[i + 2] - offset1);
			int idx2 = int(trackEvery[i + 3] - offset2);
			Point3d pt1 = ObjectDetectionContinueMaxlenFrame[i][idx1];
			Point3d pt2 = ObjectDetectionContinueMaxlenFrame[i + 1][idx2];
			Point circlePt1 = Point(int(pt1.x), int(pt1.y));
			Point circlePt2 = Point(int(pt2.x), int(pt2.y));
			circle(src, circlePt1, 10, pointcolor, -1);
			circle(src, circlePt2, 10, pointcolor, -1);
			line(src, circlePt1, circlePt2, linecolor, 20);
			namedWindow("track img", WINDOW_NORMAL);
			imshow("track img", src);
		}
		waitKey(1);
	}
}


//定位相关算法
//lat 纬度 lon 经度
void test::locationEveryTarget(Point3d ObjectDetectionFilterCorrect, Point3d & ObjectDetectionGPS, double yaw, double lat, double lon, double height, double yitaX, double yitaY, double& brngImg, string & brngImgInformation) {
	yitaX = yitaX * 3.1415926 / 180 / 2;
	yitaY = yitaX * 3.1415926 / 180 / 2;
	//006-校正
	//double heightPixelX = 1825.7974;
	//double heightPixelY = 1819.4336;
	//double CenterX = 980.5439;
	//double CenterY = 566.3955;



	//004-校正

	double heightPixelX = 1818.1278;
	double heightPixelY = 1814.64;
	double CenterX = 973.2279;
	double CenterY = 565.1970;



	//double heightPixelX = 1859.2;
	//double heightPixelY = 1860.7;
	//double CenterX = 951.8283;
	//double CenterY = 543.8695;

	double PosGps_x, PosGps_y;
	//图像上点关于中心点的位置偏移量然后，求解其相对于图像中心点的距离
	//正北方向为0度，顺时针为正0-360度
	//假定偏航角与北向的夹角，顺时针为正
	//yaw = 90;
	double yawRad = rad(yaw);
	double targetGeox = (ObjectDetectionFilterCorrect.x - CenterX) / heightPixelX * height;
	double targetGeoy = (ObjectDetectionFilterCorrect.y - CenterY) / heightPixelY * height;
	double dist = sqrt(targetGeox * targetGeox + targetGeoy * targetGeoy);

	cout << "与飞机的距离: " << dist << endl;

	//考虑偏航角，首先明确图像沿偏航角转动之后的目标点位置，逆时针旋转
	double targetGeoxAddYaw = (ObjectDetectionFilterCorrect.x - CenterX) * cos(yawRad) - (ObjectDetectionFilterCorrect.y - CenterY) * sin(yawRad);
	double targetGeoyAddYaw = (ObjectDetectionFilterCorrect.y - CenterY) * cos(yawRad) + (ObjectDetectionFilterCorrect.x - CenterX) * sin(yawRad);
	bool debug = 1;
	if (debug) {
		double targetGeoxAddYawTmp = (ObjectDetectionFilterCorrect.x - CenterX) * cos(yawRad) - (ObjectDetectionFilterCorrect.y - CenterY) * sin(yawRad) + CenterX;
		double targetGeoyAddYawTmp = (ObjectDetectionFilterCorrect.y - CenterY) * cos(yawRad) + (ObjectDetectionFilterCorrect.x - CenterX) * sin(yawRad) + CenterY;
		cout << "旋转后的x坐标: " << targetGeoxAddYawTmp << endl;
		cout << "旋转后的y坐标: " << targetGeoyAddYawTmp << endl;
		Mat srcRotate = cv::Mat(1080, 1920, CV_8UC3, Scalar(255, 255, 255));
		circle(srcRotate, Point(targetGeoxAddYawTmp, targetGeoyAddYawTmp), 5, Scalar(0, 0, 255), -1);
		circle(srcRotate, Point(CenterX, CenterY), 20, Scalar(0, 0, 0), -1);
		namedWindow("srcRotate", WINDOW_NORMAL);
		imshow("srcRotate", srcRotate);
		waitKey(1);
	}

	double brng;
	if ((targetGeoxAddYaw > 0) && (targetGeoyAddYaw < 0)) //右上
	{
		brng = deg(atan(abs(targetGeoxAddYaw / targetGeoyAddYaw)));
		brngImgInformation = "right up";
	}

	if ((targetGeoxAddYaw > 0) && (targetGeoyAddYaw > 0)) //右下
	{
		brngImgInformation = "right down";
		brng = 180 - deg(atan(abs(targetGeoxAddYaw / targetGeoyAddYaw)));

	}
	if ((targetGeoxAddYaw < 0) && (targetGeoyAddYaw > 0)) //左下
	{
		brng = deg(atan(abs(targetGeoxAddYaw / targetGeoyAddYaw))) + 180;
		brngImgInformation = "left down";

	}
	if ((targetGeoxAddYaw < 0) && (targetGeoyAddYaw < 0))  //左上
	{
		brng = 360 - deg(atan(abs(targetGeoxAddYaw / targetGeoyAddYaw)));
		brngImgInformation = "left up";

	}
	brngImg = brng;

	cout << "偏航角" << yaw << endl;
	cout << "方位角" << brng << endl;
	double lonBack, latBack;
	double yuntaiDist = 0.375;
	computerThatLonLat(lon, lat, brng, yuntaiDist, lonBack, latBack);

	computerThatLonLat(lonBack, latBack, brng, dist, PosGps_x, PosGps_y);
	double PoxGps_z = ObjectDetectionFilterCorrect.z;
	ObjectDetectionGPS = Point3d(PosGps_x, PosGps_y, PoxGps_z);
}


void test::computerThatLonLat(double lon, double lat, double brng, double dist, double& AfterLon, double& AfterLat) {

	//大地坐标系资料WGS - 84 长半径a = 6378137 短半径b = 6356752.3142 扁率f = 1 / 298.2572236
	/** 长半径a=6378137 */
	double a = 6378137;
	/** 短半径b=6356752.3142 */
	double b = 6356752.3142;
	/** 扁率f=1/298.2572236 */
	double f = 1 / 298.2572236;
	double alpha1 = rad(brng);
	double sinAlpha1 = sin(alpha1);
	double cosAlpha1 = cos(alpha1);

	double tanU1 = (1 - f) * tan(rad(lat));
	double cosU1 = 1 / sqrt((1 + tanU1 * tanU1));
	double sinU1 = tanU1 * cosU1;
	double sigma1 = atan2(tanU1, cosAlpha1);
	double sinAlpha = cosU1 * sinAlpha1;
	double cosSqAlpha = 1 - sinAlpha * sinAlpha;
	double uSq = cosSqAlpha * (a * a - b * b) / (b * b);
	double A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)));
	double B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)));

	double cos2SigmaM = 0;
	double sinSigma = 0;
	double cosSigma = 0;
	double sigma = dist / (b * A), sigmaP = 2 * 3.1415926;
	while (abs(sigma - sigmaP) > 1e-12) {
		cos2SigmaM = cos(2 * sigma1 + sigma);
		sinSigma = sin(sigma);
		cosSigma = cos(sigma);
		double deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)
			- B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)));
		sigmaP = sigma;
		sigma = dist / (b * A) + deltaSigma;
	}

	double tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1;
	double lat2 = atan2(sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
		(1 - f) * sqrt(sinAlpha * sinAlpha + tmp * tmp));
	double lambda = atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1);
	double C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha));
	double L = lambda - (1 - C) * f * sinAlpha
		* (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)));
	double revAz = atan2(sinAlpha, -tmp); // final bearing
	//std::cout << "revAz" << revAz << endl;
	//std::cout << "lon:" << lon + deg(L) << " " << deg(lat2) << endl;
	AfterLon = lon + deg(L);
	AfterLat = deg(lat2);
}


void test::GetIPPEAttitude(Mat srcImg, Rect src, Mat& rvec1, Mat& rvec2, Mat& tvec1, Mat& tvec2) {
	//Point2f Point3 = Point2f(src.x, src.y);
	//Point2f Point1 = Point2f(src.x + src.width, src.y + src.height);
	//Point2f Point0 = Point2f(src.x, src.y + src.height);
	//Point2f Point2 = Point2f(src.x + src.width, src.y);

	//cv::Mat imagePointsud(1, 4, CV_64FC2);
	//imagePointsud.ptr<Vec2d>(0)[0] = Vec2d(src.x, src.y + src.height);
	//imagePointsud.ptr<Vec2d>(0)[1] = Vec2d(src.x + src.width, src.y + src.height);
	//imagePointsud.ptr<Vec2d>(0)[2] = Vec2d(src.x + src.width, src.y);
	//imagePointsud.ptr<Vec2d>(0)[3] = Vec2d(src.x, src.y);


	int RectSize = min(src.width, src.height);

	Point2f Point3 = Point2f(src.x + src.width / 2 - RectSize / 2, src.y + src.height / 2 - RectSize / 2);
	Point2f Point1 = Point2f(src.x + src.width / 2 + RectSize / 2, src.y + src.height / 2 + RectSize / 2);
	Point2f Point0 = Point2f(src.x + src.width / 2 - RectSize / 2, src.y + src.height / 2 + RectSize / 2);
	Point2f Point2 = Point2f(src.x + src.width / 2 + RectSize / 2, src.y + src.height / 2 - RectSize / 2);


	cv::Mat imagePointsud(1, 4, CV_64FC2);
	//imagePointsud.ptr<Vec2d>(0)[0] = Vec2d(src.x, src.y + src.height);
	//imagePointsud.ptr<Vec2d>(0)[1] = Vec2d(src.x + src.width, src.y + src.height);
	//imagePointsud.ptr<Vec2d>(0)[2] = Vec2d(src.x + src.width, src.y);
	//imagePointsud.ptr<Vec2d>(0)[3] = Vec2d(src.x, src.y);

	imagePointsud.ptr<Vec2d>(0)[0] = Vec2d(src.x + src.width / 2 - RectSize / 2, src.y + src.height / 2 + RectSize / 2);
	imagePointsud.ptr<Vec2d>(0)[1] = Vec2d(src.x + src.width / 2 + RectSize / 2, src.y + src.height / 2 + RectSize / 2);
	imagePointsud.ptr<Vec2d>(0)[2] = Vec2d(src.x + src.width / 2 + RectSize / 2, src.y + src.height / 2 - RectSize / 2);
	imagePointsud.ptr<Vec2d>(0)[3] = Vec2d(src.x + src.width / 2 - RectSize / 2, src.y + src.height / 2 - RectSize / 2);
	//初始化姿态估计矩阵
	//set the length of the square object (in mm, m or whatever units you chose).
	double l = 3;//目标直径；
	float err1, err2;
	cv::Mat objectPoints;
	IPPE::PoseSolver planePoseSolver;
	planePoseSolver.generateSquareObjectCorners3D(l, objectPoints);

	//setup camera matrix:
	Mat cameraMatrix(3, 3, CV_64FC1);
	cameraMatrix.setTo(0);
	/*
   004参数
   Camera Intrinsics
                    IntrinsicMatrix: [3×3 double]
                        FocalLength: [1.7851e+03 1.7799e+03]
                     PrincipalPoint: [941.6524 546.5905]
                               Skew: 0

   Lens Distortion
                   RadialDistortion: [-0.1740 0.0739 0.0710]
               TangentialDistortion: [0 0]
	*/
	cameraMatrix.at<double>(0, 0) = 1785.1;
	cameraMatrix.at<double>(1, 1) = 1779.9;
	cameraMatrix.at<double>(0, 2) = 941.6524;
	cameraMatrix.at<double>(1, 2) = 546.5905;

	//setup distortion vector:
	cv::Mat distCoeffs(5, 1, CV_64FC1);
	distCoeffs.setTo(0); //There's no distortion in this demo.
	distCoeffs.at<double>(0) = -0.1740;
	distCoeffs.at<double>(1) = 0.0739;
	distCoeffs.at<double>(2) = 0.0710;



	//a ground truth pose:
	cv::Mat rvecGT(3, 1, CV_64FC1);
	cv::Mat tvecGT(3, 1, CV_64FC1);


	//there are two ways you can call IPPE:
	//   (1) do it directly:
	//   IPPE::PoseSolver::solveSquare(l, imagePoints,cameraMatrix,distCoeffs,rvec1,tvec1,err1,rvec2,tvec2, err2);

	//(2) first undistort the image points, then call solveGeneric. The time taken to run solveGeneric will be less because cv::undistortPoints is not required:
	//The returned RMS errors will be different because for (1) it is computed in pixels and for (2) it is computed in normalized pixels:
	//cv::undistortPoints(imagePoints, imagePointsud, cameraMatrix, distCoeffs);
	planePoseSolver.solveSquare(l, imagePointsud, cameraMatrix, distCoeffs, rvec1, tvec1, err1, rvec2, tvec2, err2);
	std::cout << " poses tvec1: (" << tvec1 << ")" << std::endl;
	std::cout << " poses rvec1: (" << rvec1 << ")" << std::endl;


	std::cout << " RMSE reprojection error of returned poses: (" << err1 << ", " << err2 << ")" << std::endl;
	Mat imagePointsRe;
	cv::projectPoints(objectPoints, rvec1, tvec1, cameraMatrix, distCoeffs, imagePointsRe);
	for (int i = 0; i < imagePointsRe.rows; i++) {
		std::cout << "改变后" << imagePointsRe.ptr<Vec2d>(0)[i] << std::endl;
		cv::circle(srcImg, cv::Point(imagePointsRe.ptr<Vec2d>(0)[i][0], imagePointsRe.ptr<Vec2d>(0)[i][1]), 5, Scalar(0, 0, 255), -1);
		cv::putText(srcImg, std::to_string(i), cv::Point(imagePointsRe.ptr<Vec2d>(0)[i][0], imagePointsRe.ptr<Vec2d>(0)[i][1]), 1, 10, 0);
	}
	//namedWindow("基于视觉位姿估计", WINDOW_NORMAL);
	//imshow("基于视觉位姿估计", srcImg);
	//waitKey(1);
}
