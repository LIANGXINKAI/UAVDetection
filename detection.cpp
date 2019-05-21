#define  _CRT_SECURE_NO_WARNINGS 
#include"detection.h"
#include"test0515.h"
#include<iostream>
#include <io.h>
//全局变量

//yolo相关变量
float confThreshold = 0.7; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

//开始识别标志字
bool detectionOpen = true;


//比赛中飞控传递信息
int g_latitude = -1;//本机的纬度
int g_longitude = -1;//本机经度
int g_height = -1;//本机高度
                  //比赛相关
int g_roll = -1;//本机的纬度
int g_yaw = -1;//本机经度
int g_pitch = -1;//本机高度


//识别结果标志字
bool Found = false;

//将原图进行一定比例的缩放,返回的图片尺寸为(w,h)

Mat letterbox_image(Mat&im, int w, int h)
{
    int new_w = im.cols;
    int new_h = im.rows;

    //在保证图像宽高比不变的情况下,计算放缩后的宽高
    if (((float)w / im.cols) < ((float)h / im.rows)) {
        //这个说明高度比例大于宽度比例,所以new_h要重新设置
        new_w = w;
        new_h = round(float(im.rows * w) / im.cols);
    }
    else {
        new_h = h;
        new_w = round(float(im.cols * h) / im.rows);
    }
    Mat resized;
    resize(im, resized,Size(new_w, new_h),INTER_AREA);
    Mat boxed = Mat::Mat(w, h, CV_8UC3, Scalar(128,128,128));
    resized.copyTo(boxed(Rect(int(float(w-1)/2- float(new_w-1) / 2), int(float(h-1)/2- float(new_h-1) / 2), new_w, new_h)));
    return boxed; //返回的图像尺寸为需要的(w,h)
}
void redNumberdetectionFilter()
{
    std::cout << "开始！！！" << endl;
    // Load names of classes
    string classesFile = "target.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3-tinyTest.cfg";

    //String modelWeights = "yolov3-tiny_111000.weights";
    String modelWeights = "yolov3-tiny_214000.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //save result
    int resultImageNum = 0;

    // Open  a camera stream.
    VideoCapture cap;
    cap.open("1.mp4");
    Mat Image;

    //mid YOLO variable
    Mat ImageWaitDetection, blob;
    double highwidthHeightRatio = 1.5, lowwidthHeightRatio = 0.75;

    //mid Track variable
    //
    int maxlen = 10;//保留maxlen帧的识别结果
    vector<vector<Point3d>> MaxlenFrameObjectDetection;//初始化保存检测识别目标的位置
    vector<Point3d>MaxlenFrameGPS;
    int trackCount = 0;
    vector<Point3d>oldObjectDetection;
    vector<Rect>oldObjectDetectionRect;
    vector<vector<int>>Track;//前两列分别为track的序号与track的分数，后面为依次帧的点序号

                             //similar compare variable
    vector<Mat> object_label;
    Mat labelImg0, labelImg1, labelImg2;
    labelImg0 = imread("1.png");
    object_label.push_back(labelImg0);
    labelImg1 = imread("2.png");
    object_label.push_back(labelImg1);
    labelImg2 = imread("3.png");
    object_label.push_back(labelImg2);

    // detection loopm

    objectDetection ObjectDetectionProcedure;
    while (detectionOpen) {
        resultImageNum = resultImageNum + 1;
        cap >> Image;
        if (Image.empty()) {
            cout << "No PICTURES Done processing !!!" << endl;
            waitKey(3000);
            break;
        }
        if (resultImageNum % 5 != 0)
            continue;
        Mat ImageWaitDetectionOrig;
        ImageWaitDetectionOrig = Image.clone();
        ImageWaitDetection=letterbox_image(ImageWaitDetectionOrig, inpWidth, inpHeight);
        imshow("ImageWaitDetection", ImageWaitDetection);
        waitKey(1);
        //经纬度数据和欧拉角数据
        int lat = g_latitude;
        int lon = g_longitude;
        int height = g_height;
        double roll = g_roll;
        double yaw = g_yaw;
        double pitch = g_pitch;
        // Create a 4D blob from a frame.
        blobFromImage(ImageWaitDetection, blob, 1 / 255.F, Size(inpWidth, inpHeight), Scalar(), true, false);
        //Sets the input to the network
        net.setInput(blob);
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        vector<String>outnames = net.getUnconnectedOutLayersNames();
        for (auto i : outnames) {
            cout << i << endl;

        }
        net.forward(outs, outnames);
        // Remove the bounding boxes with low confidence
        vector<Point3d>newObjectDetection;//直接将输出赋予objectDetectionImage变量，z存类别
        vector<Rect>newObjectDetectionRect;

        //ObjectDetectionProcedure.postprocess(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio);
        //ObjectDetectionProcedure.postprocess(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio, confThreshold,nmsThreshold);
        ObjectDetectionProcedure.postprocessAddsomeResizeSpace(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio, confThreshold, nmsThreshold, inpWidth, inpHeight, ImageWaitDetectionOrig);
        namedWindow("result ImageWaitDetectionOrig", WINDOW_NORMAL);
        imshow("result ImageWaitDetectionOrig", ImageWaitDetectionOrig);
        waitKey(1);

        // 输出单帧图像目标识别结果，并以不同颜色圈定，同时以detectedFrameResult输出
        Scalar drawcolor;
        if (newObjectDetection.size() < 1) {
            Found = false;
            std::cout << "没有找到目标" << endl;
            continue;
        }

        //ImageWaitDetectionOrig   ImageWaitDetection
        for (auto i : newObjectDetection)
        {
            std::cout << "newObjectDetection：" << i << endl;
            if (i.z == 0)
                drawcolor = Scalar(255, 0, 0);
            if (i.z == 1)
                drawcolor = Scalar(0, 255, 0);
            if (i.z == 2)
                drawcolor = Scalar(0, 255, 255);
            string objectDetectionImagelabel = format("%d ", i.z);
            circle(ImageWaitDetectionOrig, Point(i.x, i.y), 5, drawcolor, -1);
        }
        Mat detectedFrameResult;
        ImageWaitDetectionOrig.convertTo(detectedFrameResult, CV_8U);
        if (resultImageNum % 1 == 0)
        {
            string writeResultImage = to_string(resultImageNum) + "_0427Result.jpg";
            imwrite(writeResultImage, detectedFrameResult);
        }
        namedWindow("result", WINDOW_NORMAL);
        imshow("result", detectedFrameResult);
        waitKey(1);


        //利用svm对类别2与类别3进行重新判断！
        //利用svm对类别2与类别3进行重新判断！ 暂时不需要
        vector<Point3d>objectDetectionImageCorrect;
        Mat svmCompareImg = Image.clone();
        //vector<Mat> ObjectDetection23;


        for (auto i = 0; i < newObjectDetection.size(); i++) {
            objectDetectionImageCorrect.push_back(newObjectDetection[i]);
        }



        for (auto i = 0; i < newObjectDetectionRect.size(); i++)
        {
            int x_tmp = max(0, newObjectDetectionRect[i].x);
            int y_tmp = max(0, newObjectDetectionRect[i].y);
            int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
            int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);
            Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
            Mat svmtmp = svmCompareImg(tmp);
            imshow("svmtmp", svmtmp);
            waitKey(1);
        }

        //for (auto i = 0; i < newObjectDetectionRect.size(); i++)
        //{
        //  int x_tmp = max(0, newObjectDetectionRect[i].x);
        //  int y_tmp = max(0, newObjectDetectionRect[i].y);
        //  int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
        //  int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);

        //  Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
        //  Mat svmtmp = svmCompareImg(tmp);
        //  ObjectDetection23.push_back(svmtmp.clone());
        //  if ((newObjectDetection[i].z == 2) || (newObjectDetection[i].z == 1)) {
        //      if (newObjectDetection[i].z == svmHogDetection(svmtmp))
        //          objectDetectionImageCorrect.push_back(newObjectDetection[i]);
        //  }
        //}

        //for (auto i : objectDetectionImageCorrect)
        //{
        //  std::cout << "objectDetectionImageCorrect：" << i << endl;
        //  if (i.z == 0)
        //      drawcolor = Scalar(255, 0, 0);
        //  if (i.z == 1)
        //      drawcolor = Scalar(0, 255, 0);
        //  if (i.z == 2)
        //      drawcolor = Scalar(0, 255, 255);
        //  string objectDetectionImagelabel = format("%d ", i.z);
        //  circle(svmCompareImg, Point(i.x, i.y), 5, drawcolor, -1);
        //}
        //namedWindow("svm filter result", WINDOW_NORMAL);
        //imshow("svm filter result", svmCompareImg);
        //waitKey(1);

        //开始迭代更新
        // 
        if (objectDetectionImageCorrect.size() < 1)
        {
            Found = false;
            std::cout << "没有找到目标" << endl;
            continue;

        }
        bool debug = 0;
        if (objectDetectionImageCorrect.size() >= 1)
        {
            if ((MaxlenFrameObjectDetection.size() < 1) && (oldObjectDetection.size() < 1))
            {
                for (auto i = 0; i < maxlen; i++) {
                    vector<Point3d>MaxlenFrame_tmp;
                    MaxlenFrameObjectDetection.push_back(MaxlenFrame_tmp);
                }
                MaxlenFrameObjectDetection.erase(MaxlenFrameObjectDetection.begin());
                MaxlenFrameObjectDetection.push_back(objectDetectionImageCorrect);


                if (debug) {
                    cout << "MaxlenFrameObjectDetection:" << endl;
                    for (auto i : MaxlenFrameObjectDetection) {
                        for (auto j : i) {
                            cout << "识别结果 " << j << " " << endl;
                        }
                    }
                }
                vector<vector<int>>new_tracks;
                for (auto i = 0; i < objectDetectionImageCorrect.size(); i++) {
                    int tracksLen = maxlen + 2;
                    vector<int>new_tracksTmp(tracksLen, -1);
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
                }
                MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
                MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
                oldObjectDetection.swap(objectDetectionImageCorrect);
                Track.insert(Track.end(), new_tracks.begin(), new_tracks.end());
                trackCount = new_tracks.size();
                Found = false;
                continue;
            }
            else {
                ObjectDetectionProcedure.objectDetectionupdate(MaxlenFrameObjectDetection, objectDetectionImageCorrect, oldObjectDetection, Track, trackCount, maxlen);
                if (debug) {
                    cout << "Tracks:" << endl;
                    for (auto i : Track) {
                        cout << "第几个Tracks" << endl;
                        for (auto j : i) {
                            cout << "Tracks新跟踪的结果 " << j << " " << endl;
                        }
                    }
                }
                MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
                MaxlenFrameGPS.push_back(Point3d(lat, lon, height));


            }
        }



        vector<vector<int>>TrackNowSatifyLen;
        int SatifyLen = 4;
        ObjectDetectionProcedure.objectDetectionGetTrack(Track, TrackNowSatifyLen, SatifyLen);
        vector<Point3d> ObjectDetectionFilterFine;
        for (auto i : TrackNowSatifyLen) {
            vector<int>offsetAccumulate;
            ObjectDetectionProcedure.get_offsets(MaxlenFrameObjectDetection, offsetAccumulate);
            int offset = offsetAccumulate[offsetAccumulate.size()-1];
            int idx = int(i[i.size()-1] - offset);
            Point3d pt_tmp= MaxlenFrameObjectDetection[MaxlenFrameObjectDetection.size()-1][idx];
            cout << "ObjectDetectionFilterFine:" << pt_tmp << endl;
            ObjectDetectionFilterFine.push_back(pt_tmp);
        }
        Mat trackImg = Image.clone();
        ObjectDetectionProcedure.DrawobjectDetectionTrack(trackImg, MaxlenFrameObjectDetection, Track, SatifyLen);



        //利用hist相关性滤除一部分结果  后续是否可以借鉴svm呢或许可以！！！
        //Mat similarityCompareImg = Image.clone();

        //for (auto i = 0; i < newObjectDetectionRect.size(); i++)
        //{
        //  double best_dSimilarity = -1;
        //  int best_label = -1;
        //  int x_tmp = max(0, newObjectDetectionRect[i].x);
        //  int y_tmp = max(0, newObjectDetectionRect[i].y);
        //  int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
        //  int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);

        //  Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
        //  Mat first = similarityCompareImg(tmp);
        //  for (auto j = 0; j < object_label.size(); j++)
        //  {

        //      Mat second = object_label[j].clone();
        //      imshow("first", first);
        //      imshow("second", second);
        //      waitKey(1);
        //      double dSimilarity = 0;
        //      compareHistColor(first, second, dSimilarity);
        //      if (dSimilarity > best_dSimilarity)
        //          best_label = j;
        //  }
        //  if ((best_label == newObjectDetection[i].z) && (best_dSimilarity > 0.9))
        //      objectDetectionImageCorrect.push_back(newObjectDetection[i]);
        //}
    }
}
//利用跟踪结果
//经纬为度，高度为cm，姿态信息为度
void redNumberdetectionFilterTrack()
{

    std::cout << "识别过程开始！！！" << endl;
    // Load names of classes
    string classesFile = "target.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3-tinyTest.cfg";

    //String modelWeights = "yolov3-tiny_111000.weights";
    String modelWeights = "yolov3-tiny_214000.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //save result
    int resultImageNum = 0;
    string resultBackName = "_0427Result";
    // Open  a camera stream.
    VideoCapture cap;
    cap.open("1.mp4");
    Mat Image;

    //mid YOLO variable
    Mat ImageWaitDetection, blob;
    double highwidthHeightRatio = 1.5, lowwidthHeightRatio = 0.75;

    //mid Track variable
    //
    int maxlen = 10;//保留maxlen帧的识别结果
    vector<vector<Point3d>> MaxlenFrameObjectDetection;//初始化保存检测识别目标的位置
    vector<Point3d>MaxlenFrameGPS;
    vector<Point3d>MaxlenFrameAngle;
    int trackCount = 0;
    vector<Point3d>oldObjectDetection;
    vector<Rect>oldObjectDetectionRect;
    vector<vector<int>>Track;//前两列分别为track的序号与track的分数，后面为依次帧的点序号

                             //similar compare variable
    vector<Mat> object_label;
    Mat labelImg0, labelImg1, labelImg2;
    labelImg0 = imread("1.png");
    object_label.push_back(labelImg0);
    labelImg1 = imread("2.png");
    object_label.push_back(labelImg1);
    labelImg2 = imread("3.png");
    object_label.push_back(labelImg2);

    // detection loopm

    objectDetection ObjectDetectionProcedure;
    while (detectionOpen) {
        resultImageNum = resultImageNum + 1;
        cap >> Image;
        if (Image.empty()) {
            cout << "No PICTURES Done processing !!!" << endl;
            waitKey(3000);
            break;
        }
        if (resultImageNum % 5 != 0)
            continue;
        Mat ImageWaitDetectionOrig;
        ImageWaitDetectionOrig = Image.clone();
        ImageWaitDetection = letterbox_image(ImageWaitDetectionOrig, inpWidth, inpHeight);
        imshow("ImageWaitDetection", ImageWaitDetection);
        waitKey(1);
        //经纬度数据和欧拉角数据
        double lat = g_latitude;
        double lon = g_longitude;
        double height = g_height;
        double roll = g_roll;
        double yaw = g_yaw;
        double pitch = g_pitch;
        // Create a 4D blob from a frame.
        blobFromImage(ImageWaitDetection, blob, 1 / 255.F, Size(inpWidth, inpHeight), Scalar(), true, false);
        //Sets the input to the network
        net.setInput(blob);
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        vector<String>outnames = net.getUnconnectedOutLayersNames();
        for (auto i : outnames) {
            cout << i << endl;

        }
        net.forward(outs, outnames);
        // Remove the bounding boxes with low confidence
        vector<Point3d>newObjectDetection;//直接将输出赋予objectDetectionImage变量，z存类别
        vector<Rect>newObjectDetectionRect;

        //ObjectDetectionProcedure.postprocess(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio);
        //ObjectDetectionProcedure.postprocess(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio, confThreshold,nmsThreshold);
        ObjectDetectionProcedure.postprocessAddsomeResizeSpace(ImageWaitDetection, outs, newObjectDetection, newObjectDetectionRect, highwidthHeightRatio, lowwidthHeightRatio, confThreshold, nmsThreshold, inpWidth, inpHeight, ImageWaitDetectionOrig);
        namedWindow("result ImageWaitDetectionOrig", WINDOW_NORMAL);
        imshow("result ImageWaitDetectionOrig", ImageWaitDetectionOrig);
        waitKey(1);

        // 输出单帧图像目标识别结果，并以不同颜色圈定，同时以detectedFrameResult输出

        Mat detectedFrameResult;
        ImageWaitDetectionOrig.convertTo(detectedFrameResult, CV_8U);
        //没有识别结果的依然输出,且在图像上明显标记
        Scalar drawcolor;
        if (newObjectDetection.size() < 1) {
            Found = false;
            std::cout << "当前帧没有找到目标" << endl;
            string resultTxt = "当前帧没有找到目标";
            putText(detectedFrameResult, resultTxt, Point(detectedFrameResult.rows/2, detectedFrameResult.cols/2), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
            string writeResultImage = to_string(resultImageNum) + resultBackName;
            imwrite(writeResultImage, detectedFrameResult);
            continue;
        }

        for (auto i : newObjectDetection)
        {
            std::cout << "newObjectDetection：" << i << endl;
            if (i.z == 0)
                drawcolor = Scalar(255, 0, 0);
            if (i.z == 1)
                drawcolor = Scalar(0, 255, 0);
            if (i.z == 2)
                drawcolor = Scalar(0, 255, 255);
            string objectDetectionImagelabel = format("%d ", i.z);
            circle(detectedFrameResult, Point(i.x, i.y), 5, drawcolor, -1);
        }
        int resultImageSampleSave = 10; //存识别结果的周期
        if (resultImageNum % resultImageSampleSave == 0)
        {
            string writeResultImage = to_string(resultImageNum) + resultBackName;
            imwrite(writeResultImage, detectedFrameResult);
        }
        namedWindow("result", WINDOW_NORMAL);
        imshow("result", detectedFrameResult);
        waitKey(1);
        
        //以上内容为存单帧的识别结果，后续为利用跟踪的结果辅助识别判断与定位@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        bool debug = 0;
        if (newObjectDetection.size() >= 1)
        {
            if ((MaxlenFrameObjectDetection.size() < 1))
            {
                for (auto i = 0; i < maxlen; i++) {
                    vector<Point3d>MaxlenFrame_tmp;
                    MaxlenFrameObjectDetection.push_back(MaxlenFrame_tmp);
                }
                MaxlenFrameObjectDetection.erase(MaxlenFrameObjectDetection.begin());
                MaxlenFrameObjectDetection.push_back(newObjectDetection);


                if (debug) {
                    cout << "MaxlenFrameObjectDetection:" << endl;
                    for (auto i : MaxlenFrameObjectDetection) {
                        for (auto j : i) {
                            cout << "识别结果 " << j << " " << endl;
                        }
                    }
                }
                vector<vector<int>>new_tracks;
                for (auto i = 0; i < newObjectDetection.size(); i++) {
                    int tracksLen = maxlen + 2;
                    vector<int>new_tracksTmp(tracksLen, -1);
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
                    MaxlenFrameAngle.push_back(Point3d(-1, -1, -1));
                }
                MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
                MaxlenFrameGPS.push_back(Point3d(lat, lon, height));
                MaxlenFrameAngle.erase(MaxlenFrameAngle.begin());
                MaxlenFrameAngle.push_back(Point3d(pitch, yaw, roll));
                oldObjectDetection.swap(newObjectDetection);
                Track.insert(Track.end(), new_tracks.begin(), new_tracks.end());
                trackCount = new_tracks.size();
                Found = false;
                continue;
            }
            else {
                ObjectDetectionProcedure.objectDetectionupdate(MaxlenFrameObjectDetection, newObjectDetection, oldObjectDetection, Track, trackCount, maxlen);
                if (debug) {
                    cout << "Tracks:" << endl;
                    for (auto i : Track) {
                        cout << "第几个Tracks" << endl;
                        for (auto j : i) {
                            cout << "Tracks新跟踪的结果 " << j << " " << endl;
                        }
                    }
                }
                MaxlenFrameGPS.erase(MaxlenFrameGPS.begin());
                MaxlenFrameGPS.push_back(Point3d(lat, lon, height));


            }
        }



        vector<vector<int>>TrackNowSatifyLen;
        int SatifyLen = 4;
        ObjectDetectionProcedure.objectDetectionGetTrack(Track, TrackNowSatifyLen, SatifyLen);
        vector<Point3d> ObjectDetectionFilterFine;
        for (auto i : TrackNowSatifyLen) {
            vector<int>offsetAccumulate;
            ObjectDetectionProcedure.get_offsets(MaxlenFrameObjectDetection, offsetAccumulate);
            int offset = offsetAccumulate[offsetAccumulate.size() - 1];
            int idx = int(i[i.size() - 1] - offset);
            Point3d pt_tmp = MaxlenFrameObjectDetection[MaxlenFrameObjectDetection.size() - 1][idx];
            cout << "ObjectDetectionFilterFine:" << pt_tmp << endl;
            ObjectDetectionFilterFine.push_back(pt_tmp);
        }
        Mat trackImg = Image.clone();
        ObjectDetectionProcedure.DrawobjectDetectionTrack(trackImg, MaxlenFrameObjectDetection, Track, SatifyLen);



        //利用hist相关性滤除一部分结果  后续是否可以借鉴svm呢或许可以！！！
        //Mat similarityCompareImg = Image.clone();

        //for (auto i = 0; i < newObjectDetectionRect.size(); i++)
        //{
        //  double best_dSimilarity = -1;
        //  int best_label = -1;
        //  int x_tmp = max(0, newObjectDetectionRect[i].x);
        //  int y_tmp = max(0, newObjectDetectionRect[i].y);
        //  int width_tmp = min(newObjectDetectionRect[i].width, Image.cols - newObjectDetectionRect[i].x);
        //  int height_tmp = min(newObjectDetectionRect[i].height, Image.rows - newObjectDetectionRect[i].y);

        //  Rect tmp(x_tmp, y_tmp, width_tmp, height_tmp);
        //  Mat first = similarityCompareImg(tmp);
        //  for (auto j = 0; j < object_label.size(); j++)
        //  {

        //      Mat second = object_label[j].clone();
        //      imshow("first", first);
        //      imshow("second", second);
        //      waitKey(1);
        //      double dSimilarity = 0;
        //      compareHistColor(first, second, dSimilarity);
        //      if (dSimilarity > best_dSimilarity)
        //          best_label = j;
        //  }
        //  if ((best_label == newObjectDetection[i].z) && (best_dSimilarity > 0.9))
        //      objectDetectionImageCorrect.push_back(newObjectDetection[i]);
        //}
    }
}
double PI = 3.141592697;
double getAngle1(double lat_a, double lng_a, double lat_b, double lng_b) {
    double d = 0;

    lat_a = lat_a/1e+7 * PI / 180.0;

    lng_a = lng_a / 1e+7 * PI / 180.0;

    lat_b = lat_b / 1e+7 * PI / 180.0;

    lng_b = lng_b / 1e+7 * PI / 180.0;

    d = sin(lat_a) * sin(lat_b) + cos(lat_a) * cos(lat_b) * cos(lng_b - lng_a);

    d = sqrt(1 - d * d);

    d = cos(lat_b) * sin(lng_b - lng_a) / d;

    d = asin(d) * 180.0 / PI;
    return d;
}
double EARTH_RADIUS = 6378.137;
double get_distance(double lat1, double lng1, double lat2, double lng2)
{
    double radLat1 = lat1 * PI / 180.0;   
    double radLat2 = lat2 * PI / 180.0;   
    double a = radLat1 - radLat2;
    double b = lng1 * PI / 180.0 - lng2* PI / 180.0; 
    double dst = 2 * asin((sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2))));
    dst = dst * EARTH_RADIUS;
    dst = round(dst * 10000) / 10000;
    return dst;
}

int get_angle(double lat1, double lng1, double lat2, double lng2)
{
    double x = lat1 - lat2;//t d
    double y = lng1 - lng2;//z y
    int angle = -1;
    if (y == 0 && x > 0) angle = 0;
    if (y == 0 && x < 0) angle = 180;
    if (x == 0 && y > 0) angle = 90;
    if (x == 0 && y < 0) angle = 270;
    if (angle == -1)
    {
        double dislat = get_distance(lat1, lng2, lat2, lng2);
        double dislng = get_distance(lat2, lng1, lat2, lng2);
        if (x > 0 && y > 0) angle = atan2(dislng, dislat) / PI * 180;
        if (x < 0 && y > 0) angle = atan2(dislat, dislng) / PI * 180 + 90;
        if (x < 0 && y < 0) angle = atan2(dislng, dislat) / PI * 180 + 180;
        if (x > 0 && y < 0) angle = atan2(dislat, dislng) / PI * 180 + 270;
    }
    return angle;
}
double getAngle1T(double lat_a, double lng_a, double lat_b, double lng_b) {
    lat_a = lat_a / 180 * 3.1415926;
    lng_a = lng_a / 180 * 3.1415926;

    lat_b = lat_b / 180 * 3.1415926;

    lng_b = lng_b / 180 * 3.1415926;

    double y = sin(lng_b - lng_a) * cos(lat_b);
    double x = cos(lat_a)*sin(lat_b) - sin(lat_a)*cos(lat_b)*cos(lng_b - lng_a);
    double bearing = atan2(y, x);

    bearing = bearing/3.1415926*180;
    if (bearing < 0) {
        bearing = bearing + 360;
    }
    return bearing;

}


//0516--自动处理数据
string preTXT = "C:\\testDetection\\GPS\\resultTrackGPS_";
void splitTest(string& s, string& delim, vector<string>&ret) {
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != string::npos) {//查找到匹配
        ret.push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (index - last > 0)ret.push_back(s.substr(last, index - last));
}

void getFiles(string path, string exd, vector<string>& files) {
    //文件句柄
    intptr_t hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    files.push_back(pathName.assign(path).append("\\").append(fileinfo.name)); // 要得到绝对目录使用该语句
                                                                                               //如果使用
                                                                                               //files.push_back(fileinfo.name); // 只要得到文件名字使用该语句
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}
// 0516：注意：其实是因为_findfirst()返回类型为intptr_t而非long型，从“intptr_t”转换到“long”丢失了数据。
//void getFiles(string path, vector<string>& files)
//{
//  //文件句柄  
//  intptr_t   hFile = 0;
//  //文件信息  
//  struct _finddata_t fileinfo;
//  string p;
//  if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
//  {
//      do
//      {
//          //如果是目录,迭代之  
//          //如果不是,加入列表  
//          if ((fileinfo.attrib &  _A_SUBDIR))
//          {
//              if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//                  getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
//          }
//          else
//          {
//              files.push_back(p.assign(path).append("\\").append(fileinfo.name));
//          }
//      } while (_findnext(hFile, &fileinfo) == 0);
//      _findclose(hFile);
//  }
//}
void readInToMatrix(fstream &ObjectDetection, fstream &GPS, fstream &Angle, int classID,Point2d realLonLat,int IndiceName,double&errorDis, Point3d &ObjectDetectionGPS) {
    string GPSTXT = preTXT + to_string(IndiceName)+ "_GPS.txt";
    string AngleTXT = preTXT + to_string(IndiceName) + "_Angle.txt";
    string DetectionTXT = preTXT + to_string(IndiceName) + "_detection.txt";
    ObjectDetection.open(DetectionTXT, ios::in);//打开一个file
    GPS.open(GPSTXT, ios::in);//打开一个file
    Angle.open(AngleTXT, ios::in);//打开一个file
    if (!ObjectDetection.is_open()) {
        cout << "Can not find " << DetectionTXT << endl;
        system("pause");
    }
    if (!GPS.is_open()) {
        cout << "Can not find " << GPSTXT << endl;
        system("pause");
    }
    if (!Angle.is_open()) {
        cout << "Can not find " << AngleTXT << endl;
        system("pause");
    }

    string buff;

    int ObjectDetection_i=0;
    vector<vector<vector<double>>> ObjectDetectionValue;
    while (getline(ObjectDetection, buff)) {
        if (buff.empty())
            continue;
        buff.erase(0, buff.find_first_not_of("["));
        buff.erase(buff.find_last_not_of("]")+1);
        vector<string>ObjectDetectionPred;
        string splitTxt = "][";
        splitTest(buff, splitTxt, ObjectDetectionPred);

        vector<vector<double>> nums;
        for (auto i : ObjectDetectionPred)
        {
            vector<double >Onedetectionnums;
            // string->char *
            char *s_input = (char *)i.c_str();
            const char * split = ", ";
            // 以‘，’为分隔符拆分字符串
            char *p = strtok(s_input, split);
            double a;
            while (p != NULL) {
                // char * -> int
                a = atof(p);
                //cout << a << endl;
                Onedetectionnums.push_back(a);
                p = strtok(NULL, split);
            }
            nums.push_back(Onedetectionnums);
        }
        ObjectDetectionValue.push_back(nums);
        ObjectDetection_i++;
    }//end while
    ObjectDetection.close();


    int GPS_i=0;
    vector<vector<double>> GPSValue;
    while (getline(GPS, buff)) {
        if (buff.empty())
            continue;
        buff.erase(0, buff.find_first_not_of("["));
        buff.erase(buff.find_last_not_of("]") + 1);
        vector<double> nums;
        // string->char *
        char *s_input = (char *)buff.c_str();
        const char * split = ", ";
        // 以‘，’为分隔符拆分字符串
        char *p = strtok(s_input, split);
        double a;
        while (p != NULL) {
            // char * -> int
            a = atof(p);
            //cout << a << endl;
            nums.push_back(a);
            p = strtok(NULL, split);
        }//end while
        GPSValue.push_back(nums);
        GPS_i++;
    }//end while
    GPS.close();

    int Angle_i=0;
    vector<vector<double>> AngleValue;
    while (getline(Angle, buff)) {
        if (buff.empty())
            continue;
        buff.erase(0, buff.find_first_not_of("["));
        buff.erase(buff.find_last_not_of("]") + 1);
        vector<double> nums;
        // string->char *
        char *s_input = (char *)buff.c_str();
        const char * split = ", ";
        // 以‘，’为分隔符拆分字符串
        char *p = strtok(s_input, split);
        double a;
        while (p != NULL) {
            // char * -> int
            a = atof(p);
            //cout << a << endl;
            nums.push_back(a);
            p = strtok(NULL, split);
        }//end while
        AngleValue.push_back(nums);
        Angle_i++;
    }//end while
    Angle.close();



    //测试参数整理
    Point3d ObjectDetectionFilterCorrect= Point3d(-1,-1,-1);
    //for (auto i : ObjectDetectionValue) {
    //  for (auto j : i) {
    //      if (j[2] == classID)
    //          ObjectDetectionFilterCorrect = Point3d(j[0], j[1], j[2]);
    //  }
    //}
    
    for (auto i : ObjectDetectionValue[ObjectDetectionValue.size() - 1]) {
        for (auto j:i) {
            if (j== classID)
                ObjectDetectionFilterCorrect = Point3d(i[0], i[1], i[2]);
        }
    }
    if (ObjectDetectionFilterCorrect.x == -1)
    {
        cout << "没有对应类别的识别结果" << endl;
        return;
    }

    //Point3d ObjectDetectionGPS;
    double yaw = AngleValue[AngleValue.size()-1][1];


    double lat = GPSValue[GPSValue.size()-1][0]; //纬度
    double lon = GPSValue[GPSValue.size() - 1][1];//经度
    double heightError = 0.42;
    double height = GPSValue[GPSValue.size() - 1][2] - heightError;//高度

    double yitaX = 57.5;
    double yitaY = 32.5;
    double brngImg;
    string brngImgInformation;
    test testSet;
    testSet.locationEveryTarget(ObjectDetectionFilterCorrect, ObjectDetectionGPS, yaw, lat, lon, height, yitaX, yitaY, brngImg, brngImgInformation);
    double errox = ObjectDetectionGPS.x - realLonLat.x;
    double erroy = ObjectDetectionGPS.y - realLonLat.y;
    double distance = get_distance(ObjectDetectionGPS.x, ObjectDetectionGPS.y, realLonLat.x, realLonLat.y);
    errorDis = distance;
    cout << "开始!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "lat: " << std::setprecision(20) << lat << endl;
    cout << "lon: " << std::setprecision(20) << lon << endl;
    cout << "yaw: " << std::setprecision(20) << yaw << endl;
    cout << "height: " << std::setprecision(20) << height << endl;
    cout << "识别目标图像位置X: " << std::setprecision(10) << ObjectDetectionFilterCorrect.x << endl;
    cout << "识别目标图像位置Y: " << std::setprecision(10) << ObjectDetectionFilterCorrect.y << endl;
    cout << "识别目标图像类型: " << std::setprecision(10) << ObjectDetectionFilterCorrect.z << endl;
    cout << "结果" << endl;
    cout << "识别定位结果ObjectDetectionGPS: " << std::setprecision(10) << ObjectDetectionGPS << endl;
    cout << "真是目标定位结果: " << std::setprecision(10) << realLonLat << endl;
    cout << "erro lat: " << std::setprecision(20) << errox << endl;
    cout << "erro lon: " << std::setprecision(20) << erroy << endl;
    cout << "距离误差: " << distance << endl;
    cout << "结束" << endl;
}
struct FileClassmember {
    double num;
    string name;
};
bool compareFileClassmember(FileClassmember tmp1, FileClassmember tmp2) {
    return (tmp1.num < tmp2.num);
}
int main()
{
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@SVM相关@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    //构建svm 数据库
    //buildSVMdataset();
    //训练svm 模型
    //train_svm_hog();
    //测试svm 模型
    //svm_hog_detect();

    //验证累加操作！
    //vector<int>a{0,1,2,3,4};
    ////[0,1,2,3,4]
    ////[0,0,1,3,6]
    //vector<int>offsetAccumulate;
    //offsetAccumulate.push_back(0);
    //for (auto it = a.begin()+1; it != a.end(); it++)
    //{
    //  offsetAccumulate.push_back(std::accumulate(a.begin(), it, 0));
    //}
    //if (1) {
    //  cout << "offsetAccumulate:" << endl;
    //  for (auto i : offsetAccumulate) {
    //      cout << "offsetAccumulate结果 " << i << " " << endl;
    //  }
    //}
    //redNumberdetectionFilterTrack();


    //测试性能
    //构建循环读取有效的图像;
    //并实时显示对应的图像
    //for ();
    string path = "C:\\testDetection\\GPS";
    string exd = "jpg";
    vector<string>Imgfiles;

    getFiles(path, exd,Imgfiles);
    vector<double>IndiceSetName;
    vector<FileClassmember> testFilemember;
    for (auto i : Imgfiles) {
        string cc = i;
        cc.erase(i.size() - 15,i.size());
        cc.erase(0, 36);
        FileClassmember tmp;
        tmp.name = i;
        tmp.num = stod(cc);
        testFilemember.push_back(tmp);
        IndiceSetName.push_back(stod(cc));
        //cout << "文件名 " << i<< endl;
    }
    sort(testFilemember.begin(), testFilemember.end(), compareFileClassmember);
    double meanDis = 0;
    double meandisNum = 0;
    Point3d GPSsum = Point3d(0, 0,0);
    for (auto i = 0; i < testFilemember.size();i++) {
        double errorTmp = 0;
        cout << "序号名 " << testFilemember[i].name << endl;
        fstream ObjectDetection, GPS, Angle;
        int classID =1;
        //0517-RTK更新后-1号机-004
        //0517-RTK更新后-2号机-006
        //1号机前推过程：
        //Point2d realLonLat = Point2d(121.3458240, 30.8353394); //从北到南第2个3号目标
        //Point2d realLonLat = Point2d(121.3457420, 30.8352405); //从北到南第2个1号目标
        //Point2d realLonLat = Point2d(121.3458509, 30.8351886); //从北到南第1个2号目标
        //Point2d realLonLat = Point2d(121.3459859, 30.8349264); //从北到南第2个2号目标
        //Point2d realLonLat = Point2d(121.3458048, 30.8348069); //从北到南第5个1号目标
        //Point2d realLonLat = Point2d(121.3459580, 30.8346684); //从北到南第6个1号目标
        

        //1号机后退过程：
        //Point2d realLonLat = Point2d(121.3457125, 30.8349676); //从北到南第4个1号目标
        //2号机前推过程：
        //Point2d realLonLat = Point2d(121.3459525, 30.8354746); //从北到南第1个1号目标
        //Point2d realLonLat = Point2d(121.3460341, 30.8354481); //从北到南第1个3号目标
        //Point2d realLonLat = Point2d(121.3459859, 30.8349264); //从北到南第2个2号目标 重复目标
        //Point2d realLonLat = Point2d(121.3460363, 30.8352250); //从北到南第3个1号目标



        //0517-RTK更新前
        //Point2d realLonLat = Point2d(121.3458308, 30.8353592); //从北到南第2个3号目标
        //Point2d realLonLat = Point2d(121.3457528, 30.8352609); //从北到南第2个1号目标
        //Point2d realLonLat = Point2d(121.3458608, 30.8352128); //从北到南第1个2号目标
        //Point2d realLonLat = Point2d(121.3457192, 30.8349846); //从北到南第4个1号目标
        //Point2d realLonLat = Point2d(121.3458128, 30.8348284); //从北到南第5个1号目标
        //Point2d realLonLat = Point2d(121.3459676, 30.8346903); //从北到南第6个1号目标

        //Point2d realLonLat = Point2d(121.3459897, 30.8349486); //从北到南第2个2号目标
        //Point2d realLonLat = Point2d(121.3460384, 30.8354681); //从北到南第1个3号目标
        //Point2d realLonLat = Point2d(121.3460439, 30.8352501); //从北到南第3个1号目标
        //Point2d realLonLat = Point2d(121.3459590, 30.8354956); //从北到南第1个1号目标

        //0516 
        //Point2d realLonLat = Point2d(121.2071447, 31.2194879); //目标3
        Point2d realLonLat = Point2d(121.2070566, 31.2194834); //目标2
        //Point2d realLonLat = Point2d(121.2070214, 31.2194479); //目标1

        int IndiceName = testFilemember[i].num;
        Mat src = imread(testFilemember[i].name);
        Point3d ObjectGPSTmp;
        readInToMatrix(ObjectDetection, GPS, Angle, classID, realLonLat, IndiceName, errorTmp, ObjectGPSTmp);
        namedWindow("result", WINDOW_NORMAL);
        imshow("result", src);
        waitKey(0);
        if (errorTmp > 0) {
            GPSsum = GPSsum + ObjectGPSTmp;
            meanDis = meanDis + errorTmp;
            meandisNum++;
            double meanTMP = meanDis / meandisNum;
            double GPSsumMean = get_distance(GPSsum.x / meandisNum, GPSsum.y / meandisNum, realLonLat.x, realLonLat.y);
            cout << "总目标GPS信息：" << GPSsum << endl;
            cout << "平均距离误差：" << meanTMP << endl;
            cout << "叠加GPS后的平均距离误差：" << GPSsumMean << endl;
        }
    }




    test testSet;
    //testSet.redNumberdetection();
    //testSet.redNumberdetectionFilter();
    //Point3f ObjectDetectionFilterCorrect= Point3f(1254, 407, 0);
    //Point3f ObjectDetectionFilterCorrect = Point3f(916, 936, 2);
    //Point3f ObjectDetectionFilterCorrect = Point3f(1613, 558, 1);
    //Point2d ObjectDetectionFilterCorrectReal = Point2d(121.2070686,31.2194775); //2
    Point2d ObjectDetectionFilterCorrectReal = Point2f(121.2071527, 31.2194899);//3



    //0516--线程代码

    //std::thread t(&bar::foo, bar());
    //thread threadReadImages(&test::readImage, test());
    //thread threadDetectionFromCamera(&test::redNumberdetectionFilter, test());
    //threadReadImages.join();
    //threadDetectionFromCamera.join();
    
    Point3d ObjectDetectionFilterCorrect = Point3d(733, 276, 2);

    Point3d ObjectDetectionGPS;
    //double yaw = 163.5-6;
    double yaw = -107.6;
    //double lat = 31.2194463; //纬度
    //double lon = 121.2071365;//经度
    //double height = 27.94;//高度

    //double lat = 31.2194776; //纬度
    //double lon = 121.2070661;//经度

    double lat = 31.2195151; //纬度
    double lon = 121.2072224;//经度
    double heightError = 0.42;
    double height = 28.68- heightError;//高度

    double yitaX = 57.5;
    double yitaY = 32.5;
    double brngImg;
    string brngImgInformation;
    testSet.locationEveryTarget(ObjectDetectionFilterCorrect, ObjectDetectionGPS, yaw, lat, lon, height, yitaX, yitaY, brngImg, brngImgInformation);
        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@SVM相关@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    //thread threadDetection(&detection);
    //thread threadReadAndWriteFromCamera(&readANDwriteFromCamera);
    double errox = ObjectDetectionGPS.x - ObjectDetectionFilterCorrectReal.x;
    double erroy = ObjectDetectionGPS.y - ObjectDetectionFilterCorrectReal.y;
    cout << "ObjectDetectionGPS" << std::setprecision(10)<< ObjectDetectionGPS << endl;
    cout << "erro lat: " << std::setprecision(20)<< errox << endl;
    cout << "erro lon: " << std::setprecision(20) << erroy << endl;

    //thread threadDetection(&redNumberdetection);
    //threadDetection.join();
    //redNumberdetection();
    //detection();
    //threadReadAndWriteFromCamera.join();
    waitKey(0);
    return 0;
}
int main111()
{
    test tmp;
    tmp.redNumberdetectionFilterFromMovie();

    waitKey(0);
    return 0;
}