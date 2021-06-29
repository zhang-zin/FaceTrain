#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector {
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
            IDetector(),
            Detector(detector) {
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) {

        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);

    }

    virtual ~CascadeDetectorAdapter() {

    }

private:
    CascadeDetectorAdapter();

    cv::Ptr<cv::CascadeClassifier> Detector;
};

DetectionBasedTracker *getTracker() {
    String path = "E:\\c-project\\FaceTrain\\samples\\data\\cascade.xml";
//    String path = "E:\\opencv\\build\\etc\\lbpcascades\\lbpcascade_frontalface.xml";

    Ptr<CascadeClassifier> classifier = makePtr<CascadeClassifier>(path);
    //适配器
    Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(classifier);

    Ptr<CascadeClassifier> classifier1 = makePtr<CascadeClassifier>(path);
    //适配器
    Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(classifier1);
    //跟踪器
    DetectionBasedTracker::Parameters DetectorParams;
    DetectionBasedTracker *tracker = new DetectionBasedTracker(mainDetector, trackingDetector, DetectorParams);
    return tracker;
}

int main() {

#if 1
    DetectionBasedTracker *tracker = getTracker();

    //开启跟踪器
    tracker->run();

    // android不能使用这个玩意
    VideoCapture capture(0);
    Mat img;
    Mat gray;

    while (1) {

        capture >> img;

        /**
         * 预处理，去除图片噪声
         */
        // img的 颜色空间是 BGR，不像现在，早期的计算机中主流是bgr，而不是rgb
        cvtColor(img, gray, COLOR_BGR2GRAY);
        //增强对比度 (直方图均衡)
        equalizeHist(gray, gray);

        std::vector<Rect> faces;
        //处理
        tracker->process(gray);
        //获取结果
        tracker->getObjects(faces);

        for (Rect face : faces) {
            //画矩形
            //分别指定 bgra
            if (face.x < 0 || face.width < 0 || face.x + face.width > img.cols ||
                face.y < 0 || face.height < 0 || face.y + face.height > img.rows) {
                continue;
            }
            rectangle(img, face, Scalar(255, 0, 255));
#if 1
            static int i = 0;
            /**
             * 制作训练正样本........
             */
            //使用opencv自带的模型 记录你的脸作为样本
            //把找到的人脸扣出来
            Mat m;
            //把img中的脸部位拷贝到m中
            img(face).copyTo(m);
            //把人脸 重新设置为 24x24大小的图片
            resize(m, m, Size(24, 24));
            //转成灰度
            cvtColor(m, m, COLOR_BGR2GRAY);
//            char p[100];
//            sprintf(p, "E:\\c-project\\FaceTrain\\samples\\zhang/%d.jpg", i++);
//            //把mat写出为jpg文件
//            imwrite(p, m);
            m.release();
#endif
        }
        imshow("摄像头", img);
        //延迟30ms 按Esc退出 ,27 =Esc
        if (waitKey(30) == 27) {
            break;
        }
    }
    if (!img.empty()) img.release();
    if (!gray.empty()) gray.release();
    capture.release();
    tracker->stop();
    delete tracker;
#endif
    return 0;
}