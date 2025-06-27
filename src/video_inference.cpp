#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>
#include "ThreadSafeQueues.hpp"

// Uncomment ONE of these to select the task
// #define YOLO_TASK_CLASSIFY
// #define YOLO_TASK_DETECT
// #define YOLO_TASK_OBB
// #define YOLO_TASK_SEG
#define YOLO_TASK_POSE

// Include only the needed header based on the selected task
#if defined(YOLO_TASK_CLASSIFY)
#include "YOLO11Class.hpp"
#elif defined(YOLO_TASK_DETECT)
#include "YOLO11Det.hpp"
#elif defined(YOLO_TASK_OBB)
#include "YOLO11Obb.hpp"
#elif defined(YOLO_TASK_SEG)
#include "YOLO11Seg.hpp"
#elif defined(YOLO_TASK_POSE)
#include "YOLO11Pose.hpp"
#else
#error "No YOLO task defined! Please define one of: YOLO_TASK_CLASSIFY, YOLO_TASK_DETECT, YOLO_TASK_OBB, YOLO_TASK_SEG, YOLO_TASK_POSE"
#endif

int main() {
    // Set task based on defined macro
    std::string task;
    #if defined(YOLO_TASK_CLASSIFY)
        task = "classify";
    #elif defined(YOLO_TASK_DETECT)
        task = "det";
    #elif defined(YOLO_TASK_OBB)
        task = "obb";
    #elif defined(YOLO_TASK_SEG)
        task = "seg";
    #elif defined(YOLO_TASK_POSE)
        task = "pose";
    #endif

    // 模型、标签、输入视频和输出视频的路径
    const std::string modelPath = "../models/rack_block-" + task + ".onnx";
    const std::string yamlPath = "../models/rack_block-" + task + ".yaml";
    const std::string labelsPath = "../models/classes.names";
    const std::string videoPath = "../assert/sample.mp4"; // 输入视频路径
    const std::string outputPath = "../output/sample_processed.mp4"; // 输出视频路径
    const bool isGPU = false;

    // 初始化检测器
    #if defined(YOLO_TASK_CLASSIFY)
        YOLO11Classifier detector(modelPath, labelsPath, isGPU);
    #elif defined(YOLO_TASK_DETECT)
        YOLO11Detector detector(modelPath, labelsPath, isGPU);
    #elif defined(YOLO_TASK_OBB)
        YOLO11OBBDetector detector(modelPath, labelsPath, isGPU);
    #elif defined(YOLO_TASK_SEG)
        YOLOv11SegDetector detector(modelPath, labelsPath, isGPU);
    #elif defined(YOLO_TASK_POSE)
        YOLO11PoseDetector detector(modelPath, yamlPath, isGPU);
    #endif

    // 打开视频文件
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开或找到视频文件！\n";
        return -1;
    }

    // 获取视频属性
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    // 创建VideoWriter对象
    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened()) {
        std::cerr << "错误：无法打开输出视频文件进行写入！\n";
        return -1;
    }

    // 线程安全队列
    UnboundedSafeQueue<cv::Mat> frameQueue;
    #if defined(YOLO_TASK_CLASSIFY)
        UnboundedSafeQueue<std::pair<int, std::pair<cv::Mat, ClassificationResult>>> processedQueue;
    #elif defined(YOLO_TASK_DETECT)
        UnboundedSafeQueue<std::pair<int, std::pair<cv::Mat, std::vector<Detection>>>> processedQueue;
    #elif defined(YOLO_TASK_OBB)
        UnboundedSafeQueue<std::pair<int, std::pair<cv::Mat, std::vector<OBBDetection>>>> processedQueue;
    #elif defined(YOLO_TASK_SEG)
        UnboundedSafeQueue<std::pair<int, std::pair<cv::Mat, std::vector<SegmentedDetection>>>> processedQueue;
    #elif defined(YOLO_TASK_POSE)
        UnboundedSafeQueue<std::pair<int, std::pair<cv::Mat, std::vector<PosedDetection>>>> processedQueue;
    #endif

    // 指示处理完成的标志
    std::atomic<bool> processingDone(false);

    // 捕获线程
    std::thread captureThread([&]() {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame)) {
            frameQueue.enqueue(frame.clone()); // 克隆以确保线程安全
            frameCount++;
        }
        frameQueue.setFinished();
    });

    // 处理线程
    std::thread processingThread([&]() {
        cv::Mat frame;
        int frameIndex = 0;
        while (frameQueue.dequeue(frame)) {
            // 检测
            float confThreshold = 0.45f;
            float iouThreshold = 0.45f;
            auto start = std::chrono::high_resolution_clock::now();

            #if defined(YOLO_TASK_CLASSIFY)
                ClassificationResult results = detector.detect(frame);
            #elif defined(YOLO_TASK_DETECT)
                std::vector<Detection> results = detector.detect(frame, confThreshold, iouThreshold);
            #elif defined(YOLO_TASK_OBB)
                std::vector<OBBDetection> results = detector.detect(frame, confThreshold, iouThreshold);
            #elif defined(YOLO_TASK_SEG)
                std::vector<SegmentedDetection> results = detector.segment(frame, confThreshold, iouThreshold);
            #elif defined(YOLO_TASK_POSE)
                std::vector<PosedDetection> results = detector.detect(frame, confThreshold, iouThreshold);
            #endif

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            // 打印结果
            #if defined(YOLO_TASK_CLASSIFY)
                detector.printClassificationResult(results);
            #elif defined(YOLO_TASK_DETECT)
                detector.printDetectionResults(results);
            #elif defined(YOLO_TASK_OBB)
                detector.printOBBResults(results);
            #elif defined(YOLO_TASK_SEG)
                detector.printSegmentationResults(results);
            #elif defined(YOLO_TASK_POSE)
                detector.printPoseResults(results);
            #endif

            std::cout << "检测完成，用时: " << duration.count() << " 毫秒" << std::endl;

            // 可视化
            #if defined(YOLO_TASK_CLASSIFY)
                detector.drawClassificationResult(frame, results);
            #elif defined(YOLO_TASK_DETECT)
                detector.drawBoundingBox(frame, results);
            #elif defined(YOLO_TASK_OBB)
                detector.drawBoundingBox(frame, results);
            #elif defined(YOLO_TASK_SEG)
                detector.drawSegmentationsAndBoxes(frame, results);
            #elif defined(YOLO_TASK_POSE)
                detector.drawPosedBoundingBox(frame, results);
            #endif

            // 在左上角绘制耗时（白色背景，黑色字体）
            std::string timeText = "Time Cost: " + std::to_string(duration.count()) + " ms";
            cv::Size textSize = cv::getTextSize(timeText, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr);
            cv::rectangle(frame, cv::Point(10, 10), cv::Point(20 + textSize.width, 40 + textSize.height),
                         cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(frame, timeText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(0, 0, 0), 2);

            // 将处理后的帧加入队列
            #if defined(YOLO_TASK_CLASSIFY)
                processedQueue.enqueue(std::make_pair(frameIndex++, std::make_pair(frame, results)));
            #elif defined(YOLO_TASK_DETECT)
                processedQueue.enqueue(std::make_pair(frameIndex++, std::make_pair(frame, results)));
            #elif defined(YOLO_TASK_OBB)
                processedQueue.enqueue(std::make_pair(frameIndex++, std::make_pair(frame, results)));
            #elif defined(YOLO_TASK_SEG)
                processedQueue.enqueue(std::make_pair(frameIndex++, std::make_pair(frame, results)));
            #elif defined(YOLO_TASK_POSE)
                processedQueue.enqueue(std::make_pair(frameIndex++, std::make_pair(frame, results)));
            #endif
        }
        processedQueue.setFinished();
    });

    // 写入线程
    std::thread writingThread([&]() {
        #if defined(YOLO_TASK_CLASSIFY)
            std::pair<int, std::pair<cv::Mat, ClassificationResult>> processedFrame;
        #elif defined(YOLO_TASK_DETECT)
            std::pair<int, std::pair<cv::Mat, std::vector<Detection>>> processedFrame;
        #elif defined(YOLO_TASK_OBB)
            std::pair<int, std::pair<cv::Mat, std::vector<OBBDetection>>> processedFrame;
        #elif defined(YOLO_TASK_SEG)
            std::pair<int, std::pair<cv::Mat, std::vector<SegmentedDetection>>> processedFrame;
        #elif defined(YOLO_TASK_POSE)
            std::pair<int, std::pair<cv::Mat, std::vector<PosedDetection>>> processedFrame;
        #endif

        while (processedQueue.dequeue(processedFrame)) {
            // 显示处理后的帧
            cv::imshow("Detections", processedFrame.second.first);
            // 写入输出视频
            out.write(processedFrame.second.first);

            if (cv::waitKey(1) == 'q') {
                processingDone.store(true);
                frameQueue.setFinished();
                processedQueue.setFinished();
                break;
            }
        }
    });

    // 等待所有线程完成
    captureThread.join();
    processingThread.join();
    writingThread.join();

    // 释放资源
    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "视频处理成功完成。" << std::endl;

    return 0;
}