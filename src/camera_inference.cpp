#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/hpp/rs_types.hpp>
#include <librealsense2/rs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ThreadSafeQueues.hpp"

// Uncomment ONE of these to select the task
// #define YOLO_TASK_CLASSIFY
// #define YOLO_TASK_DETECT
// #define YOLO_TASK_OBB
#define YOLO_TASK_SEG
// #define YOLO_TASK_POSE

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

    // 配置参数
    const std::string modelPath = "../models/rack_block-" + task + ".onnx";
    const std::string yamlPath = "../models/rack_block-" + task + ".yaml";
    const std::string labelsPath = "../models/classes.names";
    const bool isGPU = true;

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

    // 创建管道以轻松配置和启动相机
    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_BGR8, 30);

    // 使用默认推荐设置开始流式传输
    p.start(cfg);

    // 初始化具有有界容量的队列
    const size_t max_queue_size = 2; // 双缓冲
    BoundedSafeQueue<cv::Mat> frameQueue(max_queue_size);

    #if defined(YOLO_TASK_CLASSIFY)
        BoundedSafeQueue<std::pair<cv::Mat, ClassificationResult>> processedQueue(max_queue_size);
    #elif defined(YOLO_TASK_DETECT)
        BoundedSafeQueue<std::pair<cv::Mat, std::vector<Detection>>> processedQueue(max_queue_size);
    #elif defined(YOLO_TASK_OBB)
        BoundedSafeQueue<std::pair<cv::Mat, std::vector<OBBDetection>>> processedQueue(max_queue_size);
    #elif defined(YOLO_TASK_SEG)
        BoundedSafeQueue<std::pair<cv::Mat, std::vector<SegmentedDetection>>> processedQueue(max_queue_size);
    #elif defined(YOLO_TASK_POSE)
        BoundedSafeQueue<std::pair<cv::Mat, std::vector<PosedDetection>>> processedQueue(max_queue_size);
    #endif

    std::atomic<bool> stopFlag(false);

    // 生产者线程：捕获帧
    std::thread producer([&]() {
        while (!stopFlag.load()) {
            // 等待相机的下一组帧
            rs2::frameset frames = p.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();

            // 将RealSense帧转换为OpenCV Mat
            cv::Mat frame(cv::Size(1920, 1080), CV_8UC3,
                        (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);

            if (!frameQueue.enqueue(frame))
                break; // 队列已结束
        }
        frameQueue.setFinished();
    });

    // 消费者线程：处理帧
    std::thread consumer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && frameQueue.dequeue(frame)) {
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

            std::cout << "检测完成，用时: " << duration.count() << " 毫秒" << std::endl;

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

            // 将处理后的帧加入队列
            if (!processedQueue.enqueue(std::make_pair(frame, results)))
                break;
        }
        processedQueue.setFinished();
    });

    // 显示线程
    std::thread displayThread([&]() {
        while (!stopFlag.load()) {
            #if defined(YOLO_TASK_CLASSIFY)
                std::pair<cv::Mat, ClassificationResult> item;
                if (!processedQueue.dequeue(item)) break;
                cv::imshow("Detections", item.first);
            #elif defined(YOLO_TASK_DETECT)
                std::pair<cv::Mat, std::vector<Detection>> item;
                if (!processedQueue.dequeue(item)) break;
                cv::imshow("Detections", item.first);
            #elif defined(YOLO_TASK_OBB)
                std::pair<cv::Mat, std::vector<OBBDetection>> item;
                if (!processedQueue.dequeue(item)) break;
                cv::imshow("Detections", item.first);
            #elif defined(YOLO_TASK_SEG)
                std::pair<cv::Mat, std::vector<SegmentedDetection>> item;
                if (!processedQueue.dequeue(item)) break;
                cv::imshow("Detections", item.first);
            #elif defined(YOLO_TASK_POSE)
                std::pair<cv::Mat, std::vector<PosedDetection>> item;
                if (!processedQueue.dequeue(item)) break;
                cv::imshow("Detections", item.first);
            #endif

            // 检查退出键
            if (cv::waitKey(1) == 'q') {
                stopFlag.store(true);
                frameQueue.setFinished();
                processedQueue.setFinished();
                break;
            }
        }
    });

    // 等待所有线程完成
    producer.join();
    consumer.join();
    displayThread.join();

    // 释放资源
    p.stop();
    cv::destroyAllWindows();

    return 0;
}