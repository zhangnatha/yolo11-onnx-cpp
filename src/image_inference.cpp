#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#include <dirent.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

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

bool isImageFile(const std::string &filename) {
  return filename.find(".png") != std::string::npos ||
         filename.find(".jpg") != std::string::npos ||
         filename.find(".jpeg") != std::string::npos;
}

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

  // 读取模型和 YAML 配置文件路径
  const std::string modelPath = "../models/rack_block-" + task + ".onnx";
  const std::string yamlPath = "../models/rack_block-pose.yaml";
  const std::string labelsPath = "../models/classes.names";
  #if defined(YOLO_TASK_CLASSIFY)
    const std::string folderPath = "../assert/classify";
  #else
    const std::string folderPath = "../assert";
  #endif
  const std::string resultFolder = "../output";

  bool isGPU = true;

  // 构造检测器，自动从 YAML 文件加载配置
  #if defined(YOLO_TASK_CLASSIFY)
    YOLO11Classifier detector(modelPath, labelsPath, isGPU); //分类
  #elif defined(YOLO_TASK_DETECT)
    YOLO11Detector detector(modelPath, labelsPath, isGPU); //目标检测
  #elif defined(YOLO_TASK_OBB)
    YOLO11OBBDetector detector(modelPath, labelsPath, isGPU); //旋转边界框检测
  #elif defined(YOLO_TASK_SEG)
    YOLOv11SegDetector detector(modelPath, labelsPath, isGPU); //分割
  #elif defined(YOLO_TASK_POSE)
    YOLO11PoseDetector detector(modelPath, yamlPath, isGPU); //姿态/关键点检测
  #endif

  // 检查结果文件夹是否存在，不存在则创建
  DIR *resultDir = opendir(resultFolder.c_str());
  if (resultDir == nullptr) {
#ifdef _WIN32
    if (mkdir(resultFolder.c_str()) != 0) {
#else
    if (mkdir(resultFolder.c_str(), 0755) != 0) {
#endif
      std::cerr << "错误：无法创建结果文件夹 " << resultFolder << "！"
                << std::endl;
      return 1;
    }
  } else {
    closedir(resultDir);
  }

  // 遍历目录获取图像文件
  std::vector<std::string> imageFiles;
  DIR *dir = opendir(folderPath.c_str());
  if (dir == nullptr) {
    std::cerr << "错误：无法打开文件夹 " << folderPath << "！" << std::endl;
    return 1;
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;
    if (isImageFile(filename)) {
      imageFiles.push_back(folderPath + "/" + filename);
    }
  }
  closedir(dir);
  if (imageFiles.empty()) {
    std::cerr << "错误：文件夹 " << folderPath << " 中没有找到图像文件！"
              << std::endl;
    return 1;
  }

  int cost_time_all = 0;
  // 对每张图像进行推理并保存
  for (const std::string &imagePath : imageFiles) {
    std::cout << "正在处理图像: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cerr << "错误：无法打开或加载图像 " << imagePath << "！\n";
      continue;
    }

    // 推理检测
    float confThreshold = 0.45f; // 过滤低于此置信度的框
    float iouThreshold = 0.45f;  // 非极大值抑制（NMS）的 IoU 阈值
    auto start = std::chrono::high_resolution_clock::now();

    #if defined(YOLO_TASK_CLASSIFY)
      ClassificationResult results = detector.detect(image);
    #elif defined(YOLO_TASK_DETECT)
      std::vector<Detection> results = detector.detect(image, confThreshold, iouThreshold);
    #elif defined(YOLO_TASK_OBB)
      std::vector<OBBDetection> results = detector.detect(image, confThreshold, iouThreshold);
    #elif defined(YOLO_TASK_SEG)
      std::vector<SegmentedDetection> results = detector.segment(image, confThreshold, iouThreshold);
    #elif defined(YOLO_TASK_POSE)
      std::vector<PosedDetection> results = detector.detect(image, confThreshold, iouThreshold);
    #endif

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "检测完成，用时: " << duration.count() << " 毫秒" << std::endl;

    // 可视化结果
    #if defined(YOLO_TASK_CLASSIFY)
      detector.drawClassificationResult(image, results);
    #elif defined(YOLO_TASK_DETECT)
      detector.drawBoundingBox(image, results);
    #elif defined(YOLO_TASK_OBB)
      detector.drawBoundingBox(image, results);
    #elif defined(YOLO_TASK_SEG)
      detector.drawSegmentationsAndBoxes(image, results);
    #elif defined(YOLO_TASK_POSE)
      detector.drawPosedBoundingBox(image, results);
    #endif

    // 打印结果
    // 打印检测结果
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

    // 保存结果图像
    std::string filename = imagePath.substr(imagePath.find_last_of('/') + 1);
    std::string outputPath = resultFolder + "/" + filename;
    if (!cv::imwrite(outputPath, image)) {
      std::cerr << "错误：无法保存图像 " << outputPath << "！" << std::endl;
    } else {
      std::cout << "已保存图像: " << outputPath << std::endl;
    }
    cost_time_all+=duration.count();
  }
  std::cout << "Average time(ms): " << cost_time_all / imageFiles.size() << " ms" << std::endl;

  return 0;
}