#pragma once

/**
 * @file YOLO11-OBB-Detector.hpp
 * @brief YOLO11OBBDetector 类的头文件，负责使用 YOLOv11 OBB 模型进行目标检测，
 *        优化性能以实现最小延迟。
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "Utils.hpp"

#include <cmath>

/**
 * @brief YOLO11-OBB-Detector 类处理加载 YOLO
 * 模型、预处理图像、运行推理和后处理结果。
 */
class YOLO11OBBDetector {
public:
  /**
   * @brief 使用模型和标签路径初始化 YOLO 检测器的构造函数。
   *
   * @param modelPath ONNX 模型文件的路径。
   * @param labelsPath 包含类别标签的文件路径。
   * @param useGPU 是否使用 GPU 进行推理（默认为 false）。
   */
  YOLO11OBBDetector(const std::string &modelPath, const std::string &labelsPath,
                    bool useGPU = false);

  /**
   * @brief 在提供的图像上运行检测。
   *
   * @param image 用于检测的输入图像。
   * @param confThreshold 置信度阈值，用于过滤检测（默认为 0.25）。
   * @param iouThreshold 非极大值抑制的 IoU 阈值（默认为 0.25）。
   * @return std::vector<OBBDetection> 检测结果向量。
   */
  std::vector<OBBDetection> detect(const cv::Mat &image,
                                   float confThreshold = 0.25f,
                                   float iouThreshold = 0.25);

  /**
   * @brief 根据检测结果在图像上绘制旋转边界框和标签。
   *
   * @param image 要绘制的图像。
   * @param detections 检测结果向量。
   * @param classNames 对应对象 ID 的类别名称向量。
   * @param colors 每个类别的颜色向量。
   */
  void drawBoundingBox(cv::Mat &image,
                       const std::vector<OBBDetection> &detections);

  void printOBBResults(const std::vector<OBBDetection>& results) {
    std::cout << "OBB Detection Results (" << results.size() << " objects):\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& det = results[i];
      std::cout << "  Object " << i+1 << ":\n";
      std::cout << "    Class ID: " << det.classId << "\n";
      std::cout << "    Confidence: " << det.conf << "\n";
      std::cout << "    Oriented BBox: [x:" << det.box.x << " y:" << det.box.y
                << " w:" << det.box.width << " h:" << det.box.height
                << " angle:" << det.box.angle << " rad]\n";
    }
  }

private:
  Ort::Env _env{nullptr};                       // ONNX Runtime 环境
  Ort::SessionOptions _sessionOptions{nullptr}; // ONNX Runtime 的会话选项
  Ort::Session _session{nullptr}; // 用于运行推理的 ONNX Runtime 会话
  bool _isDynamicInputShape{};    // 标志，指示输入形状是否为动态
  cv::Size _inputImageShape;      // 模型期望的输入图像形状

  // 保存分配的输入和输出节点名称的向量
  std::vector<Ort::AllocatedStringPtr> _inputNodeNameAllocatedStrings;
  std::vector<const char *> _inputNames;
  std::vector<Ort::AllocatedStringPtr> _outputNodeNameAllocatedStrings;
  std::vector<const char *> _outputNames;

  size_t _numInputNodes, _numOutputNodes; // 模型中的输入和输出节点数量

  std::vector<std::string> _classNames; // 从文件加载的类别名称向量
  std::vector<cv::Scalar> _classColors; // 每个类别的颜色向量

  /**
   * @brief 为模型推理预处理输入图像。
   *
   * @param image 输入图像。
   * @param blob 存储预处理数据的指针引用。
   * @param inputTensorShape 表示输入张量形状的向量引用。
   * @return cv::Mat 预处理后的调整大小图像。
   */
  cv::Mat preprocess(const cv::Mat &image, float *&blob,
                     std::vector<int64_t> &inputTensorShape);

  /**
   * @brief 后处理模型输出以提取包含定向边界框的检测结果。
   *
   * @param originalImageSize 原始输入图像的尺寸。
   * @param resizedImageShape 预处理后图像的尺寸。
   * @param outputTensors 模型的输出张量向量。
   * @param confThreshold 置信度阈值，用于过滤检测。
   * @param iouThreshold 非极大值抑制的 IoU 阈值（对旋转框使用 ProbIoU）。
   * @return std::vector<OBBDetection> 包含定向边界框的检测结果向量。
   */
  std::vector<OBBDetection>
  postprocess(const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold,
              int topk = 500); // 默认参数
};

// YOLO11OBBDetector 构造函数的实现
YOLO11OBBDetector::YOLO11OBBDetector(const std::string &modelPath,
                                     const std::string &labelsPath,
                                     bool useGPU) {
  // 使用警告级别初始化 ONNX Runtime 环境
  _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
  _sessionOptions = Ort::SessionOptions();

  // 设置并行处理的内部操作线程数
  _sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  _sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // 获取可用的执行提供者（例如 CPU、CUDA）
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;

  // 根据是否使用 GPU 以及其可用性配置会话选项
  if (useGPU && cudaAvailable != availableProviders.end()) {
    std::cout << "推理设备：GPU" << std::endl;
    _sessionOptions.AppendExecutionProvider_CUDA(
        cudaOption); // 添加 CUDA 执行提供者
  } else {
    if (useGPU) {
      std::cout << "你的 ONNXRuntime 构建不支持 GPU。回退到 CPU。" << std::endl;
    }
    std::cout << "推理设备：CPU" << std::endl;
  }

  // 将 ONNX 模型加载到会话中
#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  _session = Ort::Session(_env, w_modelPath.c_str(), _sessionOptions);
#else
  _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;

  // 获取输入张量形状信息
  Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShapeVec =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  _isDynamicInputShape = (inputTensorShapeVec.size() >= 4) &&
                         (inputTensorShapeVec[2] == -1 &&
                          inputTensorShapeVec[3] == -1); // 检查动态维度

  // 分配并存储输入节点名称
  auto input_name = _session.GetInputNameAllocated(0, allocator);
  _inputNodeNameAllocatedStrings.push_back(std::move(input_name));
  _inputNames.push_back(_inputNodeNameAllocatedStrings.back().get());

  // 分配并存储输出节点名称
  auto output_name = _session.GetOutputNameAllocated(0, allocator);
  _outputNodeNameAllocatedStrings.push_back(std::move(output_name));
  _outputNames.push_back(_outputNodeNameAllocatedStrings.back().get());

  // 根据模型的输入张量设置期望的输入图像形状
  if (inputTensorShapeVec.size() >= 4) {
    _inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                                static_cast<int>(inputTensorShapeVec[2]));
  } else {
    throw std::runtime_error("无效的输入张量形状。");
  }

  // 获取输入和输出节点数量
  _numInputNodes = _session.GetInputCount();
  _numOutputNodes = _session.GetOutputCount();

  // 加载类别名称并生成对应的颜色
  _classNames = utils::getClassNames(labelsPath);
  _classColors = utils::generateColors(_classNames);

  std::cout << "模型加载成功，包含 " << _numInputNodes << " 个输入节点和 "
            << _numOutputNodes << " 个输出节点。" << std::endl;
}

// 预处理函数的实现
cv::Mat YOLO11OBBDetector::preprocess(const cv::Mat &image, float *&blob,
                                      std::vector<int64_t> &inputTensorShape) {
  std::cout
      << "******************** 预处理 ********************"
      << std::endl;

  cv::Mat resizedImage;
  // 使用 letterBox 工具调整图像大小并填充
  utils::ObbLetterBox(image, resizedImage, _inputImageShape,
                      cv::Scalar(114, 114, 114), _isDynamicInputShape, false,
                      true, 32);

  // 根据调整后的图像尺寸更新输入张量形状
  inputTensorShape[2] = resizedImage.rows;
  inputTensorShape[3] = resizedImage.cols;

  // 将图像转换为浮点型并归一化到 [0, 1]
  resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

  // 为 CHW 格式的图像 blob 分配内存
  blob = new float[resizedImage.cols * resizedImage.rows *
                   resizedImage.channels()];

  // 将图像分割为单独的通道并存储在 blob 中
  std::vector<cv::Mat> chw(resizedImage.channels());
  for (int i = 0; i < resizedImage.channels(); ++i) {
    chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                     blob + i * resizedImage.cols * resizedImage.rows);
  }
  cv::split(resizedImage, chw); // 将通道分割到 blob 中

  std::cout << "预处理完成" << std::endl;

  return resizedImage;
}

std::vector<OBBDetection> YOLO11OBBDetector::postprocess(
    const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors, float confThreshold,
    float iouThreshold, int topk) {
  std::cout << "******************** 后处理 ********************" << std::endl;
  std::vector<OBBDetection> detections;

  // 获取原始输出数据和形状（假设 [1, num_features, num_detections]）
  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_features = static_cast<int>(outputShape[1]);
  int num_detections = static_cast<int>(outputShape[2]);
  if (num_detections == 0) {
    return detections;
  }

  // 确定标签/类别数量（布局：[x, y, w, h, scores..., angle]）
  int num_labels = num_features - 5;
  if (num_labels <= 0) {
    return detections;
  }

  // 计算 letterbox 参数
  float inp_w = static_cast<float>(resizedImageShape.width);
  float inp_h = static_cast<float>(resizedImageShape.height);
  float orig_w = static_cast<float>(originalImageSize.width);
  float orig_h = static_cast<float>(originalImageSize.height);
  float r = std::min(inp_h / orig_h, inp_w / orig_w);
  int padw = std::round(orig_w * r);
  int padh = std::round(orig_h * r);
  float dw = (inp_w - padw) / 2.0f;
  float dh = (inp_h - padh) / 2.0f;
  float ratio = 1.0f / r;

  // 将原始输出数据包装为 cv::Mat 并转置
  // 转置后，每行对应一个检测，布局为：
  // [x, y, w, h, score_0, score_1, …, score_(num_labels-1), angle]
  cv::Mat output = cv::Mat(num_features, num_detections, CV_32F,
                           const_cast<float *>(rawOutput));
  output = output.t(); // 形状变为：[num_detections, num_features]

  // 提取检测结果，不进行裁剪
  std::vector<OrientedBoundingBox> obbs;
  std::vector<float> scores;
  std::vector<int> labels;
  for (int i = 0; i < num_detections; ++i) {
    float *row_ptr = output.ptr<float>(i);
    // 在 letterbox 坐标空间中提取原始边界框参数
    float x = row_ptr[0];
    float y = row_ptr[1];
    float w = row_ptr[2];
    float h = row_ptr[3];

    // 提取类别得分并确定最佳类别
    float *scores_ptr = row_ptr + 4;
    float maxScore = -FLT_MAX;
    int classId = -1;
    for (int j = 0; j < num_labels; j++) {
      float score = scores_ptr[j];
      if (score > maxScore) {
        maxScore = score;
        classId = j;
      }
    }

    // 角度存储在得分之后
    float angle = row_ptr[4 + num_labels];

    if (maxScore > confThreshold) {
      // 使用 letterbox 偏移和缩放校正框坐标
      float cx = (x - dw) * ratio;
      float cy = (y - dh) * ratio;
      float bw = w * ratio;
      float bh = h * ratio;

      OrientedBoundingBox obb(cx, cy, bw, bh, angle);
      obbs.push_back(obb);
      scores.push_back(maxScore);
      labels.push_back(classId);
    }
  }

  // 将检测结果组合为 vector<OBBDetection> 以进行 NMS
  std::vector<OBBDetection> detectionsForNMS;
  for (size_t i = 0; i < obbs.size(); i++) {
    detectionsForNMS.emplace_back(OBBDetection{obbs[i], scores[i], labels[i]});
  }

  for (auto &det : detectionsForNMS) {
    det.box.x = std::min(std::max(det.box.x, 0.f), orig_w);
    det.box.y = std::min(std::max(det.box.y, 0.f), orig_h);
    det.box.width = std::min(std::max(det.box.width, 0.f), orig_w);
    det.box.height = std::min(std::max(det.box.height, 0.f), orig_h);
  }

  // 执行旋转 NMS
  std::vector<OBBDetection> post_nms_detections = OBB_NMS::nonMaxSuppression(
      detectionsForNMS, confThreshold, iouThreshold, topk);

  std::cout << "后处理完成" << std::endl;
  return post_nms_detections;
}

// 检测函数的实现
std::vector<OBBDetection> YOLO11OBBDetector::detect(const cv::Mat &image,
                                                    float confThreshold,
                                                    float iouThreshold) {
  std::cout << "******************** 旋转目标检测任务 ********************"
            << std::endl;

  float *blobPtr = nullptr; // 存储预处理图像数据的指针
  // 定义输入张量的形状（批次大小，通道数，高度，宽度）
  std::vector<int64_t> inputTensorShape = {1, 3, _inputImageShape.height,
                                           _inputImageShape.width};

  // 预处理图像并获取 blob 指针
  cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

  // 计算输入张量的总元素数
  size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

  // 从 blob 数据创建用于 ONNX Runtime 输入的向量
  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

  delete[] blobPtr; // 释放为 blob 分配的内存

  // 创建 Ort 内存信息对象（如果重复使用可以缓存）
  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // 使用预处理数据创建输入张量对象
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());

  // 使用输入张量运行推理会话并获取输出张量
  std::vector<Ort::Value> outputTensors =
      _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                   _numInputNodes, _outputNames.data(), _numOutputNodes);

  // 根据输入张量形状确定调整后的图像形状
  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));

  std::vector<OBBDetection> detections =
      postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                  iouThreshold, 100);

  return detections; // 返回检测结果向量
}

void YOLO11OBBDetector::drawBoundingBox(
    cv::Mat &image, const std::vector<OBBDetection> &detections) {
  for (const auto &detection : detections) {
    if (detection.classId < 0 ||
        static_cast<size_t>(detection.classId) >= _classNames.size())
      continue;

    // 将角度从弧度转换为度数以供 OpenCV 使用
    float angle_deg = detection.box.angle * 180.0f / CV_PI;

    cv::RotatedRect rect(cv::Point2f(detection.box.x, detection.box.y),
                         cv::Size2f(detection.box.width, detection.box.height),
                         angle_deg);

    // 将旋转矩形转换为多边形点
    cv::Mat points_mat;
    cv::boxPoints(rect, points_mat);
    points_mat.convertTo(points_mat, CV_32SC1);

    // 绘制边界框
    cv::Scalar color = _classColors[detection.classId % _classColors.size()];
    cv::polylines(image, points_mat, true, color, 3, cv::LINE_AA);

    // 准备标签
    std::string label = _classNames[detection.classId] + ": " +
                        cv::format("%.1f%%", detection.conf * 100);
    int baseline = 0;
    float fontScale = 0.6;
    int thickness = 1;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX,
                                         fontScale, thickness, &baseline);

    // 使用旋转矩形的边界矩形计算标签位置
    cv::Rect brect = rect.boundingRect();
    int x = brect.x;
    int y = brect.y - labelSize.height - baseline;

    // 如果标签超出屏幕，调整标签位置
    if (y < 0) {
      y = brect.y + brect.height;
      if (y + labelSize.height > image.rows) {
        y = image.rows - labelSize.height;
      }
    }

    x = std::max(0, std::min(x, image.cols - labelSize.width));

    // 绘制标签背景（框颜色的较暗版本）
    cv::Scalar labelBgColor = color * 0.6;
    cv::rectangle(image,
                  cv::Rect(x, y, labelSize.width, labelSize.height + baseline),
                  labelBgColor, cv::FILLED);

    // 绘制标签文本
    cv::putText(image, label, cv::Point(x, y + labelSize.height),
                cv::FONT_HERSHEY_DUPLEX, fontScale, cv::Scalar::all(255),
                thickness, cv::LINE_AA);
  }
}
