#pragma once

/**
 * @file YOLO11Detector.hpp
 * @brief YOLO11Detector 类的头文件，负责使用 YOLOv11 模型进行目标检测，
 *        优化性能以实现最小延迟。
 */
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "Utils.hpp"

/**
 * @brief YOLO11Detector 类处理加载 YOLO
 * 模型、预处理图像、运行推理和后处理结果。
 */
class YOLO11Detector {
public:
  /**
   * @brief 使用模型和标签路径初始化 YOLO 检测器的构造函数。
   *
   * @param modelPath ONNX 模型文件的路径。
   * @param labelsPath 包含类别标签的文件路径。
   * @param useGPU 是否使用 GPU 进行推理（默认为 false）。
   */
  YOLO11Detector(const std::string &modelPath, const std::string &labelsPath,
                 bool useGPU = false);

  /**
   * @brief 在提供的图像上运行检测。
   *
   * @param image 用于检测的输入图像。
   * @param confThreshold 置信度阈值，用于过滤检测（默认为 0.4）。
   * @param iouThreshold 非极大值抑制的 IoU 阈值（默认为 0.45）。
   * @return std::vector<Detection> 检测结果向量。
   */
  std::vector<Detection> detect(const cv::Mat &image,
                                float confThreshold = 0.4f,
                                float iouThreshold = 0.45f);

  /**
   * @brief 根据检测结果在图像上绘制边界框和半透明背景。
   *
   * @param image 要绘制的图像。
   * @param detections 检测结果向量。
   * @param classNames 对应对象 ID 的类别名称向量。
   * @param classColors 每个类别的颜色向量。
   * @param maskAlpha 背景透明度的 Alpha 值（默认为 0.4）。
   */
  void drawBoundingBoxMask(cv::Mat &image,
                           const std::vector<Detection> &detections,
                           float maskAlpha = 0.4f);
  /**
   * @brief 根据检测结果在图像上绘制边界框和标签。
   *
   * @param image 要绘制的图像。
   * @param detections 检测结果向量。
   * @param classNames 对应对象 ID 的类别名称向量。
   * @param colors 每个类别的颜色向量。
   */
  void drawBoundingBox(cv::Mat &image,
                       const std::vector<Detection> &detections);

    void printDetectionResults(const std::vector<Detection>& results) {
        std::cout << "Detection Results (" << results.size() << " objects):\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& det = results[i];
            std::cout << "  Object " << i+1 << ":\n";
            std::cout << "    Class ID: " << det.classId << "\n";
            std::cout << "    Confidence: " << det.conf << "\n";
            std::cout << "    Bounding Box: [x:" << det.box.x << " y:" << det.box.y
                      << " w:" << det.box.width << " h:" << det.box.height << "]\n";
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
   * @brief 后处理模型输出以提取检测结果。
   *
   * @param originalImageSize 原始输入图像的尺寸。
   * @param resizedImageShape 预处理后图像的尺寸。
   * @param outputTensors 模型的输出张量向量。
   * @param confThreshold 置信度阈值，用于过滤检测。
   * @param iouThreshold 非极大值抑制的 IoU 阈值。
   * @return std::vector<Detection> 检测结果向量。
   */
  std::vector<Detection>
  postprocess(const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold);
};

// YOLO11Detector 构造函数的实现
YOLO11Detector::YOLO11Detector(const std::string &modelPath,
                               const std::string &labelsPath, bool useGPU) {
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
cv::Mat YOLO11Detector::preprocess(const cv::Mat &image, float *&blob,
                                   std::vector<int64_t> &inputTensorShape) {
  std::cout
      << "******************** 预处理 ********************"
      << std::endl;

  cv::Mat resizedImage;
  // 使用 DetectedLetterBox 工具调整图像大小并填充
  utils::DetectedLetterBox(image, resizedImage, _inputImageShape,
                           cv::Scalar(114, 114, 114), _isDynamicInputShape,
                           false, true, 32);

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

// 后处理函数，将原始模型输出转换为检测结果
std::vector<Detection>
YOLO11Detector::postprocess(const cv::Size &originalImageSize,
                            const cv::Size &resizedImageShape,
                            const std::vector<Ort::Value> &outputTensors,
                            float confThreshold, float iouThreshold) {
  std::cout << "******************** 后处理 ********************" << std::endl;

  std::vector<Detection> detections;
  const float *rawOutput =
      outputTensors[0]
          .GetTensorData<float>(); // 从第一个输出张量提取原始输出数据
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // 确定特征数量和检测数量
  const size_t num_features = outputShape[1];
  const size_t num_detections = outputShape[2];

  // 如果没有检测结果，提前退出
  if (num_detections == 0) {
    return detections;
  }

  // 根据输出形状计算类别数量
  const int numClasses = static_cast<int>(num_features) - 4;
  if (numClasses <= 0) {
    // 无效的类别数量
    return detections;
  }

  // 为高效追加预留内存
  std::vector<BoundingBox> boxes;
  boxes.reserve(num_detections);
  std::vector<float> confs;
  confs.reserve(num_detections);
  std::vector<int> classIds;
  classIds.reserve(num_detections);
  std::vector<BoundingBox> nms_boxes;
  nms_boxes.reserve(num_detections);

  // 索引常量
  const float *ptr = rawOutput;

  for (size_t d = 0; d < num_detections; ++d) {
    // 提取边界框坐标（中心 x，中心 y，宽度，高度）
    float centerX = ptr[0 * num_detections + d];
    float centerY = ptr[1 * num_detections + d];
    float width = ptr[2 * num_detections + d];
    float height = ptr[3 * num_detections + d];

    // 找到置信度最高的类别
    int classId = -1;
    float maxScore = -FLT_MAX;
    for (int c = 0; c < numClasses; ++c) {
      const float score = ptr[d + (4 + c) * num_detections];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    // 仅当置信度超过阈值时继续
    if (maxScore > confThreshold) {
      // 将中心坐标转换为左上角 (x1, y1)
      float left = centerX - width / 2.0f;
      float top = centerY - height / 2.0f;

      // 缩放到原始图像大小
      BoundingBox scaledBox = utils::scaleCoords(
          resizedImageShape, BoundingBox(left, top, width, height),
          originalImageSize, true);

      // 四舍五入坐标以获得整数像素位置
      BoundingBox roundedBox;
      roundedBox.x = std::round(scaledBox.x);
      roundedBox.y = std::round(scaledBox.y);
      roundedBox.width = std::round(scaledBox.width);
      roundedBox.height = std::round(scaledBox.height);

      // 调整 NMS 框坐标以防止不同类别之间重叠
      BoundingBox nmsBox = roundedBox;
      nmsBox.x += classId * 7680; // 任意偏移以区分类别
      nmsBox.y += classId * 7680;

      // 添加到相应的容器中
      nms_boxes.emplace_back(nmsBox);
      boxes.emplace_back(roundedBox);
      confs.emplace_back(maxScore);
      classIds.emplace_back(classId);
    }
  }

  // 应用非极大值抑制（NMS）以消除冗余检测
  std::vector<int> indices;
  utils::DetectedNMSBoxes(nms_boxes, confs, confThreshold, iouThreshold,
                          indices);

  // 将过滤后的检测结果收集到结果向量中
  detections.reserve(indices.size());
  for (const int idx : indices) {
    detections.emplace_back(Detection{
        boxes[idx],   // 边界框
        confs[idx],   // 置信度得分
        classIds[idx] // 类别 ID
    });
  }

  std::cout << "后处理完成" << std::endl;

  return detections;
}

// 检测函数的实现
std::vector<Detection> YOLO11Detector::detect(const cv::Mat &image,
                                              float confThreshold,
                                              float iouThreshold) {
  std::cout << "******************** 检测任务 ********************"
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

  // 后处理输出张量以获取检测结果
  std::vector<Detection> detections =
      postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                  iouThreshold);

  return detections; // 返回检测结果向量
}

void YOLO11Detector::drawBoundingBox(cv::Mat &image,
                                     const std::vector<Detection> &detections) {
  // 遍历每个检测结果以绘制边界框和标签
  for (const auto &detection : detections) {

    // 确保对象 ID 在有效范围内
    if (detection.classId < 0 ||
        static_cast<size_t>(detection.classId) >= _classNames.size())
      continue;

    // 根据对象 ID 选择颜色以保持一致性
    const cv::Scalar &color =
        _classColors[detection.classId % _classColors.size()];

    // 绘制边界框矩形
    cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                  cv::Point(detection.box.x + detection.box.width,
                            detection.box.y + detection.box.height),
                  color, 2, cv::LINE_AA);

    // 准备包含类别名称和置信度百分比的标签文本
    std::string label = _classNames[detection.classId] + ": " +
                        std::to_string(static_cast<int>(detection.conf * 100)) +
                        "%";

    // 定义标签的文本属性
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = std::min(image.rows, image.cols) * 0.0008;
    const int thickness =
        std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
    int baseline = 0;

    // 计算文本大小以用于背景矩形
    cv::Size textSize =
        cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

    // 定义标签的位置
    int labelY = std::max(detection.box.y, textSize.height + 5);
    cv::Point labelTopLeft(detection.box.x, labelY - textSize.height - 5);
    cv::Point labelBottomRight(detection.box.x + textSize.width + 5,
                               labelY + baseline - 5);

    // 绘制标签的背景矩形
    cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

    // 绘制标签文本
    cv::putText(image, label, cv::Point(detection.box.x + 2, labelY - 2),
                fontFace, fontScale, cv::Scalar(255, 255, 255), thickness,
                cv::LINE_AA);
  }
}

void YOLO11Detector::drawBoundingBoxMask(
    cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha) {
  // 验证输入图像
  if (image.empty()) {
    std::cerr << "错误：提供给 drawBoundingBoxMask 的图像为空。" << std::endl;
    return;
  }

  const int imgHeight = image.rows;
  const int imgWidth = image.cols;

  // 根据图像尺寸预计算动态字体大小和粗细
  const double fontSize = std::min(imgHeight, imgWidth) * 0.0006;
  const int textThickness =
      std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

  // 创建用于混合的掩码图像（初始化为零）
  cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

  // 预过滤检测结果，仅包括置信度高于阈值且类别 ID 有效的检测
  std::vector<const Detection *> filteredDetections;
  for (const auto &detection : detections) {
    if (detection.classId >= 0 &&
        static_cast<size_t>(detection.classId) < _classNames.size()) {
      filteredDetections.emplace_back(&detection);
    }
  }

  // 在掩码图像上绘制填充矩形，用于半透明覆盖
  for (const auto *detection : filteredDetections) {
    cv::Rect box(detection->box.x, detection->box.y, detection->box.width,
                 detection->box.height);
    const cv::Scalar &color = _classColors[detection->classId];
    cv::rectangle(maskImage, box, color, cv::FILLED);
  }

  // 将掩码图像与原始图像混合以应用半透明背景
  cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);

  // 在原始图像上绘制边界框和标签
  for (const auto *detection : filteredDetections) {
    cv::Rect box(detection->box.x, detection->box.y, detection->box.width,
                 detection->box.height);
    const cv::Scalar &color = _classColors[detection->classId];
    cv::rectangle(image, box, color, 2, cv::LINE_AA);

    std::string label =
        _classNames[detection->classId] + ": " +
        std::to_string(static_cast<int>(detection->conf * 100)) + "%";
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         fontSize, textThickness, &baseLine);

    int labelY = std::max(detection->box.y, labelSize.height + 5);
    cv::Point labelTopLeft(detection->box.x, labelY - labelSize.height - 5);
    cv::Point labelBottomRight(detection->box.x + labelSize.width + 5,
                               labelY + baseLine - 5);

    // 绘制标签的背景矩形
    cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

    // 绘制标签文本
    cv::putText(image, label, cv::Point(detection->box.x + 2, labelY - 2),
                cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255),
                textThickness, cv::LINE_AA);
  }

  std::cout << "边界框和背景已绘制在图像上。" << std::endl;
}
