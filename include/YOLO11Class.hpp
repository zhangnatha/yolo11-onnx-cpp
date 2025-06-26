#pragma once

/**
 * @file YOLO11CLASS.hpp
 * @brief YOLO11Classifier 类的头文件，负责使用 ONNX
 * 模型进行图像分类，优化性能以实现最小延迟。
 */
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "Utils.hpp"
#include <iomanip> // 用于 std::fixed 和 std::setprecision
#include <sstream> // 用于 std::ostringstream

/**
 * @brief YOLO11Classifier
 * 类处理加载分类模型、预处理图像、运行推理和后处理结果。
 */
class YOLO11Classifier {
public:
  /**
   * @brief 使用模型和标签路径初始化分类器的构造函数。
   */
  YOLO11Classifier(const std::string &modelPath, const std::string &labelsPath,
                   bool useGPU = false,
                   const cv::Size &targetInputShape = cv::Size(224, 224));

  /**
   * @brief 在提供的图像上运行分类。
   */
  ClassificationResult detect(const cv::Mat &image);

  /**
   * @brief 在图像上绘制分类结果。
   */
  void
  drawClassificationResult(cv::Mat &image, const ClassificationResult &result,
                           const cv::Point &position = cv::Point(10, 10),
                           const cv::Scalar &textColor = cv::Scalar(0, 255, 0),
                           double fontScaleMultiplier = 0.0008,
                           const cv::Scalar &bgColor = cv::Scalar(0, 0, 0));

  cv::Size getInputShape() const { return _inputImageShape; }
  bool isModelInputShapeDynamic() const { return _isDynamicInputShape; }
  /**
   * @brief 通过调整大小和填充（letterbox
   * 风格）或简单调整大小，准备模型输入图像。
   */
  void preprocessImageToTensor(const cv::Mat &image, cv::Mat &outImage,
                               const cv::Size &targetShape,
                               const cv::Scalar &color = cv::Scalar(0, 0, 0),
                               bool scaleUp = true,
                               const std::string &strategy = "resize");

  void printClassificationResult(const ClassificationResult& result) {
    std::cout << "Classification Result:\n";
    std::cout << "  Class ID: " << result.classId << "\n";
    std::cout << "  Class Name: " << result.className << "\n";
    std::cout << "  Confidence: " << result.confidence << "\n";
  }
private:
  Ort::Env _env{nullptr};
  Ort::SessionOptions _sessionOptions{nullptr};
  Ort::Session _session{nullptr};

  bool _isDynamicInputShape{};
  cv::Size _inputImageShape{};

  std::vector<Ort::AllocatedStringPtr> _inputNodeNameAllocatedStrings{};
  std::vector<const char *> _inputNames{};
  std::vector<Ort::AllocatedStringPtr> _outputNodeNameAllocatedStrings{};
  std::vector<const char *> _outputNames{};

  size_t _numInputNodes{}, _numOutputNodes{};
  int _numClasses{0};

  std::vector<std::string> _classNames{};

  void preprocess(const cv::Mat &image, float *&blob,
                  std::vector<int64_t> &inputTensorShape);
  ClassificationResult
  postprocess(const std::vector<Ort::Value> &outputTensors);
};

// YOLO11Classifier 构造函数的实现
YOLO11Classifier::YOLO11Classifier(const std::string &modelPath,
                                   const std::string &labelsPath, bool useGPU,
                                   const cv::Size &targetInputShape)
    : _inputImageShape(targetInputShape) {
  _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_CLASSIFICATION_ENV");
  _sessionOptions = Ort::SessionOptions();

  _sessionOptions.SetIntraOpNumThreads(
      std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
  _sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption{};

  if (useGPU && cudaAvailable != availableProviders.end()) {
    std::cout << "尝试使用 GPU 进行推理。" << std::endl;
    _sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
  } else {
    if (useGPU) {
      std::cout
          << "警告：请求使用 GPU，但 CUDAExecutionProvider 不可用。回退到 CPU。"
          << std::endl;
    }
    std::cout << "使用 CPU 进行推理。" << std::endl;
  }

#ifdef _WIN32
  std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
  _session = Ort::Session(_env, w_modelPath.c_str(), _sessionOptions);
#else
  _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;

  _numInputNodes = _session.GetInputCount();
  _numOutputNodes = _session.GetOutputCount();

  if (_numInputNodes == 0)
    throw std::runtime_error("模型没有输入节点。");
  if (_numOutputNodes == 0)
    throw std::runtime_error("模型没有输出节点。");

  auto input_node_name = _session.GetInputNameAllocated(0, allocator);
  _inputNodeNameAllocatedStrings.push_back(std::move(input_node_name));
  _inputNames.push_back(_inputNodeNameAllocatedStrings.back().get());

  Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> modelInputTensorShapeVec = inputTensorInfo.GetShape();

  if (modelInputTensorShapeVec.size() == 4) {
    _isDynamicInputShape = (modelInputTensorShapeVec[2] == -1 ||
                            modelInputTensorShapeVec[3] == -1);
    std::cout << "模型输入张量形状来自元数据： " << modelInputTensorShapeVec[0]
              << "x" << modelInputTensorShapeVec[1] << "x"
              << modelInputTensorShapeVec[2] << "x"
              << modelInputTensorShapeVec[3] << std::endl;

    if (!_isDynamicInputShape) {
      int modelH = static_cast<int>(modelInputTensorShapeVec[2]);
      int modelW = static_cast<int>(modelInputTensorShapeVec[3]);
      if (modelH != _inputImageShape.height ||
          modelW != _inputImageShape.width) {
        std::cout << "警告：目标预处理形状 (" << _inputImageShape.height << "x"
                  << _inputImageShape.width << ") 与模型的固定输入形状 ("
                  << modelH << "x" << modelW << ") 不同。 "
                  << "图像将被预处理为 " << _inputImageShape.height << "x"
                  << _inputImageShape.width << "。"
                  << " 请考虑对齐这些以获得最佳性能/准确性。" << std::endl;
      }
    } else {
      std::cout << "模型具有动态输入高度/宽度。预处理为指定的目标： "
                << _inputImageShape.height << "x" << _inputImageShape.width
                << std::endl;
    }
  } else {
    std::cerr << "警告：模型输入张量不具有预期的 4 个维度 (NCHW)。形状： [";
    for (size_t i = 0; i < modelInputTensorShapeVec.size(); ++i)
      std::cerr << modelInputTensorShapeVec[i]
                << (i == modelInputTensorShapeVec.size() - 1 ? "" : ", ");
    std::cerr << "]。假设动态形状并继续使用目标高度x宽度： "
              << _inputImageShape.height << "x" << _inputImageShape.width
              << std::endl;
    _isDynamicInputShape = true;
  }

  auto output_node_name = _session.GetOutputNameAllocated(0, allocator);
  _outputNodeNameAllocatedStrings.push_back(std::move(output_node_name));
  _outputNames.push_back(_outputNodeNameAllocatedStrings.back().get());

  Ort::TypeInfo outputTypeInfo = _session.GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> outputTensorShapeVec = outputTensorInfo.GetShape();

  if (!outputTensorShapeVec.empty()) {
    if (outputTensorShapeVec.size() == 2 && outputTensorShapeVec[0] > 0) {
      _numClasses = static_cast<int>(outputTensorShapeVec[1]);
    } else if (outputTensorShapeVec.size() == 1 &&
               outputTensorShapeVec[0] > 0) {
      _numClasses = static_cast<int>(outputTensorShapeVec[0]);
    } else {
      for (long long dim : outputTensorShapeVec) {
        if (dim > 1 && _numClasses == 0)
          _numClasses = static_cast<int>(dim);
      }
      if (_numClasses == 0 && !outputTensorShapeVec.empty())
        _numClasses = static_cast<int>(outputTensorShapeVec.back());
    }
  }

  if (_numClasses > 0) {
    // 已更正部分用于打印 outputTensorShapeVec
    std::ostringstream oss_shape;
    oss_shape << "[";
    for (size_t i = 0; i < outputTensorShapeVec.size(); ++i) {
      oss_shape << outputTensorShapeVec[i];
      if (i < outputTensorShapeVec.size() - 1) {
        oss_shape << ", ";
      }
    }
    oss_shape << "]";
    std::cout << "模型根据输出形状预测 " << _numClasses << " 个类别： "
              << oss_shape.str() << std::endl;
    // 结束已更正部分
  } else {
    std::cerr << "警告：无法从输出形状可靠地确定类别数量： [";
    for (size_t i = 0; i < outputTensorShapeVec.size();
         ++i) { // 直接打印到 cerr
      std::cerr << outputTensorShapeVec[i]
                << (i == outputTensorShapeVec.size() - 1 ? "" : ", ");
    }
    std::cerr << "]。后处理可能不正确或假设默认值。" << std::endl;
  }

  _classNames = utils::getClassNames(labelsPath);
  if (_numClasses > 0 && !_classNames.empty() &&
      _classNames.size() != static_cast<size_t>(_numClasses)) {
    std::cerr << "警告：模型的类别数量 (" << _numClasses << ") 与 "
              << labelsPath << " 中的标签数量 (" << _classNames.size()
              << ") 不匹配。" << std::endl;
  }
  if (_classNames.empty() && _numClasses > 0) {
    std::cout << "警告：类别名称文件为空或加载失败。如果标签不可用，预测将使用"
                 "数字 ID。"
              << std::endl;
  }

  std::cout << "YOLO11Classifier 初始化成功。模型： " << modelPath << std::endl;
}

void YOLO11Classifier::preprocessImageToTensor(
    const cv::Mat &image, cv::Mat &outImage, const cv::Size &targetShape,
    const cv::Scalar &color, bool scaleUp, const std::string &strategy) {
  if (image.empty()) {
    std::cerr << "错误：输入图像到 preprocessImageToTensor 为空。" << std::endl;
    return;
  }

  if (strategy == "letterbox") {
    float r = std::min(static_cast<float>(targetShape.height) / image.rows,
                       static_cast<float>(targetShape.width) / image.cols);
    if (!scaleUp) {
      r = std::min(r, 1.0f);
    }
    int newUnpadW = static_cast<int>(std::round(image.cols * r));
    int newUnpadH = static_cast<int>(std::round(image.rows * r));

    cv::Mat resizedTemp;
    cv::resize(image, resizedTemp, cv::Size(newUnpadW, newUnpadH), 0, 0,
               cv::INTER_LINEAR);

    int dw = targetShape.width - newUnpadW;
    int dh = targetShape.height - newUnpadH;

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    cv::copyMakeBorder(resizedTemp, outImage, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);
  } else { // 默认使用 "resize"
    if (image.size() == targetShape) {
      outImage = image.clone();
    } else {
      cv::resize(image, outImage, targetShape, 0, 0, cv::INTER_LINEAR);
    }
  }
}

void YOLO11Classifier::preprocess(const cv::Mat &image, float *&blob,
                                  std::vector<int64_t> &inputTensorShape) {
  std::cout
      << "******************** 预处理 ********************"
      << std::endl;

  if (image.empty()) {
    throw std::runtime_error("输入图像到预处理为空。");
  }

  cv::Mat processedImage;
  // 1. 调整到目标输入形状（例如 224x224）。
  //    颜色 cv::Scalar(0,0,0) 用于如果使用 'letterbox' 策略时的填充；对
  //    'resize' 无用。
  preprocessImageToTensor(image, processedImage, _inputImageShape,
                          cv::Scalar(0, 0, 0), true, "resize");

  // 2. 将 BGR（OpenCV 默认）转换为 RGB
  cv::Mat rgbImageMat; // 使用不同名称以避免混淆，如果 processedImage 已是 RGB
  cv::cvtColor(processedImage, rgbImageMat, cv::COLOR_BGR2RGB);

  // 3. 转换为 float32。此时，值通常在 [0, 255] 范围内。
  cv::Mat floatRgbImage;
  rgbImageMat.convertTo(floatRgbImage, CV_32F);

  // 设置此特定输入的实际 NCHW 张量形状
  // 模型期望 NCHW：批次=1，通道=3 (RGB)，高度，宽度
  inputTensorShape = {1, 3, static_cast<int64_t>(floatRgbImage.rows),
                      static_cast<int64_t>(floatRgbImage.cols)};

  if (static_cast<int>(inputTensorShape[2]) != _inputImageShape.height ||
      static_cast<int>(inputTensorShape[3]) != _inputImageShape.width) {
    std::cerr << "严重警告：预处理后的图像尺寸 (" << inputTensorShape[2] << "x"
              << inputTensorShape[3] << ") 与目标 _inputImageShape ("
              << _inputImageShape.height << "x" << _inputImageShape.width
              << ") 在调整大小后不匹配！这表明 utils::preprocessImageToTensor "
                 "或逻辑存在问题。"
              << std::endl;
  }

  size_t tensorSize = utils::vectorProduct(inputTensorShape); // 1 * C * H * W
  blob = new float[tensorSize];

  // 4. 将像素值缩放到 [0.0, 1.0] 并将 HWC 转换为 CHW
  int h = static_cast<int>(inputTensorShape[2]);            // 高度
  int w = static_cast<int>(inputTensorShape[3]);            // 宽度
  int num_channels = static_cast<int>(inputTensorShape[1]); // 应为 3

  if (num_channels != 3) {
    delete[] blob; // 清理分配的内存
    throw std::runtime_error(
        "在 RGB 转换后，图像 blob 预期有 3 个通道，但张量形状指示： " +
        std::to_string(num_channels));
  }
  if (floatRgbImage.channels() != 3) {
    delete[] blob;
    throw std::runtime_error(
        "cv::Mat floatRgbImage 预期有 3 个通道，但得到： " +
        std::to_string(floatRgbImage.channels()));
  }

  for (int c_idx = 0; c_idx < num_channels; ++c_idx) { // 遍历 R, G, B 通道
    for (int i = 0; i < h; ++i) {                      // 遍历行（高度）
      for (int j = 0; j < w; ++j) {                    // 遍历列（宽度）
        // floatRgbImage 是 HWC（i 是行，j 是列，c_idx 是 Vec3f 中的通道索引）
        // floatRgbImage 像素值为 float，范围在 [0, 255]。
        float pixel_value = floatRgbImage.at<cv::Vec3f>(i, j)[c_idx];

        // 缩放到 [0.0, 1.0]
        float scaled_pixel = pixel_value / 255.0f;

        // 存储在 blob 中（CHW 格式）
        blob[c_idx * (h * w) + i * w + j] = scaled_pixel;
      }
    }
  }

  std::cout << "预处理完成（RGB，缩放 [0,1]）。实际输入张量形状： "
            << inputTensorShape[0] << "x" << inputTensorShape[1] << "x"
            << inputTensorShape[2] << "x" << inputTensorShape[3] << std::endl;
}

ClassificationResult
YOLO11Classifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
  std::cout << "******************** 后处理 ********************" << std::endl;

  if (outputTensors.empty()) {
    std::cerr << "错误：没有用于后处理的输出张量。" << std::endl;
    return {};
  }

  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  if (!rawOutput) {
    std::cerr << "错误：rawOutput 指针为空。" << std::endl;
    return {};
  }

  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t numScores = utils::vectorProduct(outputShape);

  // 调试输出形状
  std::ostringstream oss_shape;
  oss_shape << "输出张量形状： [";
  for (size_t i = 0; i < outputShape.size(); ++i) {
    oss_shape << outputShape[i] << (i == outputShape.size() - 1 ? "" : ", ");
  }
  oss_shape << "]";
  std::cout << oss_shape.str() << std::endl;

  // 确定有效的类别数量
  int currentNumClasses =
      _numClasses > 0 ? _numClasses : static_cast<int>(_classNames.size());
  if (currentNumClasses <= 0) {
    std::cerr << "错误：未确定有效的类别数量。" << std::endl;
    return {};
  }

  // 调试前几个原始分数
  std::ostringstream oss_scores;
  oss_scores << "前几个原始分数： ";
  for (size_t i = 0; i < std::min(size_t(5), numScores); ++i) {
    oss_scores << rawOutput[i] << " ";
  }
  std::cout << oss_scores.str() << std::endl;

  // 找到最大分数及其对应的类别
  int bestClassId = -1;
  float maxScore = -std::numeric_limits<float>::infinity();
  std::vector<float> scores(currentNumClasses);

  // 处理不同的输出形状
  if (outputShape.size() == 2 && outputShape[0] == 1) {
    // 情况 1: [1, num_classes] 形状
    for (int i = 0;
         i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i) {
      scores[i] = rawOutput[i];
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        bestClassId = i;
      }
    }
  } else if (outputShape.size() == 1 ||
             (outputShape.size() == 2 && outputShape[0] > 1)) {
    // 情况 2: [num_classes] 形状或 [batch_size, num_classes]
    // 形状（取第一个批次）
    for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores);
         ++i) {
      scores[i] = rawOutput[i];
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        bestClassId = i;
      }
    }
  }

  if (bestClassId == -1) {
    std::cerr << "错误：无法确定最佳类别 ID。" << std::endl;
    return {};
  }

  // 应用 softmax 获得概率
  float sumExp = 0.0f;
  std::vector<float> probabilities(currentNumClasses);

  // 以数值稳定性计算 softmax
  for (int i = 0; i < currentNumClasses; ++i) {
    probabilities[i] = std::exp(scores[i] - maxScore);
    sumExp += probabilities[i];
  }

  // 计算最终置信度
  float confidence = sumExp > 0 ? probabilities[bestClassId] / sumExp : 0.0f;

  // 获取类别名称
  std::string className = "未知";
  if (bestClassId >= 0 &&
      static_cast<size_t>(bestClassId) < _classNames.size()) {
    className = _classNames[bestClassId];
  } else if (bestClassId >= 0) {
    className = "ClassID_" + std::to_string(bestClassId);
  }

  std::cout << "最佳类别 ID: " << bestClassId << ", 名称: " << className
            << ", 置信度: " << confidence << std::endl;
  return ClassificationResult(bestClassId, confidence, className);
}

ClassificationResult YOLO11Classifier::detect(const cv::Mat &image) {
  std::cout << "******************** 分类任务 ********************"
            << std::endl;

  if (image.empty()) {
    std::cerr << "错误：用于分类的输入图像为空。" << std::endl;
    return {};
  }

  float *blobPtr = nullptr;
  std::vector<int64_t> currentInputTensorShape;

  try {
    preprocess(image, blobPtr, currentInputTensorShape);
  } catch (const std::exception &e) {
    std::cerr << "预处理期间发生异常： " << e.what() << std::endl;
    if (blobPtr)
      delete[] blobPtr;
    return {};
  }

  if (!blobPtr) {
    std::cerr << "错误：预处理未能产生有效的数据 blob。" << std::endl;
    return {};
  }

  size_t inputTensorSize = utils::vectorProduct(currentInputTensorShape);
  if (inputTensorSize == 0) {
    std::cerr << "错误：预处理后输入张量大小为零。" << std::endl;
    delete[] blobPtr;
    return {};
  }

  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, blobPtr, inputTensorSize, currentInputTensorShape.data(),
      currentInputTensorShape.size());

  delete[] blobPtr;
  blobPtr = nullptr;

  std::vector<Ort::Value> outputTensors;
  try {
    outputTensors =
        _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                     _numInputNodes, _outputNames.data(), _numOutputNodes);
  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime 在 Run() 期间发生异常： " << e.what()
              << std::endl;
    return {};
  }

  if (outputTensors.empty()) {
    std::cerr << "错误：ONNX Runtime Run() 未产生输出张量。" << std::endl;
    return {};
  }

  try {
    return postprocess(outputTensors);
  } catch (const std::exception &e) {
    std::cerr << "后处理期间发生异常： " << e.what() << std::endl;
    return {};
  }
}

void YOLO11Classifier::drawClassificationResult(
    cv::Mat &image, const ClassificationResult &result,
    const cv::Point &position, const cv::Scalar &textColor,
    double fontScaleMultiplier, const cv::Scalar &bgColor) {
  if (image.empty()) {
    std::cerr << "错误：提供给 drawClassificationResult 的图像为空。"
              << std::endl;
    return;
  }
  if (result.classId == -1) {
    std::cout << "由于分类结果无效，跳过绘制。" << std::endl;
    return;
  }

  std::ostringstream ss;
  ss << result.className << ": " << std::fixed << std::setprecision(2)
     << result.confidence * 100 << "%";
  std::string text = ss.str();

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
  if (fontScale < 0.4)
    fontScale = 0.4;
  const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
  int baseline = 0;

  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
  baseline += thickness;

  cv::Point textPosition = position;
  if (textPosition.x < 0)
    textPosition.x = 0;
  if (textPosition.y < textSize.height)
    textPosition.y = textSize.height + 2;

  cv::Point backgroundTopLeft(textPosition.x,
                              textPosition.y - textSize.height - baseline / 3);
  cv::Point backgroundBottomRight(textPosition.x + textSize.width,
                                  textPosition.y + baseline / 2);

  backgroundTopLeft.x = utils::clamp(backgroundTopLeft.x, 0, image.cols - 1);
  backgroundTopLeft.y = utils::clamp(backgroundTopLeft.y, 0, image.rows - 1);
  backgroundBottomRight.x =
      utils::clamp(backgroundBottomRight.x, 0, image.cols - 1);
  backgroundBottomRight.y =
      utils::clamp(backgroundBottomRight.y, 0, image.rows - 1);

  cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor,
                cv::FILLED);
  cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace,
              fontScale, textColor, thickness, cv::LINE_AA);

  std::cout << "分类结果已绘制在图像上： " << text << std::endl;
}
