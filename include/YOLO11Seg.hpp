#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "Utils.hpp"

/**
 * @brief YOLOv11SegDetector 类处理加载 YOLO
 * 模型、预处理图像、运行推理和后处理结果。
 */
class YOLOv11SegDetector {
public:
  YOLOv11SegDetector(const std::string &modelPath,
                     const std::string &labelsPath, bool useGPU = false);

  // 主要 API
  std::vector<SegmentedDetection> segment(const cv::Mat &image,
                                          float confThreshold = 0.40f,
                                          float iouThreshold = 0.45f);

  // 绘制结果
  void drawSegmentationsAndBoxes(cv::Mat &image,
                                 const std::vector<SegmentedDetection> &results,
                                 float maskAlpha = 0.5f) const;

  void drawSegmentations(cv::Mat &image,
                         const std::vector<SegmentedDetection> &results,
                         float maskAlpha = 0.5f) const;

  // 访问器
  const std::vector<std::string> &getClassNames() const { return _classNames; }
  const std::vector<cv::Scalar> &getClassColors() const { return _classColors; }
  void printSegmentationResults(const std::vector<SegmentedDetection>& results) {
    std::cout << "Segmentation Results (" << results.size() << " objects):\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& det = results[i];
      std::cout << "  Object " << i+1 << ":\n";
      std::cout << "    Class ID: " << det.classId << "\n";
      std::cout << "    Confidence: " << det.conf << "\n";
      std::cout << "    Bounding Box: [x:" << det.box.x << " y:" << det.box.y
                << " w:" << det.box.width << " h:" << det.box.height << "]\n";
      std::cout << "    Mask: " << det.mask.cols << "x" << det.mask.rows
                << " (area: " << cv::countNonZero(det.mask) << " pixels)\n";
    }
  }
private:
  Ort::Env _env;
  Ort::SessionOptions _sessionOptions;
  Ort::Session _session{nullptr};

  bool _isDynamicInputShape{false};
  cv::Size _inputImageShape;

  std::vector<Ort::AllocatedStringPtr> _inputNameAllocs;
  std::vector<const char *> _inputNames;
  std::vector<Ort::AllocatedStringPtr> _outputNameAllocs;
  std::vector<const char *> _outputNames;

  size_t _numInputNodes = 0;
  size_t _numOutputNodes = 0;

  std::vector<std::string> _classNames;
  std::vector<cv::Scalar> _classColors;

  // 辅助函数
  cv::Mat preprocess(const cv::Mat &image, float *&blobPtr,
                     std::vector<int64_t> &inputTensorShape);

  std::vector<SegmentedDetection>
  postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
              const std::vector<Ort::Value> &outputs, float confThreshold,
              float iouThreshold);
};

inline YOLOv11SegDetector::YOLOv11SegDetector(const std::string &modelPath,
                                              const std::string &labelsPath,
                                              bool useGPU)
    : _env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11Seg") {
  _sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  _sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::vector<std::string> providers = Ort::GetAvailableProviders();
  if (useGPU && std::find(providers.begin(), providers.end(),
                          "CUDAExecutionProvider") != providers.end()) {
    OrtCUDAProviderOptions cudaOptions;
    _sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    std::cout << "[信息] 使用 GPU (CUDA) 进行 YOLOv11 分割推理。\n";
  } else {
    std::cout << "[信息] 使用 CPU 进行 YOLOv11 分割推理。\n";
  }

#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  _session = Ort::Session(_env, w_modelPath.c_str(), _sessionOptions);
#else
  _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

  _numInputNodes = _session.GetInputCount();
  _numOutputNodes = _session.GetOutputCount();

  Ort::AllocatorWithDefaultOptions allocator;

  // 输入
  {
    auto inNameAlloc = _session.GetInputNameAllocated(0, allocator);
    _inputNameAllocs.emplace_back(std::move(inNameAlloc));
    _inputNames.push_back(_inputNameAllocs.back().get());

    auto inTypeInfo = _session.GetInputTypeInfo(0);
    auto inShape = inTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inShape.size() == 4) {
      if (inShape[2] == -1 || inShape[3] == -1) {
        _isDynamicInputShape = true;
        _inputImageShape = cv::Size(640, 640); // 动态情况下的默认值
      } else {
        _inputImageShape = cv::Size(static_cast<int>(inShape[3]),
                                    static_cast<int>(inShape[2]));
      }
    } else {
      throw std::runtime_error("模型输入不是 4D！预期 [N, C, H, W]。");
    }
  }

  // 输出
  if (_numOutputNodes != 2) {
    throw std::runtime_error("预期正好 2 个输出节点：output0 和 output1。");
  }

  for (size_t i = 0; i < _numOutputNodes; ++i) {
    auto outNameAlloc = _session.GetOutputNameAllocated(i, allocator);
    _outputNameAllocs.emplace_back(std::move(outNameAlloc));
    _outputNames.push_back(_outputNameAllocs.back().get());
  }

  _classNames = utils::getClassNames(labelsPath);
  _classColors = utils::generateColors(_classNames);

  std::cout << "[信息] YOLOv11Seg 已加载：" << modelPath << std::endl
            << "      输入形状：" << _inputImageShape
            << (_isDynamicInputShape ? " (动态)" : "") << std::endl
            << "      输出数量：" << _numOutputNodes << std::endl
            << "      类别数量：" << _classNames.size() << std::endl;
}

inline cv::Mat
YOLOv11SegDetector::preprocess(const cv::Mat &image, float *&blobPtr,
                               std::vector<int64_t> &inputTensorShape) {
  std::cout
      << "******************** 预处理 ********************"
      << std::endl;

  cv::Mat letterboxImage;
  utils::SegLetterBox(image, letterboxImage, _inputImageShape,
                      cv::Scalar(114, 114, 114), /*auto_=*/_isDynamicInputShape,
                      /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);

  // 如果是动态输入，更新形状
  inputTensorShape[2] = static_cast<int64_t>(letterboxImage.rows);
  inputTensorShape[3] = static_cast<int64_t>(letterboxImage.cols);

  letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);

  size_t size = static_cast<size_t>(letterboxImage.rows) *
                static_cast<size_t>(letterboxImage.cols) * 3;
  blobPtr = new float[size];

  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] =
        cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                blobPtr + c * (letterboxImage.rows * letterboxImage.cols));
  }
  cv::split(letterboxImage, channels);

  return letterboxImage;
}

std::vector<SegmentedDetection>
YOLOv11SegDetector::postprocess(const cv::Size &origSize,
                                const cv::Size &letterboxSize,
                                const std::vector<Ort::Value> &outputs,
                                float confThreshold, float iouThreshold) {
  std::cout << "******************** 后处理 ********************" << std::endl;

  std::vector<SegmentedDetection> results;

  // 验证输出数量
  if (outputs.size() < 2) {
    throw std::runtime_error("模型输出不足。预期至少 2 个输出。");
  }

  // 提取输出
  const float *output0_ptr = outputs[0].GetTensorData<float>();
  const float *output1_ptr = outputs[1].GetTensorData<float>();

  // 获取形状
  auto shape0 = outputs[0]
                    .GetTensorTypeAndShapeInfo()
                    .GetShape(); // [1, 116, num_detections]
  auto shape1 = outputs[1]
                    .GetTensorTypeAndShapeInfo()
                    .GetShape(); // [1, 32, maskH, maskW]

  if (shape1.size() != 4 || shape1[0] != 1 || shape1[1] != 32)
    throw std::runtime_error(
        "意外的 output1 形状。预期 [1, 32, maskH, maskW]。");

  const size_t num_features =
      shape0[1]; // 例如 80 类 + 4 边界框参数 + 32 分割掩码 = 116
  const size_t num_detections = shape0[2];

  // 如果没有检测结果，提前退出
  if (num_detections == 0) {
    return results;
  }

  const int numClasses =
      static_cast<int>(num_features - 4 - 32); // 修正类别数量

  // 验证类别数量
  if (numClasses <= 0) {
    throw std::runtime_error("无效的类别数量。");
  }

  const int numBoxes = static_cast<int>(num_detections);
  const int maskH = static_cast<int>(shape1[2]);
  const int maskW = static_cast<int>(shape1[3]);

  // 模型架构的常量
  constexpr int BOX_OFFSET = 0;
  constexpr int CLASS_CONF_OFFSET = 4;
  const int MASK_COEFF_OFFSET = numClasses + CLASS_CONF_OFFSET;

  // 1. 处理原型掩码
  // 将所有原型掩码存储在向量中以便访问
  std::vector<cv::Mat> prototypeMasks;
  prototypeMasks.reserve(32);
  for (int m = 0; m < 32; ++m) {
    // 每个掩码为 maskH x maskW
    cv::Mat proto(maskH, maskW, CV_32F,
                  const_cast<float *>(output1_ptr + m * maskH * maskW));
    prototypeMasks.emplace_back(proto.clone()); // 克隆以确保数据完整性
  }

  // 2. 处理检测结果
  std::vector<SegBoundingBox> boxes;
  boxes.reserve(numBoxes);
  std::vector<float> confidences;
  confidences.reserve(numBoxes);
  std::vector<int> classIds;
  classIds.reserve(numBoxes);
  std::vector<std::vector<float>> maskCoefficientsList;
  maskCoefficientsList.reserve(numBoxes);

  for (int i = 0; i < numBoxes; ++i) {
    // 提取边界框坐标
    float xc = output0_ptr[BOX_OFFSET * numBoxes + i];
    float yc = output0_ptr[(BOX_OFFSET + 1) * numBoxes + i];
    float w = output0_ptr[(BOX_OFFSET + 2) * numBoxes + i];
    float h = output0_ptr[(BOX_OFFSET + 3) * numBoxes + i];

    // 转换为 xyxy 格式
    SegBoundingBox box{static_cast<int>(std::round(xc - w / 2.0f)),
                       static_cast<int>(std::round(yc - h / 2.0f)),
                       static_cast<int>(std::round(w)),
                       static_cast<int>(std::round(h))};

    // 获取类别置信度
    float maxConf = 0.0f;
    int classId = -1;
    for (int c = 0; c < numClasses; ++c) {
      float conf = output0_ptr[(CLASS_CONF_OFFSET + c) * numBoxes + i];
      if (conf > maxConf) {
        maxConf = conf;
        classId = c;
      }
    }

    if (maxConf < confThreshold)
      continue;

    // 存储检测结果
    boxes.push_back(box);
    confidences.push_back(maxConf);
    classIds.push_back(classId);

    // 存储掩码系数
    std::vector<float> maskCoeffs(32);
    for (int m = 0; m < 32; ++m) {
      maskCoeffs[m] = output0_ptr[(MASK_COEFF_OFFSET + m) * numBoxes + i];
    }
    maskCoefficientsList.emplace_back(std::move(maskCoeffs));
  }

  // 如果置信度阈值过滤后没有框，提前退出
  if (boxes.empty()) {
    return results;
  }

  // 3. 应用非极大值抑制（NMS）
  std::vector<int> nmsIndices;
  utils::SegNMSBoxes(boxes, confidences, confThreshold, iouThreshold,
                     nmsIndices);

  if (nmsIndices.empty()) {
    return results;
  }

  // 4. 准备最终结果
  results.reserve(nmsIndices.size());

  // 计算 letterbox 参数
  const float gain =
      std::min(static_cast<float>(letterboxSize.height) / origSize.height,
               static_cast<float>(letterboxSize.width) / origSize.width);
  const int scaledW = static_cast<int>(origSize.width * gain);
  const int scaledH = static_cast<int>(origSize.height * gain);
  const float padW = (letterboxSize.width - scaledW) / 2.0f;
  const float padH = (letterboxSize.height - scaledH) / 2.0f;

  // 预计算掩码缩放因子
  const float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
  const float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

  for (const int idx : nmsIndices) {
    SegmentedDetection seg;
    seg.box = boxes[idx];
    seg.conf = confidences[idx];
    seg.classId = classIds[idx];

    // 5. 将边界框缩放到原始图像
    seg.box = utils::scaleCoords(letterboxSize, seg.box, origSize, true);

    // 6. 处理掩码
    const auto &maskCoeffs = maskCoefficientsList[idx];

    // 原型掩码的线性组合
    cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
    for (int m = 0; m < 32; ++m) {
      finalMask += maskCoeffs[m] * prototypeMasks[m];
    }

    // 应用 sigmoid 激活
    finalMask = utils::sigmoid(finalMask);

    // 裁剪掩码到 letterbox 区域，略微填充以避免边界问题
    int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
    int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
    int x2 = static_cast<int>(
        std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
    int y2 = static_cast<int>(
        std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

    // 确保坐标在掩码范围内
    x1 = std::max(0, std::min(x1, maskW - 1));
    y1 = std::max(0, std::min(y1, maskH - 1));
    x2 = std::max(x1, std::min(x2, maskW));
    y2 = std::max(y1, std::min(y2, maskH));

    // 处理裁剪可能导致零面积的情况
    if (x2 <= x1 || y2 <= y1) {
      // 跳过此掩码，因为裁剪无效
      continue;
    }

    cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat croppedMask = finalMask(cropRect).clone(); // 克隆以确保数据完整性

    // 调整到原始尺寸
    cv::Mat resizedMask;
    cv::resize(croppedMask, resizedMask, origSize, 0, 0, cv::INTER_LINEAR);

    // 阈值化并转换为二值
    cv::Mat binaryMask;
    cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);

    // 裁剪到边界框
    cv::Mat finalBinaryMask = cv::Mat::zeros(origSize, CV_8U);
    cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
    roi &= cv::Rect(0, 0, binaryMask.cols,
                    binaryMask.rows); // 确保 ROI 在掩码范围内
    if (roi.area() > 0) {
      binaryMask(roi).copyTo(finalBinaryMask(roi));
    }

    seg.mask = finalBinaryMask;
    results.push_back(seg);
  }

  return results;
}

inline void YOLOv11SegDetector::drawSegmentationsAndBoxes(
    cv::Mat &image, const std::vector<SegmentedDetection> &results,
    float maskAlpha) const {
  for (const auto &seg : results) {
    cv::Scalar color = _classColors[seg.classId % _classColors.size()];

    // -----------------------------
    // 1. 绘制边界框
    // -----------------------------
    cv::rectangle(
        image, cv::Point(seg.box.x, seg.box.y),
        cv::Point(seg.box.x + seg.box.width, seg.box.y + seg.box.height), color,
        2);

    // -----------------------------
    // 2. 绘制标签
    // -----------------------------
    std::string label = _classNames[seg.classId] + " " +
                        std::to_string(static_cast<int>(seg.conf * 100)) + "%";
    int baseLine = 0;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         fontScale, thickness, &baseLine);
    int top = std::max(seg.box.y, labelSize.height + 5);
    cv::rectangle(image, cv::Point(seg.box.x, top - labelSize.height - 5),
                  cv::Point(seg.box.x + labelSize.width + 5, top), color,
                  cv::FILLED);
    cv::putText(image, label, cv::Point(seg.box.x + 2, top - 2),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255),
                thickness);

    // -----------------------------
    // 3. 应用分割掩码
    // -----------------------------
    if (!seg.mask.empty()) {
      // 确保掩码为单通道
      cv::Mat mask_gray;
      if (seg.mask.channels() == 3) {
        cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
      } else {
        mask_gray = seg.mask.clone();
      }

      // 将掩码阈值化为二值（对象：255，背景：0）
      cv::Mat mask_binary;
      cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

      // 创建掩码的彩色版本
      cv::Mat colored_mask;
      cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
      colored_mask.setTo(color, mask_binary); // 在掩码存在的地方应用颜色

      // 将彩色掩码与原始图像混合
      cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
    }
  }
}

inline void YOLOv11SegDetector::drawSegmentations(
    cv::Mat &image, const std::vector<SegmentedDetection> &results,
    float maskAlpha) const {
  for (const auto &seg : results) {
    cv::Scalar color = _classColors[seg.classId % _classColors.size()];

    // -----------------------------
    // 仅绘制分割掩码
    // -----------------------------
    if (!seg.mask.empty()) {
      // 确保掩码为单通道
      cv::Mat mask_gray;
      if (seg.mask.channels() == 3) {
        cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
      } else {
        mask_gray = seg.mask.clone();
      }

      // 将掩码阈值化为二值（对象：255，背景：0）
      cv::Mat mask_binary;
      cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

      // 创建掩码的彩色版本
      cv::Mat colored_mask;
      cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
      colored_mask.setTo(color, mask_binary); // 在掩码存在的地方应用颜色

      // 将彩色掩码与原始图像混合
      cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
    }
  }
}

inline std::vector<SegmentedDetection>
YOLOv11SegDetector::segment(const cv::Mat &image, float confThreshold,
                            float iouThreshold) {
  std::cout << "******************** 实例分割任务 ********************"
            << std::endl;
  cv::Mat src = image.clone();
  // TODO:彩色图检测无结果使用
  if(src.channels() != 3) { cv::cvtColor(src, src, cv::COLOR_GRAY2RGB); }
  else {
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
  }
  // TODO:彩色图检测无结果使用
  float *blobPtr = nullptr;
  std::vector<int64_t> inputShape = {1, 3, _inputImageShape.height,
                                     _inputImageShape.width};
  cv::Mat letterboxImg = preprocess(src, blobPtr, inputShape);

  size_t inputSize = utils::vectorProduct(inputShape);
  std::vector<float> inputVals(blobPtr, blobPtr + inputSize);
  delete[] blobPtr;

  Ort::MemoryInfo memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
      Ort::Value::CreateTensor<float>(memInfo, inputVals.data(), inputSize,
                                      inputShape.data(), inputShape.size());

  std::vector<Ort::Value> outputs =
      _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                   _numInputNodes, _outputNames.data(), _numOutputNodes);

  cv::Size letterboxSize(static_cast<int>(inputShape[3]),
                         static_cast<int>(inputShape[2]));
  return postprocess(src.size(), letterboxSize, outputs, confThreshold,
                     iouThreshold);
}
