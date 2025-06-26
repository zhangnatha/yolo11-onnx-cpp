#pragma once

// ===================================
// 单文件 YOLOv11 姿态检测器头文件
// ===================================
// 此头文件定义了 YOLO11PoseDetector 类，用于使用 YOLOv11 模型进行姿态估计。
// 支持从 YAML 文件动态加载关键点数量、维度、类别数、类别名称和骨架定义。
// ================================

/**
 * @file YOLO11-POSE.hpp
 * @brief YOLO11PoseDetector 类的头文件，负责使用 YOLOv11 模型进行姿态估计，
 *        针对实时应用进行了性能优化，并支持动态配置。
 */
#include <onnxruntime_cxx_api.h>

// 添加 YAML 解析库
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "Utils.hpp"

#include <iomanip>
#include <sstream>
/**
 * @brief YOLO11POSEDetector 类处理加载 YOLO
 * 模型、预处理图像、运行推理和后处理结果。
 */
class YOLO11PoseDetector {
public:
  YOLO11PoseDetector(const std::string &modelPath, const std::string &yamlPath,
                     bool useGPU = false);
  std::vector<PosedDetection> detect(const cv::Mat &image,
                                     float confThreshold = 0.4f,
                                     float iouThreshold = 0.5f);
  void drawPosedBoundingBox(cv::Mat &image,
                            const std::vector<PosedDetection> &detections);
  void printPoseResults(const std::vector<PosedDetection>& results) {
    std::cout << "Pose Detection Results (" << results.size() << " objects):\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& det = results[i];
      std::cout << "  Object " << i+1 << ":\n";
      std::cout << "    Class ID: " << det.classId << "\n";
      std::cout << "    Confidence: " << det.conf << "\n";
      std::cout << "    Bounding Box: [x:" << det.box.x << " y:" << det.box.y
                << " w:" << det.box.width << " h:" << det.box.height << "]\n";
      std::cout << "    Keypoints (" << det.keypoints.size() << "):\n";
      for (size_t k = 0; k < det.keypoints.size(); ++k) {
        const auto& kp = det.keypoints[k];
        std::cout << "      " << k+1 << ": [x:" << kp.x << " y:" << kp.y
                  << " conf:" << kp.confidence << "]\n";
      }
    }
  }
private:
  Ort::Env _env{nullptr};
  Ort::SessionOptions _sessionOptions{nullptr};
  Ort::Session _session{nullptr};
  bool _isDynamicInputShape{};
  cv::Size _inputImageShape;
  std::vector<Ort::AllocatedStringPtr> _inputNodeNameAllocatedStrings;
  std::vector<const char *> _inputNames;
  std::vector<Ort::AllocatedStringPtr> _outputNodeNameAllocatedStrings;
  std::vector<const char *> _outputNames;
  size_t _numInputNodes, _numOutputNodes;

  // 从 YAML 文件加载的参数
  int _numKeypoints;
  int _featuresPerKeypoint;
  int _numClasses;
  std::vector<std::string> _class_names; // 动态加载的类别名称
  std::vector<std::pair<int, int>> _pose_skeleton;
  std::vector<int> _flip_idx;

  cv::Mat preprocess(const cv::Mat &image, float *&blob,
                     std::vector<int64_t> &inputTensorShape);
  std::vector<PosedDetection>
  postprocess(const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape,
              const std::vector<Ort::Value> &outputTensors, float confThreshold,
              float iouThreshold);
  /**
   * @brief 绘制姿态估计，包括边界框、关键点和骨架。
   *        为边界框添加带背景的类别标签和置信度，为关键点添加索引标签。
   */
  inline void
  drawPoseEstimation(cv::Mat &image,
                     const std::vector<PosedDetection> &detections,
                     const std::vector<std::pair<int, int>> &pose_skeleton,
                     const std::vector<std::string> &class_names,
                     float confidenceThreshold = 0.5, float kptThreshold = 0.5);
};

// 构造函数实现
YOLO11PoseDetector::YOLO11PoseDetector(const std::string &modelPath,
                                       const std::string &yamlPath,
                                       bool useGPU) {
  _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
  _sessionOptions = Ort::SessionOptions();
  _sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  _sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable =
      std::find(availableProviders.begin(), availableProviders.end(),
                "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;
  if (useGPU && cudaAvailable != availableProviders.end()) {
    std::cout << "推理设备：GPU" << std::endl;
    _sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
  } else {
    if (useGPU)
      std::cout << "您的 ONNXRuntime 构建不支持 GPU。回退到 CPU。" << std::endl;
    std::cout << "推理设备：CPU" << std::endl;
  }

#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  _session = Ort::Session(_env, w_modelPath.c_str(), _sessionOptions);
#else
  _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShapeVec =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  _isDynamicInputShape =
      (inputTensorShapeVec.size() >= 4) &&
      (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1);

  auto input_name = _session.GetInputNameAllocated(0, allocator);
  _inputNodeNameAllocatedStrings.push_back(std::move(input_name));
  _inputNames.push_back(_inputNodeNameAllocatedStrings.back().get());

  auto output_name = _session.GetOutputNameAllocated(0, allocator);
  _outputNodeNameAllocatedStrings.push_back(std::move(output_name));
  _outputNames.push_back(_outputNodeNameAllocatedStrings.back().get());

  if (inputTensorShapeVec.size() >= 4) {
    _inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                                static_cast<int>(inputTensorShapeVec[2]));
  } else {
    throw std::runtime_error("无效的输入张量形状。");
  }

  _numInputNodes = _session.GetInputCount();
  _numOutputNodes = _session.GetOutputCount();

  try {
    YAML::Node config = YAML::LoadFile(yamlPath);
    auto kpt_shape = config["kpt_shape"].as<std::vector<int>>();
    if (kpt_shape.size() != 2) {
      throw std::runtime_error("YAML 文件中的 kpt_shape 格式错误，应为 "
                               "[num_keypoints, features_per_keypoint]");
    }
    _numKeypoints = kpt_shape[0];
    _featuresPerKeypoint = kpt_shape[1];
    _flip_idx = config["flip_idx"].as<std::vector<int>>();
    auto skeleton = config["pose_skeleton"];
    for (const auto &pair : skeleton) {
      auto indices = pair.as<std::vector<int>>();
      if (indices.size() == 2) {
        _pose_skeleton.emplace_back(indices[0], indices[1]);
      }
    }
    auto names = config["names"];
    _numClasses = names.size();
    _class_names.resize(_numClasses);
    for (const auto &name : names) {
      int index = name.first.as<int>();
      if (index >= 0 && index < _numClasses) {
        _class_names[index] = name.second.as<std::string>();
      }
    }

    std::cout << "从 YAML 文件加载配置：关键点数量=" << _numKeypoints
              << "，特征维度=" << _featuresPerKeypoint
              << "，类别数=" << _numClasses
              << "，骨架连接数=" << _pose_skeleton.size() << "，类别名称=[";
    for (size_t i = 0; i < _class_names.size(); ++i) {
      std::cout << (i > 0 ? ", " : "") << _class_names[i];
    }
    std::cout << "]" << std::endl;
  } catch (const std::exception &e) {
    throw std::runtime_error("解析 YAML 文件失败：" + std::string(e.what()));
  }

  std::cout << "模型加载成功，包含 " << _numInputNodes << " 个输入节点和 "
            << _numOutputNodes << " 个输出节点。" << std::endl;
}

// 预处理函数实现
cv::Mat YOLO11PoseDetector::preprocess(const cv::Mat &image, float *&blob,
                                       std::vector<int64_t> &inputTensorShape) {
  std::cout
      << "******************** 预处理 ********************"
      << std::endl;
  cv::Mat resizedImage;
  utils::PosedLetterBox(image, resizedImage, _inputImageShape,
                        cv::Scalar(114, 114, 114), _isDynamicInputShape, false,
                        true, 32);
  inputTensorShape[2] = resizedImage.rows;
  inputTensorShape[3] = resizedImage.cols;
  resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);
  blob = new float[resizedImage.cols * resizedImage.rows *
                   resizedImage.channels()];
  std::vector<cv::Mat> chw(resizedImage.channels());
  for (int i = 0; i < resizedImage.channels(); ++i) {
    chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                     blob + i * resizedImage.cols * resizedImage.rows);
  }
  cv::split(resizedImage, chw);
  std::cout << "预处理完成" << std::endl;
  return resizedImage;
}

// 后处理函数实现
std::vector<PosedDetection>
YOLO11PoseDetector::postprocess(const cv::Size &originalImageSize,
                                const cv::Size &resizedImageShape,
                                const std::vector<Ort::Value> &outputTensors,
                                float confThreshold, float iouThreshold) {
  std::cout << "******************** 后处理 ********************" << std::endl;
  std::vector<PosedDetection> detections;

  const float *rawOutput = outputTensors[0].GetTensorData<float>();
  const std::vector<int64_t> outputShape =
      outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // 验证输出维度
  const size_t numFeatures = outputShape[1];
  const size_t numDetections = outputShape[2];
  size_t expectedFeatures =
      4 + _numClasses + _numKeypoints * _featuresPerKeypoint;
  if (numFeatures != expectedFeatures) {
    std::cerr << "姿态估计模型的输出形状无效，期望特征数=" << expectedFeatures
              << "，实际特征数=" << numFeatures << std::endl;
    return detections;
  }

  // 计算 PosedLetterBox 填充参数
  const float scale = std::min(
      static_cast<float>(resizedImageShape.width) / originalImageSize.width,
      static_cast<float>(resizedImageShape.height) / originalImageSize.height);
  const cv::Size scaledSize(originalImageSize.width * scale,
                            originalImageSize.height * scale);
  const cv::Point2f padding((resizedImageShape.width - scaledSize.width) / 2.0f,
                            (resizedImageShape.height - scaledSize.height) /
                                2.0f);

  // 处理每个检测
  std::vector<PosedBoundingBox> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<KeyPoint>> allKeypoints;
  std::vector<int> classIds;

  for (size_t d = 0; d < numDetections; ++d) {
    // 提取类别分数并选择最高置信度类别
    float maxConf = 0.0f;
    int maxClassId = 0;
    for (int c = 0; c < _numClasses; ++c) {
      float conf = rawOutput[(4 + c) * numDetections + d];
      if (conf > maxConf) {
        maxConf = conf;
        maxClassId = c;
      }
    }
    if (maxConf < confThreshold)
      continue;

    // 解码边界框
    const float cx = rawOutput[0 * numDetections + d];
    const float cy = rawOutput[1 * numDetections + d];
    const float w = rawOutput[2 * numDetections + d];
    const float h = rawOutput[3 * numDetections + d];

    PosedBoundingBox box;
    box.x = static_cast<int>((cx - padding.x - w / 2) / scale);
    box.y = static_cast<int>((cy - padding.y - h / 2) / scale);
    box.width = static_cast<int>(w / scale);
    box.height = static_cast<int>(h / scale);

    box.x = utils::clamp(box.x, 0, originalImageSize.width - box.width);
    box.y = utils::clamp(box.y, 0, originalImageSize.height - box.height);
    box.width = utils::clamp(box.width, 0, originalImageSize.width - box.x);
    box.height = utils::clamp(box.height, 0, originalImageSize.height - box.y);

    // 提取关键点
    std::vector<KeyPoint> keypoints;
    for (int k = 0; k < _numKeypoints; ++k) {
      const int offset = 4 + _numClasses + k * _featuresPerKeypoint;
      KeyPoint kpt;
      kpt.x = (rawOutput[offset * numDetections + d] - padding.x) / scale;
      kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padding.y) / scale;
      kpt.confidence =
          _featuresPerKeypoint == 3
              ? 1.0f / (1.0f +
                        std::exp(-rawOutput[(offset + 2) * numDetections + d]))
              : 1.0f;

      kpt.x = utils::clamp(kpt.x, 0.0f,
                           static_cast<float>(originalImageSize.width - 1));
      kpt.y = utils::clamp(kpt.y, 0.0f,
                           static_cast<float>(originalImageSize.height - 1));
      keypoints.emplace_back(kpt);
    }

    boxes.emplace_back(box);
    confidences.emplace_back(maxConf);
    classIds.emplace_back(maxClassId);
    allKeypoints.emplace_back(keypoints);
  }

  // 应用非极大值抑制
  std::vector<int> indices;
  utils::PosedNMSBoxes(boxes, confidences, confThreshold, iouThreshold,
                       indices);

  // 创建最终检测结果
  for (int idx : indices) {
    PosedDetection det;
    det.box = boxes[idx];
    det.conf = confidences[idx];
    det.classId = classIds[idx];
    det.keypoints = allKeypoints[idx];
    detections.emplace_back(det);
  }

  return detections;
}

// 检测函数实现
std::vector<PosedDetection> YOLO11PoseDetector::detect(const cv::Mat &image,
                                                       float confThreshold,
                                                       float iouThreshold) {
  std::cout << "******************** 位姿/关键点检测任务 ********************"
            << std::endl;

  float *blobPtr = nullptr;
  std::vector<int64_t> inputTensorShape = {1, 3, _inputImageShape.height,
                                           _inputImageShape.width};
  cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);
  size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
  std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);
  delete[] blobPtr;
  static Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());
  std::vector<Ort::Value> outputTensors =
      _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                   _numInputNodes, _outputNames.data(), _numOutputNodes);
  cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                             static_cast<int>(inputTensorShape[2]));
  std::vector<PosedDetection> detections =
      postprocess(image.size(), resizedImageShape, outputTensors, confThreshold,
                  iouThreshold);
  return detections;
}

void YOLO11PoseDetector::drawPoseEstimation(
    cv::Mat &image, const std::vector<PosedDetection> &detections,
    const std::vector<std::pair<int, int>> &pose_skeleton,
    const std::vector<std::string> &class_names, float confidenceThreshold,
    float kptThreshold) {
  const int min_dim = std::min(image.rows, image.cols);
  const float scale_factor = min_dim / 1280.0f;
  const int line_thickness = std::max(1, static_cast<int>(2 * scale_factor));
  const int kpt_radius = std::max(2, static_cast<int>(4 * scale_factor));
  const float font_scale = 0.5f * scale_factor;
  const float kpt_font_scale = 0.4f * scale_factor; // 关键点标签字体稍小
  const int text_thickness = std::max(1, static_cast<int>(1 * scale_factor));
  const int text_offset = static_cast<int>(10 * scale_factor);

  static const std::vector<cv::Scalar> pose_palette = {
      cv::Scalar(0, 128, 255),   cv::Scalar(51, 153, 255),
      cv::Scalar(102, 178, 255), cv::Scalar(0, 230, 230),
      cv::Scalar(255, 153, 255), cv::Scalar(255, 204, 153),
      cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
      cv::Scalar(255, 178, 102), cv::Scalar(255, 153, 51),
      cv::Scalar(153, 153, 255), cv::Scalar(102, 102, 255),
      cv::Scalar(51, 51, 255),   cv::Scalar(153, 255, 153),
      cv::Scalar(102, 255, 102), cv::Scalar(51, 255, 51),
      cv::Scalar(0, 255, 0),     cv::Scalar(255, 0, 0),
      cv::Scalar(0, 0, 255),     cv::Scalar(255, 255, 255)};

  // 动态调整颜色索引，适应任意关键点数量
  size_t numKpts = detections.empty() ? 2 : detections[0].keypoints.size();
  std::vector<int> kpt_color_indices(numKpts, 16);              // 默认绿色
  std::vector<int> limb_color_indices(pose_skeleton.size(), 9); // 默认橙色

  for (const auto &detection : detections) {
    if (detection.conf < confidenceThreshold)
      continue;
    const auto &box = detection.box;
    cv::rectangle(image, cv::Point(box.x, box.y),
                  cv::Point(box.x + box.width, box.y + box.height),
                  cv::Scalar(0, 255, 0), line_thickness);

    // 添加带背景的类别标签和置信度
    std::string class_name = (detection.classId < class_names.size())
                                 ? class_names[detection.classId]
                                 : "unknown";
    std::stringstream conf_stream;
    conf_stream << std::fixed << std::setprecision(1) << detection.conf * 100;
    std::string label = class_name + ": " + conf_stream.str() + "%";
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         font_scale, text_thickness, nullptr);
    int text_x = box.x;
    int text_y = box.y - text_offset;
    if (text_y - text_size.height < 0) { // 防止标签超出图像顶部
      text_y = box.y + text_size.height + text_offset;
    }
    // 绘制背景矩形（半透明黑色）
    cv::rectangle(image, cv::Point(text_x - 2, text_y - text_size.height - 2),
                  cv::Point(text_x + text_size.width + 2, text_y + 2),
                  cv::Scalar(0, 0, 0, 180), cv::FILLED);
    // 绘制文本
    cv::putText(image, label, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                text_thickness, cv::LINE_AA);

    std::vector<cv::Point> kpt_points(numKpts, cv::Point(-1, -1));
    std::vector<bool> valid(numKpts, false);
    for (size_t i = 0; i < numKpts; i++) {
      if (detection.keypoints[i].confidence >= kptThreshold) {
        int x = static_cast<int>(std::round(detection.keypoints[i].x));
        int y = static_cast<int>(std::round(detection.keypoints[i].y));
        kpt_points[i] = cv::Point(x, y);
        valid[i] = true;
        int color_index =
            (i < kpt_color_indices.size()) ? kpt_color_indices[i] : 0;
        cv::circle(image, cv::Point(x, y), kpt_radius,
                   pose_palette[color_index], -1, cv::LINE_AA);

        // 添加关键点索引标签
        std::string kpt_label = std::to_string(i);
        cv::Size kpt_text_size =
            cv::getTextSize(kpt_label, cv::FONT_HERSHEY_SIMPLEX, kpt_font_scale,
                            text_thickness, nullptr);
        int kpt_text_x = x + kpt_radius + 2; // 右偏移
        int kpt_text_y = y - 2;              // 略微上移
        // 绘制关键点标签背景（半透明黑色）
        cv::rectangle(
            image,
            cv::Point(kpt_text_x - 2, kpt_text_y - kpt_text_size.height - 2),
            cv::Point(kpt_text_x + kpt_text_size.width + 2, kpt_text_y + 2),
            cv::Scalar(0, 0, 0, 180), cv::FILLED);
        // 绘制关键点索引文本
        cv::putText(image, kpt_label, cv::Point(kpt_text_x, kpt_text_y),
                    cv::FONT_HERSHEY_SIMPLEX, kpt_font_scale,
                    cv::Scalar(255, 255, 255), text_thickness, cv::LINE_AA);
      }
    }
    for (size_t j = 0; j < pose_skeleton.size(); j++) {
      auto [src, dst] = pose_skeleton[j];
      if (src < numKpts && dst < numKpts && valid[src] && valid[dst]) {
        int limb_color_index =
            (j < limb_color_indices.size()) ? limb_color_indices[j] : 0;
        cv::line(image, kpt_points[src], kpt_points[dst],
                 pose_palette[limb_color_index], line_thickness, cv::LINE_AA);
      }
    }
  }
}

// 绘制边界框函数实现
void YOLO11PoseDetector::drawPosedBoundingBox(
    cv::Mat &image, const std::vector<PosedDetection> &detections) {
  float confidenceThreshold = 0.5;
  float kptThreshold = 0.5;
  drawPoseEstimation(image, detections, _pose_skeleton, _class_names,
                     confidenceThreshold, kptThreshold);
}
