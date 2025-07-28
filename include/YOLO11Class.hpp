#pragma once

/**
 * @file YOLO11Classifier.hpp
 * @brief YOLO11Classifier 类的头文件，负责使用 ONNX 模型进行图像分类，优化性能以实现最小延迟。
 */
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <iomanip>
#include <sstream>
#include "Utils.hpp"

/**
 * @brief YOLO11Classifier 类处理加载分类模型、预处理图像、运行推理和后处理结果。
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
     * @brief 在提供的图像上运行分类，返回多个分类结果。
     */
    std::vector<ClassificationResult> detect(const cv::Mat &image);

    /**
     * @brief 在图像上分行绘制分类结果。
     */
    void drawClassificationResult(cv::Mat &image, const std::vector<ClassificationResult> &results,
                                 const cv::Point &position = cv::Point(10, 10),
                                 double fontScaleMultiplier = 0.0008);

    cv::Size getInputShape() const { return _inputImageShape; }
    bool isModelInputShapeDynamic() const { return _isDynamicInputShape; }

    /**
     * @brief 通过调整大小和填充（letterbox 风格）或简单调整大小，准备模型输入图像。
     */
    void preprocessImageToTensor(const cv::Mat &image, cv::Mat &outImage,
                                const cv::Size &targetShape,
                                const cv::Scalar &color = cv::Scalar(0, 0, 0),
                                bool scaleUp = true,
                                const std::string &strategy = "resize");

    void printClassificationResult(const std::vector<ClassificationResult>& results) {
        std::cout << "Classification Results:\n";
        if (results.empty()) {
            std::cout << "  无有效分类结果。\n";
            return;
        }
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  类别 " << i + 1 << ":\n";
            std::cout << "    Class ID: " << results[i].classId << "\n";
            std::cout << "    Class Name: " << results[i].className << "\n";
            std::cout << "    Confidence: " << std::fixed << std::setprecision(4)
                      << results[i].confidence << "\n";
        }
    }
    // HSV转RGB工具函数
    static void HSVtoRGB(int *r, int *g, int *b, int h, int s, int v) {
        int i;
        float RGB_min, RGB_max;
        RGB_max = v * 2.55f;
        RGB_min = RGB_max * (100 - s) / 100.0f;
        i        = h / 60;
        int difs = h % 60;
        float RGB_Adj = (RGB_max - RGB_min) * difs / 60.0f;
        switch(i)
        {
            case 0:
                *r = RGB_max;
                *g = RGB_min + RGB_Adj;
                *b = RGB_min;
                break;
            case 1:
                *r = RGB_max - RGB_Adj;
                *g = RGB_max;
                *b = RGB_min;
                break;
            case 2:
                *r = RGB_min;
                *g = RGB_max;
                *b = RGB_min + RGB_Adj;
                break;
            case 3:
                *r = RGB_min;
                *g = RGB_max - RGB_Adj;
                *b = RGB_max;
                break;
            case 4:
                *r = RGB_min + RGB_Adj;
                *g = RGB_min;
                *b = RGB_max;
                break;
            default:
                *r = RGB_max;
                *g = RGB_min;
                *b = RGB_max - RGB_Adj;
                break;
        }
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

    void preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    std::vector<ClassificationResult> postprocess(const std::vector<Ort::Value> &outputTensors);
};

// 构造函数实现
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
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
                                  "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption{};

    if (useGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "尝试使用 GPU 进行推理。" << std::endl;
        _sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    } else {
        if (useGPU) {
            std::cout << "警告：请求使用 GPU，但 CUDAExecutionProvider 不可用。回退到 CPU。" << std::endl;
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
        std::ostringstream oss_shape;
        oss_shape << "[";
        for (size_t i = 0; i < outputTensorShapeVec.size(); ++i) {
            oss_shape << outputTensorShapeVec[i];
            if (i < outputTensorShapeVec.size() - 1) oss_shape << ", ";
        }
        oss_shape << "]";
        std::cout << "模型根据输出形状预测 " << _numClasses << " 个类别： "
                  << oss_shape.str() << std::endl;
    } else {
        std::cerr << "警告：无法从输出形状可靠地确定类别数量： [";
        for (size_t i = 0; i < outputTensorShapeVec.size(); ++i) {
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
        std::cout << "警告：类别名称文件为空或加载失败。如果标签不可用，预测将使用数字 ID。"
                  << std::endl;
    }
    std::cout << "输入节点名称： " << _inputNames[0] << std::endl;
    std::cout << "输出节点名称： " << _outputNames[0] << std::endl;
    std::cout << "YOLO11Classifier 初始化成功。模型： " << modelPath << std::endl;
}

// 预处理图像到张量
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
        if (!scaleUp) r = std::min(r, 1.0f);
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
    } else {
        if (image.size() == targetShape) {
            outImage = image.clone();
        } else {
            cv::resize(image, outImage, targetShape, 0, 0, cv::INTER_LINEAR);
        }
    }
}

// 预处理函数
void YOLO11Classifier::preprocess(const cv::Mat &image, float *&blob,
                                  std::vector<int64_t> &inputTensorShape) {
    std::cout << "******************** 预处理 ********************" << std::endl;

    if (image.empty()) {
        throw std::runtime_error("输入图像到预处理为空。");
    }

    cv::Mat processedImage;
    preprocessImageToTensor(image, processedImage, _inputImageShape,
                            cv::Scalar(0, 0, 0), true, "resize");

    cv::Mat rgbImageMat;
    cv::cvtColor(processedImage, rgbImageMat, cv::COLOR_BGR2RGB);

    cv::Mat floatRgbImage;
    rgbImageMat.convertTo(floatRgbImage, CV_32F);

    inputTensorShape = {1, 3, static_cast<int64_t>(floatRgbImage.rows),
                        static_cast<int64_t>(floatRgbImage.cols)};

    if (static_cast<int>(inputTensorShape[2]) != _inputImageShape.height ||
        static_cast<int>(inputTensorShape[3]) != _inputImageShape.width) {
        std::cerr << "严重警告：预处理后的图像尺寸 (" << inputTensorShape[2] << "x"
                  << inputTensorShape[3] << ") 与目标 _inputImageShape ("
                  << _inputImageShape.height << "x" << _inputImageShape.width
                  << ") 在调整大小后不匹配！这表明 utils::preprocessImageToTensor "
                     "或逻辑存在问题。" << std::endl;
    }

    size_t tensorSize = utils::vectorProduct(inputTensorShape);
    blob = new float[tensorSize];

    int h = static_cast<int>(inputTensorShape[2]);
    int w = static_cast<int>(inputTensorShape[3]);
    int num_channels = static_cast<int>(inputTensorShape[1]);

    if (num_channels != 3) {
        delete[] blob;
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

    for (int c_idx = 0; c_idx < num_channels; ++c_idx) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float pixel_value = floatRgbImage.at<cv::Vec3f>(i, j)[c_idx];
                float scaled_pixel = pixel_value / 255.0f;
                blob[c_idx * (h * w) + i * w + j] = scaled_pixel;
            }
        }
    }

    std::cout << "预处理完成（RGB，缩放 [0,1]）。实际输入张量形状： "
              << inputTensorShape[0] << "x" << inputTensorShape[1] << "x"
              << inputTensorShape[2] << "x" << inputTensorShape[3] << std::endl;

    std::cout << "blob 前 5 个值： ";
    for (size_t i = 0; i < std::min<size_t>(5, tensorSize); ++i) {
        std::cout << blob[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<ClassificationResult> YOLO11Classifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
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

    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t numScores = utils::vectorProduct(outputShape);

    std::ostringstream oss_shape;
    oss_shape << "输出张量形状： [";
    for (size_t i = 0; i < outputShape.size(); ++i) {
        oss_shape << outputShape[i] << (i == outputShape.size() - 1 ? "" : ", ");
    }
    oss_shape << "]";
    std::cout << oss_shape.str() << std::endl;

    int currentNumClasses = _numClasses > 0 ? _numClasses : static_cast<int>(_classNames.size());
    if (currentNumClasses <= 0) {
        std::cerr << "错误：未确定有效的类别数量。" << std::endl;
        return {};
    }

    std::cout << "完整输出分数： ";
    for (size_t i = 0; i < std::min<size_t>(numScores, currentNumClasses); ++i) {
        std::cout << rawOutput[i] << " ";
    }
    std::cout << std::endl;

    std::vector<std::pair<float, int>> scorePairs;
    if (outputShape.size() == 2 && outputShape[0] == 1) {
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i) {
            scorePairs.emplace_back(rawOutput[i], i);
            std::cout << "原始分数 [" << i << "]: " << rawOutput[i] << std::endl;
        }
    } else if (outputShape.size() == 1 || (outputShape.size() == 2 && outputShape[0] > 1)) {
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores); ++i) {
            scorePairs.emplace_back(rawOutput[i], i);
            std::cout << "原始分数 [" << i << "]: " << rawOutput[i] << std::endl;
        }
    }

    if (scorePairs.empty()) {
        std::cerr << "错误：无法收集有效的类别分数。" << std::endl;
        return {};
    }

    std::sort(scorePairs.begin(), scorePairs.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    const int maxClassesToReturn = 3;
    const float confThreshold = 0.01f;
    std::vector<ClassificationResult> results;
    float maxScore = scorePairs[0].first;
    float sumExp = 0.0f;

    for (const auto &pair : scorePairs) {
        sumExp += std::exp(pair.first - maxScore);
    }
    if (sumExp == 0.0f) {
        std::cerr << "错误：Softmax 分母为 0，可能由于输入分数无效。" << std::endl;
        return {};
    }

    for (size_t i = 0; i < std::min<size_t>(maxClassesToReturn, scorePairs.size()); ++i) {
        int classId = scorePairs[i].second;
        float score = scorePairs[i].first;
        float confidence = std::exp(score - maxScore) / sumExp;
        if (confidence < confThreshold) continue;

        std::string className = "未知";
        if (classId >= 0 && static_cast<size_t>(classId) < _classNames.size()) {
            className = _classNames[classId];
        } else {
            className = "ClassID_" + std::to_string(classId);
        }

        results.emplace_back(classId, confidence, className);
    }

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "类别 ID: " << results[i].classId
                  << ", 名称: " << results[i].className
                  << ", 置信度: " << results[i].confidence << std::endl;
    }

    return results;
}

std::vector<ClassificationResult> YOLO11Classifier::detect(const cv::Mat &image) {
    std::cout << "******************** 分类任务 ********************" << std::endl;

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
        if (blobPtr) delete[] blobPtr;
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

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, blobPtr, inputTensorSize, currentInputTensorShape.data(),
        currentInputTensorShape.size());

    delete[] blobPtr;
    blobPtr = nullptr;

    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                                    _numInputNodes, _outputNames.data(), _numOutputNodes);
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime 在 Run() 期间发生异常： " << e.what() << std::endl;
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
    cv::Mat &image, const std::vector<ClassificationResult> &results,
    const cv::Point &position, double fontScaleMultiplier) {
    if (image.empty()) {
        std::cerr << "错误：提供给 drawClassificationResult 的图像为空。" << std::endl;
        return;
    }
    if (results.empty()) {
        std::cout << "由于分类结果为空，跳过绘制。" << std::endl;
        return;
    }

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
    if (fontScale < 0.4) fontScale = 0.4;
    const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
    int baseline = 0;

    std::string sampleText = "Sample: 100.00%";
    cv::Size sampleTextSize = cv::getTextSize(sampleText, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    int lineHeight = sampleTextSize.height + baseline;

    cv::Point currentPosition = position;
    if (currentPosition.x < 0) currentPosition.x = image.cols/2;
    if (currentPosition.y < sampleTextSize.height) currentPosition.y = sampleTextSize.height + 2;
    currentPosition.x += image.cols/2;

    for (int i=0;i<results.size();i++) {
        std::ostringstream ss;
        ss << results[i].className << ": " << std::fixed << std::setprecision(2)
           << results[i].confidence * 100 << "%";
        std::string text = ss.str();

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::Point backgroundTopLeft(currentPosition.x, currentPosition.y - textSize.height - baseline / 3);
        cv::Point backgroundBottomRight(currentPosition.x + textSize.width, currentPosition.y + baseline / 2);

        backgroundTopLeft.x = utils::clamp(backgroundTopLeft.x, 0, image.cols - 1);
        backgroundTopLeft.y = utils::clamp(backgroundTopLeft.y, 0, image.rows - 1);
        backgroundBottomRight.x = utils::clamp(backgroundBottomRight.x, 0, image.cols - 1);
        backgroundBottomRight.y = utils::clamp(backgroundBottomRight.y, 0, image.rows - 1);

        // 生成HSV背景色
        int r=0,g=0,b=0;
        HSVtoRGB(&r, &g, &b, 360.0f/results.size()*i, 100, 100);
        cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, cv::Scalar(r,g,b), cv::FILLED);
        cv::putText(image, text, currentPosition, fontFace, fontScale, cv::Scalar(0,0,0), thickness, cv::LINE_AA);

        currentPosition.y += lineHeight;

        std::cout << "分类结果已绘制在图像上： " << text << std::endl;
    }
}