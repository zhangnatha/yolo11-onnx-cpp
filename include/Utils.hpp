#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <random>

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Classification ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
/**
 * @brief 表示分类结果的结构体。
 */
struct ClassificationResult {
    int classId{-1};        // 预测的类别 ID，初始化为 -1 以便于错误检查
    float confidence{0.0f}; // 预测的置信度得分
    std::string className{}; // 预测的类别名称

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};
/* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Detection ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
/**
 * @brief 表示边界框的结构体。
 */
struct BoundingBox {
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

/**
 * @brief 表示检测结果的结构体。
 */
struct Detection {
    BoundingBox box;
    float conf{};
    int classId{};
};
/* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ OBB ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
/**
 * @brief 表示以 xywhr 格式的定向边界框（OBB）的结构体。
 */
struct OrientedBoundingBox {
    float x;       // 中心 x 坐标
    float y;       // 中心 y 坐标
    float width;   // 框的宽度
    float height;  // 框的高度
    float angle;   // 旋转角度（弧度）

    OrientedBoundingBox() : x(0), y(0), width(0), height(0), angle(0) {}
    OrientedBoundingBox(float x_, float y_, float width_, float height_, float angle_)
        : x(x_), y(y_), width(width_), height(height_), angle(angle_) {}
};

/**
 * @brief 表示包含定向边界框的检测结果的结构体。
 */
struct OBBDetection {
    OrientedBoundingBox box;  // 以 xywhr 格式的定向边界框
    float conf{};             // 置信度得分
    int classId{};            // 类别 ID

    OBBDetection() = default;
    OBBDetection(const OrientedBoundingBox& box_, float conf_, int classId_)
        : box(box_), conf(conf_), classId(classId_) {}
};
/* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ POSE ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
/**
 * @brief 表示姿态估计中检测到的关键点的结构体。
 */
struct KeyPoint {
    float x;         ///< 关键点的 X 坐标
    float y;         ///< 关键点的 Y 坐标
    float confidence; ///< 关键点的置信度得分

    KeyPoint(float x_ = 0, float y_ = 0, float conf_ = 0)
        : x(x_), y(y_), confidence(conf_) {}
};

/**
 * @brief 表示目标检测边界框的结构体。
 */
struct PosedBoundingBox {
    int x;      ///< 左上角的 X 坐标
    int y;      ///< 左上角的 Y 坐标
    int width;  ///< 边界框的宽度
    int height; ///< 边界框的高度

    PosedBoundingBox() : x(0), y(0), width(0), height(0) {}
    PosedBoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

/**
 * @brief 表示图像中检测到的目标的结构体。
 */
struct PosedDetection {
    PosedBoundingBox box;           ///< 检测到的目标的边界框
    float conf{};              ///< 检测的置信度得分
    int classId{};             ///< 检测到的类别的 ID
    std::vector<KeyPoint> keypoints; ///< 关键点列表（用于姿态估计）
};
/* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ SEGMENTATION ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
struct SegBoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    SegBoundingBox() = default;
    SegBoundingBox(int _x, int _y, int w, int h)
        : x(_x), y(_y), width(w), height(h) {}

    float area() const { return static_cast<float>(width * height); }

    SegBoundingBox intersect(const SegBoundingBox &other) const {
        int xStart = std::max(x, other.x);
        int yStart = std::max(y, other.y);
        int xEnd   = std::min(x + width,  other.x + other.width);
        int yEnd   = std::min(y + height, other.y + other.height);
        int iw     = std::max(0, xEnd - xStart);
        int ih     = std::max(0, yEnd - yStart);
        return SegBoundingBox(xStart, yStart, iw, ih);
    }
};

struct SegmentedDetection {
    SegBoundingBox box;
    float       conf{0.f};
    int         classId{0};
    cv::Mat     mask;  // 单通道（8UC1）掩码，全分辨率
};
/* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */

// =============================================================
// ******************* 实用工具命名空间 ***************************
// =============================================================
/**
 * @namespace utils
 * @brief 包含实用函数的命名空间。
 */
namespace utils {
    /**
     * @brief 限制值在指定范围 [low, high] 内的稳健 clamp 函数实现。
     *
     * @tparam T 要限制的值的类型。应为算术类型（int、float 等）。
     * @param value 要限制的值。
     * @param low 范围的下界。
     * @param high 范围的上界。
     * @return const T& 限制在 [low, high] 范围内的值。
     *
     * @note 如果 low > high，函数会自动交换边界以确保有效行为。
     */
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high)
    {
        // 确保范围 [low, high] 有效；如有必要，交换边界
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        // 将值限制在 [validLow, validHigh] 范围内
        if (value < validLow) return validLow;
        if (value > validHigh) return validHigh;
        return value;
    }

    /*YOLO11Seg*/
//    template <typename T>
//    T clamp(const T &val, const T &low, const T &high) {
//        return std::max(low, std::min(val, high));
//    }

    /**
     * @brief 从给定路径加载类别名称。
     *
     * @param path 包含类别名称的文件路径。
     * @return std::vector<std::string> 类别名称的向量。
     */
    std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                // 如果存在回车符，移除（为兼容 Windows）
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        } else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        std::cout << "Loaded " << classNames.size() << " class names from " << path << std::endl;
        return classNames;
    }

    /**
     * @brief 计算向量中元素的乘积。
     *
     * @param vector 整数向量。
     * @return size_t 所有元素的乘积。
     */
    size_t vectorProduct(const std::vector<int64_t> &vector) {
        return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<size_t>());
    }

    /**
     * @brief 使用 DetectedLetterBox 方式调整图像大小以保持宽高比。
     *
     * @param image 输入图像。
     * @param outImage 输出调整大小并填充的图像。
     * @param newShape 期望的输出尺寸。
     * @param color 填充颜色（默认为灰色）。
     * @param auto_ 自动调整填充以成为步幅的倍数。
     * @param scaleFill 是否缩放以填充新形状而不保持宽高比。
     * @param scaleUp 是否允许放大图像。
     * @param stride 用于填充对齐的步幅大小。
     */
    inline void DetectedLetterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true,
                        bool scaleFill = false,
                        bool scaleUp = true,
                        int stride = 32) {
        // 计算缩放比例以使图像适应新形状
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                            static_cast<float>(newShape.width) / image.cols);

        // 如果不允许放大，防止比例超过 1
        if (!scaleUp) {
            ratio = std::min(ratio, 1.0f);
        }

        // 计算缩放后的新尺寸
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

        // 计算达到目标形状所需的填充
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            // 确保填充是步幅的倍数，以兼容模型
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        } else if (scaleFill) {
            // 缩放以填充，不保持宽高比
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                            static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        } else {
            // 在两侧均匀分配填充
            // 计算左右和上下填充，以处理奇数填充
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // 如果新尺寸不同，调整图像大小
            if (image.cols != newUnpadW || image.rows != newUnpadH) {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            } else {
                // 如果尺寸相同，避免不必要的复制
                outImage = image;
            }

            // 应用填充以达到目标形状
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
            return; // 填充已应用，提前退出
        }

        // 如果新尺寸不同，调整图像大小
        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        } else {
            // 如果尺寸相同，避免不必要的复制
            outImage = image;
        }

        // 计算左右和上下填充，以处理奇数填充
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        // 应用填充以达到目标形状
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    }



    /**
     * @brief 使用 ObbLetterBox 方式调整图像大小以保持宽高比。
     *
     * @param image 输入图像。
     * @param outImage 输出调整大小并填充的图像。
     * @param newShape 期望的输出尺寸。
     * @param color 填充颜色（默认为灰色）。
     * @param auto_ 自动调整填充以成为步幅的倍数。
     * @param scaleFill 是否缩放以填充新形状而不保持宽高比。
     * @param scaleUp 是否允许放大图像。
     * @param stride 用于填充对齐的步幅大小。
     */
    inline void ObbLetterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true,
                        bool scaleFill = false,
                        bool scaleUp = true,
                        int stride = 32) {
        // 计算缩放比例以使图像适应新形状
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                            static_cast<float>(newShape.width) / image.cols);

        // 如果不允许放大，防止比例超过 1
        if (!scaleUp) {
            ratio = std::min(ratio, 1.0f);
        }

        // 计算缩放后的新尺寸
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

        // 计算达到目标形状所需的填充
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            // 确保填充是步幅的倍数，以兼容模型
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        } else if (scaleFill) {
            // 缩放以填充，不保持宽高比
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                            static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        } else {
            // 在两侧均匀分配填充
            // 计算左右和上下填充，以处理奇数填充
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // 如果新尺寸不同，调整图像大小
            if (image.cols != newUnpadW || image.rows != newUnpadH) {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            } else {
                // 如果尺寸相同，避免不必要的复制
                outImage = image;
            }

            // 应用填充以达到目标形状
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
            return; // 填充已应用，提前退出
        }

        // 如果新尺寸不同，调整图像大小
        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        } else {
            // 如果尺寸相同，避免不必要的复制
            outImage = image;
        }

        // 计算左右和上下填充，以处理奇数填充
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        // 应用填充以达到目标形状
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    }

    /**
     * @brief 使用 PosedLetterBox 方式调整图像大小，保持纵横比。
     */
    inline void PosedLetterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true,
                        bool scaleFill = false,
                        bool scaleUp = true,
                        int stride = 32) {
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                            static_cast<float>(newShape.width) / image.cols);
        if (!scaleUp) ratio = std::min(ratio, 1.0f);
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        } else if (scaleFill) {
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                            static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        } else {
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;
            if (image.cols != newUnpadW || image.rows != newUnpadH) {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            } else {
                outImage = image;
            }
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
            return;
        }

        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        } else {
            outImage = image;
        }
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    }

    /**
     * @brief 使用 SegLetterBox 方式调整图像大小，保持纵横比。
     */
    inline void SegLetterBox(const cv::Mat &image,
                          cv::Mat &outImage,
                          const cv::Size &newShape,
                          const cv::Scalar &color     = cv::Scalar(114, 114, 114),
                          bool auto_       = true,
                          bool scaleFill   = false,
                          bool scaleUp     = true,
                          int stride       = 32) {
        float r = std::min((float)newShape.height / (float)image.rows,
                           (float)newShape.width  / (float)image.cols);
        if (!scaleUp) {
            r = std::min(r, 1.0f);
        }

        int newW = static_cast<int>(std::round(image.cols * r));
        int newH = static_cast<int>(std::round(image.rows * r));

        int dw = newShape.width  - newW;
        int dh = newShape.height - newH;

        if (auto_) {
            dw = dw % stride;
            dh = dh % stride;
        }
        else if (scaleFill) {
            newW = newShape.width;
            newH = newShape.height;
            dw = 0;
            dh = 0;
        }

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        int top = dh / 2;
        int bottom = dh - top;
        int left = dw / 2;
        int right = dw - left;
        cv::copyMakeBorder(resized, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    /**
     * @brief POSE任务-对边界框执行非极大值抑制（NMS）。
     */
    void PosedNMSBoxes(const std::vector<PosedBoundingBox>& boundingBoxes,
                const std::vector<float>& scores,
                float scoreThreshold,
                float nmsThreshold,
                std::vector<int>& indices) {
        indices.clear();
        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0) {
            std::cout << "NMS 中没有要处理的边界框" << std::endl;
            return;
        }
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i) {
            if (scores[i] >= scoreThreshold) {
                sortedIndices.push_back(static_cast<int>(i));
            }
        }
        if (sortedIndices.empty()) {
            std::cout << "没有边界框超过得分阈值" << std::endl;
            return;
        }
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                [&scores](int idx1, int idx2) { return scores[idx1] > scores[idx2]; });
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i) {
            areas[i] = static_cast<float>(boundingBoxes[i].width) * boundingBoxes[i].height;
        }
        std::vector<bool> suppressed(numBoxes, false);
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx]) continue;
            indices.push_back(currentIdx);
            const PosedBoundingBox& currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];
            for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx]) continue;
                const PosedBoundingBox& compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));
                const float interWidth = std::max(0.0f, x2 - x1);
                const float interHeight = std::max(0.0f, y2 - y1);
                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;
                if (iou > nmsThreshold) {
                    suppressed[compareIdx] = true;
                }
            }
        }
        std::cout << "NMS 完成，剩余 " + std::to_string(indices.size()) + " 个索引" << std::endl;
    }

    /**
     * @brief Segment任务-对边界框执行非极大值抑制（NMS）。
     */
    inline void SegNMSBoxes(const std::vector<SegBoundingBox> &boxes,
                         const std::vector<float> &scores,
                         float scoreThreshold,
                         float nmsThreshold,
                         std::vector<int> &indices) {
        indices.clear();
        if (boxes.empty()) {
            return;
        }

        std::vector<int> order;
        order.reserve(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (scores[i] >= scoreThreshold) {
                order.push_back((int)i);
            }
        }
        if (order.empty()) return;

        std::sort(order.begin(), order.end(),
                  [&scores](int a, int b) {
                      return scores[a] > scores[b];
                  });

        std::vector<float> areas(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            areas[i] = (float)(boxes[i].width * boxes[i].height);
        }

        std::vector<bool> suppressed(boxes.size(), false);
        for (size_t i = 0; i < order.size(); ++i) {
            int idx = order[i];
            if (suppressed[idx]) continue;

            indices.push_back(idx);

            for (size_t j = i + 1; j < order.size(); ++j) {
                int idx2 = order[j];
                if (suppressed[idx2]) continue;

                const SegBoundingBox &a = boxes[idx];
                const SegBoundingBox &b = boxes[idx2];
                int interX1 = std::max(a.x, b.x);
                int interY1 = std::max(a.y, b.y);
                int interX2 = std::min(a.x + a.width,  b.x + b.width);
                int interY2 = std::min(a.y + a.height, b.y + b.height);

                int w = interX2 - interX1;
                int h = interY2 - interY1;
                if (w > 0 && h > 0) {
                    float interArea = (float)(w * h);
                    float unionArea = areas[idx] + areas[idx2] - interArea;
                    float iou = (unionArea > 0.f)? (interArea / unionArea) : 0.f;
                    if (iou > nmsThreshold) {
                        suppressed[idx2] = true;
                    }
                }
            }
        }
    }

    /**
     * @brief 对边界框执行非极大值抑制（NMS）。
     *
     * @param boundingBoxes 边界框向量。
     * @param scores 每个边界框对应的置信度分数向量。
     * @param scoreThreshold 置信度阈值，用于过滤框。
     * @param nmsThreshold NMS 的 IoU 阈值。
     * @param indices 输出保留的索引向量。
     */
    void DetectedNMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
                const std::vector<float>& scores,
                float scoreThreshold,
                float nmsThreshold,
                std::vector<int>& indices)
    {
        indices.clear();

        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0) {
            std::cout << "NMS 中没有要处理的边界框" << std::endl;
            return;
        }

        // 步骤 1：过滤掉得分低于阈值的框，并创建按得分降序排列的索引列表
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i) {
            if (scores[i] >= scoreThreshold) {
                sortedIndices.push_back(static_cast<int>(i));
            }
        }

        // 如果阈值过滤后没有框剩余
        if (sortedIndices.empty()) {
            std::cout << "没有得分高于阈值的边界框" << std::endl;
            return;
        }

        // 根据得分降序排列索引
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                [&scores](int idx1, int idx2) {
                    return scores[idx1] > scores[idx2];
                });

        // 步骤 2：预计算所有框的面积
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i) {
            areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
        }

        // 步骤 3：抑制掩码，标记被抑制的框
        std::vector<bool> suppressed(numBoxes, false);

        // 步骤 4：遍历排序列表，抑制 IoU 高的框
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx]) {
                continue;
            }

            // 选择当前框作为有效检测
            indices.push_back(currentIdx);

            const BoundingBox& currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];

            // 比较当前框与其余框的 IoU
            for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx]) {
                    continue;
                }

                const BoundingBox& compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

                const float interWidth = x2 - x1;
                const float interHeight = y2 - y1;

                if (interWidth <= 0 || interHeight <= 0) {
                    continue;
                }

                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

                if (iou > nmsThreshold) {
                    suppressed[compareIdx] = true;
                }
            }
        }

        std::cout << "NMS 完成，保留 " + std::to_string(indices.size()) + " 个索引" << std::endl;
    }



    cv::Mat sigmoid(const cv::Mat& src) {
        cv::Mat dst;
        cv::exp(-src, dst);
        dst = 1.0 / (1.0 + dst);
        return dst;
    }

    /**
     * @brief 将检测坐标缩放回原始图像大小。
     *
     * @param imageShape 用于推理的调整后图像的形状。
     * @param bbox 要缩放的检测边界框。
     * @param imageOriginalShape 调整大小前的原始图像尺寸。
     * @param p_Clip 是否将坐标裁剪到图像边界。
     * @return BoundingBox 缩放后的边界框。
     */
    BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                            const cv::Size &imageOriginalShape, bool p_Clip) {
        BoundingBox result;
        float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                              static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

        int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
        int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

        result.x = static_cast<int>(std::round((coords.x - padX) / gain));
        result.y = static_cast<int>(std::round((coords.y - padY) / gain));
        result.width = static_cast<int>(std::round(coords.width / gain));
        result.height = static_cast<int>(std::round(coords.height / gain));

        if (p_Clip) {
            result.x = utils::clamp(result.x, 0, imageOriginalShape.width);
            result.y = utils::clamp(result.y, 0, imageOriginalShape.height);
            result.width = utils::clamp(result.width, 0, imageOriginalShape.width - result.x);
            result.height = utils::clamp(result.height, 0, imageOriginalShape.height - result.y);
        }
        return result;
    }


    inline SegBoundingBox scaleCoords(const cv::Size &letterboxShape,
                                   const SegBoundingBox &coords,
                                   const cv::Size &originalShape,
                                   bool p_Clip = true) {
        float gain = std::min((float)letterboxShape.height / (float)originalShape.height,
                              (float)letterboxShape.width  / (float)originalShape.width);

        int padW = static_cast<int>(std::round(((float)letterboxShape.width  - (float)originalShape.width  * gain) / 2.f));
        int padH = static_cast<int>(std::round(((float)letterboxShape.height - (float)originalShape.height * gain) / 2.f));

        SegBoundingBox ret;
        ret.x      = static_cast<int>(std::round(((float)coords.x      - (float)padW) / gain));
        ret.y      = static_cast<int>(std::round(((float)coords.y      - (float)padH) / gain));
        ret.width  = static_cast<int>(std::round((float)coords.width   / gain));
        ret.height = static_cast<int>(std::round((float)coords.height  / gain));

        if (p_Clip) {
            ret.x = clamp(ret.x, 0, originalShape.width);
            ret.y = clamp(ret.y, 0, originalShape.height);
            ret.width  = clamp(ret.width,  0, originalShape.width  - ret.x);
            ret.height = clamp(ret.height, 0, originalShape.height - ret.y);
        }

        return ret;
    }

    /**
     * @brief 为每个类别名称生成颜色向量。
     *
     * @param classNames 类别名称向量。
     * @param seed 用于随机颜色生成的种子，以确保可重现性。
     * @return std::vector<cv::Scalar> 颜色向量。
     */
    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42) {
        // 静态缓存，基于类别名称存储颜色，避免重复生成
        static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

        // 根据类别名称计算哈希键，以识别唯一类别配置
        size_t hashKey = 0;
        for (const auto& name : classNames) {
            hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
        }

        // 检查此类别配置的颜色是否已缓存
        auto it = colorCache.find(hashKey);
        if (it != colorCache.end()) {
            return it->second;
        }

        // 为每个类别生成独特的随机颜色
        std::vector<cv::Scalar> colors;
        colors.reserve(classNames.size());

        std::mt19937 rng(seed); // 使用固定种子初始化随机数生成器
        std::uniform_int_distribution<int> uni(0, 255); // 定义颜色值的分布

        for (size_t i = 0; i < classNames.size(); ++i) {
            colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng))); // 生成随机 BGR 颜色
        }

        // 缓存生成的颜色以供将来使用
        colorCache.emplace(hashKey, colors);

        return colorCache[hashKey];
    }
}


/**
 * @namespace OBB_NMS
 * @brief 包含 OBB 格式的非极大值抑制（NMS）函数的命名空间。
 */
namespace OBB_NMS {

    static constexpr float EPS = 1e-7f;

    /**
     * @brief 计算单个 OBB 的协方差矩阵分量。
     * @param box 输入的定向边界框。
     * @param out1 第一个分量（a）。
     * @param out2 第二个分量（b）。
     * @param out3 第三个分量（c）。
     */
    void getCovarianceComponents(const OrientedBoundingBox& box, float& out1, float& out2, float& out3) {
        float a = (box.width * box.width) / 12.0f;
        float b = (box.height * box.height) / 12.0f;
        float angle = box.angle;

        float cos_theta = std::cos(angle);
        float sin_theta = std::sin(angle);
        float cos_sq = cos_theta * cos_theta;
        float sin_sq = sin_theta * sin_theta;

        out1 = a * cos_sq + b * sin_sq;
        out2 = a * sin_sq + b * cos_sq;
        out3 = (a - b) * cos_theta * sin_theta;
    }

    /**
     * @brief 计算两组 OBB 之间的 IoU 矩阵。
     * @param obb1 第一组 OBB。
     * @param obb2 第二组 OBB。
     * @param eps 用于数值稳定的小常数。
     * @return IoU 值的二维向量。
     */
    std::vector<std::vector<float>> batchProbiou(const std::vector<OrientedBoundingBox>& obb1, const std::vector<OrientedBoundingBox>& obb2, float eps = EPS) {
        size_t N = obb1.size();
        size_t M = obb2.size();
        std::vector<std::vector<float>> iou_matrix(N, std::vector<float>(M, 0.0f));

        for (size_t i = 0; i < N; ++i) {
            const OrientedBoundingBox& box1 = obb1[i];
            float x1 = box1.x, y1 = box1.y;
            float a1, b1, c1;
            getCovarianceComponents(box1, a1, b1, c1);

            for (size_t j = 0; j < M; ++j) {
                const OrientedBoundingBox& box2 = obb2[j];
                float x2 = box2.x, y2 = box2.y;
                float a2, b2, c2;
                getCovarianceComponents(box2, a2, b2, c2);

                // 计算分母
                float denom = (a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps;

                // 项 t1 和 t2
                float dx = x1 - x2;
                float dy = y1 - y2;
                float t1 = ((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) * 0.25f / denom;
                float t2 = ((c1 + c2) * (x2 - x1) * dy) * 0.5f / denom;

                // 项 t3
                float term1 = a1 * b1 - c1 * c1;
                term1 = std::max(term1, 0.0f);
                float term2 = a2 * b2 - c2 * c2;
                term2 = std::max(term2, 0.0f);
                float sqrt_term = std::sqrt(term1 * term2);

                float numerator = (a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2);
                float denominator_t3 = 4.0f * sqrt_term + eps;
                float t3 = 0.5f * std::log(numerator / denominator_t3 + eps);

                // 计算最终 IoU
                float bd = t1 + t2 + t3;
                bd = std::clamp(bd, eps, 100.0f);
                float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
                iou_matrix[i][j] = 1.0f - hd;
            }
        }
        return iou_matrix;
    }

    /**
     * @brief 在排序后的 OBB 上执行旋转 NMS。
     * @param sorted_boxes 按置信度排序的框。
     * @param iou_thres IoU 抑制阈值。
     * @return 保留的框的索引。
     */
    std::vector<int> nmsRotatedImpl(const std::vector<OrientedBoundingBox>& sorted_boxes, float iou_thres) {
        auto ious = batchProbiou(sorted_boxes, sorted_boxes);
        std::vector<int> keep;
        const int n = sorted_boxes.size();

        for (int j = 0; j < n; ++j) {
            bool keep_j = true;
            for (int i = 0; i < j; ++i) {
                if (ious[i][j] >= iou_thres) {
                    keep_j = false;
                    break;
                }
            }
            if (keep_j) keep.push_back(j);
        }
        return keep;
    }

    /**
     * @brief 旋转框的主要 NMS 函数。
     * @param boxes 输入框。
     * @param scores 置信度得分。
     * @param threshold IoU 阈值。
     * @return 保留的框的索引。
     */
    std::vector<int> nmsRotated(const std::vector<OrientedBoundingBox>& boxes, const std::vector<float>& scores, float threshold = 0.75f) {
        // 根据得分排序索引
        std::vector<int> indices(boxes.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
            return scores[a] > scores[b];
        });

        // 创建排序后的框列表
        std::vector<OrientedBoundingBox> sorted_boxes;
        for (int idx : indices) {
            sorted_boxes.push_back(boxes[idx]);
        }

        // 执行 NMS
        std::vector<int> keep = nmsRotatedImpl(sorted_boxes, threshold);

        // 映射回原始索引
        std::vector<int> keep_indices;
        for (int k : keep) {
            keep_indices.push_back(indices[k]);
        }
        return keep_indices;
    }

    /**
     * @brief 对检测结果应用 NMS 并返回过滤后的结果。
     * @param input_detections 输入检测结果。
     * @param conf_thres 置信度阈值。
     * @param iou_thres IoU 阈值。
     * @param max_det 保留的最大检测数量。
     * @return NMS 后的过滤检测结果。
     */
    std::vector<OBBDetection> nonMaxSuppression(
        const std::vector<OBBDetection>& input_detections,
        float conf_thres = 0.25f,
        float iou_thres = 0.75f,
        int max_det = 1000) {

        // 按置信度过滤
        std::vector<OBBDetection> candidates;
        for (const auto& det : input_detections) {
            if (det.conf > conf_thres) {
                candidates.push_back(det);
            }
        }
        if (candidates.empty()) return {};

        // 提取框和得分
        std::vector<OrientedBoundingBox> boxes;
        std::vector<float> scores;
        for (const auto& det : candidates) {
            boxes.push_back(det.box);
            scores.push_back(det.conf);
        }

        // 运行 NMS
        std::vector<int> keep_indices = nmsRotated(boxes, scores, iou_thres);

        // 收集结果
        std::vector<OBBDetection> results;
        for (int idx : keep_indices) {
            if (results.size() >= max_det) break;
            results.push_back(candidates[idx]);
        }
        return results;
    }

}
