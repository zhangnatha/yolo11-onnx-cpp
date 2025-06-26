cmake_minimum_required(VERSION 3.17)
project(YOLO_ORT)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置第三方库目录路径
if(WIN32)
    set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdpartyWin")
else()
    set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdpartyU18")
endif()

# 设置OpenCV路径
if(WIN32)
    set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv")
else()
    set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv/lib/cmake/opencv4")
endif()
find_package(OpenCV 4.5.5 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV dir: ${OpenCV_DIR}")

# 设置ONNX Runtime路径
if(WIN32)
    if(BUILD_GPU)
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime-win-x64-gpu-cuda12-1.17.3")
        message(STATUS "Building with GPU support")
    else()
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime-win-x64-1.17.3")
        message(STATUS "Building with CPU support")
    endif()
else()
    if(BUILD_GPU)
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime-cuda12-1.17.3")
        message(STATUS "Building with GPU support")
    else()
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime")
        message(STATUS "Building with CPU support")
    endif()
endif()
message(STATUS "ONNX Runtime dir: ${ONNXRUNTIME_DIR}")



# 全局包含目录
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "${ONNXRUNTIME_DIR}/include"
    ${THIRDPARTY_DIR}/camera_realsense/include
    ${THIRDPARTY_DIR}/camera_realsense/include/librealsense2
)

# 仅在UNIX系统下添加yaml-cpp依赖
if(UNIX)
    include_directories(${CMAKE_SOURCE_DIR}/3rdpartyU18/yaml/include)
    link_directories(${CMAKE_SOURCE_DIR}/3rdpartyU18/yaml/lib)
endif()

# 查找pthread库（Linux需要）
if(UNIX)
    find_package(Threads REQUIRED)
endif()

# 设置FFmpeg路径（仅UNIX）
if(UNIX)
    set(FFMPEG_DIR "${THIRDPARTY_DIR}/ffmpeg")
    include_directories(${FFMPEG_DIR}/include)
    link_directories(${FFMPEG_DIR}/lib)
endif()

# 添加可执行文件
add_executable(image_inference src/image_inference.cpp)
add_executable(camera_inference src/camera_inference.cpp)
add_executable(video_inference src/video_inference.cpp)

# 设置编译特性
target_compile_features(image_inference PRIVATE cxx_std_17)
target_compile_features(camera_inference PRIVATE cxx_std_17)
target_compile_features(video_inference PRIVATE cxx_std_17)

# 链接库
if(WIN32)
    message(STATUS "Building on Windows")
    target_link_libraries(image_inference
        ${OpenCV_LIBS}
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    )
    target_link_libraries(camera_inference ${OpenCV_LIBS} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib" ${THIRDPARTY_DIR}/camera_realsense/lib/librealsense2.so.2.50.0)
    target_link_libraries(video_inference ${OpenCV_LIBS} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
else()
    message(STATUS "Building on Linux")
    target_link_libraries(image_inference
        ${OpenCV_LIBS}
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
        Threads::Threads
        avcodec avformat avutil swscale
    )
    target_link_libraries(camera_inference ${OpenCV_LIBS} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" Threads::Threads ${THIRDPARTY_DIR}/camera_realsense/lib/librealsense2.so.2.50.0 avcodec avformat avutil swscale)
    target_link_libraries(video_inference ${OpenCV_LIBS} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" Threads::Threads avcodec avformat avutil swscale)
endif()

# 仅在UNIX系统下添加yaml-cpp依赖
if(UNIX)
    target_link_libraries(image_inference yaml-cpp)
    target_link_libraries(camera_inference yaml-cpp)
    target_link_libraries(video_inference yaml-cpp)
endif()
