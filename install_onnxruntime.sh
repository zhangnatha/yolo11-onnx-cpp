#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

# 定义安装路径
SOURCE_DIR="$(pwd)"
INSTALL_GPU_DIR="$(pwd)/3rdpartyU18/onnxruntime-cuda12-1.17.3"
INSTALL_CPU_DIR="$(pwd)/3rdpartyU18/onnxruntime"

# 1. 下载onnxruntime CPU/GPU
echo "下载onnxruntime lib..."
# GPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-cuda12-1.17.3.tgz
# CPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz

# 2. 安装onnxruntime CPU/GPU
echo "解压 GPU 版本库..."
tar -xzf onnxruntime-linux-x64-gpu-cuda12-1.17.3.tgz -C "$SOURCE_DIR"
echo "解压 CPU 版本库..."
tar -xzf onnxruntime-linux-x64-1.17.3.tgz -C "$SOURCE_DIR"

# 创建目标目录（如果不存在）
mkdir -p "$INSTALL_GPU_DIR"
mkdir -p "$INSTALL_CPU_DIR"
mv "$SOURCE_DIR/onnxruntime-linux-x64-gpu-1.17.3"/* "$INSTALL_GPU_DIR/"
mv "$SOURCE_DIR/onnxruntime-linux-x64-1.17.3"/* "$INSTALL_CPU_DIR/"

# 3. 清理下载的压缩文件
echo "清除..."
rm onnxruntime-linux-x64-gpu-cuda12-1.17.3.tgz onnxruntime-linux-x64-gpu-1.17.3
rm onnxruntime-linux-x64-1.17.3.tgz onnxruntime-linux-x64-1.17.3
