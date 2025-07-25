#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

# 定义安装路径
SOURCE_DIR="$(pwd)"
INSTALL_DIR="$(pwd)/3rdpartyU18/ffmpeg"
FFMPEG_VERSION="6.1"

# 打印开始信息
echo "开始编译并安装 FFmpeg ${FFMPEG_VERSION} 到 ${INSTALL_DIR}"

# 1. 安装依赖
echo "安装必要的依赖..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git autoconf automake build-essential cmake libass-dev libfreetype6-dev \
    libsdl2-dev libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
    libxcb-xfixes0-dev pkg-config texinfo wget yasm zlib1g-dev nasm -y

# 2. 下载 FFmpeg 源码
echo "下载 FFmpeg ${FFMPEG_VERSION} 源码..."
mkdir -p "${SOURCE_DIR}"
cd "${SOURCE_DIR}"
if [ ! -d "ffmpeg-${FFMPEG_VERSION}" ]; then
    wget https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz -O ffmpeg-${FFMPEG_VERSION}.tar.gz
    tar -xzf ffmpeg-${FFMPEG_VERSION}.tar.gz
fi

# 3. 配置编译环境
echo "配置 FFmpeg..."
cd ffmpeg-${FFMPEG_VERSION}
./configure \
    --prefix="${INSTALL_DIR}" \
    --enable-shared \
    --disable-static \
    --enable-gpl \
    --enable-libass \
    --enable-libfreetype \
    --enable-libvorbis \
    --enable-nonfree \
    --disable-debug \
    --disable-doc \
    --disable-programs

# 4. 编译源码
echo "编译 FFmpeg..."
make -j$(nproc)

# 5. 安装 FFmpeg
echo "安装 FFmpeg 到 ${INSTALL_DIR}..."
make install

# 6. 清除
echo "清除..."
cd "${SOURCE_DIR}"
rm -rf ffmpeg-${FFMPEG_VERSION}
rm -rf ffmpeg-${FFMPEG_VERSION}.tar.gz
echo "FFmpeg ${FFMPEG_VERSION} 已成功安装到 ${INSTALL_DIR}"
