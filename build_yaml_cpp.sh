#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

# 定义安装路径
SOURCE_DIR="$(pwd)"
INSTALL_DIR="$(pwd)/3rdpartyU18/yaml-cpp"
YAML_CPP_VERSION="0.8.0"

# 打印开始信息
echo "开始编译并安装 yaml-cpp ${YAML_CPP_VERSION} 到 ${INSTALL_DIR}"

# 1. 安装依赖
echo "安装必要的依赖..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git cmake build-essential -y

# 2. 下载 yaml-cpp 源码
echo "下载 yaml-cpp ${YAML_CPP_VERSION} 源码..."
mkdir -p "${SOURCE_DIR}"
cd "${SOURCE_DIR}"
if [ ! -d "yaml-cpp-${YAML_CPP_VERSION}" ]; then
    wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/${YAML_CPP_VERSION}.tar.gz -O yaml-cpp-${YAML_CPP_VERSION}.tar.gz
    tar -xzf yaml-cpp-${YAML_CPP_VERSION}.tar.gz
fi

# 3. 配置编译环境
echo "配置 CMake..."
cd yaml-cpp-${YAML_CPP_VERSION}
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF \
    -DYAML_CPP_BUILD_CONTRIB=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

# 4. 编译源码
echo "编译 yaml-cpp..."
make -j$(nproc)

# 5. 安装 yaml-cpp
echo "安装 yaml-cpp 到 ${INSTALL_DIR}..."
make install

# 6. 验证安装
echo "验证安装..."
if [ -f "${INSTALL_DIR}/lib/libyaml-cpp.so" ] && [ -f "${INSTALL_DIR}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
    echo "安装成功！yaml-cpp-config.cmake 位于 ${INSTALL_DIR}/lib/cmake/yaml-cpp"
else
    echo "安装失败，请检查日志！"
    exit 1
fi

# 7. 清除
echo "清除..."
cd "${SOURCE_DIR}"
rm -rf yaml-cpp-${YAML_CPP_VERSION}
rm -rf yaml-cpp-${YAML_CPP_VERSION}.tar.gz
echo "yaml-cpp ${YAML_CPP_VERSION} 已成功安装到 ${INSTALL_DIR}"
