#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

wget https://github.com/Kitware/CMake/releases/download/v3.17.5/cmake-3.17.5-Linux-x86_64.tar.gz

tar zxvf cmake-3.17.5-Linux-x86_64.tar.gz

sudo mv cmake-3.17.5-Linux-x86_64 /opt/cmake-3.17.5

sudo ln -sf /opt/cmake-3.17.5/bin/* /usr/bin/
cmake --version

rm -rf cmake-3.17.5-Linux-x86_64.tar.gz



