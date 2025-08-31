#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

# 1. 下载cuda,并安装
echo "安装cuda..."
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

'''
[] Driver
  [] 465.19.01
'''

# 2. 下载cudnn,并安装
echo "安装cudnn..."
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
tar -xvJf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.1/include/
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.1/lib64/
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-12.1/lib64/libcudnn*

# 3. 删除文件
echo "清除..."
rm -rf cudnn-linux-x86_64-8.9.7.29_cuda12-archive cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz cuda_12.1.0_530.30.02_linux.run

# 4. 配置编译环境与验证
echo "配置环境并验证..."
echo "#for cuda12.1" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda-12.1/bin" >> ~/.bashrc
echo "export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-12.1" >> ~/.bashrc
source ~/.bashrc
nvcc -V
