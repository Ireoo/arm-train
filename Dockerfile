# 基于 NVIDIA CUDA 镜像，支持 GPU 计算
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONPATH=/workspace:/IsaacLab-2.1.0
ENV ISAAC_SIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab-2.1.0
ENV TERM=xterm
ENV FORCE_COLOR=1
ENV NO_INTERACTION=1

# 设置 OpenGL/EGL 环境变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV __GL_SYNC_TO_VBLANK=0

# 设置编译环境变量帮助构建 egl_probe
ENV CPATH="/usr/include/GL:/usr/include/EGL:$CPATH"
ENV LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

WORKDIR /

# 安装系统依赖，包括 OpenGL/EGL 开发库
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libegl1-mesa \
    libegl1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libasound2 \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 根据系统安装miniconda
# 自动识别平台并安装对应的 miniconda
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh; \
    else \
        echo "不支持的架构: $(uname -m)" && exit 1; \
    fi
RUN bash miniconda.sh -b -p /miniconda3
RUN rm miniconda.sh

# 创建 conda 环境并设置环境变量
RUN /miniconda3/bin/conda create -n isaaclab python=3.10 -y

# 设置 conda 环境变量
ENV PATH="/miniconda3/envs/isaaclab/bin:$PATH"
ENV CONDA_DEFAULT_ENV=isaaclab

# 下载isaaclab 2.1.0
RUN wget https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.1.0.tar.gz
RUN ls -alF
RUN tar -xzf v2.1.0.tar.gz
RUN rm v2.1.0.tar.gz

# 手动安装 Isaac Lab 依赖，而不是使用自动安装脚本
# 因为 Docker 环境中 Isaac Sim 路径可能不同
WORKDIR /IsaacLab-2.1.0

RUN ln -s /isaac-sim _isaac_sim

# 尝试安装 Isaac Lab，如果失败则使用备选方案
# 在 conda 环境中运行 Isaac Lab 安装脚本
RUN PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -i

# RUN PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -p -m pip install --upgrade pip

# 安装 isaacsim 到 conda 环境
# RUN PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -p -m pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com


# 安装 PyTorch (支持 CUDA) 到 conda 环境
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -p -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -p -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        echo "不支持的架构: $(uname -m)" && exit 1; \
    fi


# 创建工作目录
WORKDIR /workspace

# 复制项目文件
COPY . /workspace/

# 使用 Isaac Lab 的方式安装项目依赖
RUN cd /IsaacLab-2.1.0 && PATH="/miniconda3/envs/isaaclab/bin:$PATH" ./isaaclab.sh -p -m pip install -e /workspace/source/arm

# 创建必要的目录
RUN mkdir -p /workspace/logs/skrl/arm

RUN cp -rf /workspace/source/arm_train.usd /IsaacLab-2.1.0/source/arm_train.usd

# 给脚本添加执行权限并转换行结束符格式
RUN chmod +x /workspace/start_training_loop.sh

RUN chmod +x /workspace/*.sh

# 设置入口点
ENTRYPOINT ["/workspace/docker_entrypoint.sh"]

# 暴露常用端口（如果需要访问 TensorBoard 等）
EXPOSE 6006 8080

# 设置运行时用户（可选，为了安全考虑）
# RUN useradd -m -s /bin/bash isaac && chown -R isaac:isaac /workspace
# USER isaac 