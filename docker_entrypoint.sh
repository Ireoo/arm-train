#!/bin/bash
echo "正在启动 ARM 训练环境..."
echo "工作目录: $(pwd)"

# 激活 conda 环境
export PATH="/miniconda3/envs/isaaclab/bin:$PATH"
export CONDA_DEFAULT_ENV=isaaclab

cd /IsaacLab-2.1.0 && ./isaaclab.sh -p -m pip install -e /workspace/source/arm

echo "Python版本: $(./isaaclab.sh -p --version)"
echo "PyTorch版本: $(./isaaclab.sh -p -c \"import torch; print(torch.__version__)\")"
echo "CUDA可用性: $(./isaaclab.sh -p -c \"import torch; print(torch.cuda.is_available())\")"

# 设置显示环境变量（如果在无头模式下运行）
export DISPLAY=${DISPLAY:-:99}

# 检查 Isaac Lab 环境
if command -v isaaclab.sh &> /dev/null; then
    echo "Isaac Lab 命令可用"
else
    echo "警告: isaaclab.sh 命令不可用，请确保 Isaac Lab 已正确安装"
fi

# 运行训练循环脚本
bash /workspace/start_training_loop.sh