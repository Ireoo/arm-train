#!/bin/bash

# 检查 buildx 是否已安装
if ! docker buildx version &>/dev/null; then
  # 启用 Docker CLI 实验功能
  mkdir -p ~/.docker/cli-plugins
  curl -SL https://github.com/docker/buildx/releases/download/v0.9.1/buildx-v0.9.1.linux-amd64 -o ~/.docker/cli-plugins/docker-buildx
  chmod +x ~/.docker/cli-plugins/docker-buildx

  cat <<EOF > ~/.docker/config.json
{
  "experimental": "enabled"
}
EOF

  echo "buildx 已成功安装并配置"
else
  echo "buildx 已存在，跳过安装步骤"
fi

# 安装 binfmt 支持以便处理多平台
docker run --privileged --rm tonistiigi/binfmt --install all

# 创建并使用 Buildx 构建器
docker buildx create --name mybuilder --use

# 初始化构建器
docker buildx inspect mybuilder --bootstrap

# 使用 Buildx 构建并推送多平台镜像
docker buildx build --platform linux/arm64,linux/amd64 -t integem/arm-train:latest --push . || { echo "Failed to build $1"; exit 1; }

echo "Docker镜像构建并推送完成。"

# docker run --privileged --pull always --rm -it -p 8888:8888 integem/notebook:maix_train_mx_v5.2 bash -c "python yolov5/train.py --img 224 --epoch 30 --data duck1k_dataset.yaml --weights yolov5s.pt --workers 0 && python yolov5/export.py --weight yolov5/runs/train/exp/weights/best.pt --include onnx --img 224 320 && cp -rf yolov5/runs/train/exp/weights/best.onnx ./best.onnx && chmod +x ./light/convert_yolov5_to_cvimodel.sh && bash ./light/convert_yolov5_to_cvimodel.sh best "/workspace/datasets/duck1k_yolo/images/val" "/workspace/datasets/duck1k_yolo/images/train/11770_116.jpg" && cp -rf workspace/best_int8.cvimodel ./best_int8.cvimodel && exit"

# echo "Docker测试完成。"
