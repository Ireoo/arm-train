#!/bin/bash

# ARM训练循环启动脚本 (Bash版本)
# 使用方法: ./start_training_loop.sh

# 设置 conda 环境
export PATH="/miniconda3/envs/isaaclab/bin:$PATH"
export CONDA_DEFAULT_ENV=isaaclab

# 设置 Isaac Lab 环境变量
export ISAACLAB_PATH="/IsaacLab-2.1.0"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}ARM训练循环启动脚本 (Bash版本)${NC}"
echo -e "${GREEN}================================${NC}"

# 设置日志目录路径
LOG_DIR="/IsaacLab-2.1.0/logs/skrl/arm"

mkdir -p $LOG_DIR

# 检查日志目录是否存在
# if [ ! -d "$LOG_DIR" ]; then
#     echo -e "${RED}错误: 日志目录 $LOG_DIR 不存在${NC}"
#     read -p "按任意键退出..."
#     exit 1
# fi

# 循环计数器
LOOP_COUNT=1

while true; do
    echo -e "${YELLOW}正在查找最新的训练checkpoint...${NC}"

    # 获取最新的训练目录（按修改时间排序）
    LATEST_DIR=$(ls -td "$LOG_DIR"/*ppo_torch 2>/dev/null | head -n 1)

    if [ -z "$LATEST_DIR" ]; then
        # 如果没找到ppo_torch目录，尝试找所有子目录
        LATEST_DIR=$(ls -td "$LOG_DIR"/*/ 2>/dev/null | head -n 1)
    fi

    if [ -z "$LATEST_DIR" ]; then
        echo -e "${YELLOW}警告: 未找到任何训练目录${NC}"
        echo -e "${GREEN}将开始全新的训练${NC}"
        USE_CHECKPOINT=0
        CHECKPOINT_PATH=""
    else

        # 移除末尾的斜杠
        LATEST_DIR=${LATEST_DIR%/}

        echo -e "${GREEN}找到最新训练目录: $(basename "$LATEST_DIR")${NC}"
        echo -e "${GRAY}完整路径: $LATEST_DIR${NC}"

        # 查找最新的checkpoint文件
        CHECKPOINT_DIR="$LATEST_DIR/checkpoints"
        CHECKPOINT_PATH=""
        USE_CHECKPOINT=0
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo -e "${GRAY}检查checkpoints目录: $CHECKPOINT_DIR${NC}"
            
            # 列出所有checkpoint文件用于调试
            CHECKPOINT_FILES=$(find "$CHECKPOINT_DIR" -name "agent_*.pt" -type f 2>/dev/null)
            if [ -n "$CHECKPOINT_FILES" ]; then
                echo -e "${GRAY}找到的checkpoint文件:${NC}"
                echo "$CHECKPOINT_FILES" | while read file; do
                    echo -e "${GRAY}  - $(basename "$file")${NC}"
                done
            fi
            
            # 查找所有agent_*.pt文件，按数字排序，取最大的
            LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "agent_*.pt" -type f | \
                sed 's/.*agent_\([0-9]*\)\.pt/\1 &/' | \
                sort -n | \
                tail -n 1 | \
                cut -d' ' -f2-)
            
            if [ -n "$LATEST_CHECKPOINT" ]; then
                CHECKPOINT_PATH="$LATEST_CHECKPOINT"
                USE_CHECKPOINT=1
                echo -e "${GREEN}使用最新checkpoint: $(basename "$CHECKPOINT_PATH")${NC}"
            else
                echo -e "${YELLOW}警告: 未找到checkpoint文件，将开始新的训练${NC}"
            fi
        else
            echo -e "${YELLOW}警告: checkpoints目录不存在，将开始新的训练${NC}"
        fi
    fi
    echo ""



    # 训练循环
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}开始第 $LOOP_COUNT 次训练循环${NC}"
    echo -e "${CYAN}时间: $(date)${NC}"
    echo -e "${CYAN}================================${NC}"

    # 构建训练命令
    if [ $USE_CHECKPOINT -eq 1 ]; then
        TRAINING_COMMAND="PATH="/miniconda3/envs/isaaclab/bin:$PATH" /IsaacLab-2.1.0/isaaclab.sh -p /workspace/scripts/skrl/train.py --task Template-Arm-v0 --checkpoint \"$CHECKPOINT_PATH\" --headless"
    else
        TRAINING_COMMAND="PATH="/miniconda3/envs/isaaclab/bin:$PATH" /IsaacLab-2.1.0/isaaclab.sh -p /workspace/scripts/skrl/train.py --task Template-Arm-v0 --headless"
    fi
    
    echo -e "${WHITE}执行命令: $TRAINING_COMMAND${NC}"
    
    # 执行训练命令
    eval $TRAINING_COMMAND
    # EXIT_CODE=$?
    
    # if [ $EXIT_CODE -ne 0 ]; then
    #     echo ""
    #     echo -e "${YELLOW}警告: 训练命令执行失败 (错误代码: $EXIT_CODE)${NC}"
    #     echo -n "是否继续下一次循环? (Y/N): "
    #     read CONTINUE
    #     case $CONTINUE in
    #         [Nn]* ) 
    #             echo -e "${RED}退出训练循环${NC}"
    #             exit $EXIT_CODE
    #             ;;
    #     esac
    # fi

    echo ""
    echo -e "${GREEN}第 $LOOP_COUNT 次训练循环完成${NC}"
    
    # 检查是否生成了新的checkpoint
    echo -e "${YELLOW}检查新生成的checkpoint...${NC}"
    CURRENT_LATEST_DIR=$(ls -td "$LOG_DIR"/*ppo_torch 2>/dev/null | head -n 1)
    if [ -n "$CURRENT_LATEST_DIR" ]; then
        CURRENT_CHECKPOINT_DIR="$CURRENT_LATEST_DIR/checkpoints"
        if [ -d "$CURRENT_CHECKPOINT_DIR" ]; then
            NEWEST_CHECKPOINT=$(find "$CURRENT_CHECKPOINT_DIR" -name "agent_*.pt" -type f | \
                sed 's/.*agent_\([0-9]*\)\.pt/\1 &/' | \
                sort -n | \
                tail -n 1 | \
                cut -d' ' -f2-)
            if [ -n "$NEWEST_CHECKPOINT" ]; then
                echo -e "${GREEN}最新checkpoint: $(basename "$NEWEST_CHECKPOINT")${NC}"
            fi
        fi
    fi
    echo ""

    # 增加循环计数器
    ((LOOP_COUNT++))

    # 短暂暂停，准备下一轮训练
    echo -e "${GRAY}开始下一轮训练...${NC}"
    sleep 3
done

echo -e "${GREEN}脚本执行完成${NC}"
read -p "按任意键退出..." 