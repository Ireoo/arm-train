@echo off
setlocal enabledelayedexpansion

echo ================================
echo ARM训练循环启动脚本
echo ================================

REM 设置日志目录路径
set LOG_DIR=logs\skrl\arm

REM 检查日志目录是否存在
if not exist "%LOG_DIR%" (
    echo 错误: 日志目录 %LOG_DIR% 不存在
    pause
    exit /b 1
)

echo 正在查找最新的训练checkpoint...

REM 获取最新的训练目录
set LATEST_DIR=
set LATEST_TIME=0

for /d %%i in ("%LOG_DIR%\*") do (
    set DIR_NAME=%%~ni
    REM 提取时间戳部分 (假设格式为 YYYY-MM-DD_HH-MM-SS_ppo_torch)
    set TIMESTAMP=!DIR_NAME:~0,19!
    set TIMESTAMP=!TIMESTAMP:-=!
    set TIMESTAMP=!TIMESTAMP:_=!
    
    REM 简单的数字比较来找最新的
    if !TIMESTAMP! gtr !LATEST_TIME! (
        set LATEST_TIME=!TIMESTAMP!
        set LATEST_DIR=%%i
    )
)

if "!LATEST_DIR!"=="" (
    echo 警告: 未找到任何训练目录
    echo 将开始全新的训练
    set USE_CHECKPOINT=0
    set CHECKPOINT_PATH=
    goto TRAINING_LOOP
)

echo 找到最新训练目录: !LATEST_DIR!

REM 查找最新的checkpoint文件
set CHECKPOINT_DIR=!LATEST_DIR!\checkpoints
set CHECKPOINT_PATH=
set USE_CHECKPOINT=0

if exist "!CHECKPOINT_DIR!" (
    REM 遍历所有checkpoint文件，保存最后找到的一个
    for %%f in ("!CHECKPOINT_DIR!\agent_*.pt") do (
        set CHECKPOINT_PATH=%%f
        set USE_CHECKPOINT=1
    )
)

if !USE_CHECKPOINT! equ 1 (
    echo 使用checkpoint: !CHECKPOINT_PATH!
) else (
    echo 警告: 未找到checkpoint文件，将开始新的训练
)
echo.

REM 循环计数器
set LOOP_COUNT=1

:TRAINING_LOOP
echo ================================
echo 开始第 !LOOP_COUNT! 次训练循环
echo 时间: %date% %time%
echo ================================

REM 执行训练命令
if !USE_CHECKPOINT! equ 1 (
    echo 执行命令: isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --checkpoint "!CHECKPOINT_PATH!" --headless
    isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --checkpoint "!CHECKPOINT_PATH!" --headless
) else (
    echo 执行命令: isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --headless
    isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --headless
)

REM 检查命令执行结果
if !ERRORLEVEL! neq 0 (
    echo.
    echo 警告: 训练命令执行失败 (错误代码: !ERRORLEVEL!)
    echo 是否继续下一次循环? (Y/N)
    set /p CONTINUE=
    if /i "!CONTINUE!" neq "Y" (
        echo 退出训练循环
        pause
        exit /b !ERRORLEVEL!
    )
)

echo.
echo 第 !LOOP_COUNT! 次训练循环完成
echo.

REM 增加循环计数器
set /a LOOP_COUNT+=1

REM 短暂暂停，准备下一轮训练
echo 3秒后开始下一轮训练...
timeout /t 3 /nobreak >nul

goto TRAINING_LOOP 