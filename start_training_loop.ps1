# ARM训练循环启动脚本 (PowerShell版本)
# 使用方法: 在PowerShell中运行 .\start_training_loop.ps1

Write-Host "================================" -ForegroundColor Green
Write-Host "ARM训练循环启动脚本 (PowerShell版本)" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# 设置日志目录路径
$LogDir = "logs\skrl\arm"

# 检查日志目录是否存在
if (-not (Test-Path $LogDir)) {
    Write-Host "错误: 日志目录 $LogDir 不存在" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Write-Host "正在查找最新的训练checkpoint..." -ForegroundColor Yellow

# 获取所有训练目录并按创建时间排序
$TrainingDirs = Get-ChildItem -Path $LogDir -Directory | Sort-Object CreationTime -Descending

if ($TrainingDirs.Count -eq 0) {
    Write-Host "警告: 未找到任何训练目录" -ForegroundColor Yellow
    Write-Host "将开始全新的训练" -ForegroundColor Green
    $UseCheckpoint = $false
    $CheckpointPath = ""
} else {

    # 获取最新的训练目录
    $LatestDir = $TrainingDirs[0].FullName
    Write-Host "找到最新训练目录: $($TrainingDirs[0].Name)" -ForegroundColor Green

    # 查找最新的checkpoint文件
    $CheckpointDir = Join-Path $LatestDir "checkpoints"
    $CheckpointPath = ""
    $UseCheckpoint = $false
    
    if (Test-Path $CheckpointDir) {
        $CheckpointFiles = Get-ChildItem -Path $CheckpointDir -Filter "agent_*.pt" | 
            ForEach-Object { 
                [PSCustomObject]@{ 
                    File = $_.FullName; 
                    Number = [int]($_.BaseName -replace "agent_", "") 
                } 
            } | 
            Sort-Object Number -Descending
        
        if ($CheckpointFiles.Count -gt 0) {
            $CheckpointPath = $CheckpointFiles[0].File
            $UseCheckpoint = $true
            Write-Host "使用checkpoint: $CheckpointPath" -ForegroundColor Green
        } else {
            Write-Host "警告: 未找到checkpoint文件，将开始新的训练" -ForegroundColor Yellow
        }
    } else {
        Write-Host "警告: checkpoints目录不存在，将开始新的训练" -ForegroundColor Yellow
    }
}
Write-Host ""

# 循环计数器
$LoopCount = 1

# 训练循环
while ($true) {
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "开始第 $LoopCount 次训练循环" -ForegroundColor Cyan
    Write-Host "时间: $(Get-Date)" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan

    # 构建训练命令
    if ($UseCheckpoint) {
        $TrainingCommand = "isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --checkpoint `"$CheckpointPath`" --headless"
    } else {
        $TrainingCommand = "isaaclab.bat -p scripts/skrl/train.py --task Template-Arm-v0 --headless"
    }
    
    try {
        # 执行训练命令
        Write-Host "执行命令: $TrainingCommand" -ForegroundColor White
        Invoke-Expression $TrainingCommand
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "警告: 训练命令执行失败 (错误代码: $LASTEXITCODE)" -ForegroundColor Yellow
            $Continue = Read-Host "是否继续下一次循环? (Y/N)"
            if ($Continue -notin @('Y', 'y', 'yes', 'Yes', 'YES', '')) {
                Write-Host "退出训练循环" -ForegroundColor Red
                break
            }
        }
    }
    catch {
        Write-Host ""
        Write-Host "错误: 训练命令执行异常: $($_.Exception.Message)" -ForegroundColor Red
        $Continue = Read-Host "是否继续下一次循环? (Y/N)"
        if ($Continue -notin @('Y', 'y', 'yes', 'Yes', 'YES', '')) {
            Write-Host "退出训练循环" -ForegroundColor Red
            break
        }
    }

    Write-Host ""
    Write-Host "第 $LoopCount 次训练循环完成" -ForegroundColor Green
    Write-Host ""

    # 增加循环计数器
    $LoopCount++

    # 短暂暂停，准备下一轮训练
    Write-Host "3秒后开始下一轮训练..." -ForegroundColor Gray
    Start-Sleep -Seconds 3
}

Write-Host "脚本执行完成" -ForegroundColor Green
Read-Host "按任意键退出" 