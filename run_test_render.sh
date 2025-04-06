#!/bin/bash

# 使用数组存储不同的噪声级别
# NOISE_LEVELS=(0 0.0012 0.00152 0.0022 0.0032 0.0042)
NOISE_LEVELS=(0.0013 0.00153 0.0023 0.0033 0.0043 0.0053)

# 使用数组存储不同的ID名称
ID_NAMES=("RD_Radio34_005" "WDA_JoeManchin_000")

# 设置GPU
GPU_ID=0

# 函数：在每次测试之间清理显存
cleanup_gpu() {
    echo "清理GPU内存..."
    # 等待所有进程完成
    sleep 5
    # 可选：使用nvidia-smi重置GPU (如果有root权限)
    # sudo nvidia-smi --gpu-reset -i $GPU_ID
}

# 主循环
for id_name in "${ID_NAMES[@]}"; do
    echo "===== 开始处理 $id_name ====="
    
    for noise_level in "${NOISE_LEVELS[@]}"; do
        echo "执行噪声级别 $noise_level 的测试..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python test_with_random_noise.py \
            --idname $id_name \
            --driven_idname $id_name \
            --noise_level $noise_level
        
        # 测试后清理
        cleanup_gpu
    done
    
    echo "===== $id_name 处理完成 ====="
done

echo "所有测试完成。"