#!/bin/bash

# =============== 基础参数（按需修改） ===============
TARGET_FONT_PATH="fonts/myfont.ttf"   # 目标字体路径（用于数据集构建/分析），格式为.ttf或.otf
TRAIN_SPLIT_RATIO=0.8                  # 训练集占比（与验证占比相加应为 1.0）
VAL_SPLIT_RATIO=0.2                    # 验证集占比
RANDOM_SEED=7777                       # 随机种子（保证可复现）
BATCH_SIZE=8                           # 每批次训练的样本数量（根据显存调整）
LEARNING_RATE=1e-3                     # 初始学习率，实际学习率会根据余弦退火策略动态调整
NUM_EPOCHS=100                         # 训练轮数
DEVICE="cuda"                          # 训练设备：cuda / cpu / mps 
                                       #（cuda就是使用Nvidia GPU   mps就是使用Apple Silicon GPU）

# =============== 可选：断点续训 / 混合精度 ===============
RESUME_FROM=""                          # 若需断点续训，填入 .pth 权重路径；否则留空
ENABLE_MIXED_PRECISION=1               # 1 表示启用混合精度（需 CUDA），0 表示关闭
                                       #混合精度可以提升训练速度


# =============== 以下参数无需修改 ===============
# 由字体文件名派生输出文件名/目录
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')
MODEL_SAVE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"

# 组装可选参数
EXTRA_ARGS=""
if [ -n "$RESUME_FROM" ]; then
  EXTRA_ARGS+=" --resume_from \"$RESUME_FROM\""
fi
if [ "$ENABLE_MIXED_PRECISION" -eq 1 ]; then
  EXTRA_ARGS+=" --mixed_precision"
fi

# =============== 启动训练 ===============
python train_vqvae.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --device "$DEVICE" \
    $EXTRA_ARGS
