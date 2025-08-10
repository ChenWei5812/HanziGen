#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# =============== 基础参数（按需修改） ===============
TARGET_FONT_PATH="fonts/myfont.ttf"   # 目标字体路径（用于数据集构建/分析）
TRAIN_SPLIT_RATIO=0.8                  # 训练集占比（与验证占比相加应为 1.0）
VAL_SPLIT_RATIO=0.2                    # 验证集占比
RANDOM_SEED=9999                       # 随机种子（保证可复现）
BATCH_SIZE=16                          # 每批训练的样本数量（根据显存调整）
LEARNING_RATE=5e-4                     # 初始学习率，实际学习率会根据余弦退火策略动态调整
NUM_EPOCHS=250                         # 训练轮数
SAMPLE_STEPS=50                        # 样例图生成时的采样步数（用于可视化/评估）
IMG_SAVE_INTERVAL=10                   # 可视化图片保存间隔（单位：epoch）
LPIPS_EVAL_INTERVAL=10                 # LPIPS 评估间隔（单位：epoch）
EVAL_BATCH_SIZE=2                      # 评估批大小
DEVICE="cuda"                          # 训练设备：cuda / cpu / mps
                                       #（cuda就是使用Nvidia GPU   mps就是使用Apple Silicon GPU）

# =============== 可选：断点续训 / 混合精度 ===============
RESUME_FROM=""                          # 若需断点续训，填入 LDM .pth 权重路径；否则留空
ENABLE_MIXED_PRECISION=1               # 1 表示启用混合精度（需 CUDA），0 表示关闭
                                       #混合精度可以提升训练速度

# =============== 以下参数无需修改 ===============
# 由字体文件名派生输出文件名/目录
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

# 预训练 VQ-VAE 路径（用于编码/解码；与上次 LDM 训练保持一致）
PRETRAINED_VQVAE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

# 组装参数
ARGS=()
if [ -n "$RESUME_FROM" ]; then
  ARGS+=(--resume_from "$RESUME_FROM")
fi
if [ "$ENABLE_MIXED_PRECISION" -eq 1 ]; then
  ARGS+=(--mixed_precision)
fi

# =============== 启动训练 ===============
python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    "${ARGS[@]}"
