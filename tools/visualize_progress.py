import os
import random
import argparse
import re
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

def create_diffusion_plot(val_dir, output_path):
    """
    用法： -装val图的文件夹路径
    示例: python visualize_progress.py /workspace/HanziGen/samples_aokane/val
    Args:
        val_dir (str): 包含验证图像的目录路径。
        output_path (str): 最终拼接图的保存路径。
    """
    if not os.path.isdir(val_dir):
        print(f"错误: 找不到目录 {val_dir}")
        return

    # --- 1. 扫描并分组文件 ---
    char_epochs = defaultdict(list)
    pattern = re.compile(r"epoch_(\d+)_([a-zA-Z0-9]+)\.png")

    print(f"正在扫描目录: {val_dir}...")
    for filename in os.listdir(val_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            char_id = match.group(2)
            char_epochs[char_id].append(epoch)

    if not char_epochs:
        print("错误: 未找到有效的图像文件。请检查文件名和目录。")
        return

    # --- 2. 选择字符并找到所有相关的 epoch ---
    all_char_ids = list(char_epochs.keys())
    num_to_select = min(10, len(all_char_ids))
    if num_to_select == 0:
        print("没有找到可处理的字符。")
        return
        
    selected_char_ids = random.sample(all_char_ids, num_to_select)
    print(f"{selected_char_ids}")

    all_epochs = set()
    for char_id in selected_char_ids:
        all_epochs.update(char_epochs[char_id])
    
    sorted_epochs = sorted(list(all_epochs))
    #print(f"找到 {len(sorted_epochs)} 个独立的 epoch 步骤: {sorted_epochs}")

    CROP_SIZE = 512
    DISPLAY_SIZE = 128

    TITLE_FONTSIZE = 24      # 主标题字体大小
    EPOCH_FONTSIZE = 20      # Epoch 编号 (列标题) 字体大小
    CHAR_FONTSIZE = 20       # 字符 ID (行标签) 字体大小
    
    EPOCH_TITLE_PAD = 20     # Epoch 编号与图像的间距
    CHAR_LABEL_PAD = 35      # 字符 ID 与图像的间距

    FIG_WIDTH_MULTIPLIER = 1.2 
    FIG_HEIGHT_MULTIPLIER = 1.5
    
    fig, axes = plt.subplots(
        num_to_select, 
        len(sorted_epochs), 
        figsize=(len(sorted_epochs) * FIG_WIDTH_MULTIPLIER, num_to_select * FIG_HEIGHT_MULTIPLIER),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
    )
    
    if num_to_select == 1:
        axes = axes.reshape(1, -1)
    if not isinstance(axes, (list, tuple, plt.np.ndarray)):
        axes = plt.np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Process by Epoch", fontsize=TITLE_FONTSIZE, y=0.98)

    #print("正在生成拼接图...")
    for i, char_id in enumerate(selected_char_ids):
        for j, epoch in enumerate(sorted_epochs):
            ax = axes[i, j]
            img_path = os.path.join(val_dir, f"epoch_{epoch}_{char_id}.png")
            if not os.path.exists(img_path):
                 img_path = os.path.join(val_dir, f"epoch_{epoch:04d}_{char_id}.png")

            if i == 0:
                ax.set_title(f"E{epoch}", fontsize=EPOCH_FONTSIZE, pad=EPOCH_TITLE_PAD)

            if j == 0:
                ax.set_ylabel(char_id, fontsize=CHAR_FONTSIZE, labelpad=CHAR_LABEL_PAD, rotation=0, ha='right')

            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width >= CROP_SIZE and height >= CROP_SIZE:
                            left = width - CROP_SIZE
                            top = (height - CROP_SIZE) // 2
                            right = width
                            bottom = top + CROP_SIZE
                            cropped_img = img.crop((left, top, right, bottom))
                        else:
                            cropped_img = img
                            print(f"警告: 图像 {img_path} 小于 {CROP_SIZE}x{CROP_SIZE}，将直接缩放原图。")
                        
                        resized_img = cropped_img.resize(
                            (DISPLAY_SIZE, DISPLAY_SIZE), 
                            Image.Resampling.LANCZOS
                        )
                        
                        ax.imshow(resized_img, cmap='gray')

                except IOError:
                    print(f"警告: 无法打开图像 {img_path}。跳过。")
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"\n成功创建拼接图并保存至: {output_path}")

    except Exception as e:
        print(f"错误: 无法保存最终图像。{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="创建一个 Matplotlib 拼接图",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "val_dir",
        type=str,
        help="验证样本目录的路径 (例如, HanziGen/samples_negi/val)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./diffusion_process_plot.png",
        help="最终输出 PNG 图像的保存路径。"
    )

    args = parser.parse_args()
    create_diffusion_plot(args.val_dir, args.output)