

import os
import sys
import random
import argparse
from PIL import Image

def create_collage(folder_path, num_rows, num_cols):
    """
    从指定文件夹中随机抽取图片，将每张图片缩放，然后拼接成一张大图。

    :param folder_path: 包含图片的文件夹路径
    :param num_rows: 拼接后图片的行数
    :param num_cols: 拼接后图片的列数
    """
    # 1. 验证输入路径
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        sys.exit(1)

    # 2. 查找所有有效的图片文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"错误：在文件夹 '{folder_path}' 中未找到任何有效的图片。")
        sys.exit(1)

    # 3. 随机抽选图片
    total_needed = num_rows * num_cols
    print(f"找到 {len(image_files)} 张图片，需要 {total_needed} 张。")
    if len(image_files) < total_needed:
        print(f"警告：图片数量不足，将使用所有 {len(image_files)} 张图片进行拼接。")
        # 如果图片不足，重复使用以填满
        selected_image_paths = (image_files * (total_needed // len(image_files) + 1))[:total_needed]
    else:
        random.shuffle(image_files)
        selected_image_paths = image_files[:total_needed]

    if not selected_image_paths:
        print("没有选中任何图片，无法创建拼图。")
        return

    print("正在打开并缩放图片...")
    resized_images = []
    for path in selected_image_paths:
        try:
            with Image.open(path) as img:
                new_width = img.width // 4
                new_height = img.height // 4
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
        except Exception as e:
            print(f"警告：无法处理图片 {path}，已跳过。错误: {e}")

    if not resized_images:
        print("错误：所有选中的图片都无法处理。")
        return

    tile_width = max(img.width for img in resized_images)
    tile_height = max(img.height for img in resized_images)
    print(f"所有图片已缩放为原尺寸的一半。")
    print(f"计算得出每个单元格的尺寸为: {tile_width}x{tile_height} 像素。")

    canvas_width = tile_width * num_cols
    canvas_height = tile_height * num_rows
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    print("正在创建拼图...")

    current_image_index = 0
    for y in range(num_rows):
        for x in range(num_cols):
            if current_image_index < len(resized_images):
                img_to_paste = resized_images[current_image_index]
                
                paste_x = x * tile_width
                paste_y = y * tile_height
                

                canvas.paste(img_to_paste, (paste_x, paste_y))
                current_image_index += 1
    

    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_filename = f"{folder_name}_{num_rows}x{num_cols}_resized_collage.png"
    
    canvas.save(output_filename)
    print(f"拼图成功！已保存为: {output_filename}")

    for img in resized_images:
        img.close()

def main():
    parser = argparse.ArgumentParser(
        description="一个拼图程序",
        epilog="示例: python 拼图.py ./my_images 3 4"
    )
    
    parser.add_argument(
        "folder_path", 
        type=str, 
        help="包含源图片的文件夹路径。"
    )
    parser.add_argument(
        "rows", 
        type=int, 
        help="输出图片的行数。"
    )
    parser.add_argument(
        "columns", 
        type=int, 
        help="输出图片的列数。"
    )

    args = parser.parse_args()

    if args.rows <= 0 or args.columns <= 0:
        print("错误：行数和列数必须是正整数。")
        sys.exit(1)

    create_collage(args.folder_path, args.rows, args.columns)

if __name__ == "__main__":
    main()