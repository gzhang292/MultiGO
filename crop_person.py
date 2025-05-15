from rembg import remove
from PIL import Image
import os
from pathlib import Path
import numpy as np
import argparse

def crop_person(input_path, output_path):
    # 读取图片
    input_img = Image.open(input_path)
    
    # 使用rembg移除背景，只保留人物
    output = remove(input_img)
    
    # 将输出转换为numpy数组以便处理
    output_array = np.array(output)
    
    # 获取alpha通道
    alpha = output_array[:, :, 3]
    
    # 找到非透明像素的边界框
    coords = np.argwhere(alpha > 0)
    if len(coords) == 0:
        print(f"警告: 在图片 {input_path} 中没有检测到人物")
        return
        
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # 裁剪图片
    cropped = output.crop((x0, y0, x1, y1))
    
    # 保存裁剪后的图片
    cropped.save(output_path, 'PNG')

def main(args):
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 处理所有图片
    for img_path in Path(args.input_path).glob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            output_path = Path(args.output_path) / f"cropped_{img_path.name}"
            print(f"处理图片: {img_path}")
            try:
                crop_person(img_path, output_path)
                print(f"已保存到: {output_path}")
            except Exception as e:
                print(f"处理 {img_path} 时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将图片中的人物裁剪出来')
    
    parser.add_argument("-i", "--input-path", 
                       default='./real_img_rmbg', 
                       type=str, 
                       help="输入图片目录路径")
    
    parser.add_argument("-o", "--output-path", 
                       default='./real_img_rmbg_cropped_persons', 
                       type=str, 
                       help="输出图片目录路径")
    
    main(parser.parse_args()) 