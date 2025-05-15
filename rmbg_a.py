import rembg
import kiui
import os
import argparse

def main(source_folder, target_folder):

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    bg_remover = rembg.new_session()

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 构造文件的完整路径
        file_path = os.path.join(source_folder, filename)
        
        # 检查是否为文件且以常见图像格式结尾
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

            save_path = os.path.join(target_folder, filename)

            input_image = kiui.read_image(file_path, mode='uint8')
            # bg removal
            carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
            mask = carved_image[..., -1] > 0

            kiui.write_image(save_path.replace('.jpg','.png'), carved_image)

print("所有图像已转换为 RGBA 格式并保存到目标文件夹。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a square images given RGBA images')

    parser.add_argument("-i", "--input-path", default='./real_img', type=str, help="Input RGBA path")
    parser.add_argument("-o", "--output-path", default='./real_img_rmbg', type=str, help="Output path")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
