import rembg
import kiui
import os
import argparse

def main(source_folder, target_folder):


    os.makedirs(target_folder, exist_ok=True)

    bg_remover = rembg.new_session()


    for filename in os.listdir(source_folder):

        file_path = os.path.join(source_folder, filename)
        

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

            save_path = os.path.join(target_folder, filename)

            input_image = kiui.read_image(file_path, mode='uint8')
            # bg removal
            carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
            mask = carved_image[..., -1] > 0

            kiui.write_image(save_path.replace('.jpg','.png'), carved_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a square images given RGBA images')

    parser.add_argument("-i", "--input-path", default='./real_img', type=str, help="Input RGBA path")
    parser.add_argument("-o", "--output-path", default='./real_img_rmbg', type=str, help="Output path")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
