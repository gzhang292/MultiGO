from rembg import remove
from PIL import Image
import os
from pathlib import Path
import numpy as np
import argparse

def crop_person(input_path, output_path):
    # Read the image
    input_img = Image.open(input_path)
    
    # Use rembg to remove the background, keeping only the person
    output = remove(input_img)
    
    # Convert the output to a numpy array for processing
    output_array = np.array(output)
    
    # Get the alpha channel
    alpha = output_array[:, :, 3]
    
    # Find the bounding box of non-transparent pixels
    coords = np.argwhere(alpha > 0)
    if len(coords) == 0:
        print(f"Warning: No person detected in image {input_path}")
        return
        
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # Crop the image
    cropped = output.crop((x0, y0, x1, y1))
    
    # Save the cropped image
    cropped.save(output_path, 'PNG')

def main(args):
    # Create the output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process all images
    for img_path in Path(args.input_path).glob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            output_path = Path(args.output_path) / f"cropped_{img_path.name}"
            print(f"Processing image: {img_path}")
            try:
                crop_person(img_path, output_path)
                print(f"Saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop the person out of images')
    
    parser.add_argument("-i", "--input-path", 
                       default='./real_img_rmbg', 
                       type=str, 
                       help="Input image directory path")
    
    parser.add_argument("-o", "--output-path", 
                       default='./real_img_rmbg_cropped_persons', 
                       type=str, 
                       help="Output image directory path")
    
    main(parser.parse_args())