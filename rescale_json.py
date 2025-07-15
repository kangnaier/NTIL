import argparse
import os
import json
from time import time
from skimage.transform import rescale
import numpy as np
from PIL import Image
import gc  # 用于垃圾回收

Image.MAX_IMAGE_PIXELS = None

def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # 如果图像有 alpha 通道，去掉它
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def save_image(img, filename):
    mkdir(os.path.dirname(filename))
    Image.fromarray(img).save(filename, format='PNG')  # 保存为 PNG 格式
    print(f'Saved image to {filename}')

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]  # 如果是灰度图，两个方向的比例一致
    elif img.ndim == 3:
        scale = [scale[0], scale[1], 1]  # 对于 RGB 图像，不对颜色通道进行缩放
    else:
        raise ValueError('Unrecognized image ndim')
    
    img = rescale(img, scale, preserve_range=True)
    return img

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="Directory containing images")
    parser.add_argument('config_file', type=str, help="JSON config file with image dimensions")
    parser.add_argument('--image', action='store_true', help="Process images in the directory")
    args = parser.parse_args()
    return args

def get_image_files(directory):
    # 返回符合条件的所有图像文件名列表
    image_files = []
    for suffix in ['.jpg', '.png', '.tif']:  # 支持 .jpg, .png, .tif 格式
        image_files.extend([f for f in os.listdir(directory) if f.endswith(suffix)])
    if not image_files:
        raise FileNotFoundError('No images found in the specified directory.')
    return image_files

def main():
    args = get_args()

    # 加载配置文件，获取每张图像的尺寸信息
    config = load_config(args.config_file)

    if args.image:
        # 获取符合条件的所有图像文件名列表
        image_files = get_image_files(args.directory)

        for filename in image_files:
            # 提取文件名去掉扩展名
            base_filename = os.path.splitext(filename)[0]

            # 如果该图像在配置文件中没有对应的配置，则跳过
            if base_filename not in config:
                print(f"Skipping {filename}: No configuration found in the JSON file.")
                continue

            # 获取该图像的具体尺寸配置
            img_config = config[base_filename]
            width_raw = img_config['width_raw']
            height_raw = img_config['height_raw']
            width_target = img_config['width_target']
            height_target = img_config['height_target']

            # 计算缩放因子
            scale_x = width_target / width_raw
            scale_y = height_target / height_raw

            # 加载并处理图像
            img = load_image(os.path.join(args.directory, filename))
            img = img.astype(np.float32)
            print(f'Rescaling image {filename} (scale: {scale_x:.3f}, {scale_y:.3f})...')
            t0 = time()
            img = rescale_image(img, (scale_x, scale_y))
            print(f'Rescaling done in {int(time() - t0)} seconds')
            img = img.astype(np.uint8)

            # 保存处理后的图像为 PNG 格式
            output_filename = f"{base_filename}-scaled.png"  # 保存为 PNG 格式
            save_image(img, os.path.join(args.directory, output_filename))

            # 显式删除图像并释放内存
            del img
            gc.collect()  # 调用垃圾回收以释放内存

if __name__ == '__main__':
    main()
