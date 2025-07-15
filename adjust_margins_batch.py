import os
import numpy as np
from PIL import Image
import gc
import time

Image.MAX_IMAGE_PIXELS = None

def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    
    return img

def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement

    mode = 'edge' if pad_value is None else 'constant'
    img = crop_image(img, extent, mode=mode, constant_values=pad_value)
    return img

def process_all_images(input_dir, output_dir, pad_size=256, pad_value=255):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]

    for i, fname in enumerate(image_files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '_adjusted.png')

        print(f'[{i+1}/{len(image_files)}] Processing {fname}...')
        try:
            img = Image.open(input_path)
            img = np.array(img)

            adjusted_img = adjust_margins(img, pad_size, pad_value=pad_value)

            Image.fromarray(adjusted_img).save(output_path)
            print(f'Saved to {output_path}')

            # 清理内存并休眠
            del img
            del adjusted_img
            gc.collect()
            time.sleep(3)  # 休眠 1 秒

        except Exception as e:
            print(f'Error processing {fname}: {e}')

if __name__ == '__main__':
    input_dir = '/nas/qinzhiqing/data'
    output_dir = '/nas/qinzhiqing/pad'

    process_all_images(input_dir, output_dir, pad_size=256, pad_value=255)
