import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

# 禁用图像像素限制
Image.MAX_IMAGE_PIXELS = None

def resize_and_process_chunk(chunk, scale_factor):
    """缩放每个小块并保持图像的质量"""
    scale_factor_width, scale_factor_height = scale_factor
    new_width = int(chunk.width * scale_factor_width)
    new_height = int(chunk.height * scale_factor_height)
    return chunk.resize((new_width, new_height))

def process_chunk(chunk_data):
    """处理图像块（缩放并返回处理后的块）"""
    img, top, left, right, bottom, scale_factor = chunk_data
    chunk = img.crop((left, top, right, bottom))
    resized_chunk = resize_and_process_chunk(chunk, scale_factor)
    return resized_chunk, resized_chunk.width, resized_chunk.height, left, top

def process_image_parallel(image_path, image_dimensions, target_pixel_size, chunk_size=(1024, 1024)):
    """按块并行处理并合成最终图像"""
    # 读取图像
    img = Image.open(image_path)
    width, height = img.size

    # 获取图像的宽高对应的实际尺寸
    img_width_um, img_height_um = image_dimensions

    # 根据图像的实际尺寸和目标像素大小计算宽度和高度的缩放因子
    scale_factor_width = img_width_um / width / target_pixel_size
    scale_factor_height = img_height_um / height / target_pixel_size

    # 按块分割并生成任务
    chunks_data = []
    for top in range(0, height, chunk_size[1]):
        for left in range(0, width, chunk_size[0]):
            right = min(left + chunk_size[0], width)
            bottom = min(top + chunk_size[1], height)

            # 将每个块的处理任务加入列表
            chunks_data.append((img, top, left, right, bottom, (scale_factor_width, scale_factor_height)))
    
    # 使用多进程处理
    with Pool() as pool:
        results = pool.map(process_chunk, chunks_data)

    # 获取每个块的结果并按位置放置
    chunks = [result[0] for result in results]
    resized_widths = [result[1] for result in results]
    resized_heights = [result[2] for result in results]
    positions = [(result[3], result[4]) for result in results]

    # 计算合成图像的总宽度和高度
    num_chunks_x = (width + chunk_size[0] - 1) // chunk_size[0]  # 向上取整，确保所有块都被包括
    num_chunks_y = (height + chunk_size[1] - 1) // chunk_size[1]  # 向上取整，确保所有块都被包括

    # 计算合成图像的宽度和高度
    total_width = sum(resized_widths[:num_chunks_x])
    total_height = sum(resized_heights[::num_chunks_x])

    final_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # 将每个块按计算的正确位置粘贴
    current_y = 0
    for y in range(num_chunks_y):
        current_x = 0
        for x in range(num_chunks_x):
            index = y * num_chunks_x + x
            chunk = chunks[index]
            chunk_width, chunk_height = chunk.size
            final_img.paste(chunk, (current_x, current_y))
            current_x += chunk_width
        current_y += resized_heights[index]  # 行高应根据最后一个块的高度来计算

    # 确保图像的宽和高是256的倍数，进行填充
    final_width, final_height = final_img.size
    new_width = (final_width // 256 + 1) * 256
    new_height = (final_height // 256 + 1) * 256

    # 创建新图像并填充
    final_padded_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    final_padded_img.paste(final_img, (0, 0))

    return final_padded_img

def save_image(final_img, output_path):
    """保存合成后的图像为tif格式"""
    final_img.save(output_path, format="TIFF")

# 设置文件路径和参数
image_path = '/media/qin/Docement/Myself/Data/picture/HE/tif/125073-3.tif'  # 替换为你的图像路径
output_path = '/media/qin/Docement/Myself/Data/picture/HE/tif/HE.jpg'  # 保存合成后的图像路径

# 手动输入的图像尺寸（单位：微米）
image_dimensions = (17856.4, 22219.3)  # 图像的实际尺寸，宽度和高度
target_pixel_size = 0.5  # 每个像素对应的目标尺寸（单位：微米）

# 处理并保存图像
final_img = process_image_parallel(image_path, image_dimensions, target_pixel_size)
save_image(final_img, output_path)

print(f"图像已保存到 {output_path}")
