import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None
image_path = '/home/ioz15/work/data/HE/HE-test/adjusted_image.jpg'  # 替换成你的图像路径
img = Image.open(image_path)

img_array = np.array(img)

height, width = img_array.shape[:2]  # 如果是RGB图像，shape[2]是通道数
print(f"图像的宽度：{width} 像素")
print(f"图像的高度：{height} 像素")

# 显示图像
plt.imshow(img)
plt.axis('off')
plt.show()
