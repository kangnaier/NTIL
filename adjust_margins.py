import numpy as np
from PIL import Image



Image.MAX_IMAGE_PIXELS = None
def crop_image(img, extent, mode='edge', constant_values=None):
    """
    Crop or pad an image based on the given extent and mode.
    
    Parameters:
        img (np.ndarray or Image): The image to be cropped or padded.
        extent (np.ndarray): The extent (range) to crop or pad to, as [lower, upper] for each axis.
        mode (str): The padding mode ('edge', 'constant'). Default is 'edge'.
        constant_values (int or float): The value to use for constant padding. Default is None.
        
    Returns:
        np.ndarray: The cropped or padded image.
    """
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    
    # Calculate necessary padding
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    
    # Apply padding if necessary
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    
    # Crop the image based on the extent
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    
    return img

def adjust_margins(img, pad, pad_value=None):
    """
    Adjust the image margins to make its dimensions divisible by the given padding size.
    
    Parameters:
        img (np.ndarray or Image): The image to be adjusted.
        pad (int): The padding size to make dimensions divisible by.
        pad_value (int or float): The value to use for constant padding. Default is None (edge padding).
        
    Returns:
        np.ndarray: The padded image.
    """
    extent = np.stack([[0, 0], img.shape[:2]]).T
    
    # Make size divisible by pad without changing coordinates
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    
    # Determine padding mode
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    
    # Use the crop_image function to adjust the margins
    img = crop_image(img, extent, mode=mode, constant_values=pad_value)
    return img

# Example usage:
if __name__ == "__main__":
    # Load an image (you can replace this with your own image loading function)
    img = Image.open('/home/ioz15/work/data/HE/HE-test/he-scaled.jpg')
    img = np.array(img)
    
    # Set the padding size
    pad_size = 256
    
    # Adjust the image margins to make its dimensions divisible by 256
    adjusted_img = adjust_margins(img, pad_size, pad_value=255)  # White padding
    
    # Convert back to PIL Image and save the adjusted image
    adjusted_img_pil = Image.fromarray(adjusted_img)
    adjusted_img_pil.save('/home/ioz15/work/data/HE/HE-test/adjusted_image.jpg')
