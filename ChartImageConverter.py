import os
import constants
import numpy as np

from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize

from utils.LoggerFactory import LoggerFactory

class ChartImageConverter:
    @staticmethod 
    def resize_images(shape, logger=None):
        logger = logger or LoggerFactory.create_logger('CharImageConverter')
        
        path = os.path.join(constants.CHARTS_DIR, f'{shape[0]}x{shape[1]}')
        if not os.path.exists(path): os.makedirs(path)
        
        gray_path = os.path.join(path, 'Gray')
        if not os.path.exists(os.path.join(path, 'Gray')): os.makedirs(path)
        
        for i, name in enumerate(os.listdir(constants.CHARTS_DIR)):
            if os.path.isdir(os.path.join(constants.CHARTS_DIR, name)):
                continue
            logger.info(f'{i}) resizing {name}')
            
            image = io.imread(os.path.join(constants.CHARTS_DIR, name))
            image = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=False)
            io.imsave(os.path.join(path, name), (image*255).astype(np.uint8))
            
            image = rgba2rgb(image)
            image = rgb2gray(image)  
            io.imsave(os.path.join(gray_path, name), (image*255).astype(np.uint8))
            
if __name__ == '__main__':
    ChartImageConverter.resize_images((800//constants.IMAGE_SCALE_FACTOR, 600//constants.IMAGE_SCALE_FACTOR))