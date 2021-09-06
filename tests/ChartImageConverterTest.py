import os
import unittest
import numpy as np
import shutil

from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize

class ChartImageConverter(unittest.TestCase): 
    FILE_NAME = '101010_835934400000_3.55.png'
    TEST_IMG_DIR = './tests/200x150'
    TEST_GRAY_DIR = os.path.join(TEST_IMG_DIR, 'Gray')
    
    def setUp(self):
        if not os.path.exists(self.TEST_IMG_DIR):
            os.makedirs(self.TEST_IMG_DIR)
            os.makedirs(self.TEST_GRAY_DIR)
                
    def tearDown(self):
        if os.path.exists(self.TEST_IMG_DIR):
            shutil.rmtree(self.TEST_IMG_DIR, ignore_errors=True)
        
    def test_scale_and_save(self):
        image = io.imread(os.path.join('./data/Charts', self.FILE_NAME))
        image = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
        image = (image*255).astype(np.uint8)
        io.imsave(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME), image)
        
        image = rgba2rgb(image)
        image = rgb2gray(image)
        image = (image*255).astype(np.uint8)
        io.imsave(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME), image)
        
    def test_dimensions(self):
        
        expected = (600, 800, 4)
        actual = io.imread(os.path.join('./data/Charts', self.FILE_NAME))
        self.assertEqual(expected, actual.shape)        
        
        image = resize(actual, (actual.shape[0] // 4, actual.shape[1] // 4), anti_aliasing=True)
        image = (image*255).astype(np.uint8)
        io.imsave(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME), image)
        
        expected = (150, 200, 4)
        actual = io.imread(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME))
        self.assertEqual(expected, actual.shape)
        
        image = rgba2rgb(image)
        image = rgb2gray(image)
        image = (image*255).astype(np.uint8)
        io.imsave(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME), image)
        
        expected = (150, 200)
        actual = io.imread(os.path.join(self.TEST_IMG_DIR, self.FILE_NAME))
        self.assertEqual(expected, actual.shape)