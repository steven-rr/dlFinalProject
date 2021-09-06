import unittest
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Normalize
from data.data_loaders import *
from data.ChartDataset import *
from data.PreProcessor import PreProcessor

class MaskingTest(unittest.TestCase):
    def test_mask_2(self):
        target = CompressedChartCloseDataset(PreProcessor(charts_dir=constants.CHARTS_CLOSE_DIR))
        image, _ = target[5]
        
        plt.hist(np.array(image).ravel(), bins=50, density=True);
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of pixels");
        plt.show()
        
    def test_mask_2_normalized(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.98187733, 0.97973949, 0.97192087],
                std=[0.08062164, 0.0815032,  0.10506404]
            ),
        ])
        target = CompressedChartCloseDataset(PreProcessor(charts_dir=constants.CHARTS_CLOSE_DIR), transform=transform)
        image, _ = target[5]
        image = image.numpy()
        
        plt.hist(image.ravel(), bins=30, density=True);
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of pixels")
        plt.show()
        
        plt.imshow(image.transpose(1, 2, 0))
        plt.show()
        
    def test_calcualte_mean_std(self):
        return
        target = CompressedChartCloseDataset(PreProcessor(charts_dir=constants.CHARTS_CLOSE_DIR))
        mean, std = target.calc_mean_std()
        print()
        print(f'mean: {mean}')
        print(f'std: {std}')
        
    def test_calculate_mean_std_gru(self):
        return
        target = CompressedChartGRUDataset(PreProcessor(charts_dir=constants.CHARTS_GRU_DIR))
        mean, std = target.calc_mean_std()
        print()
        print(f'mean: {mean}')
        print(f'std: {std}')      
    
    def test_mask_gics(self):
        return
        data_loader = CompressedChartCloseProvider().get_data_loader(False, 1, False)
        for _, (data, _) in enumerate(data_loader):
            image = data[0]
            break
        
        print('\n', image.shape)
        print(torch.max(image[0]))
        print(torch.max(image[1]))
        print(torch.max(image[2]))
        
        image = image[0, 4:7, 36:] = 1.
        
        image = image.transpose(0,1).transpose(1,2)
        plt.imshow(image)
        plt.show()