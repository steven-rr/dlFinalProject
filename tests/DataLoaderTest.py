import unittest
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from data.ChartDataset import ChartDataset
from data.PreProcessor import PreProcessor


class DataLoaderTest(unittest.TestCase):
    def test_enumerate(self):
        composed = transforms.Compose([transforms.ToTensor(), transforms.Resize((80,60))]) 
        data_loader = DataLoader(ChartDataset(PreProcessor(), False, composed), batch_size=32, shuffle=True)
        length = len(data_loader)
        
        for idx, (data, target) in enumerate(data_loader):
            print(f'{idx}/{length}: {target} ({type(target)})')
            if idx > 1:
                break
        