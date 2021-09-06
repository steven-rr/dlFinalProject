
import constants
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.PreProcessor import PreProcessor
from data.ChartDataset import *

class DataLoaderProvider:
    pass
    def get_train_data_loader(self, batch_size, shuffle=True):
        return self.get_data_loader(True, batch_size, shuffle)
    def get_test_data_loader(self, batch_size):
        return self.get_data_loader(False, batch_size, False)
    def get_data_loader(self, train, batch_size, shuffle):
        pass
    @property
    def num_classes(self):
        pass
    
class ChartDataLoaderProvider(DataLoaderProvider):
    def __init__(self, class_filter=None, sampler=None):
        self.class_filter = class_filter
        self.sampler = sampler
        self.composed = transforms.Compose([transforms.ToTensor(), transforms.Resize((800//constants.IMAGE_SCALE_FACTOR, 600//constants.IMAGE_SCALE_FACTOR))]) 
                
    def get_data_loader(self, train, batch_size, shuffle):
        return DataLoader(self.get_dataset(train, class_filter=self.class_filter), batch_size=batch_size, shuffle=shuffle, num_workers=constants.DATA_LOADER_WORKERS, sampler=self.sampler if train else None)
    
    def get_dataset(self, train, class_filter=None):
        return ChartDataset(PreProcessor(), train, self.composed, class_filter=class_filter)
    
    @property
    def num_classes(self):
        return len(constants.LABEL_BIN_THRESHOLDS) + 1
    
    @property
    def num_channels(self):
        return 3    
class ChartDataLoader_MiniProvider(ChartDataLoaderProvider):
    def __init__(self, class_filter=None, sampler=None):
        super().__init__(class_filter=class_filter, sampler=sampler)
        self.composed = transforms.Compose([transforms.ToTensor()]) 
        
    def get_dataset(self, train, class_filter=None):
        return ChartDataset_Mini(PreProcessor(), train, self.composed, class_filter=class_filter)
    
    @property
    def num_channels(self):
        return 1

class CompressedChartCloseProvider(ChartDataLoaderProvider):
    def __init__(self, class_filter=None, sampler=None):
        super().__init__(class_filter=class_filter, sampler=sampler)
        self.composed = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.98187733, 0.97973949, 0.97192087],
                std=[0.08062164, 0.0815032,  0.10506404]
            ),
        ])
        
    def get_dataset(self, train, class_filter=None):
        return CompressedChartCloseDataset(PreProcessor(charts_dir=constants.CHARTS_CLOSE_DIR), train, self.composed, class_filter=class_filter)
        
class CompressedChartEstProvider(CompressedChartCloseProvider):        
    def get_dataset(self, train, class_filter=None):
        return CompressedChartEstDataset(PreProcessor(charts_dir=constants.CHARTS_EST_DIR), train, self.composed, class_filter=class_filter)
 
class CompressedChartVolProvider(CompressedChartCloseProvider):        
    def get_dataset(self, train, class_filter=None):
        return CompressedChartVolDataset(PreProcessor(charts_dir=constants.CHARTS_VOL_DIR), train, self.composed, class_filter=class_filter)
    
class CompressedChartGRUProvider(CompressedChartCloseProvider): 
    def __init__(self, class_filter=None, sampler=None):
        super().__init__(class_filter=class_filter, sampler=sampler)
        self.composed = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.84716746, 0.84716746, 0.84716746],
                std=[0.30287377, 0.30287377, 0.30287377]
            ),
        ])       
    def get_dataset(self, train, class_filter=None):
        return CompressedChartGRUDataset(PreProcessor(charts_dir=constants.CHARTS_GRU_DIR), train, self.composed, class_filter=class_filter)
    
class Cifar10DataLoaderProvider(DataLoaderProvider):
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def get_data_loader(self, train, batch_size, shuffle):
        composed = self.transform_train if train else self.transform_test
        dataset = torchvision.datasets.CIFAR10( root='./data', train=train, download=True, transform=composed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @property
    def num_classes(self):
        return 10