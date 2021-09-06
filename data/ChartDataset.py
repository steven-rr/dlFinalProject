import constants
import os.path as path
import torch
from torch.utils.data import Dataset
from data.PreProcessor import PreProcessor
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
import numpy as np
from torch.utils.data import WeightedRandomSampler

class ChartDataset(Dataset):
    '''Represents a PyTorch Dataset conatining chart images.'''
    def __init__(self, preprocessor, train=True, transform=None, class_filter=None):
        '''
        Creates a PyTorch Dataset.
        
        Args: 
            preprocessor: A data.PreProcessor instance.
            train: A value indicating whether to split this data set with the constants.TRAIN_DATA_RATIO
            transform: Optional image tranformations to perform on the data (e.g. transform to Tesnsor, separate channels).
        '''
        super().__init__()
        self.__transform=transform
        df = preprocessor.parse_file_names()
        if class_filter is not None:
            if not isinstance(class_filter, list):
                class_filter = [class_filter]
            df = df[~df.Label.isin(class_filter)]
        self.__df = self.split_data(df, train)
       
    @property 
    def df(self):
        return self.__df
        
    def __len__(self):
        ''' 
        Gets the length of the Dataset.
        
        Returns:
            an integer length.
        '''
        return self.__df.shape[0]
    
    def __getitem__(self, index):
        '''
        Allows direct indexing of the Dataset.
        
        Args:
            index: the integer index of the element to lookup.  
        
        Returns:
            A tuple of (image, label).
        '''
        if torch.is_tensor(index):
            index = index.tolist()
            
        image = self.load_image(index)     
            
        if self.__transform:
            image = self.__transform(image)
            image = image.type(constants.TENSOR_FLOAT_TYPE)
            
        return image, torch.tensor(self.__df.iloc[index]['Label'], dtype=torch.long)
    
    def split_data(self, df, train=True):
        split = int(constants.TRAIN_DATA_RATIO * df.shape[0])
        return df.sort_values(by='Date', ascending=train).head(split if train else df.shape[0]-split)
    
    def get_path(self):
        return constants.CHARTS_DIR
    
    def load_image(self, index):
        name = path.join(self.get_path(), self.__df.index[index])
        image = io.imread(name)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = rgba2rgb(image) # remove the alpha channel  
        return image 
    
    def calc_mean_std(self):
        shape = self[0][0].shape
        images = np.zeros((self.__len__(), shape[0], shape[1], shape[2]))
        for i in range(images.shape[0]):
            images[i] = self[i][0]
            if i % 1000 == 0:
                print(f'loading image {i}')
        
        mean = np.mean(images, axis=tuple(range(images.ndim-1)))
        std = np.std(images, axis=tuple(range(images.ndim-1)))
        return mean, std      
    
    
class ChartDataset_Mini(ChartDataset):
    def get_path(self):
        return f'{super().get_path()}_Mini'
    def load_image(self, index):
        image = super().load_image(index)
        return rgb2gray(image).reshape(image.shape[0], image.shape[1], 1)
        
class CompressedChartCloseDataset(ChartDataset):
    def __init__(self, preprocessor=None, train=True, transform=None, class_filter=None):
        super().__init__(preprocessor or PreProcessor(charts_dir=constants.CHARTS_CLOSE_DIR), train=train, transform=transform, class_filter=class_filter)
    def get_path(self):
        return constants.CHARTS_CLOSE_DIR
    def load_image(self, index):
        image = super().load_image(index)
        image[4:8, 36:, :] = 1.
        return image
    
class CompressedChartEstDataset(ChartDataset):
    def get_path(self):
        return constants.CHARTS_EST_DIR
class CompressedChartVolDataset(ChartDataset):
    def get_path(self):
        return constants.CHARTS_VOL_DIR
    
class CompressedChartGRUDataset(ChartDataset):
    def get_path(self):
        return constants.CHARTS_GRU_DIR