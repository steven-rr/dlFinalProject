import os
import unittest
import constants
import random
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from data.PreProcessor import PreProcessor
from data.ChartDataset import *
from data.data_loaders import *
from pandas.testing import assert_frame_equal

class ChartDatasetTest(unittest.TestCase):
    def test_len(self):
        target = ChartDataset(PreProcessor())
        
        expected = int(len([name for name in os.listdir(constants.CHARTS_DIR) if name.endswith(constants.CHART_FILE_EXT)]) * constants.TRAIN_DATA_RATIO)
        actual = target.__len__()
        self.assertEqual(expected, actual)    
        
    def test_len2(self):
        actual = ChartDataset(PreProcessor()).df.shape[0]        
        expected = int(len([name for name in os.listdir(constants.CHARTS_DIR) if name.endswith(constants.CHART_FILE_EXT)]) * constants.TRAIN_DATA_RATIO)
        self.assertEqual(expected, actual)  
        
    def test_len3(self):
        df = ChartDataset(PreProcessor()).df
        print('\n', df['Label'].unique())
        fives = df[df.Label == 5].count()[0]
        expected = df.shape[0] - fives        
        df = ChartDataset(PreProcessor(), class_filter=5).df
        print(df['Label'].unique())
        
        
    def test_getitem(self):
        return 
        target = ChartDataset(PreProcessor())
        data, _ = target[5]
        print('\n', data.shape)
        plt.imshow(data)
        plt.show()
        
    def test_getitem_Mini(self):
        return 
        target = ChartDataset_Mini(PreProcessor())
        data, _ = target[5]
        print('\n', data.shape)
        plt.imshow(data, cmap="gray")
        plt.show()
        
    def test_getitem_Cifar(self):
        target = Cifar10DataLoaderProvider().get_data_loader(False, 1, True)
        self.show_image_from_loader(target)
        
    def test_rendor_chart_from_tensor(self):
        target = ChartDataLoaderProvider().get_data_loader(False, 1, False)
        self.show_image_from_loader(target)
        
    def test_rendor_chart_from_tensor_close(self):
        target = CompressedChartCloseProvider().get_data_loader(False, 1, False)
        self.show_image_from_loader(target)
        
    def test_rendor_chart_from_tensor_gru(self):
        target = CompressedChartGRUProvider().get_data_loader(False, 1, False)
        self.show_image_from_loader(target)
        
    def show_image_from_loader(self, data_loader):
        for idx, (data, target) in enumerate(data_loader):
            image = data[0]
            image = image.transpose(0,1).transpose(1,2)
            print('\n', image.shape)
            plt.imshow(image)
            plt.show()
            break
                
    def test_split_train(self):
        target = ChartDataset(PreProcessor())
        data = [ (str(i), dt.datetime(1992, 5, 12+i)) for i in range(11)]
        columns=['name', 'Date']
        expected = pd.DataFrame(data[:8], columns=['name', 'Date']).set_index(columns[-1])
        random.shuffle(data)
        actual = target.split_data(pd.DataFrame(data, columns=['name', 'Date']).set_index(columns[-1]), train=True)
        assert_frame_equal(expected, actual)  
        
    def test_split_test(self):
        target = ChartDataset(PreProcessor())
        data = [ (str(i), dt.datetime(1992, 5, 12+i)) for i in range(11)]
        columns=['name', 'Date']
        expected = pd.DataFrame(data[8:], columns=columns).set_index(columns[-1])
        random.shuffle(data)
        actual = target.split_data(pd.DataFrame(data, columns=columns).set_index(columns[-1]), train=False)
        assert_frame_equal(expected.sort_values(by='Date'), actual.sort_values(by='Date'))   
        
if __name__ == '__main__':
    unittest.main()
        

        