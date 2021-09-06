import unittest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.resnet import resnet20
from utils.AverageMeter import AverageMeter
from utils.Metrics import Metrics
from data.data_loaders import DataLoaderProvider 

class AccuracyTest(unittest.TestCase):
    def test_hello(self):
        print("Hello")
    '''
    def test_resnet20(self):
        if False:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = resnet20().to(device)
            print('\nmade double')
            state = torch.load('./models/resnet20_epochs.1_regularization_parameter.0.0001_batch_size.64.pth')
            model.load_state_dict(state)
            
            batch_size = 10 
            test_loader = ChartDataLoaderProvider().get_test_data_loader(batch_size)
            agg = AverageMeter()
            count = len(test_loader)

            for i, (data, target) in enumerate(test_loader):
                data, target = data.float().to(device), target.float().to(device)
                print(f'{i*batch_size}/{count}...')
                out = model(data)
                batch_acc = Metrics.accuracy(out, target)
                agg.update(batch_acc, out.shape[0])            
                
            print(f'average accuracy: {agg.avg}')
    '''