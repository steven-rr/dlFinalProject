from data.ChartDataset import CompressedChartCloseDataset
import torch
import torchvision
from torchvision.models.googlenet import googlenet
import constants

from utils.LoggerFactory import LoggerFactory
from Trainer import Trainer
from data.data_loaders import ChartDataLoaderProvider, CompressedChartCloseProvider, Cifar10DataLoaderProvider
from models.resnet import resnet32, resnet20, resnet110
from utils.AverageMeter import AverageMeter
from utils.Metrics import Metrics

def get_logger():
    return LoggerFactory.create_logger('Sanity', f'sanity.log')

def train():
    logger = get_logger()
    logger.info('###############################################')
    logger.info('#                 SANITY CHECK                #')
    logger.info('###############################################')

    trainer = Trainer(logger, Cifar10DataLoaderProvider(), epochs=1)
    model = resnet32(num_classes=10)
    trainer(model, f'./models/sanity.pth')

def test():
    model = resnet32(num_classes=10)
    model.load_state_dict(torch.load(f'./models/sanity.pth')['state_dict'])
    agg = AverageMeter()
    
    for data, target in Cifar10DataLoaderProvider().get_test_data_loader(100):
        out = model(data.type(constants.TENSOR_FLOAT_TYPE))
        batch_acc = Metrics.accuracy(out, target)
        agg.update(batch_acc, out.shape[0])        
        
    logger = get_logger()
    logger.info(f'###############################################')
    logger.info(f'#       2.0 average accuracy: {agg.avg}       #')
    logger.info(f'###############################################')

def resnet_20(data_provider=None, model=None, name='resnet110'):
    logger = LoggerFactory.create_logger('Trainer', f'{name}.log')

    data_provider = data_provider or CompressedChartCloseProvider()
    logger.info(f'DataLoader: {data_provider.__class__.__name__}')
    trainer = Trainer(logger, data_provider, epochs=200, learning_rate=0.01, burn_in=0, steps=[200,201], batch_size=32, shuffle=True) 
    model = model or resnet110(in_channels=data_provider.num_channels, num_classes=data_provider.num_classes)
    trainer(model, f'./models/{name}.pth') 
    
def alex_net():
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    #model = torchvision.models.AlexNet(num_classes=len(constants.LABEL_BIN_THRESHOLDS)+1)
    #model = torchvision.models.vgg(num_classes=len(constants.LABEL_BIN_THRESHOLDS)+1)
    #model = torchvision.models.vgg19(num_classes=len(constants.LABEL_BIN_THRESHOLDS)+1)
    model = googlenet(num_classes=len(constants.LABEL_BIN_THRESHOLDS)+1)
    resnet_20(model=model, name='googlenet')
    
def retrain():
    name = 'resent100.retrain'
    logger = LoggerFactory.create_logger('Trainer', f'{name}.log')
    
    # baseline before re-training
    metrics = Metrics.from_csv('./models/resent110.csv')
    logger.info(f'accuracy: {metrics.total_accuracy}')
    logger.info(f'per class accuracy:\n{metrics.per_class_accuracy}')    
    
    # retrain
    constants.ITERATION_MODULO = 500
    model = resnet110()
    model.load_state_dict(torch.load(f'./models/resnet110.pth')['state_dict'])
    data_provider = CompressedChartCloseProvider(class_filter=[0,5])
    trainer = Trainer(logger, data_provider, epochs=1, learning_rate=0.01, burn_in=0, steps=[200,201], batch_size=32, shuffle=True) 
    trainer(model, f'./models/{name}.pth') 
    
    # new metrics
    metrics = Metrics.map_predictions(resnet110(), f'./models/{name}.pth', CompressedChartCloseProvider())
    logger.info(f'accuracy: {metrics.total_accuracy}')
    logger.info(f'per class accuracy: {metrics.per_class_accuracy}')  
    
def weighted():
    name = 'resnet100.weighted'
    logger = LoggerFactory.create_logger('Trainer', f'{name}.log')
    
    metrics = Metrics(CompressedChartCloseDataset().df)
    data_provider = CompressedChartCloseProvider(sampler=metrics.get_weighted_random_sampler())
    model = resnet110()
    
    trainer = Trainer(logger, data_provider, epochs=1, learning_rate=0.01, burn_in=0, steps=[200,201], batch_size=32, shuffle=False) 
    trainer(model, f'./models/{name}.pth') 

if __name__ == '__main__':
    weighted()
    #retrain()
    #alex_net()
    #train()
    #test()
