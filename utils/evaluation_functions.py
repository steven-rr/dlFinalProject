import torch

from utils.LoggerFactory import LoggerFactory
from Trainer import Trainer
from data.data_loaders import ChartDataLoaderProvider, Cifar10DataLoaderProvider

def trainer_eval(kwargs):
    assert 'model' in kwargs, 'A model must be provided to train'
    args = [f'{key}.{str(value)}' for key, value in kwargs.items() if key != 'model']
    file_name = f'{kwargs["model"][1]}_{"_".join(args)}'
    logger = LoggerFactory.create_logger('Trainer', f'{file_name}.log')
    
    try:   
        model, model_name = kwargs['model'][0], kwargs['model'][1]
        kwargs = dict(kwargs)
        del kwargs['model']
        logger.info(f'starting Trainer run with model {model_name} and hyper-parameters {kwargs}')      
        
        trainer = Trainer(logger, ChartDataLoaderProvider(), **kwargs)
        trainer(model)
        torch.save(model.state_dict(), f'./models/{file_name}.pth')  
          
    except Exception as e:
        logger.exception(e, exc_info=True)