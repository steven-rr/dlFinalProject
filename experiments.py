from GridSearch import GridSearch
from models.resnet import resnet20, resnet32, resnet44
from utils.evaluation_functions import trainer_eval
from utils.LoggerFactory import LoggerFactory

def main():
    try:  
        grid = GridSearch(trainer_eval)
        grid({'model':[(resnet20(), 'resnet20'), (resnet32(), 'resnet32'), (resnet44(), 'resnet44')],
              'epochs':[1],
              'regularization_parameter':[1e-4], 
              'batch_size':[64]}) 
    except Exception as e:
        LoggerFactory.create_logger('Main', 'main.log').exception(e, exc_info=True)
        
if __name__ == '__main__':
    main()