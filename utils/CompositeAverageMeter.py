import time
import constants

from utils.AverageMeter import AverageMeter

class CompositeAverageMeter:
    ITERATION_TIME = 'iteration_time'
    LOSSES = 'losses'
    ACCURACY = 'accuracy'
    
    def __init__(self, logger):
        self.logger = logger
        self.meters = {self.ITERATION_TIME:AverageMeter('Time'), self.LOSSES:AverageMeter('Loss'), self.ACCURACY:AverageMeter('Top 1')}
        self.start = time.time()
        
    @property
    def iteration_time(self):
        return self.meters[self.ITERATION_TIME]
    
    @property
    def losses(self):
        return self.meters[self.LOSSES]
    
    @property
    def accuracy(self):
        return self.meters[self.ACCURACY]        
        
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
            
    def update(self, loss, accuracy, n):
        self.meters[self.LOSSES].update(loss, n)
        self.meters[self.ACCURACY].update(accuracy, n)
        
        now = time.time()
        self.meters[self.ITERATION_TIME].update(now-self.start)
        self.start = now
        
    def log(self, epoch, idx, data_length):
        format = f'Epoch: [{epoch}][{idx}][{data_length}]    {self.iteration_time}    {self.losses}    {self.accuracy}'
        if epoch % constants.EPOCH_MODULO == 0 and idx % constants.ITERATION_MODULO == 0:
            self.logger.info(format)
        else:
            self.logger.debug(format)