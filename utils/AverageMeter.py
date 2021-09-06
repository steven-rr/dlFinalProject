class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, label='Value'):
        '''
        Creates an instance of AverageMeter
        
        Args:
            label: The label rendered when this instance is stringified.
        '''
        self.reset()
        self.label = label
        
    def __str__(self):
        '''Returns a string representation of this Average Meter'''
        return f'{self.label}: {self.val:.4f} ({self.avg:.4f})'

    def reset(self):
        '''Resets values to 0'''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        '''
        Increments running averages
        
        Args:
            val: The numeric value to update. 
            n: The weight normalizer.  
        '''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count