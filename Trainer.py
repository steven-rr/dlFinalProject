import constants
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from datetime import datetime
from utils.CompositeAverageMeter import CompositeAverageMeter
from utils.Metrics import Metrics

class Trainer:
    '''Wraps PyTorch model training.'''
    def __init__(self, logger, load_provider, epochs=10, learning_rate=0.1, momentum=0.9, regularization_parameter=5e-5, burn_in=0, steps=[6,8], batch_size=32, shuffle=True, **kwargs):
        '''Instantiates a trainer using the passed hyper parameters.'''
        self.__logger = logger
        self.__load_provider = load_provider
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__regularization_parameter = regularization_parameter
        self.__burn_in = burn_in
        assert len(steps) == 2, 'steps must have length 2'
        self.__steps = steps
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Trainer will run on {self.device}')
        logger.info(f'Trainer will run with hyperparms: {self}')
        
    def __str__(self):
        return f'{self.epochs}_{self.learning_rate}_{self.momentum}_{self.regularization_parameter}_{self.burn_in}_{self.steps[0]}_{self.steps[1]}_{self.batch_size}_{self.shuffle}'
                
    @property
    def logger(self):
        return self.__logger
    
    @property
    def load_provider(self):
        return self.__load_provider
    
    @property
    def epochs(self):
        return self.__epochs
    
    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def momentum(self):
        return self.__momentum
    
    @property
    def regularization_parameter(self):
        return self.__regularization_parameter
    
    @property
    def burn_in(self):
        return self.__burn_in
    
    @property
    def steps(self):
        return self.__steps
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def shuffle(self):
        return self.__shuffle
    
    def __call__(self, model, serialization_path=None, best_accuracy=0, best_confusion=None, best_epoch=0):
        '''
        Executes a training cycle.  If serailization_path is passed, then the most accurate model state dictionary of weights will be serialized after every epoch.
        If best_accuracy, best_confusion, and best_epoch are passed along with a deserialized model, the method will continue training with those values.  This helps
        if there is a crach, or a timeout, or you are trying to transfer learn.
        
        Args:
            model: the model to train
            serialization_path: if passed, the full file path to save the model to.  Default - None
            best_accuracy: if passed, training will resume with the value
            best_confusion: if passed, training will resume with the value
            best_epoch: if passed, training will resume with the value
        TODO:
            may wish to vary the optimizer.
        '''
        train_data_loader = self.load_provider.get_train_data_loader(self.batch_size, self.shuffle)
        test_data_loader = self.load_provider.get_test_data_loader(self.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), self.learning_rate, momentum=self.momentum, weight_decay=self.regularization_parameter)
        criterion = nn.CrossEntropyLoss()
        model = model.type(constants.TENSOR_FLOAT_TYPE).to(self.device)
        
        best_accuracy, best_confusion, best_epoch = 0, None, 0
        avgLossPerEpoch = []
        for epoch in range(best_epoch, self.epochs):
            self.decay_learning_rate(optimizer, epoch)
            meter = self.train(epoch, train_data_loader, model, optimizer, criterion)
            accuracy, confusion_matrix = self.validate(epoch, test_data_loader, model, criterion)
            avgLossPerEpoch.append(meter.losses.avg.cpu().detach().numpy())

            if accuracy > best_accuracy:
                best_accuracy, best_confusion, best_epoch = accuracy, confusion_matrix, epoch
                if serialization_path is not None:
                    torch.save({'state_dict':model.state_dict(), 'accuracy':best_accuracy, 'confusion':best_confusion, 'epoch':best_epoch}, f'{serialization_path}.best')
            
            if serialization_path is not None: 
                torch.save({'state_dict':model.state_dict(), 'accuracy':accuracy, 'confusion':confusion_matrix, 'epoch':epoch}, f'{serialization_path}.epoch{epoch}')
        
        self.logger.info(f'Best Top 1 Accuracy: {best_accuracy:.4f}')   
        per_class_accuracy = self.log_per_class_accuracy(best_confusion)   
        if serialization_path is not None: # overwrite the most recent epoch saved
            torch.save({'state_dict':model.state_dict(), 'accuracy':accuracy, 'confusion':confusion_matrix, 'epoch':epoch}, f'{serialization_path}')

        #self.plotAverageLoss(avgLossPerEpoch)

        return per_class_accuracy, best_accuracy, best_epoch
                    
    def decay_learning_rate(self, optimizer, epoch):
        '''
        Decays the learning rate with burn-in.
        
        Args:
            optimizer: the optimizer who's learing rate is updatedated.
            epoch: the current training epoch.
        '''
        epoch += 1
        if epoch <= self.burn_in:
            lr = self.__learning_rate * epoch / self.burn_in
        elif epoch > self.steps[1]:
            lr = self.learning_rate * 0.01
        elif epoch > self.steps[0]:
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def train(self, epoch, data_loader, model, optimizer, criterion):
        '''
        Executes a training loop for the passed epoch.
        
        Args:
            epoch: the epoch to train.
            data_loader: the training data_loader.
            model: the model to train.
            optimizer: the optimizer to train with.
            criterion: the criterion to train with.
        '''
        return self.run_epoch(self.train_eval, epoch, data_loader, model, optimizer, criterion)
    
    def validate(self, epoch, data_loader, model, criterion):
        '''
        Executes a validation loop for the passed epoch.
        
        Args:
            epoch: the epoch to validate.
            data_loader: the validation data_loader.
            model: the model to validate.
            criterion: the criterion to validate with.
            
        Returns:
            A tuple containing the average accuracy and the confusion matrix.
        '''
        num_classes = self.load_provider.num_classes
        confusion_matrix = torch.zeros(num_classes, num_classes).to(self.device)
        meter = self.run_epoch(self.validate_eval, epoch, data_loader, model, None, criterion, confusion_matrix)
        
        confusion_matrix /= confusion_matrix.sum(1)
        self.log_per_class_accuracy(confusion_matrix)
            
        return meter.accuracy.avg, confusion_matrix
            
    def train_eval(self, data, target, model, optimizer, criterion, state):
        '''
        Evaluates a training iteration.
        
        Args:
            data: the data to train.
            target: thr ground truth.
            model: the model to train.
            optimizer: the optimizer to train with.
            criterion: the criterion to train with.
            state: not used
            
        Returns:
            A tuple containg the model output, the ground truth, and the loss.
        '''
        out = model(data)        
        loss = criterion(out, target)       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return out, target, loss    
    
    def validate_eval(self, data, target, model, optimizer, criterion, state):
        '''
        Validates a training iteration.
        
        Args:
            data: the data to validate.
            target: thr ground truth.
            model: the model to validate.
            optimizer: not used.
            criterion: the criterion to validate with.
            state: a confusion matrix
            
        Returns:
            A tuple containg the model output, the ground truth, and the loss.
        '''
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)
            
        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            state[t.long(), p.long()] += 1
        
        return out, target, loss
            
    def run_epoch(self, eval, epoch, data_loader, model, optimizer, criterion, state=None):
        '''
        Runs a loop for one epoch        
        
        Args:
            eval: the evaluation function.
            epoch: the epoch to train.
            data_loader: the training data_loader.
            model: the model to train.
            optimizer: the optimizer to train with.
            criterion: the criterion to train with.
        '''
        meter = CompositeAverageMeter(self.logger)
        data_length = len(data_loader)
                
        for idx, (data, target) in enumerate(data_loader):                
            out, target, loss = eval(data.to(self.device), target.to(self.device), model, optimizer, criterion, state)

            batch_acc = Metrics.accuracy(out, target)
            meter.update(loss, batch_acc, out.shape[0])
            meter.log(epoch, idx, data_length)
            
        
        return meter
    
    def log_per_class_accuracy(self, confusion_matrix):
        '''
        Logs the per class accuracy.
        
        Args:
            confusion_matrix: the confusion matrix to derive per_class_accuracy from.
        '''
        per_class_accuracy = confusion_matrix.cpu().diag().detach().numpy().tolist()
        for cls, accuracy in enumerate(per_class_accuracy):
            format = f'Accuracy of Class {cls}: {accuracy:.4f}'
            self.logger.info(format)
        return per_class_accuracy

    def plotAverageLoss(self, losses):
        timestamp = datetime.fromtimestamp(datetime.now)
        timestamp = timestamp.strftime("%Y%m%d%H%M")

        plt.plot(losses)
        plt.ylabel('losses')
        plt.xlabel('epochs')
        if not os.path.exists(constants.LOSS_DIR):
            os.makedirs(constants.LOSS_DIR)
        plt.savefig(constants.LOSS_DIR + timestamp)