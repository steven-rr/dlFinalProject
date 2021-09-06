import os
import torch
import constants
import pandas as pd
from torch.utils.data import WeightedRandomSampler

class Metrics:
    def __init__(self, df):
        '''
        Wraps a df 
        '''
        assert df is not None
        self.__df = df
        
    @property
    def df(self):
        return self.__df
    
    @property
    def total_accuracy(self):
        return self.tp.sum() / self.df.shape[0]
    
    @property
    def per_class_accuracy(self):
        value = self.tp/self.per_class_instances
        return value
    
    @property
    def per_class_instances(self):
        return self.df['Label'].value_counts()
    
    @property
    def tp(self):
        return self.df[self.df.Label - self.df.Prediction == 0].groupby('Label').agg('count')['GIC']    
    
    @property
    def per_class_weights(self):
        return 1/self.per_class_instances
    
    def get_weighted_random_sampler(self):
        per_class_weights = self.per_class_weights.values
        
        self.df['Weights'] = 0
        for i in range(per_class_weights.size):
            self.df.loc[self.df['Label'] == i, 'Weights'] = per_class_weights[i]
            
        return WeightedRandomSampler(weights=self.df['Weights'].values, num_samples=self.df.shape[0], replacement=True)
        
        
    '''Metric calcualtions.  We likely want more than just accuracy here...'''
    @staticmethod
    def accuracy(output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        return correct / batch_size
    
    @classmethod 
    def from_csv(cls, path):
        df = pd.read_csv(path, index_col='Filename', parse_dates=True, dtype={'Filename':str, 'GIC':int, 'Date':str, 'Label':int, 'Return':float, 'Prediction':int})
        return cls(df)
    
    @classmethod
    def map_predictions(cls, model, state_path, provider, csv_out=None):
        '''
        Adds the predictions for a given model to a labled data set
        
        Args:
            model: The model to usse for inference
            state_path: The path to the models saved state
            provider: The Dataset provider
            csv_out: Optional output file to save the results in a csv file. Default is None
            
        Returns:
            A Dataframe containing the predictions.
        '''
        if not os.path.exists(state_path):
            print(f'No model found {state_path}')
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(state_path)['state_dict'])
        model = model.to(device)
        
        df = provider.get_dataset(False).df
        predictions = torch.ones(df.shape[0], dtype=int) * -1
        batch_size = 25
        
        for idx, (data, _) in enumerate(provider.get_data_loader(False, batch_size, False)):
            data = data.type(constants.TENSOR_FLOAT_TYPE).to(device)
            out = model(data)
            batch_index = idx*batch_size 
            _, pred = torch.max(out, dim=-1)
            predictions[batch_index:batch_index+batch_size] = pred
        
        df['Prediction'] = predictions.detach().numpy()
        if csv_out is not None:
            df.to_csv(csv_out)
        
        return cls(df)
    