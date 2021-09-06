import unittest
import unittest
import pandas

from torch.utils.data import WeightedRandomSampler
from utils.Metrics import Metrics

class PandasTest(unittest.TestCase):
    def test_distros(self):
        print('\n', self.load_df().head)
        
    def test_total_accuracy(self):
        df = self.load_df()  
        print(df[df['Label'] - df['Prediction'] == 0].shape[0] / df.shape[0])
        
    def test_per_class_accuracy(self):
        df = self.load_df()
        counts = df['Label'].value_counts()
        tp = df[df.Label - df.Prediction == 0].groupby('Label').agg('count')['GIC']
        print(df.head)
        
    def test_filter(self):
        df = self.load_df()
        fives = df[df.Label == 5].count()[0]
        epxected = df.shape[0] - fives        
        actual = df[~df.Label.isin([5])]
        actual = actual.shape[0]       
        
        self.assertEqual(epxected, actual)
    
    def test_metrics(self):
        metrics = Metrics.from_csv('./models/resent110.csv')
        print(f'accuracy: {metrics.total_accuracy}')
        print(f'per class accuracy: {metrics.per_class_accuracy}')
        print(f'per class instances: {metrics.per_class_instances}')
        print(f'TP: {metrics.tp}')
        print(f'per class weights: {metrics.per_class_weights}')
        
    def test_assign_weights(self):
        metrics = Metrics.from_csv('./models/resent110.csv')
        per_class_weights = metrics.per_class_weights.values
        
        df = metrics.df
        
        df['Weights'] = 0
        for i in range(per_class_weights.size):
            df.loc[df['Label'] == i, 'Weights'] = per_class_weights[i]
            
        print(df.head)
        
        weights = df['Weights'].values
        weighted_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=weights.size,
            replacement=True
        )
        
        self.assertIsNotNone(weighted_sampler)
        
    def load_df(self):
        csv = './models/resent110.csv'
        return pandas.read_csv(csv, index_col='Filename', parse_dates=True, dtype={'Filename':str, 'GIC':int, 'Date':str, 'Label':int, 'Return':float, 'Prediction':int})
        