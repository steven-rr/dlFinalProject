import unittest

from models.resnet import resnet110
from data.ChartDataset import CompressedChartCloseDataset
from data.data_loaders import CompressedChartCloseProvider
from utils.Metrics import Metrics

class AnalysisTest(unittest.TestCase):
    def test_distros(self):
        df = CompressedChartCloseDataset().df
        print(f'Count by bin: {df["Label"].value_counts()}')
        print(f'Distributtion %: {(df["Label"].value_counts() / df["Label"].value_counts().sum() * 100).round()}')
            
    def test_analysis(self):
        state_path = './models/resnet110.colab.pth'
        model = resnet110()
        provider = CompressedChartCloseProvider()
        csv_path = './models/resent110.colab.csv'        
        metrics = Metrics.map_predictions(model, state_path, provider, csv_path)
        self.assertIsNotNone(metrics)
        
    def test_metrics(self):
        metrics = Metrics.from_csv('./models/resent110.csv')
        print(f'accuracy: {metrics.accuracy}')
        print(f'per class accuracy: {metrics.per_class_accuracy}')
        print(f'per class instances: {metrics.per_class_instances}')
        print(f'TP: {metrics.tp}')
        