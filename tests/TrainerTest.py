from Trainer import Trainer
import unittest
from utils.MockLogger import MockLogger

class TrainerTest(unittest.TestCase):
    
    def test_kwarg_init(self):
        kwargs = {'epochs':275, 'learning_rate':.175}
        target = Trainer(MockLogger(lambda m: print(m)), None, **kwargs)
        
        self.assertEqual(kwargs['epochs'], target.epochs)
        self.assertEqual(kwargs['learning_rate'], target.learning_rate)
        self.assertEqual(5e-5, target.regularization_parameter)
