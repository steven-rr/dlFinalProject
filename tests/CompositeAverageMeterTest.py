import unittest

from utils.CompositeAverageMeter import CompositeAverageMeter
from utils.MockLogger import MockLogger

class CompositeAverageMeterTest(unittest.TestCase):            
    def test_iteration_time(self):
        target = CompositeAverageMeter(MockLogger(lambda m: self.assertEqual(m, m) ))
        actual = target.iteration_time
        self.assertIsNotNone(actual)
        self.assertEqual('Time', actual.label)
        
    def test_losses(self):
        target = CompositeAverageMeter(MockLogger(lambda m: self.assertEqual(m, m) ))
        actual = target.losses
        self.assertIsNotNone(actual)
        self.assertEqual('Loss', actual.label)
        
    def test_accuracy(self):
        target = CompositeAverageMeter(MockLogger(lambda m: self.assertEqual(m, m) ))
        actual = target.accuracy
        self.assertIsNotNone(actual)
        self.assertEqual('Top 1', actual.label)
        
    def test_update(self):
        e_val, e_avg, e_sum, e_count = 1, 1, 4, 4
        target = CompositeAverageMeter(MockLogger(lambda m: self.assertEqual(m, m) ))
        
        target.update(1, 1, 2)
        target.update(1, 1, 2)
        
        for actual in [target.losses, target.accuracy]:            
            self.assertEqual(e_val, actual.val)
            self.assertEqual(e_sum, actual.sum)
            self.assertEqual(e_count, actual.count)
            self.assertEqual(e_avg, actual.avg)
            
    def test_log(self):
        target = CompositeAverageMeter(MockLogger(lambda m: self.assertTrue(m.startswith('Epoch: [175][275][375]'))))
        target.log(175, 275, 375)
        
        
        