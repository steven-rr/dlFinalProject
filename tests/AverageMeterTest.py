import unittest

from utils.AverageMeter import AverageMeter

class AverageMeterTest(unittest.TestCase):
    def test_update(self):
        e_val, e_avg, e_sum, e_count = 1, 1, 4, 4
        actual = AverageMeter('Test')
        
        actual.update(1,2)
        actual.update(1,2)
        
        self.assertEqual(e_val, actual.val)
        self.assertEqual(e_sum, actual.sum)
        self.assertEqual(e_count, actual.count)
        self.assertEqual(e_avg, actual.avg)
        
        return actual
    
    def test_reset(self):
        actual = self.test_update()
        actual.reset()
        
        e_val, e_avg, e_sum, e_count = 0, 0, 0, 0
        
        self.assertEqual(e_val, actual.val)
        self.assertEqual(e_sum, actual.sum)
        self.assertEqual(e_count, actual.count)
        self.assertEqual(e_avg, actual.avg)
        
    def test_str(self):
        target = self.test_update()
        
        expected = 'Test: 1.0000 (1.0000)'
        actual = str(target)
        
        self.assertEqual(expected, actual)
        
        