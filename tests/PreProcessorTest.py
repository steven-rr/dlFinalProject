from data.PreProcessor import PreProcessor
from datetime import datetime
import unittest
import constants

class PreProcessorTest(unittest.TestCase):
    MS_DAY = 1000 * 60 * 60 * 24
    def test_parse_bin(self):    
        target = PreProcessor()
        expected = 3
        actual = target.parse_bin('0.0')
        self.assertEqual(expected, actual)
        
        expected = 2
        actual = target.parse_bin(str(constants.LABEL_BIN_THRESHOLDS[expected-1]+.001))
        self.assertEqual(expected, actual)
                
        expected = 0
        actual = target.parse_bin(str(min(constants.LABEL_BIN_THRESHOLDS) - .1))
        self.assertEqual(expected, actual)
        
        expected = len(constants.LABEL_BIN_THRESHOLDS)
        actual = target.parse_bin(str(max(constants.LABEL_BIN_THRESHOLDS) + 1.1))
        self.assertEqual(expected, actual)
    def test_parse_date(self):
        target = PreProcessor()
        expected = datetime.strptime('1/1/1970', '%m/%d/%Y').date()
        actual = target.parse_date('0')
        self.assertTrue(expected == actual, f'{expected} != {actual}')
        
        expected = datetime.strptime('1/3/1970', '%m/%d/%Y').date()
        actual = target.parse_date(str(self.MS_DAY*2))
        self.assertTrue(expected == actual, f'{expected} != {actual}')
        
        actual = target.parse_date(str(self.MS_DAY*3-1))
        self.assertTrue(expected == actual, f'{expected} != {actual}')
        
    def test_parse_file_name(self):
        target = PreProcessor()
        egic, edate, eret = 175, datetime.strptime('1/3/1970', '%m/%d/%Y').date(), 0.0355
        
        expected = f'{egic}_{self.MS_DAY*2+1}_{eret*100.}.png'
        
        agic, adate, abin, aret, actual = target.parse_file_name(expected)
        
        self.assertEqual(expected, actual)
        self.assertEqual(egic, agic)
        self.assertTrue(edate == adate)
        self.assertEqual(5, abin)
        self.assertEqual(eret, aret)
        
    def test_parse_file_names(self):
        target = PreProcessor()
        actual = target.parse_file_names()
        print(actual)
        print(actual.index[1])
        
        
        
if __name__ == '__main__':
    unittest.main()