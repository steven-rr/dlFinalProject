import unittest

from GridSearch import GridSearch

class GridSearchTest(unittest.TestCase):
    def test_dfs(self):
        print()
        parms = {'a':[0,1,2], 'b':[3,4], 'c':[5,6,7,8]}
        target = GridSearch(lambda x: self.assertEqual(3, len(x)))
        expected = [{'a':0,'b':3,'c':5}, {'a':0,'b':3,'c':6}, {'a':0,'b':3,'c':7}, {'a':0,'b':3,'c':8},
                    {'a':0,'b':4,'c':5}, {'a':0,'b':4,'c':6}, {'a':0,'b':4,'c':7}, {'a':0,'b':4,'c':8},
                    {'a':1,'b':3,'c':5}, {'a':1,'b':3,'c':6}, {'a':1,'b':3,'c':7}, {'a':1,'b':3,'c':8},
                    {'a':1,'b':4,'c':5}, {'a':1,'b':4,'c':6}, {'a':1,'b':4,'c':7}, {'a':1,'b':4,'c':8},
                    {'a':2,'b':3,'c':5}, {'a':2,'b':3,'c':6}, {'a':2,'b':3,'c':7}, {'a':2,'b':3,'c':8},
                    {'a':2,'b':4,'c':5}, {'a':2,'b':4,'c':6}, {'a':2,'b':4,'c':7}, {'a':2,'b':4,'c':8},]
        
        actual = target(parms) 
        
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertDictEqual(expected[i], actual[i])
        
        