import unittest

loader = unittest.TestLoader()
suite = loader.discover('./tests', '*test*.py')
unittest.TextTestRunner().run(suite)