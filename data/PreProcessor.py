import os
import constants
import numpy as np
import pandas as pd
from datetime import datetime

class PreProcessor:    
    '''Handles parsing metadata from file names and presenting chart files in an enumerable manner.'''
    def __init__(self, thresholds=None, charts_dir=None):
        '''Initializes a PreProcessor, and checks for data.'''
        self.charts_dir = charts_dir or constants.CHARTS_DIR
        assert os.path.exists(self.charts_dir), f'Make sure to download the chart data.'
        thresholds = thresholds or constants.LABEL_BIN_THRESHOLDS
        self.__thresholds = np.array(thresholds)
    
    def parse_file_names(self):
        '''
        Parses the metadata encoded in the chart file names.
        
        Returns:
            A DataFrame containing (GIC, Date, Label, Return, Filename).
        '''
        data = [self.parse_file_name(name) for name in os.listdir(self.charts_dir) if name.endswith(constants.CHART_FILE_EXT)]
        columns = ['GIC', 'Date', 'Label', 'Return', 'Filename']
        df = pd.DataFrame(data, columns=columns)
        return df.set_index(columns[-1])
    
    def parse_file_name(self, name):
        '''
        Parses the metadata encoded in a chart file name.
        
        Args:
            name: The name of the file on disk.
            
        Returns:
            A Tuple containing (GIC, Date, Label, Return, Filename).
        '''
        parts = name[:-constants.CHART_FILE_EXT_LEN].split('_')
        assert 3 == len(parts), f'{name} cannot be parsed.'
        return int(parts[0]), self.parse_date(parts[1]), self.parse_bin(parts[2]), float(parts[2])/100., name
    
    def parse_date(self, part):
        '''
        Parses the date component from a file name.
        
        Args:
            part: A string representing the date as milliseconds from 1/1/1970 0:00:00.000
        
        Returns:
            A datetime.date 
        '''
        ms = int(part)
        return datetime.utcfromtimestamp(ms//1000).date()
    
    def parse_bin(self, part):
        '''
        Parses the return and bins it into a discrete label. 
        
        Args:
            part: A string representing the 5-day percentage return
        
        Returns:
            A discrete integer label for the return.
        '''
        value = np.float(part)
        return self.__thresholds.size if value >= self.__thresholds.max() else np.argmax(value<self.__thresholds)
    
    