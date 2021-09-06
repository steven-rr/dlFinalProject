import constants
import logging
import os

class LoggerFactory:
    LOGGERS = {}
    '''Utility class for initialzing loggers.'''
    @staticmethod
    def create_logger(name, file_name=None):
        '''
        Creates a logger instance that logs to the console and optionally a file.
        
        Args:
            name: the logger name.
            file_name: if provided, a file handler will be aggregated on the logger and written to the file_name.
            
        Returns:
            The logger.
        '''
        key = f'{name}{"" if file_name is None else file_name}'
        if key in LoggerFactory.LOGGERS.keys():
            return LoggerFactory.LOGGERS[key]
        
        logger = LoggerFactory.LOGGERS[key] = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        if file_name is not None:
            file_handler = logging.FileHandler(LoggerFactory.get_log_path(file_name))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
                    
        return logger
    
    @staticmethod
    def get_log_path(file_name):
        if not os.path.exists(constants.LOGS_DIR):
            os.makedirs(constants.LOGS_DIR)
        return os.path.join(constants.LOGS_DIR, file_name)