class MockLogger:
    def __init__(self, callback):
        self.callback = callback
    def info(self, message):
        self.callback(message)
    def debug(self, message):
        self.callback(message)