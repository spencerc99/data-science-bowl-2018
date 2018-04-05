import numpy as np

class BasicModel():
    def __init__(self, config):
        self.model_name = config['model_name']
        np.random.seed(config['random_seed'])        
        self.max_iter = config['max_iter']
    
    def train(self):
        raise Exception("train method must be implemented by model subclass")
    
    def init(self):
        raise Exception("init method must be implemented by model subclass")
    
    def predict(self):
        raise Exception("predict method must be implemented by model subclass")