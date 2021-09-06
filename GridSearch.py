class GridSearch:
    def __init__(self, eval_func):
        self.__eval_func = eval_func
    
    def __call__(self, parms):
        outputs = []
        keys = parms.keys()
        self.grid_dfs({}, keys, parms, outputs)
        
        return outputs
    
    def grid_dfs(self, kwargs, keys, parms, outputs):
        if len(keys) < 1:
            self.__eval_func(kwargs)
            outputs.append(dict(kwargs))
            return
        
        keys = list(keys)         
        key = keys.pop(0)
        values = parms[key]
        assert len(values) > 0, f'The parmeter "{key}" contains no elements.'
        
        for value in values:      
            kwargs[key] = value
            self.grid_dfs(kwargs, keys, parms, outputs)