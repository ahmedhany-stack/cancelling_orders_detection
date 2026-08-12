import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f} sec")
        return result
    return wrapper