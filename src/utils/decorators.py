from functools import wraps
import time

def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
    
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        
        print(f'"{my_func.__name__}" took {(tend - tstart) * 1000} ms to execute\n')
        return output
    return timed