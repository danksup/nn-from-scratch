import numpy as np

def dot(u:list,v:list) -> float:
    """
    dot product
    """
    
    return np.dot(u,v)

def diff(func,x,h=1e-5) -> float:
    """differentiate"""
    return (-func(x + 2 *h) + 8 * func(x + h) - 8* func(x - h) + func(x - 2*h))/(12*h)