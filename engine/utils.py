def dot(u:list,v:list) -> float:
    """
    dot product
    """
    total = 0
    for i in range(len(u)):
        total += u[i] * v[i]

    return total

def diff(func,x,h=1e-5) -> float:
    """differentiate"""
    return (-func(x + 2 *h) + 8 * func(x + h) - 8* func(x - h) + func(x - 2*h))/(12*h)