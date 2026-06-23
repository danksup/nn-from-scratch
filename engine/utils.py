from engine.backend import Backend
nx = Backend()

def dot(u:list,v:list) -> float:
    """
    dot product
    """
    return nx.dot(u,v)
