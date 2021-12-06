import numpy as np

def argument_of_periastron(n, e):
    return np.arccos(np.dot(n, e))/(np.linalg.norm(n)*np.linalg.norm(e))

