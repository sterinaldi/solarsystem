import numpy as np
from numba import jit
from solarsystem.constants import *

@jit
def eccentricity_vector(r, v, m):
    return (np.sum(v**2)/mu - 1/np.linalg.norm(r, axis = 0))*r - np.dot(r, v)*v/(m*G)

@jit
def rotate_around_x(v, phi):
    M = np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
    return M@v

@jit
def rotate_around_y(v, phi):
    M = np.array([[np.cos(phi), 0, -np.sin(phi)], [0, 1, 0], [np.sin(phi), 0, np.cos(phi)]])
    return M@v

@jit
def rotate_around_z(v, theta):
    M = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return M@v

def compute_angle(v1, v2):
    '''
    v1 is the reference vector, v2 can be a list of vectors.
    WARNING: doing the opposite does not raise any warning but provides unreliable results (despite having the correct shape).
    BE CAREFUL.
    '''
    return np.arccos(np.dot(v2, v1)/(np.linalg.norm(v1)*np.linalg.norm(v2, axis = 0)))

def periapsis_precession(q, p, m):
    omega_0 = eccentricity_vector(q[0], p[0]/m, m)
    omega = np.empty(len(q), dtype = np.ndarray)
    for i in range(len(omega)):
        omega[i] = eccentricity_vector(q[i], p[i]/m, m)
    return compute_angle(omega_0, omega)
