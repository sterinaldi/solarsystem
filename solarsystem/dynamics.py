import numpy as np

from numba import jit
from tqdm import tqdm

from solarsystem.constants import *

@jit
def angular_momentum(q, p):
    L = np.zeros(3)
    for i in range(len(q)//3):
        qi = q[3*i:3*(i+1)]
        pi = p[3*i:3*(i+1)]
        L += np.array([(qi[1]*pi[2] - qi[2]*pi[1]), -(qi[0]*pi[2] - qi[2]*pi[0]), (qi[0]*pi[1] - qi[1]*pi[0])])
    return np.sqrt(np.sum(L**2))
    
@jit
def hamiltonian(q, p, m, PN):
    T = 0.
    V = 0.
    for i in range(len(m)):
        mi   = m[i]
        qi   = q[3*i:3*(i+1)]
        pi   = p[3*i:3*(i+1)]
        pi_2 = np.sum(pi**2)
        T   += pi_2/(2*mi)
        for j in range(i+1,len(m)):
            mj = m[j]
            qj = q[3*j:3*(j+1)]
            pj = p[3*j:3*(j+1)]
            dr = qi - qj
            r  = np.sqrt(np.sum(dr**2))
            pidpj  = np.dot(pi, pj)
            ndpi = np.dot(dr, pi)/r
            ndpj = np.dot(dr, pj)/r
            V  += -G*mi*mj/r
            if PN == 1.:
                V += potential_1pn(mi, mj, pi_2, r, pidpj, ndpi, ndpj)/c2
                
    return T + V, V, T

@jit
def potential_1pn(m1, m2, p1_2, r, p1dp2, ndp1, ndp2):
    return -(1./8.)*(p1_2**2)/(m1**3) + (1./8.)*(G*m1*m2/r)*(-12*p1_2/(m1**2) + 14*p1dp2/(m1*m2) + 2*(ndp1*ndp2)/(m1*m2)) + (1./4.)*(G*m1*m2/r)*(G*(m1+m2)/r)

@jit
def gradient(q, p, m, PN):
    g_q = np.zeros(len(q))
    g_p = np.zeros(len(p))
    for i in range(len(m)):
        mi = m[i]
        qi = q[3*i:3*(i+1)]
        pi = p[3*i:3*(i+1)]
        pi_2 = np.sum(pi**2)
        g_p[3*i:3*(i+1)] = pi/mi
        for j in range(i+1, len(m)):
            mj = m[j]
            qj = q[3*j:3*(j+1)]
            pj = p[3*j:3*(j+1)]
            dr = qi - qj
            r  = np.sqrt(np.sum(dr**2))
            r2 = r*r
            K  = G*mi*mj/(r*r*r)
            g_q[3*i:3*(i+1)] += + K*dr
            g_q[3*j:3*(j+1)] += - K*dr
            if PN == 1.:
                g_1pn_q = gradient_1pn_q(mi, mj, dr, r, r2, pi, pj, qi, qj)/c2
                g_1pn_p = gradient_1pn_p(mi, mj, dr, pi, pj, r, r2)/c2
                g_p[3*i:3*(i+1)] += g_1pn_p
                g_p[3*j:3*(j+1)] += g_1pn_p
                g_q[3*i:3*(i+1)] += g_1pn_q
                g_q[3*j:3*(j+1)] += g_1pn_q
    return g_q, g_p


@jit
def gradient_1pn_q(m1, m2, dr, r, r2, p1, p2, q1, q2):
     return 0.25*G**2*m1*m2*(m1 + m2)*2*(q2-q1)/(r2)**2 + 0.125*G*m1*m2*(q2-q1)*(14*np.dot(p1,p2)/(m1*m2) + 2*(np.dot(dr,p1))*(np.dot(dr,p2))/(m1*m2*r2 + (-12*np.sum(p1**2)))/m1**2)/(r**3) + 0.125*G*m1*m2*(2*p1*(np.dot(p2,dr))/(m1*m2*r2) + 2*p2*(np.dot(p1,dr))/(m1*m2*r2) + 4*(q2-q1)*(np.dot(p1,dr))*(np.dot(p2,dr))/(m1*m2*r2**2))/r
    
@jit
def gradient_1pn_p(m1, m2, dr, p1, p2, r, r2):
    return 0.125*G*m1*m2*(14*p2/(m1*m2) + 2*dr*(np.dot(dr, p2))/(m1*m2*r2) - 24*p1/m1**2)/r - 0.5*p1*(np.sum(p1**2))/m1**3

@jit
def one_step(q, p, dt, m, cn_order, PN_order):

    mid_q = q
    mid_p = p
    
    for _ in range(cn_order):
        g_q, g_p = gradient(mid_q, mid_p, m, PN_order)
        
        new_q = q + g_p*dt
        new_p = p - g_q*dt
    
        mid_q = (q + new_q)/2.
        mid_p = (p + new_p)/2.

    return new_q, new_p

def run(nsteps, dt, q0, p0, m, cn_order, PN_order):
    
    q = q0
    p = p0
    
    solution_q = np.empty(nsteps, dtype = np.ndarray)
    solution_p = np.empty(nsteps, dtype = np.ndarray)
    H          = np.empty(nsteps, dtype = np.ndarray)
    V          = np.empty(nsteps, dtype = np.ndarray)
    T          = np.empty(nsteps, dtype = np.ndarray)
    L          = np.empty(nsteps, dtype = np.ndarray)
    
    solution_q[0]    = q
    solution_p[0]    = p
    H[0], V[0], T[0] = hamiltonian(q, p, m, PN_order)
    L[0]             = angular_momentum(q, p)
    
    for i in tqdm(range(1,nsteps)):
        q, p             = one_step(q, p, dt, m, cn_order, PN_order)
        solution_q[i]    = q
        solution_p[i]    = p
        L[i]             = angular_momentum(q, p)
        H[i], V[i], T[i] = hamiltonian(q, p, m, PN_order)
    
    return solution_q, solution_p, H, V, T, L

@jit
def distance(v1, v2):
    d = np.zeros(len(v1))
    for i, (a,b) in enumerate(zip(v1, v2)):
        d[i] = np.sqrt(np.sum((a-b)**2))
    return d
