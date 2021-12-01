import numpy as np

from optparse import OptionParser

from numba import jit
from tqdm import tqdm

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy.constants import M_earth, M_sun, au, M_jup

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib import rcParams
from distutils.spawn import find_executable

if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

G = 6.67e-11
Msun = 2e30
AU = 1.5e11
day = 86400
c = 3e8
c2 = c*c

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
                g_p[3*i:3*(i+1)] += gradient_1pn_p(mi, mj, dr, pi, pj, r)/c2
                g_p[3*j:3*(j+1)] += gradient_1pn_p(mj, mi, dr, pj, pi, r)/c2
                g_q[3*i:3*(i+1)] += gradient_1pn_q(mi, mj, dr, r, r2, pi, pj, qi, qj)/c2
                g_q[3*j:3*(j+1)] += gradient_1pn_q(mj, mi, dr, r, r2, pj, pi, qj, qi)/c2
    return g_q, g_p


@jit
def gradient_1pn_q(m1, m2, dr, r, r2, p1, p2, q1, q2):
     return 0.25*G**2*m1*m2*(m1 + m2)*2*(q2-q1)/(r2)**2 + 0.125*G*m1*m2*(q2-q1)*((14*np.dot(p1,p2))/(m1*m2) + 2*(np.dot(dr,p1))*(np.dot(dr,p2))/(m1*m2*r2 + (-12*np.sum(p1**2)))/m1**2)/(r**3) + 0.125*G*m1*m2*(2*p1*(np.dot(p2,dr))/(m1*m2*r2) + 2*p2*(np.dot(p1,dr))/(m1*m2*r2) + 4*(q2-q1)*(np.dot(p1,dr))*(np.dot(p2,dr))/(m1*m2*r2**2))/r
    
@jit
def gradient_1pn_p(m1, m2, dr, p1, p2, r):
    return 0.125*G*m1*m2*(14*p2/(m1*m2) + 2*dr*(np.dot(dr, p2))/(m1*m2*r) - 24*p1/m1**2)/r - 0.5*p1*(np.sum(p1**2))/m1**3

@jit
def one_step(q, p, dt, m, cn_order, PN_order):

    dt2 = dt/2.
    mid_q = q
    mid_p = p
    
    for _ in range(cn_order):
        g_q, g_p = gradient(mid_q, mid_p, m, PN_order)
        
        new_q = q + g_p*dt2
        new_p = p - g_q*dt2
    
        mid_q = (q + new_q)/2.
        mid_p = (p + new_p)/2.

    return new_q, new_p

def run(nsteps, dt, q0, p0, m, cn_order, PN_order):
    
    q = q0
    p = p0
    
    solution = np.empty(nsteps, dtype = np.ndarray)
    H        = np.empty(nsteps, dtype = np.ndarray)
    V        = np.empty(nsteps, dtype = np.ndarray)
    T        = np.empty(nsteps, dtype = np.ndarray)
    L        = np.empty(nsteps, dtype = np.ndarray)
    
    solution[0]      = q
    H[0], V[0], T[0] = hamiltonian(q, p, m, PN_order)
    L[0]             = angular_momentum(q, p)
    
    for i in tqdm(range(1,nsteps)):
        q, p             = one_step(q, p, dt, m, cn_order, PN_order)
        solution[i]      = q
        L[i]             = angular_momentum(q, p)
        H[i], V[i], T[i] = hamiltonian(q, p, m, PN_order)
    
    return solution, H, V, T, L

@jit
def distance(v1, v2):
    d = np.zeros(len(v1))
    for i, (a,b) in enumerate(zip(v1, v2)):
        d[i] = np.sqrt(np.sum((a-b)**2))
    return d

def plot_solutions(solutions):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(solutions))))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for q in solutions:
        q = np.array(q)
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5)

    f.savefig('./n_bodies.pdf', bbox_inches = 'tight')
    
def plot_hamiltonian(t, H, V, T):

    fig, (ax, e) = plt.subplots(2,1, sharex = True)
    fig.subplots_adjust(hspace=.0)
    
    ax.plot(t, H, lw = 0.5, label = '$H$')
    e.plot(t, T - np.mean(T), lw = 0.5, color = 'g', label = '$T$')
    e.plot(t, V - np.mean(V), lw = 0.5, color = 'r', label = '$V$')
    ax.plot(t, np.ones(len(H))*H[0], lw = 0.5, ls = '--', color = 'k', label = '$H(0)$')
    
    e.set_ylabel('$E(t)$')
    e.set_xlabel('$t\ [yr]$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    e.grid(True,dashes=(1,3))
    e.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig('./kepler_hamiltonian_nbodies.pdf', bbox_inches = 'tight')

def plot_angular_momentum(t, L):
    
    fig, ax = plt.subplots()
    
    ax.plot(t, L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig('./angular_momentum_nbodies.pdf', bbox_inches = 'tight')
if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('--years', default = 1, type = 'float', help = "Number of years")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
    parser.add_option('--cn_order', default = 7, type = 'int', help = "Crank-Nicolson integrator order")
    parser.add_option('--dt', default = 1, type = 'int', help = "Number of seconds for each dt")
    parser.add_option('-p', dest = "postprocessing", default = False, action = 'store_true', help = "Postprocessing")
    parser.add_option('--PN', dest = "PN", type = 'float', default = 0, help = "Post-Newtonian order")

    (opts,args) = parser.parse_args()
    
    t = Time(datetime.now())

    m = np.array([1*Msun, (0.055*M_earth/M_sun).value*Msun])
    planet_names = ['sun', 'mercury']
    
    planets = np.array([get_body_barycentric_posvel(planet, t) for planet in planet_names])
    
    # Initial conditions
    q0 = np.concatenate([np.array([float(planet[0].x.value*AU), float(planet[0].y.value*AU), float(planet[0].z.value*AU)]) for planet in planets])
    v0 = np.concatenate([np.array([float(planet[1].x.value*AU/day), float(planet[1].y.value*AU/day), float(planet[1].z.value*AU/day)]) for planet in planets])
    
    if opts.cm:
        v_cm = np.sum([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])/np.sum(m)
        for i in range(len(m)):
            v0[3*i:3*(i+1)] -= v_cm
        
    p0 = np.concatenate([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])

    # Integrator setting
    n_years = opts.years
    nsteps = int(365*2*n_years*day/opts.dt)
    dt = opts.dt
    
    cn_order = int(opts.cn_order)
    PN_order = opts.PN
    
    if not opts.postprocessing:
        s, H, V, T, L = run(nsteps, dt, q0, p0, m, cn_order, PN_order)

        x = np.array([[si[3*i:3*(i+1)] for si in s] for i in range(len(m))])

        t = np.arange(x.shape[1])*dt
        
        #np.savetxt('./orbit_nbodies.txt', np.array([t, x1[:,0], x1[:,1], x1[:,2], x2[:,0], x2[:,1], x2[:,2], H, V, T, L]).T, header = 't x1x x1y x1z x2x x2y x2z H V T, L')
    
    else:
        sol = np.genfromtxt('./orbit_nbodies.txt', names = True)
        
        t  = sol['t']
        
        x1 = np.array([sol['x1x'], sol['x1y'], sol['x1z']]).T
        x2 = np.array([sol['x2x'], sol['x2y'], sol['x2z']]).T
        
        H  = sol['H']
        V  = sol['V']
        T  = sol['T']
        L  = sol['L']
    
    
    plot_solutions(x)
    plot_hamiltonian(t/(2*365*day), H, V, T)
    plot_angular_momentum(t/(2*365*day), L)
    
    
