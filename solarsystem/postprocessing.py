import numpy as np

from solarsystem.constants import *
from solarsystem.kepler import lenz_vector

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib import rcParams

from pathlib import Path

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

def plot_solutions(solutions, bodies, folder):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(bodies))))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for body in bodies:
        q = np.array(solutions[body])
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5)

    f.savefig(Path(folder,'trajectories.pdf'), bbox_inches = 'tight')
    
def plot_hamiltonian(t, H, V, T, folder):

    fig, (ax, e) = plt.subplots(2,1, sharex = True)
    fig.subplots_adjust(hspace=.0)
    
    ax.plot(t/(365*day), H, lw = 0.5, label = '$H$')
    e.plot(t/(365*day), T - np.mean(T), lw = 0.5, color = 'g', label = '$T$')
    e.plot(t/(365*day), V - np.mean(V), lw = 0.5, color = 'r', label = '$V$')
    ax.plot(t/(365*day), np.ones(len(H))*H[0], lw = 0.5, ls = '--', color = 'k', label = '$H(0)$')
    
    e.set_ylabel('$E(t)$')
    e.set_xlabel('$t\ [yr]$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    e.grid(True,dashes=(1,3))
    e.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig(Path(folder, 'hamiltonian.pdf'), bbox_inches = 'tight')

def plot_angular_momentum(t, L, folder):
    
    fig, ax = plt.subplots()
    
    ax.plot(t/(365*day), L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig(Path(folder, 'angular_momentum.pdf'), bbox_inches = 'tight')
    
def plot_precession(t, omega, folder):
    
    fig, ax = plt.subplots()
    print(omega[0])
    ax.plot(t/(365*day), omega[0] + (t/(365*day))*mercury_precession_GR/100, lw = 0.5, ls = '--', color = 'k', label = 'Expected')
    ax.plot(t/(365*day), omega, lw = 0.5, label = 'Reconstruced')

        
    ax.set_ylabel('$\\omega(t) - \\omega(0)\ [arcsec]$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig(Path(folder, 'perihelion_precession.pdf'), bbox_inches = 'tight')

def plot_lenz_vector(q, p, m1, m2, folder):
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')
    A = lenz_vector(q, p, m1, m2)
    ax.plot(A[:,0], A[:,1], A[:,2], lw=0.5)
    f.savefig(Path(folder,'lenz.pdf'), bbox_inches = 'tight')
    

def save_solution(q, p, H, V, T, L, planet_names, folder, dt):

    x_q = {planet: np.array([si[3*i:3*(i+1)] for si in q]) for i, planet in enumerate(planet_names)}
    x_p = {planet: np.array([si[3*i:3*(i+1)] for si in p]) for i, planet in enumerate(planet_names)}

    t = np.arange(len(x_q[planet_names[0]][:,0]))*float(dt)

    hdr = ' '.join(np.array([['q_'+ str(planet) + coord for coord in ['_x','_y','_z']] for planet in planet_names]).flatten()) + ' ' + ' '.join(np.array([['p_'+ str(planet) + coord for coord in ['_x','_y','_z']] for planet in planet_names]).flatten()) + ' t H V T L'

    np.savetxt(Path(folder, 'solutions.txt'), np.array([xi for planet in planet_names for xi in x_q[planet].T] + [xi for planet in planet_names for xi in x_p[planet].T] + [t, H, V, T, L]).T, header = hdr)

def load_solution(folder, planet_names):

    sol = np.genfromtxt(Path(folder, 'solutions.txt'), names = True)

    t  = sol['t']
    
    x_q = {planet:np.ascontiguousarray(np.array([sol['q_' + str(planet) + '_x'], sol['q_' + str(planet) + '_y'], sol['q_' + str(planet) + '_z']]).T) for planet in planet_names}
    x_p = {planet:np.ascontiguousarray(np.array([sol['p_' + str(planet) + '_x'], sol['p_' + str(planet) + '_y'], sol['p_' + str(planet) + '_z']]).T) for planet in planet_names}
    
    H  = sol['H']
    V  = sol['V']
    T  = sol['T']
    L  = sol['L']
    
    return t, x_q, x_p, H, V, T, L
