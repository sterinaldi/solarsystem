import numpy as np

from solarsystem.constants import *
from solarsystem.kepler import eccentricity_vector

from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time

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

def plot_solutions(times, solutions, bodies, folder, check_astropy = False):
    if check_astropy:
        astropy_q, astropy_p = get_astropy_posvel(bodies, times)

    colors = iter(cm.rainbow(np.linspace(0, 1, len(bodies))))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for body in bodies:
        q = np.array(solutions[body])
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5, alpha = 0.5)
        if check_astropy:
            ap_q = np.array(astropy_q[body])
            ax.plot(ap_q[:,0]/AU, ap_q[:,1]/AU, ap_q[:,2]/AU, color=c, ls = '--', lw=0.5)

    f.savefig(Path(folder,'trajectories.pdf'), bbox_inches = 'tight')
    return
    
def plot_hamiltonian(t, H, V, T, folder):

    fig, (ax, e) = plt.subplots(2,1, sharex = True)
    fig.subplots_adjust(hspace=.0)
    
    ax.plot((t-t[0])/(365*day), H, lw = 0.5, label = '$H$')
    e.plot((t-t[0])/(365*day), T - np.mean(T), lw = 0.5, color = 'g', label = '$T$')
    e.plot((t-t[0])/(365*day), V - np.mean(V), lw = 0.5, color = 'r', label = '$V$')
    ax.plot((t-t[0])/(365*day), np.ones(len(H))*H[0], lw = 0.5, ls = '--', color = 'k', label = '$H(0)$')
    
    e.set_ylabel('$E(t)$')
    e.set_xlabel('$t\ [\mathrm{yr}]$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    e.grid(True,dashes=(1,3))
    e.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig(Path(folder, 'hamiltonian.pdf'), bbox_inches = 'tight')
    return

def plot_angular_momentum(t, L, folder):
    
    fig, ax = plt.subplots()
    
    ax.plot((t-t[0])/(365*day), L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [\mathrm{yr}]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig(Path(folder, 'angular_momentum.pdf'), bbox_inches = 'tight')
    return
    
def plot_precession(t, omega, folder, pn = 0):
    k = 0.
    if pn:
        k = 1.
    fig, ax = plt.subplots()
    ax.plot((t-t[0])/(365*day), omega/arcsec, lw = 0.5, label = '$Reconstructed$')
    ax.plot((t-t[0])/(365*day), k*((t-t[0])/(365*day))*mercury_precession_GR, lw = 0.5, ls = '--', color = 'r', label = '$Expected$')

        
    ax.set_ylabel('$|\\omega(t) - \\omega(0)|\ [arcsec]$')
    ax.set_xlabel('$t\ [\mathrm{yr}]$')
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig(Path(folder, 'perihelion_precession.pdf'), bbox_inches = 'tight')
    return

def plot_eccentricity_vector(t, q, p, m1, m2, folder):
    f = plt.figure(figsize=(10,7))
    ax = f.add_subplot(111, projection = '3d')
    E = np.array([eccentricity_vector(qi, pi/m1, m1, m2) for qi, pi in zip(q,p)])
    E_modules = np.linalg.norm(E, axis = 1)
    ax.plot(E[:,0], E[:,1], E[:,2], lw=0.5)
    f.tight_layout()
    f.savefig(Path(folder,'eccentricity_3d.pdf'), bbox_inches = 'tight')
    
    f, axes = plt.subplots(4, 1, sharex = True, figsize = (10,6))
    f.subplots_adjust(hspace=.0)
    
    lab = ['$e_x$', '$e_y$', '$e_z$']
    
    for comp, ax in enumerate(axes[:-1]):
        ax.plot((t-t[0])/(365*day), E[:,comp], lw = 0.5)
        ax.set_ylabel(lab[comp])
        ax.grid(True,dashes=(1,3))
    
    axes[-1].plot((t-t[0])/(365*day), E_modules, lw = 0.5)
    axes[-1].set_ylabel('$|e|$')
    axes[-1].grid(True, dashes=(1,3))
    axes[-1].set_xlabel('$t\ [\mathrm{yr}]$')
    
    f.savefig(Path(folder,'eccentricity_components.pdf'), bbox_inches = 'tight')
    return

def plot_difference_astropy(times, x_q, x_p, planet_names, folder):
    astropy_q, astropy_p = get_astropy_posvel(planet_names, times)
    
    colors = iter(cm.rainbow(np.linspace(0, 1, len(planet_names))))
    fig, (ax_q, ax_p) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize=(6,4))
    
    for planet in planet_names:
        delta_q = np.sqrt(np.sum((x_q[planet]/AU - astropy_q[planet]/AU)**2, axis = -1))
        delta_p = np.sqrt(np.sum((x_p[planet]/(AU*masses[planet])*day - astropy_p[planet]/(AU*masses[planet])*day)**2, axis = -1))
        
        c = next(colors)
        ax_q.plot((times-times[0])/(365*day), delta_q, color = c, lw = 0.5, label = '$\\mathrm{'+planet+'}$')
        ax_p.plot((times-times[0])/(365*day), delta_p, color = c, lw = 0.5, label = '$\\mathrm{'+planet+'}$')
    
    ax_p.set_xlabel('$t\ [\mathrm{yr}]$')
    ax_q.set_ylabel('$\\Delta q(t)\ [\mathrm{AU}]$')
    ax_p.set_ylabel('$\\Delta p(t)\ [\mathrm{AU}/\mathrm{day}]$')
    
    ax_q.legend(loc = 0, frameon = False)
    ax_q.grid(visible = True, dashes=(1,3))
    ax_p.grid(visible = True, dashes=(1,3))
    
    fig.savefig(Path(folder,'astropy_difference.pdf'), bbox_inches = 'tight')
    return

def save_solution(q, p, H, V, T, L, planet_names, folder, dt, dsp, t0):
    print('Saving solution...')

    x_q = {planet: np.array([si[3*i:3*(i+1)] for si in q]) for i, planet in enumerate(planet_names)}
    x_p = {planet: np.array([si[3*i:3*(i+1)] for si in p]) for i, planet in enumerate(planet_names)}

    t = np.arange(len(x_q[planet_names[0]][:,0]))*float(dt*dsp) + t0

    hdr = ' '.join(np.array([['q_'+ str(planet) + coord for coord in ['_x','_y','_z']] for planet in planet_names]).flatten()) + ' ' + ' '.join(np.array([['p_'+ str(planet) + coord for coord in ['_x','_y','_z']] for planet in planet_names]).flatten()) + ' t H V T L'

    np.savetxt(Path(folder, 'solutions.txt'), np.array([xi for planet in planet_names for xi in x_q[planet].T] + [xi for planet in planet_names for xi in x_p[planet].T] + [t, H, V, T, L]).T, header = hdr)
    print('Saved!')
    return

def load_solution(folder, planet_names):
    print('Loading solution...')
    sol = np.genfromtxt(Path(folder, 'solutions.txt'), names = True)

    t  = sol['t']
    
    x_q = {planet:np.ascontiguousarray(np.array([sol['q_' + str(planet) + '_x'], sol['q_' + str(planet) + '_y'], sol['q_' + str(planet) + '_z']]).T) for planet in planet_names}
    x_p = {planet:np.ascontiguousarray(np.array([sol['p_' + str(planet) + '_x'], sol['p_' + str(planet) + '_y'], sol['p_' + str(planet) + '_z']]).T) for planet in planet_names}
    
    H  = sol['H']
    V  = sol['V']
    T  = sol['T']
    L  = sol['L']
    
    return t, x_q, x_p, H, V, T, L

def get_astropy_posvel(bodies, times):
    t = Time(times, format = 'gps')
    x_q = {}
    x_p = {}
    
    for body in bodies:
        posvel = get_body_barycentric_posvel(body, t)
        x_q[body] = np.ascontiguousarray(np.array([posvel[0].x.value*AU, posvel[0].y.value*AU, posvel[0].z.value*AU]).T)
        x_p[body] = np.ascontiguousarray(np.array([masses[body]*posvel[1].x.value*AU/day, masses[body]*posvel[1].y.value*AU/day, masses[body]*posvel[1].z.value*AU/day]).T)

    return x_q, x_p
