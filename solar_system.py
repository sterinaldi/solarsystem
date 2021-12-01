import numpy as np

from optparse import OptionParser

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy.constants import M_earth, M_sun, au, M_jup

from kepler_nbodies import run

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib import rcParams
from distutils.spawn import find_executable

from constants import G, Msun, AU, day

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

def plot_solutions(solutions, planet_names):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(planet_names))))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for planet in planet_names:
        q = solutions[planet]
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5, label = planet)
    
    ax.legend(loc=0,fontsize=10, bbox_to_anchor=(1.5, 1))

    f.savefig('./plot_solar_system.pdf', bbox_inches = 'tight')
    
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
    
    fig.savefig('./H_solar_system.pdf', bbox_inches = 'tight')

def plot_angular_momentum(t, L):
    
    fig, ax = plt.subplots()
    
    ax.plot(t, L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig('./L_solar_system.pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--years', default = 1, type = 'float', help = "Number of years")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
    parser.add_option('--cn_order', default = 7, type = 'int', help = "Crank-Nicolson integrator order")
    parser.add_option('--dt', default = 10, type = 'int', help = "Number of seconds for each dt")
    parser.add_option('-p', dest = "postprocessing", default = False, action = 'store_true', help = "Postprocessing")
    parser.add_option('--geo', dest = "geocentric", default = False, action = 'store_true', help = "Make plots in geocentric reference frame")

    (opts,args) = parser.parse_args()

    t = Time(datetime.now())#'2021-06-21T00:00:00')

    m = np.array([1*Msun, (M_earth/M_sun).value*Msun, (M_jup/M_sun).value*Msun, (0.055*M_earth/M_sun).value*Msun, (0.107*M_earth/M_sun).value*Msun, (0.815*M_earth/M_sun).value*Msun, (95.16*M_earth/M_sun).value*Msun, (14.54*M_earth/M_sun).value*Msun, (17.15*M_earth/M_sun).value*Msun])
    
    planet_names = ['sun', 'earth', 'jupiter', 'mercury', 'mars', 'venus', 'saturn', 'uranus', 'neptune']
    planets_to_plot = None

    planets = np.array([get_body_barycentric_posvel(planet, t) for planet in planet_names])

    # Initial conditions
    q0 = np.concatenate([np.array([float(planet[0].x.value*AU), float(planet[0].y.value*AU), float(planet[0].z.value*AU)]) for planet in planets])
    v0 = np.concatenate([np.array([float(planet[1].x.value*AU/day), float(planet[1].y.value*AU/day), float(planet[1].z.value*AU/day)]) for planet in planets])

    if opts.cm:
        v_cm = np.sum([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])/np.sum(m)
        for i in range(len(m)):
            v0[3*i:3*(i+1)] -= v_cm
        
    p0 = np.concatenate([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])

    # Integrator settings
    n_years = opts.years
    nsteps = int(365*2*n_years*day/opts.dt)
    dt = opts.dt

    order = int(opts.cn_order)

    if not opts.postprocessing:
        s, H, V, T, L = run(nsteps, dt, q0, p0, m, order)
        x = {planet: np.array([si[3*i:3*(i+1)] for si in s]) for i, planet in enumerate(planet_names)}
        t = np.arange(len(x[planet_names[0]][:,0]))*dt

        hdr = ' '.join(np.array([['q'+ str(planet) + coord for coord in ['x','y','z']] for planet in planet_names]).flatten()) + ' t H V T L'
        np.savetxt('./orbits.txt', np.array([xi for planet in planet_names for xi in x[planet].T] + [t, H, V, T, L]).T, header = hdr)
    else:
        sol = np.genfromtxt('./orbits.txt', names = True)
        
        t  = sol['t']
        
        x = {planet:np.array([sol['q' + str(planet) + 'x'], sol['q' + str(planet) + 'y'], sol['q' + str(planet) + 'z']]).T for planet in planet_names}
        
        H  = sol['H']
        V  = sol['V']
        T  = sol['T']
        L  = sol['L']

    if opts.geocentric:
        for planet in planet_names:
            x[planet] -= x['earth']
    
    if planets_to_plot == None:
        planets_to_plot = planet_names

    plot_solutions(x, planets_to_plot)
    plot_hamiltonian(t/(2*365*day), H, V, T)
    plot_angular_momentum(t/(2*365*day), L)
        
        
