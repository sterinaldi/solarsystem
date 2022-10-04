import numpy as np

from optparse import OptionParser
from pathlib import Path

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

from solarsystem.dynamics import run
from solarsystem.constants import *
from solarsystem.postprocessing import *


parser = OptionParser()
parser.add_option('-o', '--output', dest = "outfolder", default = './solution_solar_system', help = "Output folder")
parser.add_option('--years', default = 1, type = 'float', help = "Number of years")
parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
parser.add_option('--cn_order', default = 7, type = 'int', help = "Crank-Nicolson integrator order")
parser.add_option('--dt', default = 10, type = 'int', help = "Number of seconds for each dt")
parser.add_option('-p', dest = "postprocessing", default = False, action = 'store_true', help = "Postprocessing")
parser.add_option('--geo', dest = "geocentric", default = False, action = 'store_true', help = "Make plots in geocentric reference frame")
parser.add_option('--helios', dest = "heliocentric", default = False, action = 'store_true', help = "Make plots in heliocentric reference frame")
parser.add_option('--PN', dest = "PN", default = 0, type = 'int', help = "Post-Newtonian order")
parser.add_option('--dsp', dest = "dsp", default = 1, type = 'int', help = "Interval between saved steps. Default is 1 (all steps)")

(opts,args) = parser.parse_args()
out_folder  = Path(opts.outfolder).absolute()
if not out_folder.exists():
    out_folder.mkdir()

t = Time(datetime.now())
t.format = 'gps'

planet_names = ['sun', 'earth', 'jupiter', 'mercury', 'mars', 'venus', 'saturn', 'uranus', 'neptune']
planets_to_plot = None

m = np.array([masses[planet] for planet in planet_names])

planets = np.array([get_body_barycentric_posvel(planet, t) for planet in planet_names])

# Initial conditions
q0 = np.concatenate([np.array([float(planet[0].x.value*AU), float(planet[0].y.value*AU), float(planet[0].z.value*AU)]) for planet in planets])
v0 = np.concatenate([np.array([float(planet[1].x.value*AU/day), float(planet[1].y.value*AU/day), float(planet[1].z.value*AU/day)]) for planet in planets])

if opts.cm:
    v_cm = np.sum([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))], axis = 0)/np.sum(m)
    for i in range(len(m)):
        v0[3*i:3*(i+1)] -= v_cm
    
p0 = np.concatenate([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])

# Integrator settings
n_years = opts.years
nsteps = int(365*n_years*day/opts.dt)
dt = opts.dt

order = int(opts.cn_order)

if not opts.postprocessing:
    s_q, s_p, H, V, T, L = run(nsteps, dt, q0, p0, m, order, opts.PN, opts.dsp)
    save_solution(s_q, s_p, H, V, T, L, planet_names, out_folder, dt, opts.dsp, t.value)
    
t, x_q, x_p, H, V, T, L = load_solution(out_folder, planet_names)

if opts.geocentric:
    earth_pos = np.copy(x['earth'])
    for planet in planet_names:
        x[planet] -= earth_pos
elif opts.heliocentric:
    sun_pos = np.copy(x['sun'])
    for planet in planet_names:
        x[planet] -= sun_pos

if planets_to_plot == None:
    planets_to_plot = planet_names

plot_solutions(t, x_q, planets_to_plot, out_folder)
plot_hamiltonian(t, H, V, T, out_folder)
plot_angular_momentum(t, L, out_folder)
plot_difference_astropy(t, x_q, x_p, planet_names, out_folder)
