import numpy as np

# Fundamental constants
G = 6.67430e-11     # m^3 kg^-1 s^-2
Msun = 1.98847e30   # kg
Mearth = 5.97219e24 # kg
Mjup = 1.8981e27    # kg
AU = 149597870700.  # m
day = 86400         # s
c = 299792458.      # m s^-2

# Useful quantities
c2 = c*c
arcsec = 2*np.pi/(360*3600)
mercury_precession_GR = 42.9799/100. # arcsec/century
mercury_eccentricity = 0.20563069

# Solar system masses [kg] - from https://solarsystem.nasa.gov/planet-compare/
masses = {
    'sun'     : Msun,
    'earth'   : Mearth,
    'moon'    : 7.3476e22,
    'mercury' : 3.3011e23,
    'venus'   : 4.8673e24,
    'mars'    : 6.4169e23,
    'jupiter' : Mjup,
    'saturn'  : 5.68232e26,
    'uranus'  : 8.6810e25,
    'neptune' : 1.0241e26,
    'pluto'   : 1.303e22,
}
