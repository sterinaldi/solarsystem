G = 6.67e-11       # m^3 kg^-1 s^-2
Msun = 1.988e30    # kg
AU = 149597870700. # m
day = 86400        # s
Mearth = 5.972e24  # kg
c = 299792458.     # m s^-2
c2 = c*c

# Solar system masses
masses = {
    'sun'     : Msun,
    'earth'   : Mearth,
    'moon'    : 0.0123*Mearth,
    'mercury' : 0.0553*Mearth,
    'mars'    : 0.1075*Mearth,
    'venus'   : 0.815*Mearth,
    'jupiter' : 317.8*Mearth,
    'saturn'  : 95.2*Mearth,
    'uranus'  : 14.6*Mearth,
    'neptune' : 17.2*Mearth,
    'pluto'   : 0.00218*Mearth,
}
