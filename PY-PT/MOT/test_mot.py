
import sys
sys.path.insert(1, '..\\')
from prop import prop
import numpy as np
from mot import chamber, nozzle, hybrid_motor
psi2pa = 6894.76

# U.O.S. : SI units (m, kg, Pa, N)

# CEA fit coefficients
cdat = [1043.511038347968, 191.34119682349473, 21.814788271578433, -18.003201910222227, -6.777365542327267, -6.214294253776049, 0.9098598428234226, 1.7249665520956765, -0.372633420103326, 1.4156062321896583, -0.1624509950765578, -0.0840924417308857, -0.03655490987557908, 0.06440159484295371, -0.12010608442344983, 0.01099791268453032]

# Tank object ( volume, ullage, initial pressure )
tank = prop.tank()(volume=0.028, ullage=0.15, pressure=620*psi2pa)

# Injector object ( discharge area, discharge coefficient )
inj = prop.injector()( area=108*np.pi/4*(0.0016**2), cd=0.65 )

# Fuel (a, n, density, CEA coefficients)
fuel = {'a':0.00016,'n':0.5, 'density':800, 'c':cdat}

# Chamber object ( initial port radius, final port radius, grain length, fuel, C* efficiency )
chamb = chamber( ri=0.04093061, re=0.0631825, L=0.441325, fuel=fuel, eff=0.85 )

# Nozzle object ( throat area, expansion ratio )
noz = nozzle( at=np.pi/4*(0.04**2), ex=3.5)

# Hybrid motor object
mot = hybrid_motor(tank, inj, chamb, noz)

# Define time step and back pressure
mot.set_sim_properties(dt=0.01, pb=100e3)

# Reset motor values, must be done before simulation - burn()
mot.reset()

# Run motor burn simulation, print progress bar and total impulse
mot.burn(prints=True)

# Plot simulation results
mot.print()
mot.plot()
