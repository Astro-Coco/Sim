from pyCEA import cea
import matplotlib.pyplot as plt
import numpy as np


# CEA example, plot paraffin-n2o isp vs o/f curve at 400 psi chamber pressure
T = 298.15
oxids = [
    {'name':'N2O', 'wt':100, 'T':270}
]
fuels = [
    {'name':'paraffin', 'wt':50, 'T':T},
    {'name':'vybar', 'wt':50, 'h':-1494., 'T':T, 'comp':'C 300 H 600'}
]
inpts = {
         'P_CC':200,
         'P_EXT':14.6,
         'OF':7,
         'fuels':fuels,
         'oxidizers':oxids,
        }


vals = np.linspace(1, 12, 50)


m = []
for val in vals:
    inpts['OF'] = val
    res = cea(inpts)
    m.append(res['t'][0])

m = np.array(m)


inpts['P_CC'] = 300

m2 = []
for val in vals:
    inpts['OF'] = val
    res = cea(inpts)
    m2.append(res['t'][0])

m2 = np.array(m2)

inpts['P_CC'] = 400

m3 = []
for val in vals:
    inpts['OF'] = val
    res = cea(inpts)
    m3.append(res['t'][0])

m3 = np.array(m3)


ax = plt.gca()
ax.tick_params(direction='in')
ax.set_xlabel('O/F')
ax.set_ylabel('T (K)')
ax.plot(vals, m, '-k')
ax.plot(vals, m2, '--k')
ax.plot(vals, m3, '-.k')
ax.legend(['Pc = 200 PSIA','Pc = 300 PSIA','Pc = 400 PSIA'])
plt.show()

