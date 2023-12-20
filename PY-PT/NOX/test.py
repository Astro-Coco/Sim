from inj import inject
import numpy as np
import matplotlib.pyplot as plt

Pin = 2.2e6
xin = 0.1

deltaP = np.linspace(1e4, 5e5, 25)
G = deltaP*0

for k in range(len(deltaP)):
    Pout = Pin - deltaP[k]

    Gi, P, Psat, x = inject(
        Pin=Pin,
        xin=0.1,
        Pout=Pout,
    )

    G[k] = Gi

fig, ax = plt.subplots()
ax.plot(deltaP/1e6, G, '-k')
ax.set_xlabel('$\Delta P$ $(MPa)$')
ax.set_ylabel('$G_{ox}$ $(kg/m^2s)$')
fig.set_size_inches(4.5, 4)

plt.show()
