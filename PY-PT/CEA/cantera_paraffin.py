import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from pyCEA import cea
psi2pa = 6894.76


def CP_CV_EQ(g):
    dT = 1e-3
    dP = 1e-3

    # CP
    H0 = g.enthalpy_mole
    g.TP = g.T+dT, g.P
    g.equilibrate("TP")
    H1 = g.enthalpy_mole
    g.TP = g.T-dT, g.P
    g.equilibrate("TP")
    CP = (H1 - H0)/dT

    # dlogVdlogT_P
    logV0 = np.log(1/g.density_mole)
    logT0 = np.log(g.T)
    g.TP = g.T+dT, g.P
    g.equilibrate("TP")
    logV1 = np.log(1/g.density_mole)
    logT1 = np.log(g.T)
    g.TP = g.T-dT, g.P
    g.equilibrate("TP")
    dlogVdlogT_P = (logV1-logV0)/(logT1-logT0)

    # dlogVdlogP_T
    logV0 = np.log(1/g.density_mole)
    logP0 = np.log(g.P)
    g.TP = g.T, g.P+dP
    g.equilibrate("TP")
    logV1 = np.log(1/g.density_mole)
    logP1 = np.log(g.P)
    g.TP = g.T, g.P-dP
    g.equilibrate("TP")
    dlogVdlogP_T = (logV1-logV0)/(logP1-logP0)

    CV = CP + g.P/(g.density_mole*g.T)*dlogVdlogT_P**2 / dlogVdlogP_T


    return CP*1e-3, CV*1e-3



T0 = 300.
P0 = 25*1e5

# CANTERA SETUP
#gas = ct.Solution("thermo.yaml", "gas")
#gas.TPX = T0, P0, "N2O:1"

#paraffin = ct.Solution("thermo.yaml", "paraffin")
#paraffin.TP = T0, P0

#M_F = paraffin.mean_molecular_weight
#M_OX = gas.mean_molecular_weight

#mix = ct.Mixture([(gas, 1), (paraffin, 1)])
gas = ct.Solution("thermo.yaml", "gas")


# CEA SETUP
oxids = [
    {'name':'O2', 'wt':100, 'T':T0}
]
fuels = [
    {'name':'H2', 'wt':100, 'T':T0},
    #{'name':'vybar', 'wt':50, 'h':-1494., 'T':T0, 'comp':'C 300 H 600'}
]
inpts = {
         'P_CC':P0/psi2pa,
         'P_EXT':14.6,
         'OF':7,
         'fuels':fuels,
         'oxidizers':oxids,
        }


OF = np.linspace(1, 10)
T_CEA = OF*0
T_CANTERA = OF*0
CS_CEA = OF*0
CS_CANTERA = OF*0
k_CEA = OF*0
k_CANTERA = OF*0


for k in range(len(OF)):

    # CANTERA
    #gas.T = 300.
    #gas.P = 25*1e5
    #gas.species_moles = f"paraffin:{1/M_F}, N2O:{OF[k]*1/M_OX}"
    gas.TPY = T0, P0, f"H2:1, O2:{OF[k]}"

    gas.equilibrate("HP")

    CP, CV = CP_CV_EQ(gas)
    gamma = CP/CV
    M = gas.mean_molecular_weight*1e-3
    T_CANTERA[k] = gas.T
    CS_CANTERA[k] = np.sqrt(gamma*8.3145/M*gas.T)/(gamma*np.sqrt((2/(gamma+1))**((gamma+1)/(gamma-1))))
    k_CANTERA[k] = gamma

    # CEA
    inpts['OF'] = OF[k]
    res = cea(inpts)
    T_CEA[k] = res["t"][0]
    CS_CEA[k] = res["cstar"][-1]
    k_CEA[k] = res["gamma"][0]
    print(res['gamma'][0], gamma)


fig, ax = plt.subplots()
ax.plot(OF, T_CANTERA, "--k")
ax.plot(OF, T_CEA, ":r")
ax.set_xlabel("O/F (kg/kg)")
ax.set_ylabel("T (K)")
ax.legend(("CANTERA", "CEA"))

fig, ax = plt.subplots()
ax.plot(OF, CS_CANTERA, "--k")
ax.plot(OF, CS_CEA, ":r")
ax.set_xlabel("O/F (kg/kg)")
ax.set_ylabel("C* (m/s)")
ax.legend(("CANTERA", "CEA"))

fig, ax = plt.subplots()
ax.plot(OF, k_CANTERA, "--k")
ax.plot(OF, k_CEA, ":r")
ax.set_xlabel("O/F (kg/kg)")
ax.set_ylabel("Cp/Cv (-)")
ax.legend(("CANTERA", "CEA"))

plt.show()
