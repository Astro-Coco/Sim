import n2o
import matplotlib.pyplot as plt
from inj import injector


def newt_volume(T_ini, M_tot, V_tot, H):
    def V(T, H, m, V_tank):
        h_liq = n2o.hLiqSat(T)
        h_vap = n2o.hVapSat(T)
        rho_liq = n2o.rhoLiqSat(T)
        rho_vap = n2o.rhoVapSat(T)
        
        x=((H/m)-h_liq)/(h_vap-h_liq)
        
        return m*((1-x)/(rho_liq)+ x/rho_vap)-V_tank
    
    eps = 1e-6
    tol = 1e-4
    N = 25
    n = 1
    T = T_ini
    dT = 1

    while (abs(dT) >= tol)&(n<N):
        F = V(T, H, M_tot, V_tot)
        dF = ( V(T+abs(eps*T),H,M_tot,V_tot)-F )/abs(eps*T)
        dT = -F/dF

        T += dT
        n += 1
    
    return T


class nox_tank:

    def __init__(self, volume, ullage, pressure=-1, temp=-1):
        if temp == -1:
            temp = n2o.TSat(pressure)
        elif pressure == -1:
            pressure = n2o.PSat(temp)
        self.T = temp
        self.P = pressure
        self.T_0 = temp
        self.V = volume
        self.x = ullage
        self.x_0 = ullage
        self.get_mass()

        h_vap = n2o.hVapSat(self.T)
        h_liq = n2o.hLiqSat(self.T)
        self.h = self.ml*h_liq + self.mv*h_vap
    
    def reset(self):
        self.x = self.x_0
        self.T = self.T_0
        self.P = n2o.PSat(self.T)
        self.get_mass()

    def get_mass(self):
        # Get mixture volume mass
        rhol = n2o.rhoLiqSat(self.T)
        rhov = n2o.rhoVapSat(self.T)
        # Get individual masses
        self.mv = self.x*self.V*rhov
        self.ml = (1-self.x)*self.V*rhol
        # Get total tank mass
        self.m = self.mv + self.ml

    def step(self, dt, injector, auto_update=True):
        
        # Get injector mass flow rate given conditions
        if auto_update:
            injector.set_0(self.P, self.x)
        m_dot, po = injector.inj()

        # Update tank total mass an enthalpy
        h_liq = n2o.hLiqSat(self.T)
        self.m -= m_dot*dt
        self.h -= m_dot*dt*h_liq

        # Get tank temperature
        self.T = newt_volume(self.T, self.m, self.V, self.h)
        self.P = n2o.PSat(self.T)

        # Update properties
        rhol = n2o.rhoLiqSat(self.T)
        rhov = n2o.rhoVapSat(self.T)
        h_liq = n2o.hLiqSat(self.T)
        h_vap = n2o.hVapSat(self.T)

        # Update tank ullage and liquid mass
        self.x = (self.h/self.V - rhol*h_liq)/(rhov*h_vap - rhol*h_liq)
        self.ml = (1-self.x)*self.V*rhol
        self.mv = self.m - self.ml

        # Do exit checks
        if self.ml <= 0:
            return 1
        else:
            return 0
        
    def empty(self, dt, injector):
        # Initialise loop and stored variables
        Ps = [self.P/6894.76]
        Ts = [self.T]
        Ms = [self.m]
        LMs = [self.ml]
        VMs = [self.mv]
        Xs = [self.x]
        times = [0]
        while self.x < 0.7:
            self.step(dt, injector)
            Ps.append(self.P/6894.76)
            Ts.append(self.T)
            Ms.append(self.m)
            LMs.append(self.ml)
            VMs.append(self.mv)
            Xs.append(self.x)
            times.append( times[-1]+dt )
            print(self.ml, self.T)
        
        fig, ax = plt.subplots(4)
        ax[0].plot(times, LMs)
        ax[1].plot(times, VMs)
        ax[2].plot(times, Ps)
        ax[3].plot(times, Xs)
        plt.show()


if __name__=='__main__':
    tank = nox_tank(0.03, 0.2, temp=295)
    a = 3.1415927 * ( (25.4*0.6/2000)**2 )
    inj = injector( a, 0.65 )
    tank.empty(0.1, inj)
