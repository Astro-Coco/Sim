import n2o
import matplotlib.pyplot as plt
import numpy as np
import sys, getopt
from math import isnan, sqrt


class injector:

    def __init__(self, area, cd):
        self.a = area
        self.cd = cd
        self.cda = area*cd
        self.m_dot = 0
        self.p1 = -1
    
    def set_0(self, p0, x0):
        self.p0 = p0
        self.x0 = x0
    
    def set_p1(self, p1):
        self.p1 = p1
    
    def inj(self):
        if self.p1 == -1:
            G, Po = quick_inj(self.p0, self.x0)
        else:
            G, Po = quick_inj(self.p0, self.x0, self.p1)
        self.m_dot = G*self.cda
        self.po = Po
        if self.p1 > self.p0:
            self.m_dot = 0
        return self.m_dot, Po




def inject(Pin, xin, Pout = 0):
    P = Pin*1.0
    # Find initial properties
    x = xin       # Initial vapor mass fraction
    u = 1 # Initial velocity (must be non-zero)

    dP = -1*Pin/500 # pressure step

    T = n2o.TSat(P)
    vL = 1/n2o.rhoLiqSat(T)
    vV = 1/n2o.rhoVapSat(T)
    v = vL*(1-x) + vV*x
    G = u/v
    Gold = G-1
    
    Gsave = [G, 0, 0]
    Psave = [P, 0, 0]
    step = 0
    while (Gold < G)&(P > Pout):
        # Compute du/dP = f(P, u) = - 1/(rho*u)
        # RK4 explicit
        T = n2o.TSat(P)
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        G = u/v
        k1 = - v/u
        
        T = n2o.TSat(P+dP/2)
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        k2 = - v/(u + dP*k1/2)
        k3 =  - v/(u + dP*k2/2)

        T = n2o.TSat(P+dP)
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        k4 = - v/(u + dP*k3)
        du = (1/6)*dP*(k1 + 2*k2 + 2*k3 + k4)

        # Compute dx/dP = f(P, x) = x*vvl/hvl - Cp*T*vvl/hvl^2
        T = n2o.TSat(P)
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        hL = n2o.hLiqSat(T)
        hV = n2o.hVapSat(T)
        Cp = n2o.CpLiqSat(T)*(1-x) + n2o.CpVapSat(T)*x
        k1 = x*vV/(hL-hV) - Cp*T*(vV-vL)/( (hL-hV)**2 )

        T = n2o.TSat(P+dP/2)
        xn = x+dP*k1/2
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        hL = n2o.hLiqSat(T)
        hV = n2o.hVapSat(T)
        Cp = n2o.CpLiqSat(T)*(1-xn) + n2o.CpVapSat(T)*xn
        k2 = xn*vV/(hL-hV) - Cp*T*(vV-vL)/( (hL-hV)**2 )

        xn = x+dP*k2/2
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        hL = n2o.hLiqSat(T)
        hV = n2o.hVapSat(T)
        Cp = n2o.CpLiqSat(T)*(1-xn) + n2o.CpVapSat(T)*xn
        k3 = xn*vV/(hL-hV) - Cp*T*(vV-vL)/( (hL-hV)**2 )

        T = n2o.TSat(P+dP)
        xn = x+dP*k3
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        hL = n2o.hLiqSat(T)
        hV = n2o.hVapSat(T)
        Cp = n2o.CpLiqSat(T)*(1-xn) + n2o.CpVapSat(T)*xn
        k4 = xn*vV/(hL-hV) - Cp*T*(vV-vL)/( (hL-hV)**2 )

        dx = (1/6)*dP*(k1 + 2*k2 + 2*k3 + k4)

        u += du
        x += dx
        P += dP
        
        step += 1
        vL = 1/n2o.rhoLiqSat(T)
        vV = 1/n2o.rhoVapSat(T)
        v = vL*(1-x) + vV*x
        Gold = G
        G = u/v

        if step > 2:
            Gsave[0] = Gsave[1]
            Psave[0] = Psave[1]
            Gsave[1] = Gsave[2]
            Psave[1] = Psave[2]
        elif step == 2:
            Gsave[1] = Gsave[2]
            Psave[1] = Psave[2]
        Gsave[2] = G
        Psave[2] = P
    
    # Find actual maxima via quadratic fit
    ky = Gsave[2] - Gsave[0] - (Gsave[1]-Gsave[0])*(Psave[2] - Psave[0])/(Psave[1] - Psave[0])
    kx = Psave[2]**2 - Psave[0]**2 - (Psave[2]-Psave[0])*(Psave[1]+Psave[0])
    a2 = ky/kx
    a1 = (Gsave[1]-Gsave[0]-a2*(Psave[1]**2-Psave[0]**2))/(Psave[1]-Psave[0])
    a0 = Gsave[0] - a1*Psave[0] - a2*Psave[0]**2
    #print('a2 : ', a2)
    P = - a1/(2*a2) # Best fit
    G = a0 + a1*P + a2*P**2

    return G, P, n2o.TSat(P), x


def quick_inj(Pin, Xin, Pout=0):
    def lin_func(k):
        # Important, iterator order function
        a = (sqrt(1+8*k)-1)/2
        b = a - int(a)

        i = int(a*b+0.5)
        j = int(a-i)
        return (i, j)

    def quad_surf(x, y, c):
        z = 0
        for k in range(len(c)):
            i, j = lin_func(k)
            z += c[k]*(x**i)*(y**j)
        return z
    
    cG = [ 5.02865216e+02, -8.39696372e+02,  6.98867232e+01, -4.76223028e+02, -9.96605997e+01, -1.07070940e-01, -1.00594931e+02, 8.53305494e+01, 8.02101206e-02,  1.43688826e-04,  2.05267361e+02, -2.26080276e+01, -4.72371279e-02, -6.14712785e-06, -8.41735073e-08,  4.43797475e+02]
    cP = [ 9.90407303e+04, -8.16870972e+05,  4.94309617e+03,  1.21966664e+06, -2.40816197e+02,  2.91214780e-01, -5.81517183e+05,  4.62759154e+01]
    #cX = [ 1.72612758e-01,  6.39276296e-01, -2.74093887e-03,  3.69993131e-01, 3.11052195e-03,  1.36768281e-05,  1.31925596e-01, -1.98569966e-03, -6.57632851e-06, -2.27890405e-08, -5.35153475e-02,  8.84737607e-04, 5.57169869e-07,  4.84382229e-09,  1.23884236e-11, -1.95281477e-01]
    P = Pin/6894.76
    G = quad_surf(P, Xin, cG)
    Po = quad_surf(P, Xin, cP)

    if Pout > Po:
        if (Pin-Po) == 0:
            G = 0
        else:
            x = (Pin-Pout)/(Pin-Po) + 0j
            G = G*(1-(x-1)**2)**0.5
            G = G.real

    return G, Po


if __name__=='__main__':

    Cd = -1
    d = -1
    Pinj = -1
    ud = ''
    up = ''
    try:
        opts, args = getopt.getopt(sys.argv, "hc:d:p:u:", ["cd=","diameter=","pressure=","units="])
    except getopt.GetoptError:
        print('inj.py -c <Cd> -d <hydraulicDiameter(m)> -p <pressure(Pa)> -u <d_units-p_units>')
        sys.exit(2)
    i = 0
    while i < (len(args)-1)/2:
        i += 1
        opt = args[2*i-1]
        if len(args)>2:
            arg = args[2*i]
        if opt == "-h":
            print('inj.py -c <Cd> -d <hydraulicDiameter(m)> -p <pressure(Pa)> -u <d_units-p_units>')
            sys.exit()
        elif opt in ("-c","--cd"):
            Cd = float(arg)
        elif opt in ("-d","--diameter"):
            d = float(arg)
        elif opt in ("-p","--pressure"):
            Pinj = float(arg)
        elif opt in ("-u","--units"):
            un = arg.split('-')
            ud = un[0]
            up = un[1]
    
    if Cd == -1:
        Cd = 0.8
        print('No Cd value entered, assuming Cd = 0.8')
    if d == -1:
        print('Error: enter hydraulic diameter')
        print('inj.py -c <Cd> -d <hydraulicDiameter(m)> -p <pressure(Pa)> -u <d_units-p_units>')
        sys.exit(2)
    if Pinj == -1:
        print('Error: enter injection pressure')
        print('inj.py -c <Cd> -d <hydraulicDiameter(m)> -p <pressure(Pa)> -u <d_units-p_units>')
        sys.exit(2)

    # Convert to SI units
    def udf(d):
        return d
    def udfi(d):
        return d
    def upf(p):
        return p
    def upfi(p):
        return p
    if ud in ('in', 'po', 'inch', 'pouce', 'pouces'):
        def udf(d):
            return d*0.0254
        def udfi(d):
            return d/0.0254
    elif ud in ('mm', 'millimeter'):
        def udf(d):
            return d*0.001
        def udfi(d):
            return d*1000
    elif ud in ('cm', 'centimeter'):
        def udf(d):
            return d*0.01
        def udfi(d):
            return d*100
    if up in ('PSI', 'PSIA', 'psi', 'psia'):
        def upf(p):
            return p*6894.76
        def upfi(p):
            return p/6894.76
    elif up in ('PSIG', 'psig'):
        def upf(p):
            return (p + 14.6959) * 6894.76
        def upfi(p):
            return p/6894.76 - 14.6959
    elif up in ('kpa', 'kPa'):
        def upf(p):
            return p*1000
        def upf(p):
            return p/1000
    d = udf(d)
    Pinj = upf(Pinj)

    CdA = Cd * 3.14159265359*((d/2)**2)  # m**2
    Gs = []
    Ps = []
    xs = []
    for i in range(20):
        G, P, T, x = inject(Pinj, i/30)
        xs.append(i/30)
        Gs.append(G*CdA)
        Ps.append(upfi(P))
    
    if ud == '':
        ud = 'm'
    if up == '':
        up = 'Pa'
    fig, ax = plt.subplots()
    axp = ax.twinx()
    ax.plot(xs, Gs, 'k')
    ax.set_ylabel('Choked mass flow (kg/s)')
    axp.plot(xs, upfi(Pinj)-np.array(Ps), color='tab:red')
    axp.set_ylabel('Choking delta P ('+up+')', color='tab:red')
    axp.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Pre-inj. vapor fraction\nDiameter = '+str(udfi(d))+' ('+ud+'), Pre-inj. pressure = '+str(upfi(Pinj))+' ('+up+'), Cd = '+str(Cd))
    plt.show()
