import sys, getopt
sys.path.insert(1, '..\\')
sys.path.insert(1, '..\\DATA')
from pyCEA import cea
import numpy as np
from math import sqrt
import json


def read_json(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.read())
    return data

def write_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


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


def quick_cs(p, of, c=[], c_file='./c.prop'):
    # if c is empty, read c file
    if c == []:
        cf = read_json(c_file)
        c = cf['c']
    
    return quad_surf(p/1e6, of, c)

def quick_k(p, of, c=[], c_file='./c.prop'):
    # if c is empty, read c file
    if c == []:
        kf = read_json(c_file)
        k = kf['c']
    
    return quad_surf(p/1e6, of, c)


def generate_fit(p_in, of_in, c_file=''):
    # Generate 2d arrays for pressure, o/f and C*
    nx, ny = (10, 20)
    p = np.linspace(p_in[0], p_in[1], nx)
    of = np.linspace(of_in[0], of_in[1], ny)
    p, of = np.meshgrid(p, of)
    cs = np.zeros((ny, nx))

    fuels = [{'name':'paraffin', 'T':293, 'comp':'C 32 H 66'}]
    oxids = [{'name':'N2O', 'T':293}]
    inpts = {
            'P_CC':1,
            'P_EXT':14.5,
            'OF':1,
            'fuels':fuels,
            'oxidizers':oxids
            }

    # Compute C* using CEA for every point
    for i in range(len(cs)):
        for j in range(len(cs[0])):
            inpts['P_CC'] = p[i][j]/6894.76
            inpts['OF'] = of[i][j]
            cs[i][j] = cea(inpts)['cstar'][-1]

    # Fit quadratic surface

    # Start by generating A matrix and b solution vector
    N = 16  # Number of polynomial terms in fit surface
    A = np.zeros((nx*ny, N))
    b = np.zeros(nx*ny)
    k = 0
    for i in range(ny):
        for j in range(nx):
            for n in range(N):
                ii, jj = lin_func(n)
                A[k][n] = ((p[i][j]/1e6)**ii)*(of[i][j]**jj)
            b[k] = cs[i][j]
            k += 1
    
    # Get least squares solution to linear system
    c = np.linalg.lstsq(A, b,rcond=None)[0]

    # Save file
    if c_file != '':
        write_json(c_file,{'c':list(c)})
    
    return c


if __name__=='__main__':
    # Get system arguments for pressure and of range
    of = [-1, -1]
    p = [-1, -1]
    up = ''
    try:
        opts, args = getopt.getopt(sys.argv, "ho:p:u:", ["oxidFuelRatio=","pressure=","units="])
    except getopt.GetoptError:
        print('cea_fit.py -o <min_o/f>-<max_o/f> -p <pressure_min>-<pressure_max(Pa)> -u <p_units>')
        sys.exit(2)
    i = 0
    while i < (len(args)-1)/2:
        i += 1
        opt = args[2*i-1]
        if len(args)>2:
            arg = args[2*i]
        if opt == "-h":
            print('cea_fit.py -o <min_o/f>-<max_o/f> -p <pressure_min>-<pressure_max(Pa)> -u <p_units>')
            sys.exit()
        elif opt in ("-o","--oxidFuelRatio"):
            of = [float(i) for i in arg.split('-')]
        elif opt in ("-p","--pressure"):
            p = [float(i) for i in arg.split('-')]
        elif opt in ("-u","--units"):
            up = arg
    
    # Convert to SI units
    def upf(p):
        return p
    def upfi(p):
        return p
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
    p = [upf(i) for i in p]

    # Generate and save coefficients
    generate_fit(p, of, './c.json')

    
    
    
