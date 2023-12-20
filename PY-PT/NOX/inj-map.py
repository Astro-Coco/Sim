from inj import inject
import numpy as np
import matplotlib.pyplot as plt

psi2pa = 6894.76

def lin_func(k):
    # Important, iterator order function
    a = (np.sqrt(1+8*k)-1)/2
    b = a - np.floor(a)

    i = round(a*b)
    j = np.floor(a-i)
    return (i, j)

def quad_surf(x, y, c):
    z = 0
    for k in range(len(c)):
        i, j = lin_func(k)
        z += c[k]*(x**i)*(y**j)
    return z

def quick_inj(Pin, Xin):
    cG = [ 5.02865216e+02, -8.39696372e+02,  6.98867232e+01, -4.76223028e+02, -9.96605997e+01, -1.07070940e-01, -1.00594931e+02, 8.53305494e+01, 8.02101206e-02,  1.43688826e-04,  2.05267361e+02, -2.26080276e+01, -4.72371279e-02, -6.14712785e-06, -8.41735073e-08,  4.43797475e+02]
    cP = [ 9.90407303e+04, -8.16870972e+05,  4.94309617e+03,  1.21966664e+06, -2.40816197e+02,  2.91214780e-01, -5.81517183e+05,  4.62759154e+01]
    cX = [ 1.72612758e-01,  6.39276296e-01, -2.74093887e-03,  3.69993131e-01, 3.11052195e-03,  1.36768281e-05,  1.31925596e-01, -1.98569966e-03, -6.57632851e-06, -2.27890405e-08, -5.35153475e-02,  8.84737607e-04, 5.57169869e-07,  4.84382229e-09,  1.23884236e-11, -1.95281477e-01]
    P = Pin/6894.76
    G = quad_surf(P, Xin, cG)
    Po = quad_surf(P, Xin, cP)
    Xo = quad_surf(P, Xin, cX)
    return G, Po, n2o.TSat(Po), Xo

num_p = 40
num_x = 10
pin = np.linspace(1, 1000, num_p)
pin = np.multiply(pin, psi2pa)
xin = np.linspace(0, 1, num_x)
pin, xin = np.meshgrid(pin, xin)
var = np.zeros((num_x, num_p))


for i in range(len(pin)):
    for j in range(len(pin[0])):
        gox2, pout, t_out, var[i][j] = inject(pin[i][j], xin[i][j])
    print(i)

X = np.multiply(pin, 1/psi2pa)
Y = xin
Z = var
nx = num_p
ny = num_x

# Start by generating A matrix and b solution vector
N = 16  # Number of polynomial terms in fit surface
A = np.zeros((nx*ny, N))
b = np.zeros(nx*ny)
k = 0
for i in range(ny):
    for j in range(nx):
        for n in range(N):
            ii, jj = lin_func(n)
            A[k][n] = (X[i][j]**ii)*(Y[i][j]**jj)
        b[k] = Z[i][j]
        k += 1

# Get least squares solution to linear system
c = np.linalg.lstsq(A, b,rcond=None)[0]
print(c)

Z2 = np.zeros((ny, nx))
for i in range(ny):
    for j in range(nx):
        Z2[i][j] = quad_surf(X[i][j], Y[i][j], c)

# Plot result
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 200)
ax.contour3D(X, Y, Z2, 200)
plt.show()

print('end')