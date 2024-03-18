from inj import inject
from inj import quick_inj
import math
import matplotlib.pyplot as plt	
import numpy as np


P_atm = 101325

def A_section(diameter):
    diameter_m = diameter*2.54/100
    return (math.pi*diameter_m**2)/4

A1 = A_section((3/8))
A2 = A_section((3/64))

#psi to pa factor
converse_factor = 6894.75729

def my_bisection(f, a, b, tol = 1e-10): 
    # approximates a root, R, of f bounded 
    # by a and b to within tolerance 
    # | f(m) | < tol with m the midpoint 
    # between a and b Recursive implementation
    
    # check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception(
         "The scalars a and b do not bound a root")
        
    # get midpoint
    m = (a + b)/2
    guesses.append(m)
    print('TOL  : ', tol)
    print('f(m) : ', f(m))
    if np.abs(f(m)) < tol:
        print('between tol')
        # stopping condition, report m as root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a. 
        # Make recursive call with a = m
        return my_bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b. 
        # Make recursive call with b = m
        return my_bisection(f, a, m, tol)


def fonction(x):
    
    sol = inject(Pin = 5.5158e6, xin= 0.2, Pout = x )
    mdot1 = sol[0]*A1

    mdot2 = inject(Pin = x, xin =  sol[-1], Pout = P_atm)[0]*A2

    return (mdot1 - mdot2)

guesses = []

def newton(f, guess=1):
    # Set the tolerance and epsilon for convergence and derivative calculation
    eps = 1e-6  # Epsilon to avoid division by zero in the derivative
    tol = 1e-4  # Tolerance to determine when the root is sufficiently refined
    N = 500  # Maximum number of iterations to prevent infinite loops
    n = 0  # Iteration counter
    x = guess  # Initial guess for the root
    dx = 1  # Initial delta x

    # Loop until the change in x is smaller than the tolerance or max iterations reached
    while (abs(dx) >= tol) & (n < N):
        F = f(x)  # Evaluate the function at the current guess
        dxi = max(
            [abs(eps * x), eps]
        )  # Increment for numerical derivative, guarding against zero
        dF = 0.5 * (f(x + dxi) - f(x - dxi)) / dxi  # Central difference for derivative
        dx = -1 * F / dF  # Newton's method update

        x += dx  # Update the guess
        guesses.append(x)
        n += 1  # Increment the iteration counter

    # If the maximum number of iterations is reached, warn the user
    if n >= N:
        print("WARNING - Newton overflow, x:", x, "F:", F)
    return x  # Return the refined root

#result = newton(f = fonction, guess =800*converse_factor)
result = my_bisection(f = fonction, a = -1000*converse_factor, b = 1000*converse_factor)
print(result)

plt.scatter(range(len(guesses)), np.array(guesses)/converse_factor)
plt.show()
