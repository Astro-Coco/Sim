import sys

local_py_pt = r'c:\Users\colin\SIM_REPO\Sim\PY-PT'


paths = ["", "\\", "\\CEA", "\\MOT", "\\NOX", "\\ANIM",'\\DATA']
for specific_path in paths:
    sys.path.insert(1, local_py_pt + specific_path)

import utils
from cea_fit import quick_cs
from inj import injector
from tank import nox_tank
from motor_animation import motor_animation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import getopt
import json
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

psi_conversion_factor = 6894.76  # psi to Pa


# Newton root-rinding method for arbitrary f(x)
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
        n += 1  # Increment the iteration counter

    # If the maximum number of iterations is reached, warn the user
    if n >= N:
        print("WARNING - Newton overflow, x:", x, "F:", F)
    return x  # Return the refined root


# Newton-Raphson method for arbirary f(x) = [f1(x), f2(x), ... fn(x)], x = [x1, x2, ... xn]
def raphson(f, guess, relf=0.5):
    eps = 1e-6
    tol = 1e-4
    N = 500
    n = 0
    x = guess
    dx = 1
    F = 1

    while (np.linalg.norm(F) >= tol) & (n < N):
        F = f(x)
        J = []
        for i in range(len(F)):
            dxi = x * 0
            dxi[i] = max(abs(eps * x[i]), eps)
            dFi = 0.5 * (f(x + dxi) - f(x - dxi)) / dxi[i]
            J.append(dFi)

        dx = -relf * np.linalg.solve(J, F)

        x += dx
        n += 1
    if n >= N:
        print("WARNING - Raphson overflow, x:", x, "F:", F)
    return x


# Nozzle class
# Allow to get CF and exit pressure/back pressure ratio
class nozzle:
    def __init__(self, throat_area, expansion_ratio):
        self.throat_area = throat_area
        self.expansion_ratio = expansion_ratio  # Expansion ratio Ae/At (from geometry)
        self.mach = 2  # Initial guess for Mach and will change regards time

    # Find exit Mach number
    def find_mach(self, k):
        # Iterate on exit Mach number with Newton Method with optimization for expansion ratio - ARM = 0
        def f_expansion_ratio(m):
            # Theorical equation for expansion ratio (isentropic flow)
            # Exit area / Throat area = f(Exit mach, Specific heat ratio)
            expansion_ratio_eq = (
                (k / 2 + 0.5) ** (-(k / 2 + 0.5) / (k - 1))
                * 1
                / m
                * (1 + (k / 2 - 0.5) * m**2) ** ((k / 2 + 0.5) / (k - 1))
            )

            return expansion_ratio_eq - self.expansion_ratio

        self.mach = newton(f_expansion_ratio, self.mach)  # Solve with Newton method

    # Compute CF at current chamber conditions and calculate Pe/Pb ratio
    def calculate_cf_pepbratio(self, pbpc_ratio, heat_capacity_ratio):
        k = heat_capacity_ratio  # Specific heat ratio Cp/Cv of gaz
        self.find_mach(k)
        pept_ratio = (1 + 0.5 * (k - 1) * self.mach**2) ** (
            -k / (k - 1)
        )  # Exit pressure/total pressure ratio (note : Pt = Pc)

        # Calculate thrust coefficient and pepb ratio
        cf = k * np.sqrt(
            2
            / (k - 1)
            * (2 / (k + 1)) ** ((k + 1) / (k - 1))
            * (1 - pept_ratio ** (1 - 1 / k))
        ) + self.expansion_ratio * (pept_ratio - pbpc_ratio)
        pepb_ratio = pept_ratio / pbpc_ratio  # Exit pressure/Back pression ratio

        return cf, pepb_ratio


# Combustion chamber class
# Used for simulation to calculate fuel mass flow rate during burn
class chamber:
    def __init__(
        self, initial_port_radius, final_port_radius, grain_length, fuel, efficiency
    ):
        # Chamber objects
        self.r_init = initial_port_radius  # initial port radius
        self.r_final = final_port_radius  # final port radius (max)
        self.r = initial_port_radius
        self.grain_length = grain_length  # grain length
        self.fuel = fuel
        self.last_r_dot = 0
        self.pressure_chamber = 0  # Pressure chamber
        self.efficiency = efficiency  # C* efficiency

    # Compute fuel mass flow rate given oxidizer mass flow rate, using r_dot = a*G_ox**n
    def m_dot_fuel(self, m_dot_ox):
        port_area = np.pi * (self.r**2)
        G_ox = m_dot_ox / port_area + 0j  # Oxidizer mass flow rate per area
        r_dot = self.fuel["a"] * (G_ox ** self.fuel["n"])  # fuel regression rate
        fuel_inside_area = 2 * np.pi * self.r * self.grain_length
        fuel_mass_flow_rate = fuel_inside_area * self.fuel["density"] * r_dot.real
        return fuel_mass_flow_rate

    # Reset chamber properties (port radius and chamber pressure)
    def reset(self):
        self.r = self.r_init
        self.r_dot = 0
        self.last_r_dot = 0
        self.pressure_chamber = 0

    # Step forward in simulation, given time step dt
    def step(self, dt, injector):
        m_dot_ox = injector.m_dot  # Get m dot oxydizer value from injector simulator
        port_area = np.pi * (self.r**2)  # Current port area
        G_ox = m_dot_ox / port_area + 0j
        r_dot = self.fuel["a"] * (G_ox ** self.fuel["n"])

        self.r_dot = r_dot.real  # 0.5*(3*r_dot - self.last_r_dot)
        self.r += dt * self.r_dot
        self.last_r_dot = self.r_dot


# Tubing class [In progress]
# Get pressure drop between tank and injector and oxidizer gas fraction at right before injector
class tubing:
    def __init__(self, r, l, cd, mu_c, rho_c, cv):
        self.r = r
        self.cd = cd
        self.l = l
        self.cv = cv
        self.mu_c = mu_c
        self.rho_c = rho_c

    def get_p_loss(self, m_dot_ox, p_tank):
        p_loss = m_dot_ox * 175 * psi_conversion_factor / 2.2
        x = m_dot_ox * 0.1 / 2.2
        return p_loss, x


# Rocket motor class [major class]
# Simulate the motor burn and store data
class hybrid_motor:
    def __init__(self, tank, tubing, injector, chamber, nozzle, motor_animation):
        self.tank = tank
        self.tubing = tubing
        self.injector = injector
        self.chamber = chamber
        self.nozzle = nozzle
        self.motor_animation = motor_animation

    def set_sim_properties(self, dt, pb):
        # Set time step and back pressure
        self.dt = dt
        self.back_pressure = pb

    def reset(self):
        # Reset properties
        self.time = 0
        self.impulse = 0
        self.chamber.reset()
        self.tank.reset()
        self.motor_animation.set_up(
            ox_max=self.tank.ml, r_max=self.chamber.r_final, ri=self.chamber.r_init
        )
        self.first_time_step = True

        # Reset vectors for plot
        self.data = {
            "Time      (s)": [],
            "ISP     (m/s)": [],
            "C*      (m/s)": [],
            "Pe/Pb     (-)": [],
            "Thrust    (N)": [],
            "Impulse  (Ns)": [],
            "P tank (psia)": [],
            "P inj. (psia)": [],
            "P comb (psia)": [],
            "P crit (psia)": [],
            "O/F       (-)": [],
            "m. ox. (kg/s)": [],
            "Gox (kg/s-m2)": [],
            "r.     (mm/s)": [],
            "Ullage       ": [],
        }

    # Solve non-linear multivariable problem for chamber pressure and oxidizer mass flow rate
    # Equations:
    #   m_dot_ox = f(p_inj, pressure_chamber)   (1)
    #   CS = Pc*At/m_dot                        (2)
    # Default guess point : pressure chamber = 100 psi and m_dot_ox = 2 kg/s
    def solve_cs(self, guess_pc=100 * psi_conversion_factor, guess_mdo=2):
        # Get initial guesses or last values
        if self.first_time_step:  # At first time step
            last_pressure_chamber = guess_pc
            last_m_dot_ox = guess_mdo
            relf = 0.3  # relaxation coefficient
            self.first_time_step = False
        else:
            last_pressure_chamber = self.chamber.pressure_chamber
            last_m_dot_ox = self.injector.m_dot
            relf = 0.6  # Increase relaxation coefficient

        # function 1, oxidixer mass flow rate model
        def f_mox(pc, m_dot_ox):
            self.injector.set_p1(pc)  # Assign pressure chamber to injector
            p_loss, x_loss = self.tubing.get_p_loss(
                m_dot_ox, self.tank.P
            )  # Pressure loss by tubing and vapor fraction
            self.injector.set_0(
                self.tank.P - p_loss, x_loss
            )  # Assign pressure before injector
            (
                m_dot_ox_i,
                po,
            ) = self.injector.inj()  # m_dot_ox and current pressure after injector
            return m_dot_ox_i - m_dot_ox

        # function 2, C* equation, values taken from CEA with a surface polynomial fit
        def f_cs(pc, m_dot_ox):
            of = m_dot_ox / self.chamber.m_dot_fuel(m_dot_ox)
            m_dot = m_dot_ox * (1 + 1 / of)  # Total mass flow rate
            cs = self.chamber.efficiency * quick_cs(
                pc, of, self.chamber.fuel["c"]
            )  # estimated C* with CEA
            return (cs - pc * self.nozzle.throat_area / m_dot) / 1e3

        # Merge function, f(x) = [f_mox, f_cs],
        # x = [chamber pressure/1e6, oxidizer mass flow]
        def f(x):
            pc = x[0] * 1e6  # Pa in eqt but kPa for Newton method
            m_dot_ox = x[1]
            return np.array([f_mox(pc, m_dot_ox), f_cs(pc, m_dot_ox)])

        # Find root using newton-raphson method
        x = raphson(f, np.array([last_pressure_chamber / 1e6, last_m_dot_ox]), relf)

        # Save chamber pressure
        self.chamber.pressure_chamber = x[0] * 1e6
        return x[0] * 1e6

    def step(self, thrust):
        # Find chamber pressure using newton-raphson loop
        self.solve_cs()

        # Update tank and chamber properties
        self.tank.step(self.dt, self.injector, auto_update=False)
        self.chamber.step(self.dt, self.injector)

        self.motor_animation.step(
            ox_mass=self.tank.ml,
            r=self.chamber.r,
            thrust=thrust,
            dt=self.dt,
        )

    

        """    def ignition(self, m_dot_ox, chamber_pressure):

        def ignition_data(max_thrust,dt = 0.001):
            import math
            coeff = [-29118.76867808, 31342.97234717, -1605.90936194,  410.15048635]
            # trouver la formule mathématique, pourrait être bcp plus simplifiée probablement
            a,b,c,d = (-29118.76867808, 31342.97234717, -1605.90936194,  410.15048635)
            x_min_fit = (-2*b + math.sqrt((2*b)**2 - 4*3*a*c))/(6*a)
            x_max_fit = (-2*b - math.sqrt((2*b)**2-4*3*a*c))/(6*a)
            ignition_time = x_max_fit-x_min_fit
            print(ignition_time)
            #print(dt)
            points = np.linspace(0, ignition_time, round((1/dt)*ignition_time))
            print('coeff = ', coeff)
            fit = np.polyval(coeff,points)
            fit_at_zero = fit - fit.min()
            fit_scaled = fit_at_zero*(max_thrust/fit_at_zero.max())
            return fit_scaled , points

        print(m_dot_ox)
        ignition_m_dot, ignition_time = ignition_data(m_dot_ox)
        ignition_P_comb, ignition_time2 = ignition_data(chamber_pressure)
        return ignition_time, ignition_m_dot, ignition_P_comb"""

    # Simulate a motor burn
    def burn(self, prints=False, animate=True):
        time = 0
        thrust = 1

        # Progress bar handling
        if prints:
            pb = utils.progressBar(25)
            ml0 = self.tank.ml

        # Main loop, stop when no liquid nox is left in tank, or when all fuel grain has beed burned
        while (self.tank.ml > 0) & (self.chamber.r < self.chamber.r_final):
            self.step(thrust)
  
            # Step forward simulation
            # Compute and save properties for plots
            of = self.injector.m_dot / self.chamber.m_dot_fuel(
                self.injector.m_dot
            )  # O/F
            cs = self.chamber.efficiency * quick_cs(
                self.chamber.pressure_chamber, of, self.chamber.fuel["c"]
            )  # C*
            heat_capacity_ratio = 1.3  # to modify for a new calculation with CEA
            cf, pepb = self.nozzle.calculate_cf_pepbratio(
                self.back_pressure / self.chamber.pressure_chamber,
                heat_capacity_ratio,
            )  # CF and pe/pb
            isp = cs * cf  # Specific impulse
            thrust = isp * self.injector.m_dot * (1 + 1 / of)
            self.impulse += thrust * self.dt  # Impulse for the time step

            self.data["Time      (s)"].append(time)
            self.data["r.     (mm/s)"].append(self.chamber.r_dot * 1000)
            self.data["m. ox. (kg/s)"].append(self.injector.m_dot)
            self.data["P comb (psia)"].append(
                self.chamber.pressure_chamber / psi_conversion_factor
            )
            self.data["P inj. (psia)"].append(self.injector.p0 / psi_conversion_factor)
            self.data["P tank (psia)"].append(self.tank.P / psi_conversion_factor)
            self.data["C*      (m/s)"].append(cs)
            self.data["ISP     (m/s)"].append(isp)
            self.data["Pe/Pb     (-)"].append(pepb)
            self.data["Thrust    (N)"].append(thrust)
            self.data["O/F       (-)"].append(of)
            self.data["Gox (kg/s-m2)"].append(
                self.injector.m_dot / (np.pi * self.chamber.r**2)
            )
            self.data["P crit (psia)"].append(self.injector.po / psi_conversion_factor)
            self.data["Impulse  (Ns)"].append(self.impulse)
            self.data["Ullage       "].append(self.tank.x)

            # Step time forward
            time += self.dt

            if prints:
                # Progress bar update. Progress is computed using port radius or liquid oxidizer mass
                progr = (self.chamber.r - self.chamber.r_init) / (
                    self.chamber.r_final - self.chamber.r_init
                )
                progm = (ml0 - self.tank.ml) / ml0
                prog = max([progm, progr])
                if prog > 1:
                    prog = 1
                pb.set_progress(prog)

        # Print end state, no fuel or no oxidizer
        if prints:
            pb.end()  # Progress bar handling
            if self.tank.ml < 0:  # No more oxidizer
                print(
                    "Oxidizer tank empty\n" + "Final grain radius",
                    "{:.5f}".format(self.chamber.r),
                    "m ["
                    + "{:.1f}".format(
                        100
                        * (self.chamber.r - self.chamber.r_init)
                        / (self.chamber.r_final - self.chamber.r_init)
                    ),
                    "% used]",
                )
            else:  # No more fuel
                print(
                    "Fuel grain exhausted\n" + "Liquid oxidizer mass left:",
                    "{:.3f}".format(self.tank.ml),
                    "kg [" + "{:.1f}".format(100 * (ml0 - self.tank.ml) / ml0),
                    "% used]",
                )
            print("Burn time : " + "{:.3f}".format(time) + " s")

    def save_data_to_csv(self, filename):
        filename = filename.split('\\')[-1]
        filename = filename.split(".")[0]
        dataframe_to_save = pd.DataFrame(self.data)
        try:
            dataframe_to_save.to_csv(os.path.join(saving_path, "data_" + filename + ".csv"), index=False)
        except:
            dataframe_to_save.to_csv( "data_" + filename + ".csv", index=False)
        return dataframe_to_save

    # Plot simulation results
    def plot(self):
        # First figure, pressure plot
        fig, ax = plt.subplots()
        ax.plot(
            self.data["Time      (s)"],
            self.data["P tank (psia)"],
            color="tab:blue",
            label="Tank",
        )
        ax.plot(
            self.data["Time      (s)"],
            self.data["P inj. (psia)"],
            color="k",
            label="Inj.",
        )
        ax.plot(
            self.data["Time      (s)"],
            self.data["P comb (psia)"],
            color="tab:red",
            label="Comb.",
        )
        ax.plot(
            self.data["Time      (s)"],
            self.data["P crit (psia)"],
            color="tab:orange",
            label="Crit.",
        )
        ax.set_ylabel("Pressure (PSIA)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True)

        # Second figure, oxidizer flow rate, O/F and regression rate
        fig, ax = plt.subplots(2)
        ax[0].plot(self.data["Time      (s)"], np.array(self.data["r.     (mm/s)"]))
        ax[0].grid(True)
        ax[1].plot(self.data["Time      (s)"], self.data["m. ox. (kg/s)"], "k")
        ax[1].set_ylabel("Oxidizer mass flow (kg/s)")
        ax01 = ax[1].twinx()
        ax01.plot(
            self.data["Time      (s)"], self.data["O/F       (-)"], color="tab:blue"
        )
        ax01.set_ylabel("O/F")
        ax01.tick_params(axis="y", labelcolor="tab:blue")
        ax[0].set_ylabel("r dot (mm/s)")
        ax[1].set_xlabel("Time (s)")

        # Third figure, thrust, ISP, C* and expansion properties
        fig, ax = plt.subplots(2)
        ax[0].plot(
            self.data["Time      (s)"],
            np.array(self.data["Thrust    (N)"]) / 1e3,
            color="k",
        )
        ax[0].set_ylabel("Thrust (kN)")
        ax[0].grid(True)
        ax[1].plot(
            self.data["Time      (s)"],
            self.data["ISP     (m/s)"],
            color="tab:orange",
            label="ISP",
        )
        ax[1].plot(
            self.data["Time      (s)"],
            self.data["C*      (m/s)"],
            color="tab:red",
            label="C*",
        )
        ax[1].tick_params(axis="y", labelcolor="tab:red")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Velocity (m/s)")
        ax[1].legend()
        ax11 = ax[1].twinx()
        ax11.plot(self.data["Time      (s)"], self.data["Pe/Pb     (-)"], "k")
        ax11.set_ylabel("Pe/Pb")

        plt.show()

    # Print informations (min, average and max) in terminal
    def print(self):
        # find minimal value, average of list and maximal value
        def min_mean_max(values):
            return [min(values), np.mean(values), max(values)]

        # Print info about simulation with header
        print("")
        print("                |    min    |    mean   |    max    |")
        print("|---------------|-----------|-----------|-----------|")
        for key, value in self.data.items():
            if (
                key != "Time      (s)" and key != "P crit (psia)"
            ):  # No needs to print time and Pcrit
                if key == "Impulse  (Ns)":  # Only last value for impulse
                    print(f"| {key} |     -     |     -     | {value[-1]:5.3e} |")
                else:
                    value = min_mean_max(value)  # Min, average and max
                    print(
                        f"| {key} | {value[0]:5.3e} | {value[1]:5.3e} | {value[2]:5.3e} |"
                    )
        print("|---------------|-----------|-----------|-----------|")


# Motor definition fron json file
def mot_from_json(prm):
    tank = nox_tank(
        volume=prm["tank"]["v"], ullage=prm["tank"]["x"], pressure=prm["tank"]["p"]
    )
    # TO DO - Add tubing section
    # tub = tubing(r=prm['tubing']['r'], l=prm['tubing']['l'], cd=prm['tubing']['cd'], mu_c=prm['tubing']['mu'],
    #             rho_c=prm['tubing']['rho'], cv=prm['tubing']['cv'])
    tub = tubing(r=0, l=0, cd=0, mu_c=0, rho_c=0, cv=0)
    inj = injector(area=prm["injector"]["a"], cd=prm["injector"]["cd"])
    chamb = chamber(
        initial_port_radius=prm["chamber"]["ri"],
        final_port_radius=prm["chamber"]["re"],
        grain_length=prm["chamber"]["l"],
        fuel=prm["chamber"]["fuel"],
        efficiency=prm["chamber"]["efficiency"],
    )
    noz = nozzle(
        throat_area=prm["nozzle"]["at"], expansion_ratio=prm["nozzle"]["expan"]
    )
    anim = motor_animation()
    return hybrid_motor(tank, tub, inj, chamb, noz, anim)


if __name__ == "__main__":
    # Input handling
    try:
        opts, args = getopt.getopt(sys.argv, "f:p:", ["file=", "plot="])
    except getopt.GetoptError:
        print("mot.py -f <filename> -p")
        sys.exit(2)

    filename = ""
    plot_type = False
    save_data = True

    i = 0
    while i < (len(args) - 1) / 2:
        i += 1
        opt = args[2 * i - 1]
        if len(args) > 2:
            arg = args[2 * i]
        if opt == "-h":
            print("mot.py -f <filename> -s <saving_path>")
            sys.exit()
        elif opt in ("-f", "--file"):
            filename = arg
        elif opt in ("-s", "--save"):  # New option for saving path
            saving_path = arg
        elif opt in ("-p", "--plot"):
            plot_type = True

    print(f"filename : {filename}")
    if filename != "":
        try:
            with open(filename, "r") as file:
                parameters = json.loads(file.read())
        except:
            with open(os.path.join(saving_path,filename), "r") as file:
                parameters = json.loads(file.read())

        # Create motor object from parameters
        motor = mot_from_json(parameters)

        # Simulation
        motor.set_sim_properties(dt=parameters["dt"], pb=parameters["pb"])
        motor.reset()
        motor.burn(prints=True)

        # Plot simulation results
        motor.print()
        if plot_type:
            plt.ioff()
            motor.plot()
        if save_data:
            print('DATA SAVED')
            motor.save_data_to_csv(filename)
