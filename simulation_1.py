'''
This script simulates in the setting where
there is no drag. The simulation is in 1 dimension
'''


import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import yaml


def ode(t, y):
    output = np.zeros_like(y)
    output[0] = y[1]
    output[1] = 1 / y[2] * (-y[3]) * y[4] + y[5]
    output[2] = -y[3]
    return output


def plot(t, y, acceleration):
    fig = plt.figure()
    fig.set_size_inches(10, 15)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("height(m)")
    ax1.plot(t, y[0, :], 'b', label="height")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_ylabel("velocity(m/s)")
    ax2.plot(t, y[1, :], 'r', label="velocity")
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_ylabel("acceleration(m/s^2)")
    ax3.plot(t[:-2], acceleration, 'g', label="acceleration")

    fig.tight_layout()
    # fig.legend()
    # fig.show()
    fig.savefig("plot.png")


def main():
    with open("toy.yaml", 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    y0 = np.array([0, 0, config["dry_mass"]+config["fuel_mass"], 
    config["fuel_consumption"]*config["fuel_density"], config["isp"]*config["g"], config["g"]])
    
    t_eval = np.arange(config["tmax"]+1)
   
    solver = integrate.solve_ivp(ode, (0, config["tmax"]), y0, t_eval=t_eval)
    acceleration = np.zeros(config["tmax"]-1)
    for i in range(config["tmax"]-1):
        acceleration[i] = solver.y[1, i+1] - solver.y[1, i]
    plot(solver.t, solver.y, acceleration)



if __name__ == "__main__":
    main()