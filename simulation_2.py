'''
This script simulates in the setting where
there is drag, and the front of the rocket
is a semi-sphere. The simulation is in 1 dimension
'''


import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import yaml


def ode(t, y, config):
    output = np.zeros_like(y)

    g = config["g"] * ((config["r_earth"] / (config["r_earth"] + y[0])) ** 2)

    if config["lapse_rate"] * y[0] / config["temperature0"] >= 1:
        air_density = 0
    else:
        air_density = (config["pressure0"] * config["molar_mass"]) / (config["gas_cst"] * config["temperature0"]) * \
            np.power(1 - (config["lapse_rate"] * y[0] / config["temperature0"]), 
            (g * config["molar_mass"]) / (config["gas_cst"] * config["lapse_rate"]) - 1)
    drag = -air_density * np.linalg.norm(y[1]) * np.pi * config["diameter"] * y[1]
    

    output[0] = y[1]
    if y[2] > config["dry_mass"]:
        output[1] = (1 / y[2]) * (-config["fuel_consumption"] * config["fuel_density"]) * (config["isp"] * -g) + \
            -g + drag / y[2]
        output[2] = -config["fuel_consumption"] * config["fuel_density"]
    else:
        output[1] = -g + drag / y[2]
        output[2] = 0

    return output

def terminate(t, y):
    if y[0] <= 0:
        return 0
    else:
        return y[0]


def plot(t, y, accelerations):
    fig = plt.figure()
    fig.set_size_inches(10, 15)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("altitude(m)")
    ax1.grid()
    ax1.plot(t, y[0, :], 'b', label="altitude")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_ylabel("velocity(m/s)")
    ax2.grid()
    ax2.plot(t, y[1, :], 'r', label="velocity")
    
    # ax3 = fig.add_subplot(3, 1, 3)
    # ax3.set_ylabel("acceleration(m/s^2)")
    # ax3.grid()
    # ax3.plot(t[:-1], acceleration[:-1], 'g', label="acceleration")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_ylabel("acceleration(m/s^2)")
    ax3.grid()
    ax3.plot(t, accelerations[0], label="g")
    ax3.plot(t, accelerations[1], label='a_drag')
    ax3.plot(t, accelerations[2], label='a_burn')
    ax3.plot(t, accelerations[3], label='a_total')
    ax3.legend()

    fig.tight_layout()
    # fig.legend()
    # fig.show()
    fig.savefig("plot2.png")


def get_acceleration(y, config):
    g = config["g"] * ((config["r_earth"] / (config["r_earth"] + y[0, :])) ** 2)
    g_accleeration = -g
    
    air_density = (config["pressure0"] * config["molar_mass"]) / (config["gas_cst"] * config["temperature0"]) * \
            np.power(1 - (config["lapse_rate"] * y[0, :] / config["temperature0"]), 
            (g * config["molar_mass"]) / (config["gas_cst"] * config["lapse_rate"]) - 1)
    air_density[config["lapse_rate"] * y[0, :] / config["temperature0"] >= 1] = 0
    drag = -air_density * np.absolute(y[1, :]) * np.pi * config["diameter"] * y[1, :]
    drag_acceleration = drag / y[2, :]

    burn_acceleration = 1 / y[2, :] * (-config["fuel_consumption"] * config["fuel_density"]) * (config["isp"] * -g)
    burn_acceleration[y[2, :] <= config["dry_mass"]] = 0

    total_acceleration = g_accleeration + drag_acceleration + burn_acceleration

    return (g_accleeration, drag_acceleration, burn_acceleration, total_acceleration)



if __name__ == "__main__":
    with open("config_2.yaml", 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    y0 = np.array([1e-12, 0, config["dry_mass"]+config["fuel_mass"]])
    
    t_eval = np.arange(0, config["tmax"], 0.5)
    terminate.terminal = True
    # print(t_eval)
    solver = integrate.solve_ivp(lambda t, y: ode(t, y, config), (0, config["tmax"]), y0, t_eval=t_eval, events=terminate)
    # print(solver.t)
    # acceleration = np.zeros_like(solver.t)
    # for i in range(acceleration.size - 1):
    #     acceleration[i] = (solver.y[1, i+1] - solver.y[1, i]) / 0.5
    # plot(solver.t, solver.y, acceleration)
    accelerations = get_acceleration(solver.y, config)
    plot(solver.t, solver.y, accelerations)
