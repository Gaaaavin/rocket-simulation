'''
This script simulates in the setting where
there is drag, and the front of the rocket
is a cone. The simulation is in 1 dimension
'''


import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml


def ode(t, y, config, theta):
    output = np.zeros_like(y)

    g = config["g"] * ((config["r_earth"] / (config["r_earth"] + y[0])) ** 2)

    if config["lapse_rate"] * y[0] / config["temperature0"] >= 1:
        air_density = 0
    else:
        air_density = (config["pressure0"] * config["molar_mass"]) / (config["gas_cst"] * config["temperature0"]) * \
            np.power(1 - (config["lapse_rate"] * y[0] / config["temperature0"]), 
            (g * config["molar_mass"]) / (config["gas_cst"] * config["lapse_rate"]) - 1)
    drag = -air_density * np.linalg.norm(y[1]) * np.pi * config["diameter"] * y[1] * np.sin(theta)
    

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


def plot(t, y, accelerations, thetas):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    fig.set_size_inches(15, 10)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("altitude(m)")
    ax1.grid()
    for i in range(len(t)):
        ax1.plot(t[i], y[i][0, :], label="theta={:.4f}".format(thetas[i]))
    ax1.plot(t[9], 44330*np.ones_like(t[9]), color='grey', linestyle='dashed', label="boundry of atmosphere")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_ylabel("velocity(m/s)")
    ax2.grid()
    for i in range(len(t)):
        ax2.plot(t[i], y[i][1, :], label="theta={:.4f}".format(thetas[i]))
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_ylabel("acceleration(m/s^2)")
    ax3.grid()
    for i in range(len(t)):
        ax3.plot(t[i], accelerations[i][3], label="theta={:.4f}".format(thetas[i]))
    ax3.legend()

    fig.tight_layout()
    # fig.legend()
    # fig.show()
    fig.savefig("presentation_plot/plot_3.png")


def get_acceleration(y, config, theta):
    g = config["g"] * ((config["r_earth"] / (config["r_earth"] + y[0, :])) ** 2)
    g_accleeration = -g
    
    air_density = (config["pressure0"] * config["molar_mass"]) / (config["gas_cst"] * config["temperature0"]) * \
            np.power(1 - (config["lapse_rate"] * y[0, :] / config["temperature0"]), 
            (g * config["molar_mass"]) / (config["gas_cst"] * config["lapse_rate"]) - 1)
    air_density[config["lapse_rate"] * y[0, :] / config["temperature0"] >= 1] = 0
    drag = -air_density * np.absolute(y[1, :]) * np.pi * config["diameter"] * y[1, :] * np.sin(theta)
    drag_acceleration = drag / y[2, :]

    burn_acceleration = 1 / y[2, :] * (-config["fuel_consumption"] * config["fuel_density"]) * (config["isp"] * -g)
    burn_acceleration[y[2, :] <= config["dry_mass"]] = 0

    total_acceleration = g_accleeration + drag_acceleration + burn_acceleration

    return (g_accleeration, drag_acceleration, burn_acceleration, total_acceleration)


if __name__ == "__main__":
    with open("config_3.yaml", 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    y0 = np.array([1e-12, 0, config["dry_mass"]+config["fuel_mass"]])
    
    t_eval = np.arange(0, config["tmax"], 0.5)
    terminate.terminal = True

    t_records = []
    y_records = []
    accelerations_records = []
    thetas = np.linspace(np.pi / 2, 0, 10, endpoint=False)
    for theta in thetas:
        solver = integrate.solve_ivp(lambda t, y: ode(t, y, config, theta), (0, config["tmax"]), y0, t_eval=t_eval, events=terminate)
        accelerations = get_acceleration(solver.y, config, theta)
        t_records.append(solver.t)
        y_records.append(solver.y)
        accelerations_records.append(accelerations)
    
    plot(t_records, y_records, accelerations_records, thetas)
