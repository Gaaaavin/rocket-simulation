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


def plot(t, y):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("height", color=color)
    ax1.plot(t, y[0, :], label="height", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("velocity", color=color)
    ax2.plot(t, y[1, :], label="velocity", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # ax3 = ax1.twinx()
    # color = "tab:green"
    # ax3.set_ylabel("mass", color=color)
    # ax3.plot(t, y[2, :], label="mass", color=color)
    # ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend()
    # fig.show()
    fig.savefig("plot.png")


def main():
    with open("toy.yaml", 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    y0 = np.array([0, 0, config["dry_mass"]+config["fuel_mass"], 
    config["fuel_consumption"]*config["fuel_density"], config["isp"]*config["g"], config["g"]])
    
    t_eval = np.arange(config["tmax"]+1)
   
    solver = integrate.solve_ivp(ode, (0, config["tmax"]), y0, t_eval=t_eval)
    plot(solver.t, solver.y)



if __name__ == "__main__":
    main()