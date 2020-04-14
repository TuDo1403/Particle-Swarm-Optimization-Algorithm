import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def get_plot_data(func_inf):
    x = np.arange(func_inf.DOMAIN[0], func_inf.DOMAIN[1], 0.25)
    y = np.arange(func_inf.DOMAIN[0], func_inf.DOMAIN[1], 0.25)
    x, y = np.meshgrid(x, y)

    z = None
    if not func_inf.MULTI_DIM_PARAMS:
        z = func_inf.F_FUNC([x, y])
    else:
        n = len(x)
        z = [func_inf.F_FUNC(np.vstack((x[i], y[i]))) for i in range(n)]
        z = np.array(z)

    opt_x = np.array([func_inf.GLOBAL_OPTIMA[0]])
    opt_y = np.array([func_inf.GLOBAL_OPTIMA[1]])

    return (x, y, z), (opt_x, opt_y)

def contour_plot(plt_points, go_point, domain, contour_dens):
    
    plt.contour(plt_points[0], plt_points[1], plt_points[2], contour_dens, cmap=cm.seismic)
    plt.plot(go_point[0], go_point[1], "rx", label="Global Optimum ({}, {})".format(go_point[0][0], go_point[1][0]), markersize=10)

    plt.xlim([domain[0], domain[1]])
    plt.ylim([domain[0], domain[1]])

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc=1)

def scatter_plot(swarm, title="", f_name=""):
    x, y = swarm[:, 0], swarm[:, 1]
    plt.plot(x, y, "b.")
    plt.suptitle(f_name, size=15)
    plt.title(title, loc='center')