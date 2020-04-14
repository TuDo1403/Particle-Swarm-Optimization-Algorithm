import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def get_plot_data(func, domain, global_optima, multivar=False):
    x = np.arange(domain[0], domain[1], 0.25)
    y = np.arange(domain[0], domain[1], 0.25)
    x, y = np.meshgrid(x, y)

    z = None
    if multivar:
        z = func([x, y])
    else:
        n = len(x)
        z = [func(np.vstack((x[i], y[i]))) for i in range(n)]
        z = np.array(z)

    opt_x = np.array([global_optima[0]])
    opt_y = np.array([global_optima[1]])

    return (x, y, z), (opt_x, opt_y)

def contour_plot(plt_points, go_point, domain, contour_dens=30):
    CS = plt.contour(plt_points[0], plt_points[1], plt_points[2], contour_dens, cmap=cm.seismic)
    plt.plot(go_point[0], go_point[1], "rx", label="Global Optimum")

    plt.xlim([domain[0], domain[1]])
    plt.ylim([domain[0], domain[1]])

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend(loc=1)


def scatter_plot(swarm, title=""):
    x, y = swarm[:, 0], swarm[:, 1]
    plt.plot(x, y, "b.")
    plt.suptitle(title)