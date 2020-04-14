import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def surface_plot(func_inf, contour=True):
    domain = func_inf.DOMAIN
    multi_dim_params = func_inf.MULTI_DIM_PARAMS
    func = func_inf.F_FUNC
    title = func_inf.NAME
    global_optima = func_inf.GLOBAL_OPTIMA
    
    x = np.arange(domain[0], domain[1], 0.25)
    y = np.arange(domain[0], domain[1], 0.25)
    x, y = np.meshgrid(x, y)

    z = None
    if not multi_dim_params:
        z = func([x, y])
    else:
        n = len(x)
        z = [func(np.vstack((x[i], y[i]))) for i in range(n)]
        z = np.array(z)

    print("## Plotting...")
    fig = plt.figure()
    ax = Axes3D(fig)

    if contour:
        opt_x = np.array([global_optima[0]])
        opt_y = np.array([global_optima[1]])
        opt_x, min_y = np.meshgrid(opt_x, opt_y)
        if not multi_dim_params:
            opt_z = func([opt_x, opt_y])
        else:
            opt_z = func(np.vstack((opt_x, opt_y)))

        surf = ax.contour3D(x, y, z, 500, cmap=cm.rainbow)
        ax.scatter3D(opt_x, opt_y, opt_z, marker="o", label="Global Optimum")
    else:
        surf = ax.plot_surface(x, y, z, cmap=cm.rainbow)
    
    ax.set_xlim([domain[0]+1, domain[1]+1])
    ax.set_ylim([domain[0]+1, domain[1]+1])

    ax.set_xticks(np.linspace(domain[0], domain[1], 5))
    ax.set_yticks(np.linspace(domain[0], domain[1], 5))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("F(x, y)")

    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.suptitle(title)

    ax.legend(loc="lower right")

    plt.show()

# import fitness_function as ff

# surface_plot(ff.booth_function, domain=(-10, 10), global_optima=(1, 3), 
#             title="Booth Function", multi_dim_var=False, contour=True)
# print("Program paused")
# wait = input("PRESS ENTER TO CONTINUE")


# surface_plot(ff.beale_function, domain=(-4.5, 4.5), global_optima=(3, 0.5),
#             title="Beale Function", multi_dim_var=False, contour=False)
# print("Program paused")
# wait = input("PRESS ENTER TO CONTINUE")


# surface_plot(ff.rastrigin_function, domain=(-5.12, 5.12), global_optima=(0, 0), 
#             title="Rastrigin Function", multi_dim_var=True, contour=True)

# print("## Done")
