import numpy as np
from numpy.random import uniform
from numpy.random import rand
from numpy import round


def initialize_swarm(num_p, p_dim, l_bound, h_bound):
    swarm = uniform(low=l_bound, high=h_bound, size=(num_p, p_dim))
    return round(swarm, 4)

def swarm_evaluation(swarm, f_func):
    f_P = np.array(list(map(f_func, swarm)))
    return f_P[:, np.newaxis]

def best_positions(current_pos, prev_pos, f_func):
    f_current = swarm_evaluation(current_pos, f_func)
    f_prev = swarm_evaluation(prev_pos, f_func)
    return np.where(f_current < f_prev)[0]

def ring_topo_selection(p, f_p):
    g = np.zeros((np.shape(p)))
    n = len(g)
    for i in range(n):
        neighbors = [f_p[i-1], f_p[i], f_p[i+1]] if i < n-1 else [f_p[i-1], f_p[i], f_p[0]]
        idx = np.where(f_p == min(neighbors))[0]
        g[i] = p[np.random.choice(idx)]

    return g

np.set_printoptions(suppress=True)  # Prevent numpy exponential notation on print, default False
import matplotlib.pyplot as plt
from copy import deepcopy
import contour_plot as cp

def pso(user_config, f_func, num_dim, domain, 
        global_optima, multi_dim_var=False, ring_topo=False, contour_dens=50):
    plottable = True if num_dim == 2 else False
    if plottable:
        plt_points, go_point = cp.get_plot_data(f_func, domain, global_optima, multi_dim_var)

    p_dim = num_dim
    num_p = user_config.NUM_PARTICLES.value
    l_bound, h_bound = domain[0], domain[1]

    P = initialize_swarm(num_p, p_dim, l_bound, h_bound)
    p = deepcopy(P)
    w = user_config.INERTIA_WEIGHT.value
    c1, c2 = user_config.ACCEL_CONST.value[0], user_config.ACCEL_CONST.value[1]
    v = uniform(low=-abs(h_bound-l_bound), high=abs(h_bound-l_bound), size=(num_p, p_dim))
    
    num_gens = user_config.NUM_GENS.value
    for gen in range(num_gens):
        better_indices = best_positions(P, p, f_func)
        p[better_indices] = P[better_indices]
        f_p = swarm_evaluation(p, f_func)

        g = ring_topo_selection(p, f_p) if ring_topo else p[np.argmin(f_p, axis=0)]

        r_p, r_g = rand(), rand()
        v = w*v + c1*r_p * (p - P) + c2*r_g * (g - P)
        P = round(P+v, 4)
        
        if plottable:
            plt.clf()
            cp.contour_plot(plt_points, go_point, domain, contour_dens)
            cp.scatter_plot(P, "Gen:" + str(gen+1))
            plt.pause(0.00001)
        else:
            f_g = swarm_evaluation(g, f_func)
            if ring_topo:
                global_f_g = min(f_g)
                global_g = g[np.argmin(f_g, axis=0)]
                print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, global_g.flatten(), global_f_g))
            else:
                print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, g.flatten(), f_g))

    plt.show()

from enum import Enum
class PSOConfig(Enum):
    INERTIA_WEIGHT = 0.7298
    ACCEL_CONST = (1.49618, 1.49618)
    NUM_GENS = 100
    NUM_PARTICLES = 50

import fitness_function as ff

np.random.seed(1)
inertia_weight = 0.7298
accel_const = (1.49618, 1.49618)
num_gens = 100
num_particles = 50

## Booth function minimization test
user_config = PSOConfig
pso(user_config, ff.booth_function, 
    num_dim=2, domain=(-10,10), ring_topo=True,
    global_optima=(1, 3))

wait = input("PRESS ENTER TO CONTINUE")

# ## Rastrigin function minimization test
# pso(num_particles, inertia_weight, accel_const, num_gens, ff.rastrigin_function,
#     num_dim=2, domain=(-5.12, 5.12), ring_topo=False,
#     global_optima=(0,0), contour_dens=20)

# wait = input("PRESS ENTER TO CONTINUE")

# ## Beale function minimization test
# pso(num_particles, inertia_weight, accel_const, num_gens, ff.beale_function,
#     num_dim=2, domain=(-4.5,4.5), ring_topo=False, 
#     global_optima=(3, 0.5), multi_dim_var=False, contour_dens=100)

# print("## Done")

