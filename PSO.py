import numpy as np
from numpy.random import uniform
from numpy.random import rand
from numpy import round


def initialize_swarm(num_p, p_params, l_bound, h_bound):
    swarm = uniform(low=l_bound, high=h_bound, size=(num_p, p_params))
    return round(swarm, 5)

def swarm_evaluation(swarm, f_func):
    f_P = np.array(list(map(f_func, swarm)))
    return f_P[:, np.newaxis]

def better_sol(current_pos, prev_pos, f_func):
    f_current = swarm_evaluation(current_pos, f_func)
    f_prev = swarm_evaluation(prev_pos, f_func)
    return np.where(f_current < f_prev)[0], f_current, f_prev

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
def particle_swarm_optimization(user_options, func_inf, plot=True, contour_dens=50):
    plottable = True if plot and func_inf.NUM_PARAMS == 2 else False
    if plottable:
        data_points, go_point = cp.get_plot_data(func_inf)

    p_params = func_inf.NUM_PARAMS
    num_p = user_options.NUM_PARTICLES
    l_bound, h_bound = func_inf.DOMAIN[0], func_inf.DOMAIN[1]

    P = initialize_swarm(num_p, p_params, l_bound, h_bound)
    p = deepcopy(P)
    w = user_options.INERTIA_WEIGHT
    c1, c2 = user_options.ACCEL_CONST[0], user_options.ACCEL_CONST[1]
    v = uniform(low=-abs(h_bound-l_bound), high=abs(h_bound-l_bound), size=(num_p, p_params))
    
    best_sol = None
    print("## Searching...")
    gens = range(user_options.NUM_GENS)
    for gen in gens:
        better_indices, f_P, f_p = better_sol(P, p, func_inf.F_FUNC)

        p[better_indices] = P[better_indices]
        f_p[better_indices] = f_P[better_indices]

        g = ring_topo_selection(p, f_p) if user_options.RING_TOPO else p[np.argmin(f_p, axis=0)]
        f_g = None                          #

        r_p, r_g = rand(), rand()
        v = w*v + c1*r_p * (p - P) + c2*r_g * (g - P)
        P = round(P+v, 5)
        
        if plottable:
            plot_data(data_points, go_point, func_inf.DOMAIN, func_inf.NAME, contour_dens, P, gen)
        else:
            f_g = pso_print_gen_result(g, func_inf.F_FUNC, user_options.RING_TOPO, gen)

        best_sol = g[np.argmin(f_g, axis=0)] if user_options.RING_TOPO else p[np.argmin(f_p, axis=0)]

    if plottable:
        plt.show()

    return (best_sol.flatten(), round(func_inf.F_FUNC(best_sol.T)[0], 4))

def plot_data(data_points, go_point, domain, f_name, contour_dens, P, gen):
    plt.clf()
    cp.contour_plot(data_points, go_point, domain, contour_dens)
    cp.scatter_plot(P, "Gen:" + str(gen+1), f_name)
    plt.pause(0.00001)

def pso_print_gen_result(g, f_func, ring_topo, gen):
    f_g = swarm_evaluation(g, f_func)
    if ring_topo:
        global_f_g = min(f_g)
        global_g = g[np.argmin(f_g, axis=0)]
        print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, global_g.flatten(), global_f_g.flatten()[0]))
    else:
        print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, g.flatten(), f_g.flatten()[0]))
    
    return f_g


class PSOConfig:
    INERTIA_WEIGHT = 0.7298
    ACCEL_CONST = (1.49618, 1.49618)
    NUM_GENS = 100
    NUM_PARTICLES = 50
    RING_TOPO = True    # Using Ring-Topology selection as default selection mode
    
    def __init__(self, inertia_weight, accel_const, num_gens, num_particles, ring_topo):
        self.INERTIA_WEIGHT = inertia_weight
        self.ACCEL_CONST = accel_const
        self.NUM_GENS = num_gens
        self.NUM_PARTICLES = num_particles
        self.RING_TOPO = ring_topo

