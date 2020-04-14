import numpy as np
from numpy.random import uniform
from numpy.random import rand
from numpy import round


def initialize_swarm(num_p, p_params, l_bound, h_bound):
    swarm = uniform(low=l_bound, high=h_bound, size=(num_p, p_params))
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
    
    print("## Searching...")
    gens = range(user_options.NUM_GENS)
    for gen in gens:
        better_indices = best_positions(P, p, func_inf.F_FUNC)
        p[better_indices] = P[better_indices]
        f_p = swarm_evaluation(p, func_inf.F_FUNC)

        g = ring_topo_selection(p, f_p) if user_options.RING_TOPO else p[np.argmin(f_p, axis=0)]

        r_p, r_g = rand(), rand()
        v = w*v + c1*r_p * (p - P) + c2*r_g * (g - P)
        P = round(P+v, 4)
        
        if plottable:
            plot_data(data_points, go_point, func_inf.DOMAIN, func_inf.NAME, contour_dens, P, gen)
        else:
            pso_print_gen_result(g, func_inf.F_FUNC, user_options.RING_TOPO, gen)

    plt.show()

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
        print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, global_g.flatten(), global_f_g))
    else:
        print("## Gen {}:\n Best Position: {} \n Fitness: {}.".format(gen+1, g.flatten(), f_g))


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


import fitness_function as ff
from fitness_function import FuncInf
from surface_plot import surface_plot

## Set up useful parameters
np.random.seed(1)
user_options = PSOConfig    
##

## Booth function minimization test
func_info = FuncInf(name="Booth function", num_params=2, multi_dim_params=False, 
                    domain=(-10, 10), global_optima=(1, 3), func=ff.booth_function)

# Visualize function with surface plot
surface_plot(func_info, contour=True)

# Search for global optima
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=50)
wait = input("PRESS ENTER TO CONTINUE")
##



## Rastrigin function minimization test
func_info = FuncInf(name="Rastrigin function", num_params=2, multi_dim_params=True, 
                    domain=(-5.12, 5.12), global_optima=(0, 0), func=ff.rastrigin_function)

# Visualize function with surface plot
surface_plot(func_info, contour=False)

# Search for global optima
user_options.RING_TOPO = False
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=20)
wait = input("PRESS ENTER TO CONTINUE")
##



## Beale function minimization test
func_info = FuncInf(name="Beale function", num_params=2, multi_dim_params=False, 
                    domain=(-4.5, 4.5), global_optima=(3, 0.5), func=ff.beale_function)

# Visualize function with surface plot
surface_plot(func_info, contour=True)

# Search for global optima
PSOConfig.NUM_GENS = 50
PSOConfig.RING_TOPO = False     # Use Star-Topology selection method
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=70)
##

print("## Done")

