import fitness_function as ff
from fitness_function import FuncInf
from surface_plot import surface_plot
from PSO import PSOConfig
from PSO import particle_swarm_optimization
import numpy as np

np.set_printoptions(suppress=True)  # Prevent numpy exponential notation on print, default False


## Set up useful parameters
np.random.seed(18521578)
user_options = PSOConfig   
user_options.NUM_PARTICLES = 32
user_options.NUM_GENS = 50
##


## Cross-in tray function minimization test
user_options.RING_TOPO = False
global_optima = ([1.34941, 1.34941, -1.34941, -1.34941], [-1.34941, 1.34941, 1.34941, -1.34941])
func_info = FuncInf(name="Cross-in tray function", num_params=2, multi_dim_params=False, 
                    domain=(-10, 10), global_optima=global_optima, func=ff.cross_in_tray_function)

# Visualize function with surface plot
surface_plot(func_info, contour=False)

# Search for global optima
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=20)
wait = input("PRESS ENTER TO CONTINUE")
##


## Himmelblau function minimization test
user_options.RING_TOPO = False
global_optima = ([3, -2.805118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.848126])
func_info = FuncInf(name="Himmelblau function", num_params=2, multi_dim_params=False, 
                    domain=(-5, 5), global_optima=global_optima, func=ff.himmelblau_function)

# Visualize function with surface plot
surface_plot(func_info, contour=False)

# Search for global optima
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=50)
wait = input("PRESS ENTER TO CONTINUE")
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
user_options.RING_TOPO = False
func_info = FuncInf(name="Rastrigin function", num_params=2, multi_dim_params=True, 
                    domain=(-5.12, 5.12), global_optima=(0, 0), func=ff.rastrigin_function)

# Visualize function with surface plot
surface_plot(func_info, contour=False)

# Search for global optima
# user_options.RING_TOPO = False  # Use Star-Topology selection method
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=20)
wait = input("PRESS ENTER TO CONTINUE")
##



## Beale function minimization test
user_options.RING_TOPO = False
func_info = FuncInf(name="Beale function", num_params=2, multi_dim_params=False, 
                    domain=(-4.5, 4.5), global_optima=(3, 0.5), func=ff.beale_function)

# Visualize function with surface plot
surface_plot(func_info, contour=False)

# Search for global optima
particle_swarm_optimization(user_options, func_info, plot=True, contour_dens=70)
##


print("## Done")