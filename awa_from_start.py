import numpy as np
import re
import matplotlib.pyplot as plt
import h5py as h5
import os
import glob
import multiprocessing
import functools
import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
import contextlib
from scipy.integrate import simps
from scipy.interpolate import CubicSpline
plt.rcParams['figure.dpi'] =200
matplotlib.rcParams.update({'font.size': 12})
from scipy.integrate import odeint
import prediction_functions as pf
n0_base = 2e20
n_pe= n0_base
kpn1=pf.calculate_kpn1(n_pe)
print(kpn1)
window_size=6
num_points=500
# Declare intrinsic parameters
emit_nx = 200e-6
emit_ny = 2e-6
sigma_z = 600e-6
Q = 2e-9
E = 58e6
gamma = E/0.511e6 + 1
gamma_b = gamma
#Define plasma parameters
points = 30
#emittance_n_x = 0.9e-6
length = 1e-2
sigma = 1e-2
#extra_length = 5e-2
zs = np.linspace(0,0.04, points)

#ellipticity = 2
ellipticity_2 = 1
plasma_upramp_end = 0
model='gaussian'
initial_conditions = [1/pf.optimize_kbeta_parameters_single_gaussian(n_pe, E, emit_nx, emit_ny, sigma_z, Q, 0)[0], 0,1/pf.optimize_kbeta_parameters_single_gaussian(n_pe, E, emit_nx, emit_ny, sigma_z, Q, 0)[1],0]  # Initial values of x and y
initial_conditions_axi = [1/pf.kbeta_matched_x(n_pe, gamma_b, 1), 0,1/pf.kbeta_matched_y(n_pe, gamma_b, 1),0]  # Initial values of x and y
print(f'Inside plasma parameters: {initial_conditions_axi}')
# Create a time grid
#t_span = np.linspace(0, 10, 100)
# Solve the ODE using odeint
density = []
xp_all=[]
yp_all=[]
ellipticity_all=[]
density_ellipticity_map = {}
zs_all = []
tolerance = 5e-3
points = 60
zs_drift = np.linspace(zs.min(),zs.max(),points)
density_pic=[]
for z in zs_drift:
    density_pic.append(pf.plasma_density(z,model,plasma_upramp_end,sigma,length))
#zs_drift_2 = np.linspace(zs.max(),zs.min(),points)
#solution = odeint(pf.ode_plasma_estimate_ellipticity, initial_conditions, zs_drift,  args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density))
solution_axi = odeint(pf.ode_plasma, initial_conditions_axi, zs_drift,  args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b,1))
#beta_x= solution[:,0]
#beta_y = solution[:,2]
#plt.plot(zs,beta_x*1e2)
#plt.plot(zs,beta_y*1e2)
#plt.savefig('Initial solution.png')
#plt.close()
# Extract the solution
# Extract the solution axisymmetric
beta_x_values_axi = solution_axi[:, 0]
alpha_x_values_axi = solution_axi[:, 1]
beta_y_values_axi = solution_axi[:, 2]
alpha_y_values_axi = solution_axi[:, 3]


beta_x_end_axi = beta_x_values_axi[-1]
alpha_x_end_axi = alpha_x_values_axi[-1]
beta_y_end_axi = beta_y_values_axi[-1]
alpha_y_end_axi = alpha_y_values_axi[-1]

#plt.plot(zs_drift,beta_x_values)
print(density_pic)
plt.plot(zs_drift,beta_x_values_axi)
#plt.plot(zs_drift,beta_y_values)
plt.plot(zs_drift,beta_y_values_axi)
plt.plot(zs_drift,density_pic)
plt.show()
zs_drift_2 = -np.sort(np.asarray(zs_drift))[1:]
print(zs_drift)
initial_drift = [beta_x_end,alpha_x_end,beta_y_end,alpha_y_end]
initial_drift_axi = [beta_x_end_axi,alpha_x_end_axi,beta_y_end_axi,alpha_y_end_axi]
initial_drift_kpn1 = [beta_x_end/kpn1[0],alpha_x_end,beta_y_end/kpn1[0],alpha_y_end]
initial_drift_kpn1_axi = [beta_x_end_axi/kpn1[0],alpha_x_end_axi,beta_y_end_axi/kpn1[0],alpha_y_end_axi]
initial_sigma_x = pf.sigma_beam(beta_x_end,emit_nx,gamma)
initial_sigma_y = pf.sigma_beam(beta_y_end,emit_ny,gamma)
initial_sigma_x_axi = pf.sigma_beam(beta_x_end_axi,emit_nx,gamma)
initial_sigma_y_axi = pf.sigma_beam(beta_y_end_axi,emit_ny,gamma)
print(f'Charge: {Q}')
beam_density = pf.calculate_beam_density(Q,initial_sigma_x,initial_sigma_y, sigma_z)
beam_density_axi = pf.calculate_beam_density(Q,initial_sigma_x_axi,initial_sigma_y_axi, sigma_z)
normalized_beam = beam_density/n_pe
normalized_beam_axi = beam_density_axi/n_pe
print(r'Normalized beam density initially:' + str(normalized_beam))
print(r'Normalized beam density axi initially:' + str(normalized_beam_axi))
print(rf'Initial drift parameters ($\beta_x$,$\alpha_x$,$\beta_y$,$\alpha_y$): {initial_drift}')
print(rf'Initial drift parameters axi ($\beta_x$,$\alpha_x$,$\beta_y$,$\alpha_y$): {initial_drift_axi}')
print(rf'Initial drift parameters at PIC ($\beta_x/kpn1$,$\alpha_x$,$\beta_y/kpn1$,$\alpha_y$): {initial_drift_kpn1}')
print(rf'Initial drift parameters at PIC axi ($\beta_x/kpn1$,$\alpha_x$,$\beta_y/kpn1$,$\alpha_y$): {initial_drift_kpn1_axi}')
print(rf'Initial spot size PIC ($\sigma_x/kpn1$,$\sigma_y$/kpn1,sigma_z/kpn1): {initial_sigma_x/kpn1[0],initial_sigma_y/kpn1[0],sigma_z/kpn1[0]}')
print(rf'emit_nx (PIC): {emit_nx/kpn1[0]}')
print(rf'emit_ny (PIC): {emit_ny/kpn1[0]}')
print(rf'emit_nx axi (PIC): {np.sqrt(emit_nx*emit_ny)/kpn1[0]}')
print(rf'emit_ny axi (PIC): {np.sqrt(emit_nx*emit_ny)/kpn1[0]}')

