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
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
plt.rcParams['figure.dpi'] =200
matplotlib.rcParams.update({'font.size': 12})
from scipy.integrate import odeint
import prediction_functions as pf
n0_base = 1e20
n_pe= n0_base
kpn1=pf.calculate_kpn1(n_pe)
print(kpn1)
window_size=5
num_points=300
# Declare intrinsic parameters
emit_nx = 400e-6
emit_ny = 20e-6
sigma_z = 600e-6
Q =2e-9
E = 58e6
gamma = E/0.511e6 + 1
gamma_b = gamma
#Define plasma parameters
points = 100
#emittance_n_x = 0.9e-6
length = 1e-2
sigma = 0.25e-2
#extra_length = 5e-2
zs = np.linspace(0,length+4*sigma, points)
#ellipticity = 2
ellipticity_2 = 1
plasma_upramp_end = 0
model='gaussian'
initial_conditions = [1/pf.optimize_kbeta_parameters_single_gaussian(n_pe, E, emit_nx, emit_ny, sigma_z, Q, 0,window_size,num_points)[0], 0,1/pf.optimize_kbeta_parameters_single_gaussian(n_pe, E, emit_nx, emit_ny, sigma_z, Q, 0,window_size,num_points)[1],0]  # Initial values of x and y
initial_conditions_axi = [1/pf.kbeta_matched_x(n_pe, gamma_b, 1), 0,1/pf.kbeta_matched_y(n_pe, gamma_b, 1),0]  # Initial values of x and y
print(f'Inside plasma parameters: {initial_conditions}')
# Create a time grid
#t_span = np.linspace(0, 10, 100)
# Solve the ODE using odeint
density = []
xp_all=[]
yp_all=[]
ellipticity_all=[]
density_ellipticity_map = {}
zs_all = []
tolerance = 1e-4
for z in zs:
    density_value = pf.plasma_density(z, model, plasma_upramp_end, sigma, length)
    if density_value not in density_ellipticity_map and density_value>tolerance:
        xp, yp, ellipticity = pf.optimize_boundary_parameters_gaussian_single(density_value*n_pe, E, emit_nx,
                                                           emit_ny, sigma_z, Q, 0, window_size, num_points)
        density_ellipticity_map[density_value] = ellipticity
        density.append(density_value)
        # Append results to the lists
        zs_all.append(z)
        xp_all.append(xp)
        yp_all.append(yp)
        ellipticity_all.append(ellipticity)
#plt.plot(density, xp_all)
#plt.plot(density, yp_all)
#plt.plot(density, ellipticity_all)
#plt.savefig('ellipticity.png')
# Create the interpolation function for ellipticity

# Extract unique densities and their corresponding ellipticities
unique_densities = np.array(list(density_ellipticity_map.keys()))
ellipticity_all = np.array(list(density_ellipticity_map.values()))

# Create the interpolation function for ellipticity
ellipticity_interpolator = interp1d(unique_densities, ellipticity_all, kind='linear', fill_value=(1, 1), bounds_error=False)
min_density = unique_densities.min()
print(f'Minimum density: {min_density}')
zs = np.linspace(0,length+4*sigma, points)
zs_drift = np.linspace(zs.min(),zs.max(),points)
#zs_drift_2 = -np.sort(np.asarray(zs_drift))[1:]
#zs_drift = np.concatenate((zs_drift,zs_drift_2))
density_pic=[]
for z in zs_drift:
    density_pic.append(pf.plasma_density(z,model,plasma_upramp_end,sigma,length))
solution = odeint(pf.ode_plasma_estimate_ellipticity, initial_conditions, zs_drift,  args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density))
solution_axi = odeint(pf.ode_plasma, initial_conditions_axi, zs_drift,  args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b,1))
#beta_x= solution[:,0]
#beta_y = solution[:,2]
#plt.plot(zs,beta_x*1e2)
#plt.plot(zs,beta_y*1e2)
#plt.savefig('Initial solution.png')
#plt.close()
# Extract the solution
beta_x_values = solution[:, 0]

alpha_x_values = solution[:, 1]
beta_y_values = solution[:, 2]
alpha_y_values = solution[:, 3]

# Extract the solution axisymmetric
beta_x_values_axi = solution_axi[:, 0]
alpha_x_values_axi = solution_axi[:, 1]
beta_y_values_axi = solution_axi[:, 2]
alpha_y_values_axi = solution_axi[:, 3]

beta_x_start = beta_x_values[0]
alpha_x_start = alpha_x_values[0]
beta_y_start = beta_y_values[0]
alpha_y_start = alpha_y_values[0]

beta_x_end = beta_x_values[-1]
alpha_x_end = alpha_x_values[-1]
beta_y_end = beta_y_values[-1]
alpha_y_end = alpha_y_values[-1]

beta_x_end_axi = beta_x_values_axi[-1]
alpha_x_end_axi = alpha_x_values_axi[-1]
beta_y_end_axi = beta_y_values_axi[-1]
alpha_y_end_axi = alpha_y_values_axi[-1]
"""
plt.plot(zs_drift,beta_x_values*1e2, label ='Beta x (cm)')
plt.plot(zs_drift,beta_x_values_axi*1e2, label = 'Beta x (axi) (cm)')
plt.plot(zs_drift,beta_y_values*1e2,label = 'Beta y (cm)')
#plt.plot(zs_drift,beta_y_values_axi)
plt.plot(zs_drift,density_pic,label='Density profile')
plt.plot(zs_all,ellipticity_all,label='Ellipticity profile')
plt.legend()
plt.show()
plt.savefig('ellipticity_.png')
"""
#plt.plot(zs,beta_x_values)
zs_drift = np.linspace(zs.max(),zs.min(),points)
zs_drift_2 = -np.sort(np.asarray(zs_drift))[1:]
#extra_array = np.linspace(zs.max()+extra_length,zs.max(),points)
#print(zs_2)
#print(extra_array)
zs_drift = np.concatenate((zs_drift,zs_drift_2))

density_pic=[]
for z in zs_drift:
    density_pic.append(pf.plasma_density(z,model,plasma_upramp_end,sigma,length))
pf.plasma_export_PIC(density_pic,np.flip(zs_drift+zs_drift[0]),n_pe,1e-4)
pf.PIC_corrector('plasma_density_PIC.txt')
print(zs_drift)
initial_drift = [beta_x_end,alpha_x_end,beta_y_end,alpha_y_end]
initial_drift_axi = [beta_x_end_axi,alpha_x_end_axi,beta_y_end_axi,alpha_y_end_axi]
initial_drift_kpn1 = [beta_x_end/kpn1[0],alpha_x_end,beta_y_end/kpn1[0],alpha_y_end]
initial_drift_kpn1_axi = [beta_x_end_axi/kpn1[0],alpha_x_end_axi,beta_y_end_axi/kpn1[0],alpha_y_end_axi]
initial_sigma_x = pf.sigma_beam(beta_x_end,emit_nx,gamma)
initial_sigma_y = pf.sigma_beam(beta_y_end,emit_ny,gamma)
final_sigma_x = pf.sigma_beam(beta_x_start,emit_nx,gamma)
final_sigma_y = pf.sigma_beam(beta_y_start,emit_ny,gamma)
initial_sigma_x_axi = pf.sigma_beam(beta_x_end_axi,emit_nx,gamma)
initial_sigma_y_axi = pf.sigma_beam(beta_y_end_axi,emit_ny,gamma)
print(f'Charge: {Q}')
beam_density = pf.calculate_beam_density(Q,initial_sigma_x,initial_sigma_y, sigma_z)
beam_density_axi = pf.calculate_beam_density(Q,initial_sigma_x_axi,initial_sigma_y_axi, sigma_z)
#beam_density_axi_final = pf.calculate_beam_density(Q,final_sigma_x_axi,final_sigma_y_axi, sigma_z)
beam_density_final = pf.calculate_beam_density(Q,final_sigma_x,final_sigma_y, sigma_z)
normalized_beam = beam_density/n_pe
normalized_beam_axi = beam_density_axi/n_pe
normalized_beam_final = beam_density_final/n_pe
print(r'Normalized beam density initially (PIC):' + str(normalized_beam))
print(r'Normalized beam density finally:' + str(normalized_beam_final))
print(r'Normalized beam density axi initially:' + str(normalized_beam_axi))
print(r'Normalized beam density axi finally:' + str(normalized_beam_final))
print(rf'Initial drift parameters ($\beta_x$,$\alpha_x$,$\beta_y$,$\alpha_y$): {initial_drift}')
print(rf'Initial drift parameters axi ($\beta_x$,$\alpha_x$,$\beta_y$,$\alpha_y$): {initial_drift_axi}')
print(rf'Initial drift parameters at PIC ($\beta_x/kpn1$,$\alpha_x$,$\beta_y/kpn1$,$\alpha_y$): {initial_drift_kpn1}')
print(rf'Initial drift parameters at PIC axi ($\beta_x/kpn1$,$\alpha_x$,$\beta_y/kpn1$,$\alpha_y$): {initial_drift_kpn1_axi}')
print(rf'Initial spot size PIC ($\sigma_x/kpn1$,$\sigma_y$/kpn1,sigma_z/kpn1): {initial_sigma_x/kpn1[0],initial_sigma_y/kpn1[0],sigma_z/kpn1[0]}')
print(rf'emit_nx (PIC): {emit_nx/kpn1[0]}')
print(rf'emit_ny (PIC): {emit_ny/kpn1[0]}')
print(rf'emit_nx axi (PIC): {np.sqrt(emit_nx*emit_ny)/kpn1[0]}')
print(rf'emit_ny axi (PIC): {np.sqrt(emit_nx*emit_ny)/kpn1[0]}')
#print(rf'sigma_vx: {emit_nx/initial_sigma_x}')
#print(rf'sigma_vy: {emit_ny/initial_sigma_y}')
#print(rf'sigma_vx axi: {emit_nx/initial_sigma_x_axi}')
#print(rf'sigma_vy axi: {emit_ny/initial_sigma_y_axi}')
#print(rf'Drift parameters at waist ($\beta_x$,$\beta_y$): {initial_drift}')
#initial_wrong = [beta_x_end,alpha_x_end,beta_y_end,alpha_y_end,gamma_b]
solution_drift = odeint(pf.ode_drift, initial_drift, zs_drift)
solution_plasma = odeint(pf.ode_plasma_estimate_ellipticity, initial_drift, zs_drift, args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density))
solution_plasma_wrong = odeint(pf.ode_plasma_estimate_ellipticity, initial_drift_axi, zs_drift, args=(model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density))
#Drift
beta_x_drift = solution_drift[:,0]
alpha_x_drift = solution_drift[:,1]
beta_y_drift = solution_drift[:,2]
alpha_y_drift = solution_drift[:,3]
#Plasma
beta_x_plasma = solution_plasma[:,0]
alpha_x_plasma = solution_plasma[:,1]
beta_y_plasma = solution_plasma[:,2]
alpha_y_plasma = solution_plasma[:,2]
#Plasma
beta_x_plasma_wrong = solution_plasma_wrong[:,0]
alpha_x_plasma_wrong = solution_plasma_wrong[:,1]
beta_y_plasma_wrong = solution_plasma_wrong[:,2]
alpha_y_plasma_wrong = solution_plasma_wrong[:,3]
#gamma_b_plasma_wrong = solution_plasma_wrong[:,4]
print(rf'Drift parameters at waist ($\beta_x$,$\beta_y$): {min(beta_x_drift)},{min(beta_y_drift)}')
print(rf'Spot size at waist ($\sigma_x$,$\sigma_y$): {np.sqrt(min(beta_x_drift)*emit_nx/gamma)},{np.sqrt(min(beta_y_drift)*emit_ny/gamma)}')
print(rf'Spot size at waist ($\sigma_x$,$\sigma_y$) PICYe: {np.sqrt(min(beta_x_drift)*emit_nx/gamma)/kpn1[0]},{np.sqrt(min(beta_y_drift)*emit_ny/gamma)/kpn1[0]}')
print(rf'Drift waist ($z_x$,$z_y$): {zs_drift[np.argmin(beta_x_drift)]},{zs_drift[np.argmin(beta_y_drift)]}')

# Calculate spot sizes
sigma_x = np.sqrt(beta_x_plasma*emit_nx/gamma_b)
sigma_y = np.sqrt(beta_y_plasma*emit_ny/gamma_b)
sigma_x_wrong = np.sqrt(beta_x_plasma_wrong*emit_nx/gamma_b)
sigma_y_wrong = np.sqrt(beta_y_plasma_wrong*emit_ny/gamma_b)
sigma_x_drift = np.sqrt(beta_x_drift*emit_nx/gamma_b)
sigma_y_drift = np.sqrt(beta_y_drift*emit_ny/gamma_b)
density_asym = pf.calculate_beam_density(Q,sigma_x,sigma_y,sigma_z)/np.multiply(density_pic,n_pe)
density_axi = pf.calculate_beam_density(Q,sigma_x_wrong,sigma_y_wrong,sigma_z)/np.multiply(density_pic,n_pe)
# Plot the solution
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
# Adjust space between subplots
plt.subplots_adjust(hspace=0.06)
# First subplot for sigma_x and sigma_y
zs_all = np.flip(zs_drift+zs_drift[0])
density_all = []
ellipticity_all = []
for z in zs_all:
    density_all.append(pf.plasma_density(z, model, length+4*sigma,sigma, length))
for current_density in density_all:
    # Interpolate the ellipticity based on the current plasma density
    if current_density < min_density:
        ellipticity = 1.0
    else:
        ellipticity = ellipticity_interpolator(current_density)
    ellipticity_all.append(ellipticity)
sigma_x_drift_line, = ax1.plot(zs_all, sigma_x_drift*1e6, color='orange', label='x (vacuum)')
sigma_x_line, = ax1.plot(zs_all, sigma_x*1e6, color='red', label=r'x ($\alpha_p$)')
sigma_x_wrong_line, = ax1.plot(zs_all, sigma_x_wrong*1e6, color='red', linestyle='dotted', label=r'x (axi)')
sigma_y_drift_line, = ax1.plot(zs_all, sigma_y_drift*1e6, color='blue', label='y (vacuum)')
sigma_y_line, = ax1.plot(zs_all, sigma_y*1e6, color='green', label=r'y ($\alpha_p$)')
sigma_y_wrong_line, = ax1.plot(zs_all, sigma_y_wrong*1e6, color='green', linestyle='dotted', label=r'y (axi)')
density_line, =  ax2.plot(zs_all, density_asym,color='pink', label=r'$n_b$ ($\alpha_p$)')
density_axi_line, =  ax2.plot(zs_all, density_axi, color='pink', linestyle='dotted', label=r'$n_b$ (axi)')
density_line_2, =  ax1.plot([],[], color='pink', label=r'$n_b$ ($\alpha_p$)')
density_axi_line_2, =  ax1.plot([],[], color='pink', linestyle='dotted', label=r'$n_b$ (axi)')
# Plot density and ellipticity on ax2
#ax1.plot(zs_all, ellipticity_all, label='Ellipticity')
#ax1.plot(zs_all, density_all, color='purple', label='Density')
#ax1.plot(zst,ellipticity_all, label=r'$\alpha_p$ (flattop)')
#blank_handle, = ax1.plot(np.NaN, np.NaN, '-', color='none', label='')

handles, labels = ax1.get_legend_handles_labels()
#handles.append(blank_handle)  # Adding the blank line to the handles list
#labels.append('')  # Adding an empty string as a label

#density_line, = ax1.plot(-zs_drift, np.multiply(density, 100), color='purple', label='Plasma profile (a.u.)')
#handles.append(density_line)  # Adding the blank line to the handles list
labels.append('Plasma profile (a.u.)')
#ax1.set_ylim(0, 8e3)
#ax2.set_yticks(np.linspace(0, 200, num=5))
#ax1.set_xlim(-0.4, 0.4)

# Add labels for the first subplot
ax1.set_ylabel(r'$\sigma$ ($\mu$m)')

# Second subplot for beta_x and beta_y
beta_x_drift_line, = ax2.plot(zs_all, beta_x_drift*1e2, color='orange', label='x (vacuum)')
#ax2.plot(-zs_drift,ellipticity_all, label='$\alpha_p$')
beta_x_line, = ax2.plot(zs_all, beta_x_plasma*1e2, color='red', label='x ($\alpha_p$)')
beta_x_wrong_line, = ax2.plot(zs_all, beta_x_plasma_wrong*1e2, color='red', linestyle='dotted', label='x (axi)')
beta_y_drift_line, = ax2.plot(zs_all, beta_y_drift*1e2, color='blue', label='y (vacuum)')
beta_y_line, = ax2.plot(zs_all, beta_y_plasma*1e2, color='green', label='y ($\alpha_p$)')

beta_y_wrong_line, = ax2.plot(zs_all, beta_y_plasma_wrong*1e2, color='green', linestyle='dotted', label='y (axi)')
#ellipticity_all = np.array(ellipticity_all)
density_all = np.array(density_all)
#zs_all_full = np.concatenate([-zs_all[::-1], zs_all])
#ellipticity_all_full = np.concatenate([ellipticity_all[::-1], ellipticity_all])
#density_full = np.concatenate([density[::-1], density])  # if symmetric

# Plot
ax2.plot(zs_all, ellipticity_all, label='Ellipticity')
ax2.plot(zs_all, density_all, color='purple', label='Plasma density')

# Plot density and ellipticity on ax2
#ax2.plot(density, ellipticity_all, color='purple', label='Ellipticity')
#density_line2, = ax2.plot(-zs_drift, np.multiply(density, 1), color='purple', label='Plasma profile (a.u.)')
ax2.set_ylim(0, 3)
#ax2.set_xlim(-0.06, 0.06)
#ax2.set_yticks(np.linspace(0, 2, num=5))
#plt.xticks(np.linspace(-0.04, 0.04, num=3))

# Add labels for the second subplot
ax2.set_ylabel(r'$\beta$ (cm)')
ax1.set_xlabel('z (m)')

#blank_handle2, = ax1.plot(np.NaN, np.NaN, '-', color='none', label='')

# Add a single legend
fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.44, 1.02), fancybox=True, shadow=True, ncol=3,fontsize=12)

# Save the plot
plt.savefig(f'sigma_beta.png', bbox_inches='tight')
plt.show()