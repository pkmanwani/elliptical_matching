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
from scipy.integrate import simpson
from scipy.integrate import odeint
def Kr(gamma):
    return 1/(2*gamma)

def Kx(xp, yp, gamma):
    return 2*Kr(gamma)/(1+xp**2/yp**2)

def Ky(xp, yp, gamma):
    return 2*Kr(gamma)*(xp**2/yp**2)/(1+xp**2/yp**2)

def ab(xp, yp, gamma, epsx):
    return np.sqrt(1/np.sqrt(Kx(xp, yp, gamma))*epsx)

def bb(xp, yp, gamma, epsy):
    return np.sqrt(1/np.sqrt(Ky(xp, yp, gamma))*epsy)


def ce(xp, yp):
    return np.sqrt(xp**2 - yp**2)
def z(x, y, xp, yp):
    return ((x**2-y**2-ce(xp, yp)**2)**2+4*x**2*y**2)**(1/4)


def fex(x, y, x_p, y_p):
    return -x*y_p**2/(x_p**2+y_p**2)

def fey(x, y, x_p, y_p):
    return -y*x_p**2/(x_p**2+y_p**2)

def fez(x, y, x_p, y_p, x_pp, y_pp):
    psip = (x_p*y_p)/(x_p**2+y_p**2)**2*(x_pp*y_p*(x**2-y**2+y_p**2)+x_p*y_pp*(-x**2+y**2+x_p**2))
    return psip
    
# field from the ion column
def eix(x, y, x_p, y_p):
    return -x*y_p/(x_p+y_p)

def eiy(x, y, x_p, y_p):
    return -y*x_p/(x_p+y_p)

def ebx(nb, x, y, xp, yp, threshold=1e-8):
    inside = x**2 - y**2 + z(x, y, xp, yp)**2 - ce(xp, yp)**2

    valid_mask = inside >= threshold

    result = np.zeros_like(inside)

    valid_inside = inside[valid_mask]
    result[valid_mask] = nb*xp*yp/ce(xp, yp)**2*(x[valid_mask] - np.sign(x[valid_mask]) * np.sqrt((valid_inside)/2))

    return result 
    
def eby(nb, x, y, xp, yp, threshold=1e-8):
    inside = y**2 - x**2 + z(x, y, xp, yp)**2 + ce(xp, yp)**2
    
    # Create a mask for where 'inside' is above the threshold
    valid_mask = inside >= threshold
    
    # Initialize the result array with zeros
    result = np.zeros_like(inside)
    
    # Only perform the calculation where 'inside' is valid
    valid_inside = inside[valid_mask]
    result[valid_mask] = nb*xp*yp/ce(xp, yp)**2*(np.sign(y[valid_mask]) * np.sqrt((valid_inside)/2) - y[valid_mask])
    
    return result

def parameters(n0, E, epsxn, epsyn, sigma_z, Q, z):
    """
    n0 is the density of the plasma, (m^-3)
    E is the energy of the beam, (eV)
    epsxn and epsyn are the normalized emittance of the beam, (m)
    sigma_z is the bunch length, (m)
    Q is the total charge of the beam, (C)
    z is the longitudinal position of the transverse slice, (kp-1)
    """
    e0 = 1.602*10**(-19)
    m0 = 9.109*10**(-31)
    c = 299792458 
    epsilon0 = 8.854*10**(-12)
    gamma = E/(5.11*10**5)
    wp = np.sqrt((n0*e0**2)/(m0*epsilon0))
    kp = wp/c

    I_b_m = Q/(n0*e0*np.sqrt(2*np.pi)*sigma_z)
    I_b_m_n = I_b_m*kp**2
    # print(I_b_m)

    I_b_n = I_b_m_n*np.exp(-z**2/(2*(kp*sigma_z)**2))
    epsx = epsxn/gamma*kp
    epsy = epsyn/gamma*kp

    return I_b_n, epsx, epsy, gamma
    
def compute_sumtot_self(params, Q, epsx, epsy, gamma, window, num_points): 
    num_points = num_points
    
    x = np.linspace(-window, window, num_points)
    y = np.linspace(-window, window, num_points)
    
    xp, yp = params
    
    a = np.sqrt(2) * ab(xp, yp, gamma, epsx)
    b = np.sqrt(2) * bb(xp, yp, gamma, epsy)
    if xp<0.9*a or yp<0.9*b or xp<0.999*yp:
        sumtot=np.inf
    else:
        nb = Q/(np.pi*a*b)
        
        vz = nb*a*b/((xp+1)*(yp+1))
        
        t = np.linspace(0.01, 0.01 + 2*np.pi, num_points//10)
        X_ellipse = xp * np.cos(t)
        Y_ellipse = yp * np.sin(t)
        # print(1)
        diff_ellipse_x = (1 + vz)*fex(X_ellipse, Y_ellipse, xp, yp) - vz*eix(X_ellipse, Y_ellipse, xp, yp) + (1 - vz) * ebx(nb, X_ellipse, Y_ellipse, a, b)
        diff_ellipse_y = (1 + vz)*fey(X_ellipse, Y_ellipse, xp, yp) - vz*eiy(X_ellipse, Y_ellipse, xp, yp) + (1 - vz) * eby(nb, X_ellipse, Y_ellipse, a, b)
        
        sumx = np.sum(np.multiply(diff_ellipse_x,1/xp)**2)
        sumy = np.sum(np.multiply(diff_ellipse_y,1/yp)**2)
        
        sumtot = sumx + sumy
        
        #print(nb, a, b, params, sumtot)
    return sumtot

def compute_sumtot_g_self(params, Q, epsx, epsy, gamma, window, num_points): 
    num_points = num_points
    
    x = np.linspace(-window, window, num_points)
    y = np.linspace(-window, window, num_points)
    
    xp, yp = params
    
    a = ab(xp, yp, gamma, epsx)
    b = bb(xp, yp, gamma, epsy)
    if xp<2*a or yp<2*b or xp<0.999*yp:
        sumtot=np.inf
    else:
        nb = Q/(np.pi*a*b*2)
        
        t = np.linspace(0.01, 0.01 + 2*np.pi, num_points//10)
        X_ellipse = xp * np.cos(t)
        Y_ellipse = yp * np.sin(t)
    
        vz = nb*a*b/((xp+1)*(yp+1))
        efieldx_asym, efieldy_asym= efield_gaussian(a, b, num_points, nb, window)[2:]
        efieldx_interpolator = RectBivariateSpline(x, y, efieldx_asym)
        efieldy_interpolator = RectBivariateSpline(x, y, efieldy_asym)
        
        diff_ellipse_x = (1+vz)*fex(X_ellipse, Y_ellipse, xp, yp) - vz*eix(X_ellipse, Y_ellipse, xp, yp) + (1-vz)*efieldx_interpolator.ev(Y_ellipse, X_ellipse)
        diff_ellipse_y = (1+vz)*fey(X_ellipse, Y_ellipse, xp, yp) - vz*eiy(X_ellipse, Y_ellipse, xp, yp) + (1-vz)*efieldy_interpolator.ev(Y_ellipse, X_ellipse)
        
        sumx = np.sum((diff_ellipse_x/xp)**2)
        sumy = np.sum((diff_ellipse_y/yp)**2)
        
        sumtot = sumx + sumy
        #print(nb, a, b, params, sumtot)
    return sumtot

def ellipse_shape_self(Q_k, epsx_k, epsy_k, gamma_k, xp, yp, window_k, num_points_k):
    
    # Constraint to ensure xp > a
    constraint_xp_positive = {'type': 'ineq', 'fun': lambda params: params[0] - ab(params[0], params[1], gamma_k, epsx_k)}
    
    # Constraint to ensure yp > b
    constraint_yp_positive = {'type': 'ineq', 'fun': lambda params: params[1] - bb(params[0], params[1], gamma_k, epsy_k)}

    # Original constraint to ensure xp > yp
    constraint_xp_greater_yp = {'type': 'ineq', 'fun': lambda params: params[0] - params[1]}
    
    # Combine all constraints in a list, including the new positivity constraints
    constraints = [constraint_xp_positive, constraint_yp_positive, constraint_xp_greater_yp]
    
    partial_func = functools.partial(compute_sumtot_self, Q = Q_k, epsx = epsx_k, epsy = epsy_k, gamma = gamma_k, window = window_k, num_points = num_points_k)

    # print(1)
    initial_guess = [xp, yp]
    # Pass the constraint to the minimize function
    result = minimize(partial_func, initial_guess, constraints=constraints)
    optimal_xp, optimal_yp = result.x
    cost = result.fun
    
    a = np.sqrt(2) * ab(optimal_xp, optimal_yp, gamma_k, epsx_k)
    b = np.sqrt(2) * bb(optimal_xp, optimal_yp, gamma_k, epsy_k)
    
    return cost, optimal_xp, optimal_yp, a, b
    
def guess(n0, E, epsxn, epsyn, sigma_z, Q):
    #Assuming blowout
    gamma = E/(0.511e6) + 1
    sigma_x = calculate_matched_beam(n0, gamma, epsxn)[0]
    sigma_y = calculate_matched_beam(n0, gamma, epsyn)[0]
    #print(sigma_x,sigma_y,sigma_z)
    nb = calculate_beam_density(Q,sigma_x,sigma_y,sigma_z)/n0
    #print(nb)
    a = np.sqrt(2)*sigma_x
    b = np.sqrt(2)*sigma_y
    alpha_b = a/b
    xp = a*np.sqrt((alpha_b**2-1)/(2*(alpha_b)**2))*np.sqrt(np.sqrt(1 + ((4*(alpha_b*nb)**2)/((alpha_b**2-1)**2)))+1)
    yp = a*np.sqrt((alpha_b**2-1)/(2*(alpha_b)**2))*np.sqrt(np.sqrt(1 + ((4*(alpha_b*nb)**2)/((alpha_b**2-1)**2)))-1)
    xp_g = xp/calculate_kpn1(n0)[0]
    yp_g = yp/calculate_kpn1(n0)[0]
    return xp_g,yp_g

def optimize_ellipse_parameters(n0, E, epsxn, epsyn, sigma_z, Q, z,window_size=5, num_points=2000):
    # Calculate parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, z)
    #Get guess
    xp_g,yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)
    
    # Optimize ellipse shape parameters
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)
    
    return cost, xp_op, yp_op, a_op, b_op
    
def ellipse_shape_g_self(Q_k, epsx_k, epsy_k, gamma_k, xp, yp, window_k, num_points_k):
    
    # Constraint to ensure xp > a
    constraint_xp_positive = {'type': 'ineq', 'fun': lambda params: params[0] - ab(params[0], params[1], gamma_k, epsx_k)}
    
    # Constraint to ensure yp > b
    constraint_yp_positive = {'type': 'ineq', 'fun': lambda params: params[1] - bb(params[0], params[1], gamma_k, epsy_k)}

    # Original constraint to ensure xp > yp
    constraint_xp_greater_yp = {'type': 'ineq', 'fun': lambda params: params[0] - params[1]}

    # Original constraint to ensure xp / yp < sqrt(epsx/epsy)
    # constraint_ellip = {'type': 'ineq', 'fun': lambda params: 1 - params[0]/params[1]}
    
    # Combine all constraints in a list, including the new positivity constraints
    # constraints = [constraint_xp_positive, constraint_yp_positive, constraint_xp_greater_yp, constraint_ellip]
    constraints = [constraint_xp_positive, constraint_yp_positive, constraint_xp_greater_yp]
    
    partial_func = functools.partial(compute_sumtot_g_self, Q = Q_k, epsx = epsx_k, epsy = epsy_k, gamma = gamma_k, window = window_k, num_points = num_points_k)

    # print(1)
    initial_guess = [xp, yp]
    # Pass the constraint to the minimize function
    result = minimize(partial_func, initial_guess, constraints=constraints)
    optimal_xp, optimal_yp = result.x
    cost = result.fun
    
    a = ab(optimal_xp, optimal_yp, gamma_k, epsx_k)
    b = bb(optimal_xp, optimal_yp, gamma_k, epsy_k)
    
    return cost, optimal_xp, optimal_yp, a, b
def optimize_ellipse_parameters_gaussian(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,window_size=2, num_points=500):
    # Calculate parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position)

    #Get guess
    xp_g,yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)
    
    #Use it for Gaussian case
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_g_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)
    
    return cost, xp_op, yp_op, a_op, b_op

def efield_gaussian(a, b, N, nb, window):
    x,y = np.meshgrid(np.linspace(-window, window, num=N, dtype=float),np.linspace(window, -window, num=N, dtype=float))

    s = nb * np.exp(-(x/a)**2 / 2 - (y/b)**2 / 2)

    #fourier transform of s
    sfft = np.fft.fft2(s)

    # obtain the frequency plane
    freq = 2*np.pi*(np.fft.fftfreq(N, d=window*2/(N-1)))
    #print(freq)
    #freqplane_sq =freq**2+(freq[:,None])**2
    freqx = get_freqx(freq)
    freqy = get_freqy(freq)
    freqplane_sq = (freqx)**2+(freqy)**2

    phifft = -sfft/freqplane_sq

    # Setting the value for the invalid one
    phifft[0,0] = 0
    phi = np.real(np.fft.ifft2(phifft))

    # Calculating the electric field
    efieldfftx = 1j*phifft * freq
    efieldffty = 1j*phifft * freq[:,None]
    efieldx = np.real(np.fft.ifft2(efieldfftx))
    efieldy = np.real(np.fft.ifft2(efieldffty))
    
    return s, phi, efieldx, efieldy    


def get_freqx(freq):
    freqx = np.zeros((int(len(freq)), int(len(freq))))
    for i in range(len(freq)):
        for j in range(len(freq)):
            freqx[i,j] = freq[j]
    return freqx

def get_freqy(freq):
    freqy = np.zeros((int(len(freq)), int(len(freq))))
    for i in range(len(freq)):
        for j in range(len(freq)):
            freqy[i,j] = freq[i]
    return freqy

#################MATCHING FUNCTIONS###########################
#Growth rate
global c,epsilon_0, e, m_e, r_e
c = 299792458 
epsilon_0 = 8.8542e-12
e = 1.6022e-19
m_e = 9.109e-31
r_e = 2.82e-15
#beta = velocity of beam
beta = 1
#alpha = Normalized beam density
#alpha = alpha
#gamma = lorentz factor
#flip and concatenate
def f_c(arr1):
    flipped_arr1 = np.flip(arr1)
    result = np.concatenate((flipped_arr1,arr1))
    #print(result)
    return result
    
def get_size(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

def calculate_kpn1(n_pe):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    kpn1 = c/omega_pe
    lambda_pe = 2*np.pi*kpn1
    return kpn1, lambda_pe

def calculate_omegap(n_pe):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    return omega_pe

def calculate_corresponding_B(n_pe):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    B = omega_pe*m_e/e
    return B

def calculate_kp(n_pe):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    kp = omega_pe/c
    #lambda_pe = 2*np.pi*kpn1
    return kp

def calculate_electric_wb(n_pe):
    kpn1 = calculate_kpn1(n_pe)[0]
    e_wb = m_e*(c**2)/(kpn1*e)
    return e_wb 


def calculate_matched_beam(n_pe, gamma_b, emittance_normalized):
    kpn1 = calculate_kpn1(n_pe)[0]
    beta = kpn1*np.sqrt(2.0*gamma_b)
    emittance = emittance_normalized/gamma_b
    spot_size = np.sqrt(emittance*beta)
    return spot_size, beta

def beta_beam(gamma_b, sigma, emittance_normalized):
    emittance = emittance_normalized/gamma_b
    beta = (sigma**2)/emittance
    return beta
def sigma_beam(beta_beam, emittance_normalized, gamma_beam):
    emittance = emittance_normalized/gamma_beam
    result = np.sqrt(beta_beam*emittance)
    return result
def calculate_beam_density(charge, sigma_x, sigma_y, sigma_z):
    beam_density = charge/(e*(15.7496*sigma_x*sigma_y*sigma_z))
    return beam_density
def calculate_beam_density_cylindrical(charge, sigma_x, sigma_z):
    beam_density = charge/(e*(np.sqrt(8*(np.pi**3))*(np.pi*sigma_x**2)*sigma_z))
    return beam_density
def gaussian_ish(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma * sigma))

def ramp(z):
    plasma_upramp_start = 0
    plasma_upramp_end = 0.2
    plasma_upramp_sigma = (plasma_upramp_end - plasma_upramp_start) / 3
    plasma_downramp_start = 0.4
    plasma_downramp_end = 0.6
    plasma_downramp_sigma = (plasma_downramp_end - plasma_downramp_start) / 3
    if z < plasma_upramp_end:
        return gaussian_ish(z, plasma_upramp_end, plasma_upramp_sigma)
    elif z < plasma_downramp_start:
        return 1
    else:
        return gaussian_ish(z, plasma_downramp_start, plasma_downramp_sigma)


def plasma_export_PIC(r, z, npe, limit):
    #find kpn1
    kpn1 = calculate_kpn1(n_pe)[0]
    with open('plasma_density_PIC.txt', 'w+') as f:
        #f.write('[')
        for i in range(len(z)):
            rx = r[i]
            if (rx > limit):
                zx = z[i] / kpn1
                f.write(f'{zx:.4f}')
                f.write(',')
        fsize = get_size(f)
        f.truncate(fsize - 1)
        #f.write(']')
        f.write('\n')
        for i in range(len(z)):
            rx = r[i]
            if (rx > limit):
                zx = z[i] / kpn1
                f.write(f'{rx:.4f}')
                f.write(',')
        fsize = get_size(f)
        f.truncate(fsize - 1)
        #f.write(']')


        
    
def kbeta(n_pe, gamma_b, model, z, sigma, length, plasma_upramp_end):
    n_pe = plasma_density(model,z, sigma, length, plasma_upramp_end)*n_pe
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe/np.sqrt(2*gamma_b)
    result = omega_beta/c
    return result
    
def twiss_0(beta_waist, z_final, z0):
    alpha_waist = 0
    gamma_waist = 1/beta_waist
    beta = beta_waist - 2*(z_final-z0)*alpha_waist + ((z_final-z0)**2)*gamma_waist
    alpha = alpha_waist - (z_final-z0)*gamma_waist
    gamma = (1 + alpha**2)/beta
    return alpha, beta, gamma

def calc_gamma(alpha,beta):
    result = (1+alpha**2)/beta
    return result

def f(y, z, n_pe, gamma_b, model, sigma, length, plasma_upramp_end):
    k = kbeta(n_pe, gamma_b, model, z, sigma, length, plasma_upramp_end)
    return (y[1], (-2*(k)**2)*y[0] + (2/y[0])*(1+(y[1]/2)**2))

def calculate_charge_using_current(peak_current, spot_size_z):
    sigma_t = spot_size_z/c
    q = np.sqrt(2*np.pi)*sigma_t*peak_current
    return q

def occurences(mylist, value = 1.000, tol =1e-08):
    d = 0
    for num,item in enumerate(mylist):
        if np.abs(item - value) < tol and d==0:
            #print(item)
            start = num
            d = d+1
        elif np.abs(item - value) < tol and d!=0:
            stop = num
    return start,stop
    
def plasma_export_PIC(r, z, n_pe, limit):
    #find kpn1
    kpn1 = calculate_kpn1(n_pe)[0]
    with open('plasma_density_PIC.txt', 'w+') as f:
        #f.write('[')
        for i in range(len(z)):
            rx = r[i]
            if (rx > limit):
                zx = z[i] / kpn1
                f.write(f'{zx:.4f}')
                f.write(',')
        fsize = get_size(f)
        f.truncate(fsize - 1)
        #f.write(']')
        f.write('\n')
        for i in range(len(z)):
            rx = r[i]
            if (rx > limit):
                zx = z[i] / kpn1
                f.write(f'{rx:.4f}')
                f.write(',')
        fsize = get_size(f)
        f.truncate(fsize - 1)
        #f.write(']')

def PIC_corrector(filename):
    a_0 = []
    with open(filename, 'r+') as f:
        contents = f.read()
        #contents.replace('\x00', '')
        #print(contents)
        #print(contents)
        content_list = contents.splitlines()
        a = content_list[0].split(',')
        b = content_list[1].split(',')
        
        a = [float(x.rstrip('\x00')) for x in a]
        b = [float(x.rstrip('\x00')) for x in b]
        first_occur, last_occur = occurences(b)
        print(first_occur)
        print(last_occur)
        del a[first_occur+1:last_occur]
        del b[first_occur+1:last_occur]
        a_0 = a
        tp = len(a_0)
        a_0 = [str(element) for element in a_0]
        a_0 = ",".join(a_0)
        print(a_0)        
        b = [str(element) for element in b]
        b     = ",".join(b)
        with open(filename.split('.')[0] + '_density.txt','w+') as out:
            out.write(str(tp))
            out.write("[")
            out.write(b)
            out.write("]")
            out.close()
        with open(filename.split('.')[0] + '_z.txt','w+') as out:
            out.write("[")
            out.write(a_0)
            out.write("]")
            out.close()
        f.close()
    
def gaussian_ish(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma * sigma))
    
def kbeta(z,model, sigma, length,n_pe, gamma_b):
    n_pe = plasma_density(z, model, sigma, length)*n_pe
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe/np.sqrt(2*gamma_b)
    result = omega_beta/c
    return result

def plasma_density(z, model,  plasma_upramp_end, sigma, length):
    if model == 'gaussian':        
        if z < plasma_upramp_end-length:
            return gaussian_ish(z, plasma_upramp_end - length, sigma)
        elif z < plasma_upramp_end + length:
            return 1
        else:
            return gaussian_ish(z, plasma_upramp_end + length, sigma)
        
def kbeta(z,model, sigma, length,n_pe, gamma_b):
    n_pe = plasma_density(z, model, plasma_upramp_end,sigma, length)*n_pe
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe/np.sqrt(2*gamma_b)
    result = omega_beta/c
    return result

def kbeta_x(z,model, plasma_upramp_end,sigma, length,n_pe, gamma_b,ellipticity):
    n_pe = plasma_density(z, model, plasma_upramp_end,sigma, length)*n_pe
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe/np.sqrt((1+ellipticity**2)*gamma_b)
    result = omega_beta/c
    return result

def kbeta_y(z,model, plasma_upramp_end,sigma, length,n_pe, gamma_b,ellipticity):
    n_pe = plasma_density(z, model,plasma_upramp_end, sigma, length)*n_pe
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe*(ellipticity)/np.sqrt((1+ellipticity**2)*gamma_b)
    result = omega_beta/c
    return result


def kbeta_x_estimate(z, model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density):
    # Calculate the current plasma density
    current_density = plasma_density(z, model, plasma_upramp_end, sigma, length)

    # Interpolate the ellipticity based on the current plasma density
    if current_density < min_density:
        ellipticity = 1.0
    else:
        ellipticity = ellipticity_interpolator(current_density)

    # Calculate omega_pe and omega_beta
    omega_pe = np.sqrt(current_density*n_pe* e ** 2 / (epsilon_0 * m_e))
    omega_beta = omega_pe/ np.sqrt((1 + ellipticity ** 2) * gamma_b)

    # Calculate and return kbeta_y
    kbeta_x_value = omega_beta / c
    return kbeta_x_value



def kbeta_y_estimate(z, model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator, min_density):
    # Calculate the current plasma density
    current_density = plasma_density(z, model, plasma_upramp_end, sigma, length)

    # Interpolate the ellipticity based on the current plasma density
    if current_density < min_density:
        ellipticity = 1.0
    else:
        ellipticity = ellipticity_interpolator(current_density)

    # Calculate omega_pe and omega_beta
    omega_pe = np.sqrt(current_density*n_pe * e ** 2 / (epsilon_0 * m_e))
    omega_beta = omega_pe*ellipticity/ np.sqrt((1 + ellipticity ** 2) * gamma_b)

    # Calculate and return kbeta_y
    kbeta_y_value = omega_beta / c
    return kbeta_y_value

def kbeta_matched_x(n_pe, gamma_b,ellipticity):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe/np.sqrt((1+(ellipticity**2))*gamma_b)
    result = omega_beta/c
    return result

def kbeta_matched_y(n_pe, gamma_b,ellipticity):
    omega_pe = np.sqrt(n_pe*e**2/(epsilon_0*m_e))
    omega_beta = omega_pe*ellipticity/np.sqrt((1+(ellipticity**2))*gamma_b)
    result = omega_beta/c
    return result
def optimize_kbeta_parameters_single(n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size=5, num_points=2000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    
    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """
    n0 = n_pe
    
    # Calculate initial parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position)
    
    # Get initial guesses for optimization
    xp_g, yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)
    
    # Optimize ellipse shape parameters
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)
    
    # Calculate ellipticity (assumed to be needed for kbeta calculation)
    ellipticity = xp_op / yp_op
    
    # Calculate kbeta parameters
    kbeta_x = kbeta_matched_x(n0, gamma, ellipticity)
    kbeta_y = kbeta_matched_y(n0, gamma, ellipticity)
    
    return kbeta_x, kbeta_y

def optimize_kbeta_parameters_single_gaussian(n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size=2, num_points=500):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    """
    n0 = n_pe

    gamma = E/0.511e6 + 1
    xp_op,yp_op,ellipticity = optimize_boundary_parameters_gaussian_single(n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size, num_points)
    
    # Calculate kbeta parameters
    kbeta_x = kbeta_matched_x(n0, gamma, ellipticity)
    kbeta_y = kbeta_matched_y(n0, gamma, ellipticity)
    
    return kbeta_x, kbeta_y
    
def optimize_kbeta_parameters(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size=4, num_points=2000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    
    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """
    # Calculate n0 based on the given parameters
    xp_op,yp_op,ellipticity = optimize_boundary_parameters_gaussian(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size, num_points)
    gamma = E / 0.511e6 + 1
    # Calculate kbeta parameters
    kbeta_x = kbeta_matched_x(n0, gamma, ellipticity)
    kbeta_y = kbeta_matched_y(n0, gamma, ellipticity)
    
    return kbeta_x, kbeta_y

def optimize_kbeta_parameters_gaussian(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size=4, num_points=1000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    
    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """

    xp_op,yp_op,ellipticity = optimize_boundary_parameters_gaussian(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size, num_points)
    gamma = E / 0.511e6 + 1
    # Calculate kbeta parameters
    kbeta_x = kbeta_matched_x(n0, gamma, ellipticity)
    kbeta_y = kbeta_matched_y(n0, gamma, ellipticity)
    
    return kbeta_x, kbeta_y
    
def optimize_boundary_parameters(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size=4, num_points=2000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    
    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """
    # Calculate n0 based on the given parameters
    #print(n_pe)
    #print(plasma_density(zs, model, plasma_upramp_end, sigma, length))
    n0 = plasma_density(zs, model, plasma_upramp_end, sigma, length)*n_pe
    
    # Calculate initial parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position)
    
    # Get initial guesses for optimization
    xp_g, yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)
    
    # Optimize ellipse shape parameters
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)
    
    # Calculate ellipticity (assumed to be needed for kbeta calculation)
    ellipticity = xp_op / yp_op
    
    return xp_op, yp_op, ellipticity

def optimize_boundary_parameters_gaussian(zs, model, sigma,  length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position,  window_size=4, num_points=2000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.
    
    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.
    
    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """
    # Calculate n0 based on the given parameters
    #print(n_pe)
    #print(plasma_density(zs, model, plasma_upramp_end, sigma, length))
    n0 = plasma_density(zs, model, plasma_upramp_end, sigma, length)*n_pe
    
    # Calculate initial parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position)
    
    # Get initial guesses for optimization
    xp_g, yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)
    
    # Optimize ellipse shape parameters
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_g_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)
    
    # Calculate ellipticity (assumed to be needed for kbeta calculation)
    ellipticity = xp_op / yp_op
    
    return xp_op, yp_op, ellipticity


def optimize_boundary_parameters_gaussian_single(n_pe, E, epsxn, epsyn, sigma_z,
                                          Q, longitudinal_position, window_size=4, num_points=2000):
    """
    Optimize k-beta parameters based on input physical parameters and optimization settings.

    Parameters:
    n0 (float): Initial particle density
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    z (float): Longitudinal position
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.

    Returns:
    tuple: Optimized kbeta_x and kbeta_y parameters
    """
    # Calculate n0 based on the given parameters
    # print(n_pe)
    # print(plasma_density(zs, model, plasma_upramp_end, sigma, length))
    n0 = n_pe

    # Calculate initial parameters using the provided inputs
    I_b_n, epsx, epsy, gamma = parameters(n0, E, epsxn, epsyn, sigma_z, Q, longitudinal_position)

    # Get initial guesses for optimization
    xp_g, yp_g = guess(n0, E, epsxn, epsyn, sigma_z, Q)

    # Optimize ellipse shape parameters
    cost, xp_op, yp_op, a_op, b_op = ellipse_shape_g_self(I_b_n, epsx, epsy, gamma, xp_g, yp_g, window_size, num_points)

    # Calculate ellipticity (assumed to be needed for kbeta calculation)
    ellipticity = xp_op / yp_op

    return xp_op, yp_op, ellipticity


def ode_plasma_elliptical_model(vars, zs, model, plasma_upramp_end, sigma, length, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size=4, num_points=2000):
    """
    Calculate the derivatives for the ODE system in the plasma, including the optimization of k-beta parameters.

    Parameters:
    vars (list): List containing the current values of [beta_x, alpha_x, beta_y, alpha_y]
    zs (float): Longitudinal position in the plasma
    model (str): Plasma model being used
    plasma_upramp_end (float): End of the plasma upramp
    sigma (float): Standard deviation of the plasma density profile
    length (float): Length of the plasma
    n_pe (float): Electron density in the plasma
    gamma_b (float): Lorentz factor of the beam
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.

    Returns:
    list: List of derivatives [dbeta_xdt, dalpha_xdt, dbeta_ydt, dalpha_ydt]
    """
    beta_x, alpha_x, beta_y, alpha_y = vars

    # Optimize k-beta parameters
    k_x, k_y = optimize_kbeta_parameters(zs, model, sigma, length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size, num_points)

    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = -(1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y

    return [dbeta_xdt, dalpha_xdt, dbeta_ydt, dalpha_ydt]

def ode_plasma_elliptical_model_gaussian(vars, zs, model, plasma_upramp_end, sigma, length, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size=4, num_points=2000):
    """
    Calculate the derivatives for the ODE system in the plasma, including the optimization of k-beta parameters.

    Parameters:
    vars (list): List containing the current values of [beta_x, alpha_x, beta_y, alpha_y]
    zs (float): Longitudinal position in the plasma
    model (str): Plasma model being used
    plasma_upramp_end (float): End of the plasma upramp
    sigma (float): Standard deviation of the plasma density profile
    length (float): Length of the plasma
    n_pe (float): Electron density in the plasma
    gamma_b (float): Lorentz factor of the beam
    E (float): Energy
    epsxn (float): Normalized emittance in the x-direction
    epsyn (float): Normalized emittance in the y-direction
    sigma_z (float): Bunch length
    Q (float): Charge
    window_size (int, optional): Window size for the optimization algorithm. Default is 5.
    num_points (int, optional): Number of points for the optimization. Default is 2000.

    Returns:
    list: List of derivatives [dbeta_xdt, dalpha_xdt, dbeta_ydt, dalpha_ydt]
    """
    beta_x, alpha_x, beta_y, alpha_y = vars

    # Optimize k-beta parameters
    k_x, k_y = optimize_kbeta_parameters_gaussian(zs, model, sigma, length, plasma_upramp_end, n_pe, E, epsxn, epsyn, sigma_z, Q, longitudinal_position, window_size, num_points)

    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = -(1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y

    return [dbeta_xdt, dalpha_xdt, dbeta_ydt, dalpha_ydt]

def ode_plasma(vars,zs,model,plasma_upramp_end,sigma,length,n_pe,gamma_b,ellipticity):
    beta_x, alpha_x, beta_y, alpha_y  = vars
    #define plasma density
    k_x = kbeta_x(zs, model,plasma_upramp_end,sigma,length,n_pe,gamma_b,ellipticity)
    k_y = kbeta_y(zs, model,plasma_upramp_end,sigma,length,n_pe,gamma_b,ellipticity)
    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = -(1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y
    #print(dalpha_xdt,dalpha_ydt)
    return [dbeta_xdt, dalpha_xdt,dbeta_ydt,dalpha_ydt]


def ode_plasma_estimate_ellipticity(vars, z, model, plasma_upramp_end, sigma, length, n_pe, gamma_b,
                                    ellipticity_interpolator, min_density):
    beta_x, alpha_x, beta_y, alpha_y = vars

    # Estimate kbeta_x and kbeta_y using the new functions
    k_x = kbeta_x_estimate(z, model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator,
                           min_density)
    k_y = kbeta_y_estimate(z, model, plasma_upramp_end, sigma, length, n_pe, gamma_b, ellipticity_interpolator,
                           min_density)

    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = -(1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y

    return [dbeta_xdt, dalpha_xdt, dbeta_ydt, dalpha_ydt]

def ode_plasma_energy(vars,zs,model,plasma_upramp_end,sigma,length,n_pe,gamma_b_func,ellipticity):
    # Calculate gamma_b based on the given function and parameters
    #gamma_b = gamma_b_func(zs, plasma_density(zs, model, plasma_upramp_end, sigma, length))
    beta_x, alpha_x, beta_y, alpha_y,gamma_b  = vars
    #define plasma density
    plasma_density_value = plasma_density(zs, model, plasma_upramp_end, sigma, length)
    k_x = kbeta_x(zs, model,plasma_upramp_end,sigma,length,n_pe,gamma_b,ellipticity)
    k_y = kbeta_y(zs, model,plasma_upramp_end,sigma,length,n_pe,gamma_b,ellipticity)
    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = - (1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y
    dgamma_dt = calculate_electric_wb(n_pe*plasma_density_value)/(0.511*1e6)
    #print(dgamma_dt)
    # Ensure gamma_b does not go below 1
    if gamma_b <= 1:
        dgamma_dt = 0
        gamma_b = 1  # Set gamma_b to 1 if it goes below 1
    #print(dgamma_dt)
    #print(f"z: {zs}, gamma_b: {gamma_b}, plasma_density: {plasma_density_value}, k_x: {k_x}, k_y: {k_y}")

    return [dbeta_xdt, dalpha_xdt,dbeta_ydt,dalpha_ydt,dgamma_dt]

def ode_drift(vars,zs):
    beta_x, alpha_x, beta_y, alpha_y  = vars
    #define plasma density
    k_x = 0
    k_y =0
    dbeta_xdt = -2 * alpha_x
    dbeta_ydt = -2 * alpha_y
    dalpha_xdt = - (1 + (-alpha_x) ** 2) / beta_x + (k_x ** 2) * beta_x
    dalpha_ydt = -(1 + (-alpha_y) ** 2) / beta_y + (k_y ** 2) * beta_y
    return [dbeta_xdt, dalpha_xdt,dbeta_ydt,dalpha_ydt]

  
def calculate_electric_wb(n_pe):
    kpn1 = calculate_kpn1(n_pe)[0]
    #print(kpn1)
    e_wb = m_e*(c**2)/(kpn1*e)
    return e_wb