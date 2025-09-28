import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import astropy.units as u
from dust_extinction.parameter_averages import G23
from scipy.optimize import curve_fit
from scipy import integrate

# Blackbody function
# lambdaArr in microns
# T in kelvin
def planck_function(lambdaArr, T):
    h = const.h.value
    c = const.c.value
    k = const.k_B.value

    lambdaArr = lambdaArr * 1e-6

    I1 = (2*h*(c**2))/(lambdaArr**5)
    I2 = 1/np.expm1((h*c)/(lambdaArr*k*T))
    I = np.pi * I1 * I2

    return I * 1e-6

# Calculates extinction
# model is any extinction model (I use G23())
# A_v in magnitudes
def extinction(model, A_v):
    return 10**(-0.4 * model * A_v)

# Models the star's spectrum
# Accounts for the underlying main sequence star spectrum,
# extinction, and blue black-body from mass accretion
# Star model is a reference spectrum similar to the star
# wl_arr in microns
# T in Kelvin
# Av in magnitudes
# s and s_acc are scalars
# Rv is the ratio Av/E(B-V)
def F_model(star_model, wl_arr, T, Av, s, s_acc, Rv):
    x = wl_arr * u.micron
    model = G23(Rv=Rv)
    black_body = planck_function(wl_arr, T)
    reddening = extinction(model(x), Av)
    return (s*star_model + s_acc*black_body) * reddening

# Wrapper function to utilize scipy.curve_fit
def F_model_for_fit(wl_microns, T, Av, s, s_acc, a1, Rv):
    color = (1.0 + a1 * (wl_microns - 1.6))
    return F_model(color*leo_fit, wl_fit, T, Av, s, s_acc, Rv)

# Loading T Tauri data into an array
TTau_arr = np.loadtxt('TTau_merged_microns.txt')

# Separating into wavelength, flux, and flux error
wl_microns_tt = TTau_arr[:, 0]  # microns
flux_tt = TTau_arr[:, 1]  # Wm^-2/micron
flux_uncert_tt = TTau_arr[:, 2]

# Loading reference star data (HD 100006/86 Leo)
leo_arr = np.loadtxt('86Leo_microns.txt')

# Separating into wavelength, flux, and flux error
wl_microns_leo = leo_arr[:, 0]  # microns
flux_leo = leo_arr[:, 1]  # Wm^-2/micron
flux_uncert_leo = leo_arr[:, 2]

# Interpolating 86 Leo values to match the T Tauri x range
flux_leo_on_tt = np.interp(wl_microns_tt, wl_microns_leo, flux_leo, left=np.nan, right=np.nan)

# Making sure no infinite values
mask = np.isfinite(flux_tt) & np.isfinite(flux_leo_on_tt)
wl_fit = wl_microns_tt[mask]
flux_fit = flux_tt[mask]
leo_fit = flux_leo_on_tt[mask]
sigma_fit = flux_uncert_tt[mask]

# Initiating extinction model for testing
x = wl_fit * u.micron
ext_model = G23(Rv=3.1)

# Intermediate plots to confirm all components look as expected
plt.plot(x, ext_model(x), label='G23')
plt.xlabel('Wavelength [microns]')
plt.ylabel(r'$A(x)/A(V)$')
plt.title('G23')
plt.legend()
plt.show()
plt.clf()

plt.plot(wl_fit, extinction(ext_model(x), 5))
plt.xlabel('Wavelength [microns]')
plt.ylabel('Transmission')
plt.title('Extinction')
plt.show()
plt.clf()

plt.plot(wl_fit, planck_function(wl_fit, 8500))
plt.title('Black-body (bluer)')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.show()
plt.clf()

plt.plot(wl_fit, flux_fit)
plt.ylabel('Flux [Wm^-2/micron]')
plt.xlabel('Wavelength [microns]')
plt.title('T Tauri')
plt.show()
plt.clf()

plt.plot(wl_fit, leo_fit)
plt.title('86 Leo')
plt.ylabel('Flux [Wm^-2/micron]')
plt.xlabel('Wavelength [microns]')
plt.show()
plt.clf()

plt.plot(wl_fit, flux_fit, label='T Tauri')
# plt.plot(wl_fit, extinction(ext_model(x), 5), label='Extinction')
plt.plot(wl_fit, leo_fit, label='86 Leo')
plt.plot(wl_fit, (3.23e-21)*planck_function(wl_fit, 11435), label='Black-body')
# plt.xlim(0.65, 2.6)
plt.ylabel('Flux [Wm^-2/micron]')
plt.xlabel('Wavelength [microns]')
plt.title('All Components')
plt.legend()
plt.show()
plt.clf()

# Initiating free parameters for curve_fit function
s0 = np.median(flux_fit / np.clip(leo_fit, 1e-30, None))
p0 = [5000.0, 2.0, s0, 3e-21, 0, 3.1]
boundsL = [2000.0, 0.0, s0/100, 1e-23, -1.0, 2.3]
boundsU = [12000.0, 6.0, s0*100, 5e-21, 1.0, 5.6]

# Identifying optimal values for closest fit of the data
popt, pcov = curve_fit(
    F_model_for_fit,
    wl_fit,
    flux_fit,
    p0=p0,
    bounds=(boundsL, boundsU),
    sigma=sigma_fit,
    absolute_sigma=True,
)

T_best, Av_best, s_leo_best, s_acc_best, a1_best, Rv_best = popt

print(f'T = {T_best:.0f} K')
print(f'Av = {Av_best:.2f} mag')
print(f's = {s_leo_best:.3g}')
print(f's_acc = {s_acc_best:.3g}')
print(f'a1 = {a1_best:.3g}')
print(f'Rv = {Rv_best:.3g}\n')

# Plotting the closest fit
model_best = F_model_for_fit(wl_fit, *popt)

plt.plot(wl_fit, flux_fit, label="T Tauri (data)")
plt.plot(wl_fit, model_best, label="Best-fit model")
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.legend()
plt.show()
plt.clf()

############Measuring mass accretion rate##################

# Hydrogen emission lines
PaB = 1.2818  # microns
Bry = 2.1661  # microns
# T Tauri constants
M_tt = 4.17e30  # kg
R_tt = 3.136e10  # m
d = (140 * u.pc).to(u.m).value  # pc
# Luminosity of the sun
L_sun = 3.828e26  # W

# Identifying emission lines on spectrum
plt.plot(wl_fit, flux_fit)
plt.axvline(x=PaB, ymin=0.6, ymax=0.7, color='red', linestyle='-', linewidth=2, label="PaB")
plt.axvline(x=Bry, ymin=0.23, ymax=0.33, color='green', linestyle='-', linewidth=2, label='Bry')
plt.title('T Tauri')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.legend()
plt.show()
plt.clf()

# Zooming in on emission lines to determine bounds
plt.plot(wl_fit, flux_fit)
plt.xlim(PaB-0.0008, PaB+0.0017)
plt.title('PaB')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.show()
plt.clf()

plt.plot(wl_fit, flux_fit)
plt.xlim(Bry-0.0021, Bry+0.0019)
plt.title('Bry')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.show()
plt.clf()

# Integrating over isolated emission line windows and subtracting continuum
PaB_index_low, PaB_index_high = (np.where(wl_fit >= PaB-0.0008)[0][0],
                                 np.where(wl_fit <= PaB+0.0017)[0][-1])
PaB_integrated = integrate.simpson(flux_fit[PaB_index_low:PaB_index_high],
                                   x=wl_fit[PaB_index_low:PaB_index_high])
PaB_cont_index_low, PaB_cont_index_high = (np.where(wl_fit >= PaB-0.0033)[0][0],
                                           np.where(wl_fit <= PaB-0.0008)[0][-1])
continuum_PaB = integrate.simpson(flux_fit[PaB_cont_index_low:PaB_cont_index_high],
                                  wl_fit[PaB_cont_index_low:PaB_cont_index_high])

Bry_index_low, Bry_index_high = (np.where(wl_fit >= Bry-0.0021)[0][0],
                                 np.where(wl_fit <= Bry+0.0019)[0][-1])
Bry_integrated = integrate.simpson(flux_fit[Bry_index_low:Bry_index_high],
                                   x=wl_fit[Bry_index_low:Bry_index_high])
Bry_cont_index_low, Bry_cont_index_high = (np.where(wl_fit >= Bry-0.0056)[0][0],
                                           np.where(wl_fit <= Bry-0.0016)[0][-1])
continuum_Bry = integrate.simpson(flux_fit[Bry_cont_index_low:Bry_cont_index_high],
                                  wl_fit[Bry_cont_index_low:Bry_cont_index_high])

# Correcting for extinction and calculating line luminosities
ext_model = G23(Rv=Rv_best)
reddening_PaB = extinction(ext_model(PaB * u.micron), Av_best)
reddening_Bry = extinction(ext_model(PaB * u.micron), Av_best)

line_lum_PaB = 4 * np.pi * d**2 * (PaB_integrated - continuum_PaB) * reddening_PaB
line_lum_Bry = 4 * np.pi * d**2 * (Bry_integrated - continuum_Bry) * reddening_Bry
L_bb = 4 * np.pi * d**2 * integrate.simpson(s_acc_best * planck_function(wl_fit, T_best), wl_fit)

# Calculating accretion luminosity
L_acc_PaB = 10**3.15 * (line_lum_PaB / L_sun)**1.14 * L_sun
L_acc_Bry = 10**4.43 * (line_lum_Bry / L_sun)**1.25 * L_sun

print(f'Luminosity of black-body: {L_bb:.2e} W')
print(f'Accretion luminosity (PaB): {L_acc_PaB:.2e} W')
print(f'Accretion luminosity (Bry): {L_acc_Bry:.2e} W\n')

# Calculating mass accretion rates
M_acc_PaB = (L_acc_PaB * R_tt) / (const.G.value * M_tt)
M_acc_Bry = (L_acc_Bry * R_tt) / (const.G.value * M_tt)

print(f'Mass accretion rate (PaB): {M_acc_PaB:.2e} kg/s')
print(f'Mass accretion rate (Bry): {M_acc_Bry:.2e} kg/s\n')

#################Measuring wind outflow velocity####################

# Locating P Cygni profile
plt.plot(wl_fit, flux_fit)
plt.xlim(1.081, 1.085)
plt.title('He I')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Flux [Wm^-2/micron]')
plt.show()
plt.clf()

# Calculating outflow velocities using P Cygni profile, PaB, and Bry
v_pcygni = 3e5 * (1.082 - 1.083) / 1.083  # Outflow velocity from P Cygni profile in km/s
v_PaB = 3e5 * (1.281 - 1.2818) / 1.2818  # Outflow velocity from PaB emission line in km/s
v_Bry = 3e5 * (2.165 - 2.1661) / 2.1661  # Outflow velocity from PaB emission line in km/s

print(f'Outflow velocity (HeI/PCygni): {v_pcygni:.2f} km/s')
print(f'Outflow velocity (PaB): {v_PaB:.2f} km/s')
print(f'Outflow velocity (Bry): {v_Bry:.2f} km/s')
