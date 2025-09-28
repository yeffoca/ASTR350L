from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import sys

def vega_to_ab(m_vega, const):
    return m_vega + const

def ab_to_freq(m_ab):
    m_ab = np.array(m_ab).astype(np.float64)
    return 10**((m_ab + 48.6) / -2.5)

def freq_to_lambda(f_v, l):
    c = const.c.cgs.value
    return (f_v * c) / l**2

def planck_function(lambdaArr, T):
    h = const.h.cgs.value
    c = const.c.cgs.value
    k = const.k_B.cgs.value

    I1 = (2*h*(c**2))/(lambdaArr**5)
    I2 = 1/(np.exp((h*c)/(lambdaArr*k*T))-1)
    I = np.pi * I1 * I2

    return I

def scaled_planck(lambda_arr, T, S):
    return S * planck_function(lambda_arr, T)

###############################Data Prep##############################
# List of stars used in this lab
target_list = ['Betelgeuse', 'Alioth',
              'Rigel', 'alf Cen A']

# Names produced by simbad
simbad_ids = ['* alf Ori', '* eps UMa',
              '* bet Ori', '* alf Cen A']

# Collecting Simbad data (optical)
Simbad.add_votable_fields('allfluxes', 'parallax')
return_table = Simbad.query_objects(target_list)
sim_photo_df = return_table.to_pandas()

# print(sim_photo_df.loc[sim_photo_df['main_id'] == '* bet Ori', 'ra'].iloc[0])

# Collecting Allwise/Vizier data (infrared)
allwise_df = pd.DataFrame()
for star in simbad_ids:
    # Pulling RA and dec from simbad (sim_photo_df)
    target_ra = sim_photo_df.loc[sim_photo_df['main_id'] == star, 'ra'].iloc[0]
    target_dec = sim_photo_df.loc[sim_photo_df['main_id'] == star, 'dec'].iloc[0]

    # Pulling allwise data from vizier
    vizier = Vizier(columns=['*'])  # Initializing vizier class with default columns
    return_table = Vizier.query_object(star, catalog='II/328/allwise')[0]  # Querying for desired object
    viz_photo_df = return_table.to_pandas()  # Converting table to pandas df

    # Finding closest match to Simbad coordinates
    abs_diff_ra = (viz_photo_df['RAJ2000'] - target_ra).abs()  # Calculating difference in ra for each column abs(allwise_ra - simbad_ra)
    abs_diff_dec = (viz_photo_df['DEJ2000'] - target_dec).abs()  # Calculating difference in dec for each column abs(allwise_dec - simbad_dec)
    hyp = np.sqrt((abs_diff_ra**2 + abs_diff_dec**2))  # Calculating distance between simbad coords and allwise coords

    # identifying row with smallest ra difference
    closest_index = hyp.idxmin()  # Identifying closest row
    closest_row = viz_photo_df.loc[closest_index].to_frame().T  # Turning row into single row df
    closest_row.insert(0, "main_id", star)  # adding simbad id column for easy merging

    allwise_df = pd.concat([allwise_df, closest_row], ignore_index=True)

# Combining simbad data with allwise data
master_df = pd.merge(sim_photo_df, allwise_df, on='main_id')

# converting numeric values to floats
filters = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K',
                    'W1mag', 'W2mag', 'W3mag', 'W4mag'])
master_corrected = master_df.copy()
master_corrected[filters] = master_df[filters].apply(pd.to_numeric, errors='coerce')

master_df = master_corrected
master_df_j = master_df.copy()

##################Data Cleaning####################
vega_conversions = {'U': 0.79, 'B': -0.09,
                       'V': 0.02, 'R': 0.21,
                       'I': 0.45, 'J': 0.91,
                       'H': 1.39, 'K': 1.85}

zeromag = {'W1mag': 2.699, 'W2mag': 3.339,
              'W3mag': 5.174, 'W4mag': 6.620}

# Effective lambda in microns
eff_lambda = {'U': 0.36, 'B': 0.438,
                 'V': 0.545, 'R': 0.641,
                 'I': 0.798, 'J': 1.22,
                 'H': 1.63, 'K': 2.19,
                 'W1mag': 3.4, 'W2mag': 4.6,
                 'W3mag': 12, 'W4mag': 22}

# Converting UBVRIJHK magnitudes to physical units
for filter in vega_conversions:
    valid_indices = ~np.isnan(master_df[filter])
    master_df_j.loc[valid_indices, filter] = ab_to_freq(vega_to_ab(master_df[filter][valid_indices], vega_conversions[filter]))

#  Converting wise magnitudes to physical units
for w_filter in zeromag:
    valid_indices = ~np.isnan(master_df[w_filter])
    master_df_j.loc[valid_indices, w_filter] = ab_to_freq(vega_to_ab(master_df[w_filter][valid_indices], zeromag[w_filter]))

betel_series = master_df_j.iloc[0]
alioth_series = master_df_j.iloc[1]
rigel_series = master_df_j.iloc[2]
alpha_series = master_df_j.iloc[3]

lambda_arr_mu = np.array(list(eff_lambda.values()))  # Creating effective lambda array in microns
lambda_arr_cm = lambda_arr_mu * 1e-4  # Creating effective lambda array in cm

betel_flux_list = freq_to_lambda(np.array(betel_series[filters]), lambda_arr_cm)
rigel_flux_list = freq_to_lambda(np.array(rigel_series[filters]), lambda_arr_cm)
alioth_flux_list = freq_to_lambda(np.array(alioth_series[filters]), lambda_arr_cm)
alpha_flux_list = freq_to_lambda(np.array(alpha_series[filters]), lambda_arr_cm)

# Including Procyon B data
procb_fv_arr = np.array([1.06, 22.4, 46.6, 102.3, 145.3, 138.7, 149.1,
                         144.4, 150.9, 171.1, 172.2, 165.0, 145.6, 136.2]) * 1e-26
procb_lambda_arr_cm = np.array([1491, 2189, 2587, 3341, 4300, 4695, 4865,
                                5013, 5614, 6306, 6564, 6591, 6697, 7828]) * 1e-8  # cm
procb_lambda_arr_mu = np.array([1491, 2189, 2587, 3341, 4300, 4695, 4865,
                                5013, 5614, 6306, 6564, 6591, 6697, 7828]) * 1e-4  # microns
procb_flux_list = freq_to_lambda(procb_fv_arr, procb_lambda_arr_cm)
procb_dist = 3.5

betel_dist = (1000 / betel_series['plx_value']) * 3.086e18  # cm
alioth_dist = (1000 / alioth_series['plx_value']) * 3.086e18  # cm
rigel_dist = (1000 / rigel_series['plx_value']) * 3.086e18  # cm
alpha_dist = (1000 / alpha_series['plx_value']) * 3.086e18  # cm


star_flux_tup_list = [(betel_flux_list, betel_dist, 'ʻAua', 3500), (rigel_flux_list, rigel_dist, 'Hōkū kau ʻōpae', 10000),
                      (alioth_flux_list, alioth_dist, 'Hiku lima', 9100), (alpha_flux_list, alpha_dist, 'Ka maile hope', 5800),]
T0 = 5000
final_vals_list = []
col_names = ['Name', 'Total Flux [cgs]', 'Luminosity [cgs]', 'Radius [cm]', 'T [K]', 'T_err [K]']
for star_flux_tup in star_flux_tup_list:
    star_flux, dist, name, measured_temp = star_flux_tup
    star_flux = star_flux.astype(float)
    valid_indices = ~np.isnan(star_flux)
    lambda_mu_filtered = lambda_arr_mu[valid_indices]
    lambda_cm_filtered = lambda_arr_cm[valid_indices]
    filters_filtered = list(filters[valid_indices])
    star_flux_filtered = star_flux[valid_indices]

    S0 = (star_flux_filtered[filters_filtered.index('V')] /  # Scaling factor to account for distance from Earth
          planck_function(lambda_cm_filtered[filters_filtered.index('V')], T0))
    popt, pcov = curve_fit(scaled_planck, lambda_cm_filtered, star_flux_filtered, p0=[T0, S0])

    temp = int(popt[0])
    total_flux = simpson(star_flux_filtered, lambda_cm_filtered)
    lum = 4 * total_flux * np.pi * dist**2
    r = np.sqrt(lum / (4*np.pi*const.sigma_sb.cgs.value*(temp**4)))
    final_vals_list.append((name, "{:.2e}".format(total_flux), "{:.2e}".format(lum),
                            "{:.2e}".format(r), temp, abs(temp-measured_temp)))

    # Plotting each star
    plt.plot(lambda_mu_filtered, star_flux_filtered, label=f'{name}: {temp} K')

# Finding temperature, total flux, and radius for Procyon B
S0 = (procb_flux_list[7] / planck_function(procb_lambda_arr_cm[7], T0))  # Using I band since data does not have V band
popt, pcov = curve_fit(scaled_planck, procb_lambda_arr_cm, procb_flux_list, p0=[T0, S0])
temp = int(popt[0])
total_flux = simpson(procb_flux_list, procb_lambda_arr_cm)
lum = 4 * total_flux * np.pi * dist**2
r = np.sqrt(lum / (4*np.pi*const.sigma_sb.cgs.value*(temp**4)))
final_vals_list.append(('Ka ʻōnohi aliʻi', "{:.2e}".format(total_flux),
                        "{:.2e}".format(lum), "{:.2e}".format(r), temp, abs(temp-7740)))

# Plotting Procyon B on graph with other stars
plt.plot(procb_lambda_arr_mu, procb_flux_list, label=f'Ka ʻōnohi aliʻi: {temp} K')

plt.xlim(0.36, 4)
plt.legend()
plt.ylabel('Spectral Flux Density [cgs]')
plt.xlabel('Wavelength [microns]')
plt.title('B_lambda vs Wavelength')
plt.savefig('Blambda_vs_wavelength.png')
plt.show()
plt.clf()

# Producing table of all calculated values
final_vals_df = pd.DataFrame(final_vals_list, columns=col_names)
plt.subplots(figsize=(12, 4))
plt.axis('off')
plt.table(cellText=final_vals_df.values,
          colLabels=final_vals_df.columns,
          loc='center',
          cellLoc='center').scale(1.2, 3)
plt.savefig('calculated_values_table.png')
plt.show()
plt.clf()

# Plotting Procyon B on its own to get a better representation of its SED
plt.plot(procb_lambda_arr_mu, procb_flux_list, label=f'Ka ʻōnohi aliʻi: {temp} K')
plt.legend()
plt.ylabel('Spectral Flux Density [cgs]')
plt.xlabel('Wavelength [microns]')
plt.title('B_lambda vs Wavelength (Ka ʻōnohi aliʻi)')
plt.savefig('Blambda_vs_wavelength_procb.png')
plt.show()
plt.clf()




