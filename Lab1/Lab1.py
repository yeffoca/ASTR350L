from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c

def vega_to_jansky_wise(m_vega, f_v0):
    return f_v0 * 10**(-m_vega/2.5)

def vega_to_ab(m_vega, const):
    return m_vega + const

def ab_to_jansky(m_ab):
    return 10**((m_ab + 48.6) / -2.5)

def freq_to_lambda(f_v, l):
    return (f_v * c.value * (10**2)) / (l * (10**(-4)))**2

###############################Data Prep##############################
# List of stars used in this lab
target_list = ['Betelgeuse', 'Alioth',
              'Rigel', 'alf Cen A']

# Names produced by simbad
simbad_ids = ['* alf Ori', '* eps UMa',
              '* bet Ori', '* alf Cen A']

# Collecting Simbad data (optical)
Simbad.add_votable_fields('allfluxes')
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
master_df_j = master_df.copy()
# master_df.to_csv('master.csv', index=False)

##################Data Cleaning####################
vega_conversions = {'U': 0.79, 'B': -0.09,
                       'V': 0.02, 'R': 0.21,
                       'I': 0.45, 'J': 0.91,
                       'H': 1.39, 'K': 1.85}

# zeromag = {'W1': 309.540, 'W2': 171.787,
#               'W3': 31.674, 'W4': 8.363}

zeromag = {'W1': 2.699, 'W2': 3.339,
              'W3': 5.174, 'W4': 6.620}

eff_lambda = {'U': 0.36, 'B': 0.438,
                 'V': 0.545, 'R': 0.641,
                 'I': 0.798, 'J': 1.22,
                 'H': 1.63, 'K': 2.19,
                 'W1': 3.4, 'W2': 4.6,
                 'W3': 12, 'W4': 22}
eff_lambda_df = pd.DataFrame.from_dict(eff_lambda, orient='index')

master_df_j['U'] = ab_to_jansky(vega_to_ab(master_df['U'], vega_conversions['U']))
master_df_j['B'] = ab_to_jansky(vega_to_ab(master_df['B'], vega_conversions['B']))
master_df_j['V'] = ab_to_jansky(vega_to_ab(master_df['V'], vega_conversions['V']))
master_df_j['R'] = ab_to_jansky(vega_to_ab(master_df['R'], vega_conversions['R']))
master_df_j['I'] = ab_to_jansky(vega_to_ab(master_df['I'], vega_conversions['I']))
master_df_j['J'] = ab_to_jansky(vega_to_ab(master_df['J'], vega_conversions['J']))
master_df_j['K'] = ab_to_jansky(vega_to_ab(master_df['K'], vega_conversions['K']))
master_df_j['H'] = ab_to_jansky(vega_to_ab(master_df['H'], vega_conversions['H']))

master_df_j['W1mag'] = ab_to_jansky(vega_to_ab(master_df['W1mag'], zeromag['W1']))
master_df_j['W2mag'] = ab_to_jansky(vega_to_ab(master_df['W2mag'], zeromag['W2']))
master_df_j['W3mag'] = ab_to_jansky(vega_to_ab(master_df['W3mag'], zeromag['W3']))
master_df_j['W4mag'] = ab_to_jansky(vega_to_ab(master_df['W4mag'], zeromag['W4']))


# master_df_j['W1mag'] = vega_to_jansky_wise(master_df['W1mag'], zeromag['W1'])
# master_df_j['W2mag'] = vega_to_jansky_wise(master_df['W2mag'], zeromag['W2'])
# master_df_j['W3mag'] = vega_to_jansky_wise(master_df['W3mag'], zeromag['W3'])
# master_df_j['W4mag'] = vega_to_jansky_wise(master_df['W4mag'], zeromag['W4'])

betel_series = master_df_j.iloc[0]
alioth_series = master_df_j.iloc[1]
rigel_series = master_df_j.iloc[2]
alpha_series = master_df_j.iloc[3]

filters = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'W1mag', 'W2mag', 'W3mag', 'W4mag']
lambda_arr = np.array(eff_lambda_df.values)

betel_flux_list = freq_to_lambda(np.array(betel_series[filters]), lambda_arr[0])
plt.plot(lambda_arr, betel_flux_list)
plt.title('Betelgeuse')
plt.show()
plt.clf()

alioth_flux_list = freq_to_lambda(np.array(alioth_series[filters]), lambda_arr[0])
plt.plot(lambda_arr, alioth_flux_list)
plt.title('Alioth')
plt.show()
plt.clf()

rigel_flux_list = freq_to_lambda(np.array(rigel_series[filters]), lambda_arr[0])
plt.plot(lambda_arr, rigel_flux_list)
plt.title('Rigel')
plt.show()
plt.clf()

alpha_flux_list = freq_to_lambda(np.array(alpha_series[filters]), lambda_arr[0])
plt.plot(lambda_arr, alpha_flux_list)
plt.title('Alpha')
plt.show()
plt.clf()









