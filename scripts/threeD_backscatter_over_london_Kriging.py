"""
Script to carry out Kriging on 3D modelled backscatter over London.

Created by Elliott Warren Fri 14 Dec 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import os
import math

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
# from pykrige.ok3d import OrdinaryKriging3D
from pykrige.rk import Krige
from pykrige.compat import GridSearchCV

# Kriging Doc: https://media.readthedocs.org/pdf/pykrige/latest/pykrige.pdf

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')
from threeD_backscatter_over_london import calculate_corner_locations, rotate_lon_lat_2D

# great circle formula
def Haversine_formula(lats, lons):

    # Haversine Formual - Calculate the great circle distance between two points on the earth)
    # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

    # approximate radius of earth in km
    R = 6373.0

    # grid center
    lat1 = math.radians(lats[0, 0])
    lon1 = math.radians(lons[0, 0])

    for lon2 in lons:
        for lat2 in lats:
            lon2 = math.radians(lon2)
            lat2 = math.radians(lat2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = math.sin(dlat / 2.0) ** 2.0 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2.0
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c

            # print("Result:", distance)
            # print("Should be:", 278.546, "km")
    return

def convert_deg_to_km(longitude, latitude):

    """
    Convert latitude and longitude from degrees to km from the bottom left most location (which will become 0 km in
    lat and lon)
    :param lontitude
    :param latitude
    :return:
    """

    r = 6370.0  # [km] radius of Earth
    c = 2.0 * math.pi * r  # [km] circumference of Earth

    # bottom left grid cell
    lon_0 = longitude[0]
    lat_0 = latitude[0]

    # Empty arrays to fill
    rotlon_km = np.empty(longitude.shape)
    rotlon_km[:] = np.nan
    rotlat_km = np.empty(latitude.shape)
    rotlat_km[:] = np.nan

    # [degrees]
    for i, lat_i in enumerate(latitude):
        dlat = lat_i - lat_0
        rotlat_km[i] = (dlat / 360.0) * c

    for j, lon_j in enumerate(longitude):
        dlon = lon_j - lon_0
        rotlon_km[j] = (dlon / 360.0) * c

    # rotlon2d = np.array([rotlon_km] * rotlat_km.shape[0])
    # rotlat2d = np.transpose(np.array([rotlat_km] * rotlon_km.shape[0]))

    rotlon2d = np.array([rotlon_km] * rotlat_km.shape[0])
    rotlat2d = np.transpose(np.array([rotlat_km] * rotlon_km.shape[0]))


    return rotlon2d, rotlat2d, rotlon_km, rotlat_km

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # data variable to plot
    data_var = 'backscatter'
    # data_var = 'RH'

    height_range = np.arange(0,30) # only first set of heights
    lon_range = np.arange(30, 65) # only London area (right hand side of larger domain -35:

    # save?
    numpy_save = True

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    modDatadir = datadir + model_type + '/'
    savedir = maindir + 'figures/model_runs/variograms/'
    twodrangedir = maindir + 'figures/model_runs/2D_range/'
    npysavedir = datadir + 'npy/'

    # intial test case
    # daystr = ['20180406']
    daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # current set (missing 20180215 and 20181101)
    # daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507',
    #           '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
    #           '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
    #           '20180901','20180902','20180903','20181007','20181010','20181020','20181023']
    days_iterate = eu.dateList_to_datetime(daystr)

    # save name
    # savestr = day.strftime('%Y%m%d') + '_3Dbackscatter.npy'

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # print data variable to screen
    print data_var

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    # empty arrays to fill  (day, hour, height)
    range = np.empty((len(days_iterate), 24, 41))
    range[:] = np.nan

    U_mean = []
    aer = []

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d')

        # calculate the 3D backscatter field across London
        # .shape = (hour, height, lat, lon)
        mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, res, 905, allvars=True)

        # rotate the lon and lats onto a normal geodetic grid (WGS84) [degrees] and expands lon and lat by 1 so it can
        # be used in plt.pcolormesh() which wants corner edges not center points
        # rotLon2d_deg, rotLat2d_deg = rotate_lon_lat_2D(mod_data['longitude'], mod_data['latitude'], model_type)

        # convert lons and lats to distance [km] from the bottom left corner of the grid
        unrotLon2d, unrotLat2d, unrotLon1d, unrotLat1d= convert_deg_to_km(mod_data['longitude'][lon_range], mod_data['latitude'])

        # find important days (low wind, high aerosol
        # U wind from ground to 955 m - try to find day with lowest wind values (more local source emissions)
        # alternative - try when murk was high!
        U = np.sqrt((mod_data['u_wind'][:, :16, :, :]**2.0) + (mod_data['v_wind'][:, :16, :, :]**2.0))
        U_mean += [np.nanmean(U)]
        aer += [np.nanmean(mod_data['aerosol_for_visibility'][:, :16, :, :])]

        # ==============================================================================
        # Kriging
        # ==============================================================================
        # .shape(time, height, lat, lon) # definately lat then lon ...

        for height_idx, height_i in enumerate(mod_data['level_height'][height_range]): # [:20]
            print 'h = ' + str(height_i)
            for hr_idx, hr in enumerate(mod_data['time'][:-1]): # ignore midnight next day

                # extract 2D cross section for this time
                # just London area
                if data_var == 'backscatter':# or
                    data = np.log10(mod_data[data_var][hr_idx, height_idx, :, lon_range]) # [-35:]
                    #print 'data logged'
                elif data_var == 'specific_humidity':
                    data = np.log10(mod_data[data_var][hr_idx, height_idx, :, lon_range]*1e3) # [g kg-1]
                else:
                    data = mod_data[data_var][hr_idx, height_idx, :, lon_range]

                #ToDo Find best model to fit on the variogram, then pass best model into the main OrdinaryKriging() function
                # use the kriging guide to help with this
                #ToDo Use OK.update_variogram_model to save computational resources

                # # test which model type to fit to the data using cross validation
                # # PyKrige documentation 4.1: Krige CV
                # param_dict = {"method": ["ordinary"],
                #               "variogram_model": ["linear", "power", "spherical"],
                #               "nlags": [20],
                #               # "weight": [True, False]
                #               }
                #
                # estimator = GridSearchCV(Krige(), param_dict, verbose=True)

                # # data - needs to be X : array-like, shape = [n_samples, n_features]
                # # Training vector, where n_samples is the number of samples and n_features is the number of features.
                # # location [lat, lon]
                # #! Note: import to make sure reshaped axis keep the data in the correct order so X_i(lon_i, lat_i) and not
                # #   some random lon or lat point. Be careful when stacking and rotating - always do checks agains the
                # #   original, input variable! e.g. check with y[30] = [lon[30,0], lat[30,0]]
                # y = data.flatten()
                # # data = [col0=lon, col1=lat]
                # X = np.stack((unrotLon2d.flatten(), unrotLat2d.flatten()), axis=1)
                #
                # estimator.fit(X=X, y=y)

                # # print results
                # if hasattr(estimator, 'best_score_'):
                #     print('best_score R2 = {:.3f}'.format(estimator.best_score_))
                #     print('best_params = ', estimator.best_params_)
                #     print('\nCV results::')
                # if hasattr(estimator, 'cv_results_'):
                #     for key in ['mean_test_score', 'mean_train_score',
                #                 'param_method', 'param_variogram_model']:
                #         print(' - {} : {}'.format(key, estimator.cv_results_[key]))


                # choose variogram model based on cross validation test reslts
                variogram_model = 'spherical'

                # Ordinary Kriging: 3.1.1 in pykrige documentation
                # Function fails when given 2D arrays so pass in flattened 1D arrays of the 2D data.
                OK = OrdinaryKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(),
                                     variogram_model=variogram_model, nlags=35, weight=True) # verbose=True,enable_plotting=True,

                # # Takes a long time ...
                # UK = UniversalKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(), enable_plotting=True,
                #                      variogram_model=variogram_model, nlags=20, weight=True, verbose=True) # verbose=True

                # ax = plt.gca()
                # plt.suptitle(hr.strftime('%Y-%m-%d_%H') + ' beta; height=' + str(mod_data['level_height'][height_idx]) + 'm')
                # ax.set_xlabel('Distance [km]')
                # ax.set_ylabel('Semi-variance')
                # savesubdir = savedir + hr.strftime('%Y-%m-%d') + '/'
                # savename = hr.strftime('%Y-%m-%d_%H') +  '_{:05.0f}'.format(mod_data['level_height'][height_idx]) + 'm_variogram'
                #
                # if os.path.exists(savesubdir) == False:
                #     os.mkdir(savesubdir)
                # plt.savefig(savesubdir + savename)
                # plt.close()

                # # look at variogram_model_parameters to find the nugget, sill etc.
                # #! list order varies, depending on the variogram_model used!
                # #! gives back partial sill (full sill - nugget), not the full sill
                if variogram_model == 'spherical':
                    range[d, hr_idx, height_idx] = OK.variogram_model_parameters[1]

        # plt.figure()
        # plt.hist(data.flatten(), bins=50)

    # np.argsort(U_mean)
    # np.array(U_mean)[np.argsort(U_mean)]

    # ==============================================================================
    # Plotting
    # ==============================================================================

    # One day case
    time = [i.hour for i in mod_data['time'][:-1]]
    fig, ax = plt.subplots(1, figsize=(7,5))
    plt.pcolormesh(time, mod_data['level_height'][height_range], range[0, : , height_range] , vmin=0, vmax=65.0)
    #, cmap=cm.get_cmap('jet'))
    plt.colorbar()
    plt.suptitle(mod_data['time'][0].strftime('%Y-%m-%d') + ' ' + data_var)
    plt.savefig(twodrangedir + mod_data['time'][0].strftime('%Y-%m-%d_')+data_var+'.png')

    # multiple days
    time = [i.hour for i in mod_data['time'][:-1]]
    range_avg = np.nanmedian(range[:, : , height_range], axis=0)
    fig, ax = plt.subplots(1, figsize=(7,5))
    plt.pcolormesh(time, mod_data['level_height'][height_range], np.transpose(range_avg) , vmin=0, vmax=65.0)
    #, cmap=cm.get_cmap('jet'))
    plt.colorbar()
    plt.suptitle(str(len(days_iterate)) + 'days: ' + data_var)
    plt.savefig(twodrangedir +data_var+'_28daymed.png')

    print 'END PROGRAM'


# # 3D kriging code (currently runs out of memory during...)
# # testing for 3D kriging (only bot 10 levels):
# # transpose them so they are the correcy shape:(lon, lat, height) = (35L, 65L, 41L)
# rotlon3d = np.transpose(np.array([unrotLon2d] * 10), axes=[1, 2, 0])
# rotlat3d = np.transpose(np.array([unrotLat2d] * 10), axes=[1, 2, 0])
# rotheight3d = np.transpose(np.array([[mod_data['level_height'][:10]] * unrotLat2d.shape[0]] * unrotLat2d.shape[1]),
#                            axes=[1, 2, 0])

# # test 3D kriging (bottom 10 levels only)
# data3d = np.transpose(np.log10(mod_data['backscatter'][t, :10, :, :]), axes=[1, 2, 0])

# # Carry out the kriging process
# OK3d = OrdinaryKriging3D(rotlon3d.flatten(), rotlat3d.flatten(), rotheight3d.flatten() / 1000.0,
#                          data3d.flatten(), enable_plotting=True,
#                          variogram_model='spherical', nlags=20, weight=True, verbose=True)  # verbose=True