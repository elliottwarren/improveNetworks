"""
Script to carry out Kriging on 3D modelled backscatter over London.

Created by Elliott Warren Fri 14 Dec 2018
"""

# multiple import when on MO machine
import sys
#sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')
# from threeD_backscatter_over_london import calculate_corner_locations, rotate_lon_lat_2D

# workaround while PYTHONPATH plays up on MO machine
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/Utils') #aerFO
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ellUtils') # general utils
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ceilUtils') # ceil utils

import matplotlib
#matplotlib.use('Agg') # needed as SPICE does not have a monitor and will crash otherwise if plotting is used


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import math
import datetime as dt
import iris

#from threeD_backscatter_over_london_by_hour import all_semivariance
#import threeD_backscatter_over_london_by_hour as td

import dask.multiprocessing
import dask
#import dask.array as da
from dask import compute, delayed, visualize
# setting dask.config does not work as it is only present in the developers version
#    apparently: https://github.com/dask/dask/issues/3531
# dask.config.set(scheduler='processes') # doesn't work

#import multiprocessing
#os.system('echo multiprocessing.cpu_count()')
#c = str(multiprocessing.cpu_count())
#os.system('echo '+ c)
#print multiprocessing.cpu_count()
#print''

from ellUtils import ellUtils as eu
from ceilUtils import ceilUtils as ceil
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

# import ellUtils as eu
# import ceilUtils as ceil
# from Utils import FOUtils as FO
# from Utils import FOconstants as FOcon

from pykrige.ok import OrdinaryKriging
# from pykrige.uk import UniversalKriging
# from pykrige.ok3d import OrdinaryKriging3D
from pykrige.rk import Krige
from pykrige.compat import GridSearchCV

# Kriging Doc: https://media.readthedocs.org/pdf/pykrige/latest/pykrige.pdf


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


def rotate_lon_lat_2D(longitude, latitude, model_type, corner_locs=False):
    """
    Create 2D array of lon and lats in rotated space (WGS84)

    :param mod_data:
    :keyword corner_locs (bool): return lon and lat corner locations so (0,0) is bottom left corner
            of bottom left box
    :return: rotlon2D, rotlat2D (2D arrays): 2D arrays of rotate lon and latitudes (rot. to WSG84)
    """

    import cartopy.crs as ccrs
    import iris

    # test corners with a plot if necessary
    def test_create_corners(rotlat2D, rotlon2D, corner_lats, corner_lons):
        """ quick plotting to see whether the functions within the script have created the corner lat and lon properly.
        Original and extrap/interp data together."""

        # copy and pasted out of main functions. Put back if this function is to be used.
        # # 3D array with a slice with nothing but 0s...
        # normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
        # lons_orig = normalgrid[:, :, 0]
        # lats_orig = normalgrid[:, :, 1]
        # # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...
        #
        # # 3D array with a slice with nothing but 0s...
        # normalgrid = ll.transform_points(rotpole, corner_lons, corner_lats)
        # lons = normalgrid[:, :, 0]
        # lats = normalgrid[:, :, 1]

        plt.figure()
        plt.scatter(rotlat2D, rotlon2D, color='red')
        plt.scatter(corner_lats, corner_lons, color='blue')
        # plt.scatter(a, d, color='green')
        # plt.scatter(lats, lons, color='red')
        # plt.scatter(lats_orig, lons_orig, color='blue')
        plt.figure()
        plt.pcolormesh(corner_lats, vmin=np.nanmin(corner_lats), vmax=np.nanmax(corner_lats))
        plt.colorbar()

        return

    # duplicate the single column array but number of rows in latitude
    # rotlat2D needs to be transposed as rows need to be latitude, and columns longitude.
    # np.transpose() does transpose in the correct direction:
    # compare mod_all_data['latitude'][n] with rotlat2D[n,:]

    rotlon2D = np.array([longitude] * latitude.shape[0])
    rotlat2D = np.transpose(np.array([latitude] * longitude.shape[0]))

    # Get model transformation object (ll) to unrotate the model data later
    # Have checked rotation is appropriate for all models below
    if (model_type == 'UKV') | (model_type == '55m') | (model_type == 'LM'):
        rotpole = (iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
            6371229.0))).as_cartopy_crs()  # rot grid
        #rotpole2 = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        ll = ccrs.Geodetic()  # to normal grid
    else:
        raise ValueError('model_type not set as UKV!')

    # # calculate corners of grid, from their centre positions, of unrotated data
    # # ---------------------
    # if corner_locs == True:
    #     rotlat2D = calculate_corner_locations(rotlat2D)
    #     rotlon2D = calculate_corner_locations(rotlon2D)

    # 3D array with a slice with nothing but 0s...
    normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
    lons = normalgrid[:, :, 0]
    lats = normalgrid[:, :, 1]
    # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...

    # test to see if the functions above have made the corner lat and lons properly
    # test_create_corners(rotlat2D, rotlon2D, corner_lats, corner_lons)

    return lons, lats


# plotting

def twoD_range_one_day(day_time, day_height, day_data, mlh_obs, data_var, pltsavedir, U_str, aer_str, rh_str, dir_wind_str):

    """
    Plot today's 2D range today with the mixing layer heights
    :param day_time:
    :param day_height:
    :param day_data:
    :param mlh_obs (dict): mixing layer height observations
    :return: fix, ax
    """

    # Get discrete colourbar
    vmin = 0.0
    vmax = 65.0
    cmap = cm.get_cmap('viridis_r', 13)

    day_hrs = [i.hour for i in day_time]
    fig, ax = plt.subplots(1, figsize=(7, 5))

    plt.pcolormesh(day_hrs, day_height, day_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    for site_id in mlh_obs.iterkeys():
        plt.plot(day_hrs, mlh_obs[site_id]['MH'], marker='o', label=site_id)
    plt.legend(loc='upper left')
    plt.ylabel('height [m]')
    plt.xlabel('time [hr]')

    plt.suptitle(mod_data['time'][0].strftime('%Y-%m-%d') + ' ' + data_var +
                 '; WS=' + U_str + '; aer='+aer_str+'; RH='+rh_str+'; Udir='+dir_wind_str)
    plt.savefig(pltsavedir + mod_data['time'][0].strftime('%Y-%m-%d_') + data_var + '.png')
    print 'saved: '+pltsavedir + mod_data['time'][0].strftime('%Y-%m-%d_') + data_var + '.png'
    plt.close()

    return fig, ax

def twoD_sill_one_day(day_time, day_height, day_data, mlh_obs, data_var, pltsavedir, U_str, aer_str, rh_str, dir_wind_str):

    """
    Plot today's 2D range today with the mixing layer heights
    :param day_time:
    :param day_height:
    :param day_data:
    :param mlh_obs (dict): mixing layer height observations
    :return: fix, ax
    """

    # Get discrete colourbar
    cmap = cm.get_cmap('jet')

    day_hrs = [i.hour for i in day_time]
    fig, ax = plt.subplots(1, figsize=(7, 5))
    #plt.pcolormesh(day_hrs, day_height, day_data, vmin=vmin, vmax=vmax, cmap=cmap)

    plt.pcolormesh(day_hrs, day_height, day_data, cmap=cm.get_cmap('jet'), norm=LogNorm(vmin=1e-4, vmax=1e0))
    # , cmap=cm.get_cmap('jet'))
    plt.colorbar()
    for site_id in mlh_obs.iterkeys():
        plt.plot(day_hrs, mlh_obs[site_id]['MH'], marker='o', label=site_id)
    plt.legend(loc='upper left')
    plt.ylabel('height [m]')
    plt.xlabel('time [hr]')

    plt.suptitle(mod_data['time'][0].strftime('%Y-%m-%d') + ' ' + data_var + '; WS=' + U_str +
                 '; aer='+aer_str+'; RH='+rh_str+'; Udir='+dir_wind_str)
    plt.savefig(pltsavedir + mod_data['time'][0].strftime('%Y-%m-%d_') + data_var + '.png')
    print 'saved: '+pltsavedir + mod_data['time'][0].strftime('%Y-%m-%d_') + data_var + '.png'
    plt.close()

    return fig, ax

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # data variable to plot
    data_var = 'backscatter'
    # data_var = 'RH'
    # <= ~2000 m - same as PCA heights
    height_range = np.arange(0, 24)
    # N-S extract over central London for UKV, given larger domain
    #lon_range = np.arange(41, 47)
    lr_s = 41
    lr_e = 47

    # save?
    numpy_save = True
    
    # debugging? - shrink data size later on
    test_mode = True
    
    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    variogramsavedir = maindir + 'figures/model_runs/variograms/'
    twodrangedir = maindir + 'figures/model_runs/2D_range/'
    twodsilledir = maindir + 'figures/model_runs/2D_sill/'
    twodRangeCompositeDir = twodrangedir + 'composite/'
    krigingsavedir = datadir + 'npy/kriging/'

    # intial test case
    # daystr = ['20181020']
    # daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # current set (missing 20180215 and 20181101) # 08-03
    # daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507',
    #           '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
    #           '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
    #           '20180901','20180902','20180903','20181007','20181010','20181020','20181023']

    # daystr = ['20180514','20180515','20180519','20180520','20180622','20180623','20180624',
    #           '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
    #           '20180901','20180902','20180903','20181007','20181010','20181020','20181023']

    daystr = ['20180803','20180804','20180805','20180806',
              '20180901','20180902','20180903','20181007','20181010','20181020','20181023']
    days_iterate = eu.dateList_to_datetime(daystr)

    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']

    # get height information for all the sites
    site_bsc = ceil.extract_sites(all_sites, height_type='agl')

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # print data variable to screen
    print data_var

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    # empty arrays to fill  (day, hour, height)
    sill = np.empty((len(days_iterate), 24, len(height_range)))
    sill[:] = np.nan

    # empty arrays to fill  (day, hour, height)
    v_range = np.empty((len(days_iterate), 24, len(height_range)))
    v_range[:] = np.nan

    # empty arrays to fill  (day, hour, height)
    nugget = np.empty((len(days_iterate), 24, len(height_range)))
    nugget[:] = np.nan

    U_mean = np.empty((len(days_iterate)))
    U_mean[:] = np.nan

    aer_mean = np.empty((len(days_iterate)))
    aer_mean[:] = np.nan

    rh_mean = np.empty((len(days_iterate)))
    rh_mean[:] = np.nan

    dir_wind_mean = np.empty((len(days_iterate)))
    dir_wind_mean[:] = np.nan

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d')

        # times to match to, so the time between days will line up
        start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
        end = start + dt.timedelta(days=1) - dt.timedelta(minutes=60)
        time_match = eu.date_range(start, end, 1, 'hour')

        # calculate the 3D backscatter field across London
        # .shape = (hour, height, lat, lon)
        mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, metvars=True)

        # rotate the winds out from rotated grid to normal WGS84
        # iris.analysis.cartography.rotate_winds()

        # # reduce domain size to match UKV extract
        # # domain edges found using eu.nearest compared to the UKV extract domain edges
        # if model_type == 'UKV':
        #     mod_data['longitude'] = mod_data['longitude'][lon_range]
        #     mod_data[data_var] = mod_data[data_var][:, :, :, lon_range]
        #     mod_data['u_wind'] = mod_data['u_wind'][:, :, :, lon_range]
        #     mod_data['v_wind'] = mod_data['v_wind'][:, :, :, lon_range]
        #     mod_data['RH'] = mod_data['RH'][:, :, :, lon_range]
        #     mod_data['aerosol_for_visibility'] = mod_data['aerosol_for_visibility'][:, :, :, lon_range]
            
        # # shrink data size to help with debugging
        # if (model_type == 'LM') & (test_mode == True):
        #     mod_data['longitude'] = mod_data['longitude'][:50]
        #     mod_data['latitude'] = mod_data['latitude'][:50]
        #     mod_data[data_var] = mod_data[data_var][:, :, :50, :50]

        # testing
        # rotate the lon and lats onto a normal geodetic grid (WGS84) [degrees] and expands lon and lat by 1 so it can
        # be used in plt.pcolormesh() which wants corner edges not center points
        rotLon2d_deg, rotLat2d_deg = rotate_lon_lat_2D(mod_data['longitude'], mod_data['latitude'], model_type)

        # convert lons and lats to distance [km] from the bottom left corner of the grid
        # unrotLon2d, unrotLat2d, unrotLon1d, unrotLat1d= convert_deg_to_km(mod_data['longitude'][lon_range], mod_data['latitude'])
        unrotLon2d, unrotLat2d, unrotLon1d, unrotLat1d = \
            convert_deg_to_km(mod_data['longitude'][lr_s:lr_e], mod_data['latitude'])

        # # unrotate u and v wind components
        # # get u and v wind, lon and lats. In netCDF, u_wind = x_wind, v_wind = y_wind
        # rot_cs = iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(6371229.0))
        #
        # # normal lon_lats to regrid u and v winds onto
        # reg_cube = iris.cube.Cube(mod_data['aerosol_for_visibility'])
        # aer_lat = iris.coords.DimCoord(mod_data['latitude'], 'grid_latitude', coord_system=rot_cs)
        # aer_lon = iris.coords.DimCoord(mod_data['longitude'], 'grid_longitude', coord_system=rot_cs)
        # reg_cube.add_dim_coord(aer_lat, 2)
        # reg_cube.add_dim_coord(aer_lon, 3)
        #
        # # get u_winds with their lons and lats
        # u_wind_data = eu.netCDF_read('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/data' \
        # '/UKV/extract_London_prodm_op_ukv_20180802_21_full.nc',
        #                             vars=['latitude','longitude_xwind'])
        # u_cube = iris.cube.Cube(mod_data['u_wind'])
        # u_lat = iris.coords.DimCoord(u_wind_data['latitude'], 'grid_latitude', coord_system=rot_cs)
        # u_lon = iris.coords.DimCoord(u_wind_data['longitude_xwind'], 'grid_longitude', coord_system=rot_cs)
        # u_cube.add_dim_coord(u_lat, 2)
        # u_cube.add_dim_coord(u_lon, 3)
        # u_cube.coord('grid_latitude').convert_units('degrees')
        # u_cube.coord('grid_longitude').convert_units('degrees')
        # u_cube.regridded(reg_cube, mode='bilinear')
        #
        # # get v_winds with their lons and lats
        # v_wind_data = eu.netCDF_read('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/data' \
        # '/UKV/extract_London_prodm_op_ukv_20180802_21_full.nc',
        #                             vars=['latitude_ywind','longitude'])
        # v_cube = iris.cube.Cube(mod_data['v_wind'])
        # v_lat = iris.coords.DimCoord(mod_data['latitude_ywind'], 'grid_latitude', coord_system=rot_cs)
        # v_lon = iris.coords.DimCoord(mod_data['longitude'], 'grid_longitude', coord_system=rot_cs)
        # v_cube.add_dim_coord(v_lat, 2)
        # v_cube.add_dim_coord(v_lon, 3)


        # set n_lags equal to the latitude (longer dimension of the data)
        n_lags = len(mod_data['latitude'])

        # find important days (low wind, high aerosol
        # U wind from ground to 955 m - try to find day with lowest wind values (more local source emissions)
        # alternative - try when murk was high!
        u_wind = mod_data['u_wind'][:, :, :, lr_s:lr_e]
        v_wind = mod_data['v_wind'][:, :, :, lr_s:lr_e]
        U = np.sqrt((u_wind**2.0) + (v_wind**2.0))
        U_mean[d] = np.mean(U)
        # where blowing from # tested u_wind=-3.125, v_wind=0.875 => dir_wind=105.642...
        # take mean of wind components, then calculate direction
        u_wind_i = np.mean(mod_data['u_wind'][:, :, :, lr_s:lr_e])
        v_wind_i = np.mean(mod_data['v_wind'][:, :, :, lr_s:lr_e])
        dir_wind_mean[d] = 180 + ((180/np.pi)*np.arctan2(u_wind_i, v_wind_i))

        aer_mean[d] = np.mean(mod_data['aerosol_for_visibility'][:, :, :, lr_s:lr_e])
        rh_mean[d] = np.mean(mod_data['RH'][:, :, :, lr_s:lr_e])
        # read in MLH data
        mlh_obs = ceil.read_all_ceils(day, site_bsc, ceilDatadir, 'MLH', timeMatch=time_match)

        # ==============================================================================
        # Kriging
        # ==============================================================================
        # .shape(time, height, lat, lon) # definately lat then lon ...

        for height_idx, height_i in enumerate(mod_data['level_height'][height_range]): # [:20]
            print 'h = ' + str(height_i)
            for hr_idx, hr in enumerate(mod_data['time'][:-1]): # ignore midnight next day
                print 'hr_idx = ' + str(hr_idx)

                savesubdir = variogramsavedir + hr.strftime('%Y-%m-%d') + '/'
                if os.path.exists(savesubdir) == False:
                    os.mkdir(savesubdir)

                # extract 2D cross section for this time
                # just London area
                if data_var == 'backscatter':# or
                    data = np.log10(mod_data[data_var][hr_idx, height_idx, :, lr_s:lr_e])
                    #print 'data logged'
                elif data_var == 'specific_humidity':
                    data = np.log10(mod_data[data_var][hr_idx, height_idx, :, lr_s:lr_e]*1e3) # [g kg-1]
                else:
                    data = mod_data[data_var][hr_idx, height_idx, :, lr_s:lr_e]

                # # test which variogram model type to fit to the data using cross validation.
                # #   Kriging will use ordinary method and fixed number of lags regardless
                # # PyKrige documentation 4.1: Krige CV
                # param_dict = {"method": ["ordinary"],
                #               #'anisotropy_scaling':[0.5, 1, 2, 3],
                #               #'anisotropy_angle': [0, 90, 180, 270],
                #               "variogram_model": ["linear", "power", "spherical", 'exponential'],
                #               "nlags": [n_lags]}
                #               #"weight":[False]}
                #
                # # R^2 (coefficient of determination) regression score function which can
                # #   go negative and below -1.0 (see examples in first link below).
                # # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
                # # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
                # estimator = GridSearchCV(Krige(), param_dict) # , verbose=True
                #
                # # data - needs to be X : array-like, shape = [n_samples, n_features]
                # # Training vector, where n_samples is the number of samples and n_features is the number of features.
                # # location [lat, lon]
                # #! Note: import to make sure reshaped axis keep the data in the correct order so X_i(lon_i, lat_i) and not
                # #   some random lon or lat point. Be careful when stacking and rotating - always do checks agains the
                # #   original, input variable! e.g. check with y[30] = [lon[30,0], lat[30,0]]
                #
                y = data.flatten()
                X = np.stack((unrotLon2d.flatten(), unrotLat2d.flatten()), axis=1)
                # estimator.fit(X=X, y=y)

                #  test whether with or without weight is best with spherical
                # if estimator.best_params_['variogram_model'] == 'spherical':
                param_dict = {"method": ["ordinary"],
                              "variogram_model": ["spherical"],
                              "nlags": [n_lags],
                              'weight':[True, False]}
                estimator = GridSearchCV(Krige(), param_dict)
                estimator.fit(X=X, y=y)

                # # print results
                # if hasattr(estimator, 'best_score_'):
                #     print('best_score R2 = {:.3f}'.format(estimator.best_score_))
                #     print('best_params = ', estimator.best_params_)
                #     print('\nCV results::')
                # if hasattr(estimator, 'cv_results_'):
                #     for key in ['mean_test_score', 'mean_train_score',
                #                 'param_method', 'param_variogram_model']:
                #         print(' - {} : {}'.format(key, estimator.cv_results_[key]))

                
                # extract out the best parameters based on the cross validation. Remove 'method' key as it cannot
                #   be taken as a keyword by OrdinaryKriging() (which itself uses the ordinary method)
                best_params = estimator.best_params_
                del best_params['method']

                # Ordinary Kriging: 3.1.1 in pykrige documentation
                # Function fails when given 2D arrays so pass in flattened 1D arrays of the 2D data.
                OK = OrdinaryKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(), # enable_plotting=True,
                                      **best_params)

                # OK = OrdinaryKriging(unrotLon2d[5:-5,:].flatten(), unrotLat2d[5:-5,:].flatten(), data[5:-5,:].flatten(),
                #                      variogram_model='spherical', nlags=35, weight=True, verbose=True,enable_plotting=True) # verbose=True,enable_plotting=True,

                # OK = OrdinaryKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(),
                #                      variogram_model='spherical', nlags=35, verbose=True,enable_plotting=True)

                # OK = OrdinaryKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(),
                #                      variogram_model='spherical', nlags=35, #weight=True,
                #                      anisotropy_angle=np.mean(dir_wind), #anisotropy_scaling=np.mean(U),
                #                      verbose=True,enable_plotting=True) #

                # start plotting data
                plt.figure()
                OK.semivariance # y values
                OK.lags # x values
                plt.scatter(OK.lags, OK.semivariance)
                plt.plot(OK.lags, OK.variogram_function(OK.variogram_model_parameters, OK.lags), 'k-')
                ax = plt.gca()
                plt.suptitle(hr.strftime('%Y-%m-%d_%H') + ' beta; height=' + str(mod_data['level_height'][height_idx]) + 'm')
                ax.set_xlabel('Distance [km]')
                ax.set_ylabel('Semi-variance')
                savename = hr.strftime('%Y-%m-%d') + '_{:04.1f}m_'.format(height_i) + hr.strftime('hr%H') +'_variogram.png'
                plt.savefig(savesubdir + savename)
                plt.close()

                # # look at variogram_model_parameters to find the nugget, sill etc.
                # #! list order varies, depending on the variogram_model used!
                # #! gives back partial sill (full sill - nugget), not the full sill
                if best_params['variogram_model'] == 'spherical':
                    sill[d, hr_idx, height_idx] = OK.variogram_model_parameters[0]
                    v_range[d, hr_idx, height_idx] = OK.variogram_model_parameters[1]
                    nugget[d, hr_idx, height_idx] = OK.variogram_model_parameters[2]
                else:
                    sill[d, hr_idx, height_idx] = np.nan
                    v_range[d, hr_idx, height_idx] = np.nan
                    nugget[d, hr_idx, height_idx] = np.nan

        # ==============================================================================
        # Plotting
        # ==============================================================================

        # this day's variables to plot
        day_time = mod_data['time'][:-1]
        day_height = mod_data['level_height'][height_range]
        day_sill_data = sill[d, :, height_range]
        day_range_data = v_range[d, :, height_range]
        U_str = '{:.4}'.format(U_mean[d])
        dir_wind_str = '{:.4}'.format(dir_wind_mean[d])
        aer_str = '{:.4}'.format(aer_mean[d])
        rh_str = '{:.4}'.format(rh_mean[d])

        # plot 2D range and sill for today
        fig, ax = twoD_sill_one_day(day_time, day_height, day_sill_data, mlh_obs, data_var, twodsilledir,
                                    U_str, aer_str, rh_str, dir_wind_str)

        fig, ax = twoD_range_one_day(day_time, day_height, day_range_data, mlh_obs, data_var, twodrangedir,
                                     U_str, aer_str, rh_str, dir_wind_str)

        # save data in numpy array


    # # multiple days
    # time = [i.hour for i in mod_data['time'][:-1]]
    # range_comp = np.nanmedian(range[:, : , height_range], axis=0)
    # fig, ax = plt.subplots(1, figsize=(7,5))
    # plt.pcolormesh(time, mod_data['level_height'][height_range], np.transpose(range_comp) , vmin=0, vmax=65.0)
    # #, cmap=cm.get_cmap('jet'))
    # plt.colorbar()
    # plt.suptitle(str(len(days_iterate)) + 'days: ' + data_var)
    # plt.savefig(twodRangeCompositeDir +data_var+'_28daymed.png')

    print 'END PROGRAM'

    ## Trash code - kept here encase needed

    # # Create the semivariance and lags
    # # appending dmax += 0.001 ensure maximum bin is included
    # nlags = np.max(list(data.shape))
    # dmin = unrotLat2d[1, 0]  # equidistant grid, therefore the first box across ([1,0]) will have the minimum distance
    # dmax = np.sqrt(
    #     (np.amax(unrotLat2d) ** 2) + (np.amax(unrotLon2d) ** 2))  # [km] - diag distance to opposite corner of domain
    #
    # dd = (dmax - dmin) / nlags  # average nlag spacing
    # bins = [dmin + n * dd for n in range(nlags)]
    # dmax += 0.001
    # bins.append(dmax)
    #
    # # load in lags to limit computation expense (save if needed)
    # # lags = np.load(npy_savedir + 'lags/'+model_type+'_lags.npy')
    # # np.save(npy_savedir + 'lags/'+model_type+'_lags.npy', lags_full)
    #
    #
    # # set up semivariance array ready
    # # semivariance = np.zeros(nlags) # semivariance within each lag bin
    # # semivariance[:] = np.nan
    #
    # # sample size for each lag. Start at 0 and add them up as more idx pairs are found
    # # m = np.zeros(nlags)
    #
    # # maximum idx position for each dimension
    # idx_max = [j - 1 for j in data.shape]
    #
    # # create euclidean distance matrix (from point [0,0])
    # # only creates distances for one quadrant (top right) effectively
    # distance = np.sqrt((unrotLat2d ** 2) + (unrotLon2d ** 2))
    #
    # #                 plt.figure()
    # #                 plt.pcolormesh(distance)
    # #                 plt.colorbar()
    # #                 plt.show()
    #
    # os.system('echo calculating semivariance @ ' + str(dt.datetime.now()))
    #
    # # prepare all_semicariance() inputs by making them dask objects
    # #    hopefully make the dask delayed processes work better...
    # # data =
    #
    # semivariance, m, lags = all_semivariance(bins, data, distance, idx_max)
    #
    # # print 'semivariance_full'
    # # print semivariance_full
    # # print 'm_full'
    # # print m_full
    #
    # # fig = d.visualise()
    # # plt.savefig(daskmapsavedir + 'debugging_map.png')
    # # a = d.compute()
    #
    # # print'\n\n\n\n\n'
    #
    #
    # os.system('echo about to make the variogram @ ' + str(dt.datetime.now()))
    #
    # if (height_idx == 0) & (hr_idx == 0):
    #     print 'lags:'
    #     print lags
