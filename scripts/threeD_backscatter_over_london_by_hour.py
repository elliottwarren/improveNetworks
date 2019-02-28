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
sys.path.append('/home/mm0100/ewarren/miniconda2/lib/python2.7/site-packages')

import matplotlib
matplotlib.use('Agg') # needed as SPICE does not have a monitor and will crash otherwise if plotting is used
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors

import numpy as np
import os
import math
import datetime as dt

import dask.multiprocessing
import dask.array as da

import ellUtils as eu
import ceilUtils as ceil
import FOUtils as FO
import FOconstants as FOcon

os.system('echo '+sys.platform)

from pykrige.ok import OrdinaryKriging
from pykrige.core import _calculate_variogram_model
from pykrige import variogram_models
# from pykrige.uk import UniversalKriging
# from pykrige.ok3d import OrdinaryKriging3D
# from pykrige.rk import Krige
# from pykrige.compat import GridSearchCV

# Kriging Doc: https://media.readthedocs.org/pdf/pykrige/latest/pykrige.pdf

#import resource
#os.system('max resources allowed (kB):')
#resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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

# plotting

def twoD_range_one_day(day_time, day_height, day_range, mlh_obs, data_var, U_str, aer_str):

    """
    Plot today's 2D range today with the mixing layer heights
    :param day_time:
    :param day_height:
    :param day_range:
    :param mlh_obs (dict): mixing layer height observations
    :return: fix, ax
    """

    # Get discrete colourbar
    vmin = 0.0
    vmax = 65.0
    #cmap, norm = eu.discrete_colour_map(vmin, vmax, 14) # cmap=plt.cm.viridis

    cmap = cm.get_cmap('viridis_r', 13)
    # cmap = cm.get_cmap('jet', 13)

    day_hrs = [i.hour for i in day_time]
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plt.pcolormesh(day_hrs, day_height, day_range, vmin=vmin, vmax=vmax, cmap=cmap)
    # , cmap=cm.get_cmap('jet'))
    plt.colorbar()
    for site_id in mlh_obs.iterkeys():
        plt.plot(day_hrs, mlh_obs[site_id]['MH'], marker='o', label=site_id)
    plt.legend(loc='upper left')
    plt.ylabel('height [m]')
    plt.xlabel('time [hr]')

    plt.suptitle(mod_data['time'][0].strftime('%Y-%m-%d') + ' ' + data_var + '; WS=' + U_str + '; aer='+aer_str)
    plt.savefig(twodrangedir + mod_data['time'][0].strftime('%Y-%m-%d_') + data_var + '.png')
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
    
    #height_range = np.arange(0,30) # only first set of heights
    #lon_range = np.arange(30, 65) # only London area (right hand side of larger domain -35:
    
    # save?
    numpy_save = True
    
    # ------------------
    
    # which modelled data to read in
    model_type = '55m' # 'UKV'
    res = FOcon.model_resolution[model_type]
    
    # directories
    maindir = os.environ.get('HOME') + '/Documents/AerosolBackMod/scripts/improveNetworks/'
    datadir = os.environ.get('DATADIR') + '/suite_forecasts/'
    ceilDatadir = os.environ.get('DATADIR') + '/MLH/'
    ceilmetadatadir = os.environ.get('DATADIR') + '/metadata/'
    variogramsavedir = maindir + 'figures/variograms/'+model_type+'/'
    twodrangedir = maindir + 'figures/2D_range/'+model_type+'/'
    npy_savedir = os.environ.get('DATADIR') + '/npy/'
    
    # intial test case
    daystr = ['20160913'] # 55 m case
    # daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # current set (missing 20180215 and 20181101) # 08-03
    days_iterate = eu.dateList_to_datetime(daystr)
    
    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-E_NK']
    #     all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']
    
    # get height information for all the sites
    site_bsc = ceil.extract_sites(all_sites, height_type='agl')
    
    if model_type == '55m':
        max_height_num = 76
        # max_hour_num = 10
    
    # passed in from bash script
    hr_str = sys.argv[1]
    hr_int = int(hr_str)
    
    # ==============================================================================
    # Read and process data
    # ==============================================================================
    
    # print data variable to screen
    print data_var
    
    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(ceilmetadatadir, ceilsitefile)
    
    # empty arrays to fill  (day, hour, height)
    v_range = np.empty((max_height_num))
    v_range[:] = np.nan
    
    # empty arrays to fill  (day, hour, height)
    sill = np.empty((max_height_num))
    sill[:] = np.nan
    
    # empty arrays to fill  (day, hour, height)
    nugget = np.empty((max_height_num))
    nugget[:] = np.nan
    
    heights_processed = np.empty((max_height_num))
    heights_processed[:] = np.nan
    
    #     U_mean = np.empty((len(days_iterate)))
    #     U_mean[:] = np.nan
    #     
    #     aer_mean = np.empty((len(days_iterate)))
    #     aer_mean[:] = np.nan
    
    day = days_iterate[0]

    print 'day = ' + day.strftime('%Y-%m-%d')

    for height_idx in np.arange(max_height_num):

        # date directory save for variogram
        savesubdir = variogramsavedir + day.strftime('%Y%m%d')
        if os.path.exists(savesubdir) == False:
            os.mkdir(savesubdir)


        print 'working on height_idx: '+str(height_idx)
        
        # # times to match to, so the time between days will line up
        # if model_type == 'UKV':
        #     start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
        #     end = start + dt.timedelta(days=1) - dt.timedelta(minutes=60)
        #     time_match = eu.date_range(start, end, 1, 'hour')
        # else:
        #     # start = dt.datetime(day.year, day.month, day.day, 9, 0, 0)
        #     start = dt.datetime(day.year, day.month, day.day, 9, 0, 0)
        #     end = start + dt.timedelta(hours=max_height_num)
        #     time_match = eu.date_range(start, end, 1, 'hour')
        
        # directory for today's data
        modDatadir = datadir + model_type + '/London/' + day.strftime('%Y%m%d') + '/'
        
        #for hr_idx, hr in enumerate(time_match):
            
        # use hour given in bash script
        hr = dt.datetime(day.year, day.month, day.day, hr_int, 0, 0)
        
        mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, res, 905, Z='03', allvars=False, hr=hr, height_extract_idx=height_idx)
        
        # store which height is being processed for .npy save later
        heights_processed[height_idx] = mod_data['level_height'][0]
        
        # reduce domain size to match UKV extract
        # domain edges found using eu.nearest compared to the UKV extract domain edges
        if model_type == '55m':
            mod_data['longitude'] = mod_data['longitude'][293:1213]
            mod_data['latitude'] = mod_data['latitude'][170:1089]
            mod_data[data_var] = mod_data[data_var][:, :, 170:1089, 293:1213]
        
        # # process mod_data if it doesn't yet exist, otherwise read it in from .npy save.
        # # calculate the 3D backscatter field across London
        # # .shape = (hour, height, lat, lon)
        # np_mod_data_savename = model_type + '_mod_data_processed.npy'
        # if os.path.exists(npy_savedir + np_mod_data_savename) == True:
        #     mod_data = np.load(npy_savedir + np_mod_data_savename).flat[0]
        # else:
        #     mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, res, 905, Z='03', allvars=False, hr=hr)
        #     np.save(npy_savedir + np_mod_data_savename, mod_data)

        print ''
        print 'read in mod_data ok!'
        print 'backscatter shape:'
        print mod_data['backscatter'].shape

        os.system('echo mod_data keys: '+str(mod_data.keys()))

        # rotate the lon and lats onto a normal geodetic grid (WGS84) [degrees] and expands lon and lat by 1 so it can
        # be used in plt.pcolormesh() which wants corner edges not center points
        # rotLon2d_deg, rotLat2d_deg = rotate_lon_lat_2D(mod_data['longitude'], mod_data['latitude'], model_type)

        # convert lons and lats to distance [km] from the bottom left corner of the grid
        unrotLon2d, unrotLat2d, _, _ = convert_deg_to_km(mod_data['longitude'], mod_data['latitude'])

        # find important days (low wind, high aerosol
        # U wind from ground to 955 m - try to find day with lowest wind values (more local source emissions)
        # alternative - try when murk was high!
        # U = np.sqrt((mod_data['u_wind'][:, :16, :, :]**2.0) + (mod_data['v_wind'][:, :16, :, :]**2.0))
        #U_mean[d] = np.nanmean(U)
        #aer_mean[d] = np.nanmean(mod_data['aerosol_for_visibility'][:, :16, :, :])

        # read in MLH data
        #mlh_obs = ceil.read_all_ceils(day, site_bsc, ceilDatadir, 'MLH', timeMatch=time_match)

        # ==============================================================================
        # Kriging
        # ==============================================================================
        # .shape(time, height, lat, lon) # definately lat then lon ...

        # for hr_idx, hr in enumerate(mod_data['time'][:-1]): # ignore midnight next day

        # read in mod_data for this hour
        
        # any prep....

        # height_idx = 0; height_i = mod_data['level_height'][0]


        # extract 2D cross section for this time
        # just London area
        if data_var == 'backscatter':
            data = np.log10(mod_data[data_var][0, 0, :, :])
        else:
            data = mod_data[data_var][0, 0, :, :]
                

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
        # #   some random lon or lat point. Be careful when stacking and rotating - always do checks against the
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


        # Create the semivariance and lags
        # appending dmax += 0.001 ensure maximum bin is included
        nlags = np.max(list(data.shape))
        dmin = unrotLat2d[1,0] # equidistant grid, therefore the first box across ([1,0]) will have the minimum distance 
        dmax = np.sqrt((np.amax(unrotLat2d)**2) + (np.amax(unrotLon2d)**2)) # [km] - diag distance to opposite corner of domain
        
        dd = (dmax - dmin) / nlags # average nlag spacing
        bins = [dmin + n * dd for n in range(nlags)]
        dmax += 0.001 
        bins.append(dmax)
        
        # setup arrays to be filled
        lags = np.zeros(nlags) # average actual distance between points within each lag bin (e.g could be 1.31 km for bin range 1 - 2 km)
        
        # set up semivariance array ready
        semivariance = np.zeros(nlags) # semivariance within each lag bin
        semivariance[:] = np.nan
        
        # sample size for each lag. Start at 0 and add them up as more idx pairs are found
        m = np.zeros(nlags)
        
        # maximum idx position for each dimension
        idx_max = [j - 1 for j in data.shape]
        
        
        
        # create euclidean distance matrix (from point [0,0])
        # only creates distances for one quadrant (top right) effectively
        distance = np.sqrt((unrotLat2d**2) + (unrotLon2d**2))
        
        # find idx linked to this h
        for h, bin_start in enumerate(bins[:-1]):
            
            print ' h = '+str(h)
            
            bin_end = bins[h+1]
            
            # idx relative to the current position [0,0] (just top right quadrant idx)
            idx = list(np.where((distance >= bin_start) & (distance < bin_end)))

            # combine 4 versions of idx that would represent all 4 quadrants and not just top right from the origin
            idx_origin = [np.hstack([idx[0], -idx[0], idx[0], -idx[0]]), np.hstack([idx[1], -idx[1], -idx[1], idx[1]])]
            # plt.scatter(idx_all[0], idx_all[1]) to see that it centre is over 0,0
            
        
            
#         def calc_semivariance(data, bins, nlags):
#         
#             """
#             Calculate the semivariance for the 2D data slice. Currently evenly spaced with nlags = domain width
#             
#             :param data (2D np.array): data to calculate semivariance from
#             :param bins (1D np.array): bin edges 
#             :param nlags (int): number of lags 
#             
#             """
            

            
            # square_differences - part of the equation for semi_variance
            # add onto 0, as there will be too many values to keep in an array and sum at the end
            square_diff = 0
            
            # paired distance differernces - need to find out average distance in this bin
            paired_dist_diff = 0
            
            for z_idx, z_i in np.ndenumerate(data):
                
                # adjust idx origin for this z_idx
                idx_from_z_idx = [z_idx[0] + idx_origin[0], z_idx[1] + idx_origin[1]]
                                                       
                
                # remove idx positions past the edges of the domain (below 0 or above the last idx for the data array)
                # split if statements up to reduce computational expense
                # super inefficient... 
                idx_pairs_keep = []
                for idx_i in idx_pairs_all:
                    if idx_i[0] >= 0:
                        if idx_i[0] <= idx_max[0]:
                            if idx_i[1] >= 0:
                                if idx_i[1] <= idx_max[1]:
                                    idx_pairs_keep += [idx_i]
                        
                # add on the number of pairs for z_i to the total in this lag
                m[h] += len(idx_pairs_keep)
                
                # calculate square difference between z_i and paired values. Add to the square_diff list to be summed after
                # iterating through all z_i
                square_diff += np.sum([(z_i-data[idx_i[0], idx_i[1]])**2 for idx_i in idx_pairs_keep])
                
                # find paired distances between all points
                lat_diffs = [unrotLat2d[z_idx[0], z_idx[1]] - unrotLat2d[idx_i[0], idx_i[1]] for idx_i in idx_pairs_keep]
                lon_diffs = [unrotLon2d[z_idx[0], z_idx[1]] - unrotLon2d[idx_i[0], idx_i[1]] for idx_i in idx_pairs_keep]
                paired_dist_diff += np.sum([np.sqrt((lat_diff_i**2) + (lon_diff_i**2)) for (lat_diff_i, lon_diff_i) in zip(lat_diffs, lon_diffs)])
                
            # finish the semivariance calculation (semivariance = effectivly half of the average square_diff)
            semivariance[h] = 0.5 * square_diff / m[h]
            lags[h] = paired_dist_diff / m[h] # /m finishes calculation of the mean
                                
        # return semivariance, lags, m
        # 
        # semivariance, lags, m = calc_semivariance(data, bins, nlags)

        
        # choose variogram model based on cross validation test reslts
        variogram_model = 'spherical'
        variogram_function = variogram_models.spherical_variogram_model
        weight = True

        # with the bins and semivariance, do the fitting
        variogram_model_parameters = \
                _calculate_variogram_model(lags, semivariance, variogram_model,
                                           variogram_function, weight)

        # plot variogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(lags, semivariance, 'r*')
        ax.plot(lags,
                variogram_function(variogram_model_parameters,
                                        lags), 'k-')

        ax = plt.gca()
        plt.suptitle(hr.strftime('%Y-%m-%d_%H') + ' beta; height=' + str(mod_data['level_height'][0]) + 'm')
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Semi-variance')
        savesubdir = variogramsavedir + hr.strftime('%Y-%m-%d') + '/'
        savename = hr.strftime('%Y-%m-%d_%H') +  '_{:05.0f}'.format(mod_data['level_height'][0]) + 'm_variogram'
        
        if os.path.exists(savesubdir) == False:
            os.mkdir(savesubdir)
        plt.savefig(savesubdir + savename)
        plt.close()

#         # Ordinary Kriging: 3.1.1 in pykrige documentation
#         # Function fails when given 2D arrays so pass in flattened 1D arrays of the 2D data.
#         OK = OrdinaryKriging(unrotLon2d.flatten(), unrotLat2d.flatten(), data.flatten(),
#                              variogram_model=variogram_model, nlags=1440, weight=True, enable_plotting=True) # verbose=True,enable_plotting=True,


        ax = plt.gca()
        plt.suptitle(hr.strftime('%Y-%m-%d_%H') + ' beta; height=' + str(mod_data['level_height'][0]) + 'm')
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Semi-variance')
        savesubdir = variogramsavedir + hr.strftime('%Y-%m-%d') + '/'
        savename = hr.strftime('%Y-%m-%d_%H') +  '_{:05.0f}'.format(mod_data['level_height'][0]) + 'm_variogram'
        
        if os.path.exists(savesubdir) == False:
            os.mkdir(savesubdir)
        plt.savefig(savesubdir + savename)
        plt.close()

        # # look at variogram_model_parameters to find the nugget, sill etc.
        # #! list order varies, depending on the variogram_model used!
        # #! gives back partial sill (full sill - nugget), not the full sill
        if variogram_model == 'spherical':
             sill[height_idx] = OK.variogram_model_parameters[0]
             v_range[height_idx] = OK.variogram_model_parameters[1]
             nugget[height_idx] = OK.variogram_model_parameters[2]

        # plt.figure()
        # plt.hist(data.flatten(), bins=50)

        # np.argsort(U_mean)
        # np.array(U_mean)[np.argsort(U_mean)]

        # ==============================================================================
        # Plotting
        # ==============================================================================

    # simple save encase everything goes wrong
    np_savename_hr = model_type +'_'+data_var+'_'+hr.strftime('%Y%m%d_%H')+'.npy'
    np_save_dict_hr = {'v_range': v_range, 'sill':sill, 'nugget':nugget, 
                        'day': day.strftime('%Y%m%d'), 'model_type':model_type, 'hr': hr, 'data_var':data_var, 'heights': heights_processed}
    np.save(npy_savedir + np_savename_hr, np_save_dict_hr)

    # # this day's variables to plot
    #day_time = mod_data['time'][:-1]
    #day_height = mod_data['level_height'][height_range]
    #day_range = v_range[d, :, height_range]
    #U_str = '{:.4}'.format(U_mean[d])
    #aer_str = '{:.4}'.format(aer_mean[d])

    # plot 2D range for today
    #fig, ax = twoD_range_one_day(day_time, day_height, day_range, mlh_obs, data_var, U_str, aer_str)

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