"""
Script to forward model backscatter for several sites within London and intercompare them in the same
way as the ceilometer observations.

Created by Elliott Warren Mon 3 Dec 2018
"""

import numpy as np
from scipy.stats import spearmanr
import datetime as dt
import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from copy import deepcopy

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')
from ceil_comparison import setup_statistics, nearest_heights

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # correlate in 'height' or in 'time'?
    corr_type = 'time'

    #main_site = 'IMU'
    #main_site = 'RGS'
    #main_site = 'MR'
    #main_site = 'SWT'
    main_site = 'NK'

    # min and max height to cut off backscatter (avoice clouds above BL, make sure all ceils start fairly from bottom)
    min_height = 0.0
    max_height = 2000.0

    # save?
    numpy_save = True

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    # modDatadir = datadir + model_type + '/'
    modDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/'\
                 + model_type + '/'
    savedir = maindir + 'figures/model_runs/'
    npysavedir = datadir + 'npy/model_runs/'

    # 2016 and 2017 cases from MorningBL work that went unused. Ignoring first few cases from MorningBL as they have
    # fewer variables in them.
    daystrList = ['20161125', '20161129', '20161130', '20161204', '20161205', '20161227', '20161229', '20170105',
                  '20170117', '20170118', '20170119', '20170120', '20170121', '20170122', '20170325', '20170330',
                  '20170408', '20170429', '20170522', '20170524', '20170526', '20170601', '20170614', '20170615',
                  '20170619', '20170620', '20170626', '20170713', '20170717', '20170813', '20170827', '20170828',
                  '20170902']

    days_iterate = eu.dateList_to_datetime(daystrList)
    [i.strftime('%Y%j') for i in days_iterate]

    # import all site names and heights
    all_sites = ['IMU', 'RGS', 'MR', 'SWT', 'NK']


    # all ceilometers excluding the main_ceilometer. These will be paired with the main ceilometer, one at a time.
    paired_sites = deepcopy(all_sites)
    paired_sites.remove(main_site)

    # which axis of backscatter to slice into based on corr_type
    print 'corr_type = ' + corr_type
    if corr_type == 'time':
        axis = 0
    elif corr_type == 'height':
        axis = 1

    # minimum number of pairs to have in a correlation
    min_corr_pairs = 6

    # save info?
    savestr = main_site + '_aerFO_' + corr_type + '_statistics.npy'

    print 'main ceilometer: ' + corr_type + '_' +main_site


    # ==============================================================================
    # Read data
    # ==============================================================================

    # # np load existing data
    # np_dict = np.load(npysavedir + savestr).flat[0]
    # statistics = np_dict['statistics']

    # set up statistics dictionary
    statistics = setup_statistics(corr_type, paired_sites, daystrList, 'ukv')

    # Read Ceilometer metadata
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d (%j)')

        # calculate aerFO backscatter and other variables
        mod_data = FO.mod_site_extract_calc(day, ceil_metadata, modDatadir, model_type, res, 905,
                                            allvars=True)

        # get list of all sites present this day, excluding the main site
        paired_sites_today = deepcopy(mod_data.keys())
        paired_sites_today.remove(main_site)

        # if the main ceil is present, then create statistics for it
        if main_site in mod_data.keys():

            # post process mod data?
            # for site in mod_data.keys():
            #     process stuff
            # end loop

            for paired_site_i in paired_sites:

                # match heights from each aerFO site
                # get unique pairs of paired ceilometer to the main ceilometer
                x_hc_unique_pairs, y_hc_unique_pairs, \
                x_unique_pairs_heights, y_unique_pairs_heights = \
                    nearest_heights(mod_data[main_site]['level_height'], mod_data[paired_site_i]['level_height'], max_height)

                # extract out all unique pairs below the upper height limit
                # these are time and height matched now
                main_backscatter = mod_data[main_site]['bsc_attenuated'][:, x_hc_unique_pairs]
                pair_backscatter = mod_data[paired_site_i]['bsc_attenuated'][:, y_hc_unique_pairs]
                x_height = mod_data[main_site]['level_height'][x_hc_unique_pairs]

                # idx position of where to store the statistic.
                #   time idx will simply be all the idx positions ([0, 1, ... len(time)])
                #   height will be the extracted heights based on the height matching above
                if corr_type == 'time':
                    stat_store_idx = np.arange(len(mod_data[main_site]['time']))
                elif corr_type == 'height':
                    stat_store_idx = x_hc_unique_pairs
                else:
                    raise ValueError('corr_type not time or height!')


                # carry out statistics on the profiles
                # for i, _ in enumerate(dimension):
                for i, stat_store_idx_i in enumerate(stat_store_idx):

                    # extract out 1d arrays from each, either in time or height
                    x = np.take(main_backscatter, i, axis=axis)
                    y = np.take(pair_backscatter, i, axis=axis)

                    # # extract out 1d arrays from each, either in time or height
                    # x = np.take(bsc_obs[main_ceil_name]['backscatter'], i, axis=axis)
                    # y = np.take(bsc_obs[paired_site_i]['backscatter'], i, axis=axis)

                    # mainly for the correlation functions which can misbehave when nans are involved...
                    finite_bool = np.isfinite(x) & np.isfinite(y)

                    # do stats
                    # stats_func = 1
                    # store sample size
                    statistics[paired_site_i]['n'][stat_store_idx_i, d] = np.sum(finite_bool)
                    statistics[paired_site_i]['mean_diff'][stat_store_idx_i, d] = \
                        np.nanmean(x[finite_bool] - y[finite_bool])
                    statistics[paired_site_i]['median_diff'][stat_store_idx_i, d] = \
                        np.nanmedian(x[finite_bool] - y[finite_bool])

                    # if the number of pairs to correlate is high enough ... correlate
                    if np.sum(finite_bool) >= min_corr_pairs:

                        # spearman correlation
                        statistics[paired_site_i]['corr_rs'][stat_store_idx_i, d], \
                        statistics[paired_site_i]['corr_ps'][stat_store_idx_i, d] = \
                            spearmanr(x[finite_bool], y[finite_bool])


    # save statistics in numpy array
    if numpy_save == True:
        save_dict = {'statistics': statistics, 'cases': days_iterate, 'paired_sites': paired_sites, 'main_site': main_site,
                     'height': mod_data[main_site]['level_height']}
        np.save(npysavedir + savestr, save_dict)
        print 'data saved!: ' + npysavedir + savestr
















    print 'END PROGRAM'