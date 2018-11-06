"""
Compare the observed ceilometer backscatter from multiple instruments against each other.

Created by Elliott Tues 23rd Oct '18
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import datetime as dt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon
import ceilUtils as ceil


def setup_statistics(corr_type):
    """
    Set up the statistics dictionary. Arrays of predefined size, based on the corr_type
    :param corr_type:
    :return:
    """

    if corr_type == 'time':
        dim_length = 5761
    if corr_type == 'height':
        dim_length = 770
    else:
        raise ValueError('corr_type not set as time or height!')

    statistics = {}

    for var in ['corr_rs', 'corr_ps']:
        statistics[var] = np.empty((dim_length, len(daystrList)))
        statistics[var][:] = np.nan

    return statistics

def remove_cloud_effected_backscatter(cld_data, var, backscatter, cbh_lower_gates, max_height):

    """
    Make smoothed backscatter nan that has some cloud backscatter in it
    1) lower all CBH down by x range gates as set by cbh_lower_gates, as Vaisala algorithm often puts CBH
           within the cloud. This creates a sort of buffer and lower limit to the cloud extent.
    2) for each new CBH below 'max_height', find effect backscatter around it, and make it nan
    3) for all backscatter above CBH, make it nan

    :param cld_data (dictionary):
    :param var (str): the key to the cld_data dictionary
    :param backscatter:
    :param cbh_lower_gates:
    :param max_height [m]: cut off limit to do the cbh correction to. As a cut off is used elsewhere in code,
                            this allows some iterations to be skipped.
    :return: backscatter (array)
    """

    a=1

    # 1) reduce CBH to below cloud
    cld_data[var] = cld_data[var] - (cbh_lower_gates * cld_data['range'][0])

    # get time based idx of cbh
    cld_idx = np.where(~np.isnan(cld_data[var]))[0]

    for cbh_t, cld_idx_t in zip(cld_data[var][cld_idx], cld_idx):

        # if the cbh would be below the max_height and therefore effect the remaining backscatter profiles ...
        # 10.0 is based on CL31 gate range
        if cbh_t - (cbh_lower_gates * 10.0) <= max_height:
            # 2) nan "contaminated" backscatter values around CBH
            #   time smoothing was 25 min (101 time steps, therefore find idx 50 before and after)
            #   make sure lowest (highest) idx position is 0 (len(time)) (don't go off the edge!)
            #   (50+1) as range needs an extra 1 to include the extra idx position
            t_s = np.max([cld_idx_t-50, 0])
            t_e = np.min([cld_idx_t+50+1, len(cld_data['time'])])

            # get height range of effected backscatter
            cbh_height_idx = eu.binary_search(cld_data['height'], cbh_t)
            # h_range = np.arange(np.max([cbh_height_idx-5, 0]), max_height_idx, dtype='int64')
            h_s = np.max([cbh_height_idx-5, 0])
            h_e = max_height_idx

            # nan effected profiles
            backscatter[t_s:t_e, h_s:h_e] = np.nan

    return backscatter

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/KSK15S/'
    ceilCLDDatadir = datadir + 'L0/KSK15S/'
    savedir = maindir + 'figures/obs_intercomparison/paired/'
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'

    # wavelength (int) [nm]
    ceil_lambda_nm = 905

    # # 12 clear days for the 2008 KSK15S pair
    # daystrList = ['20080208', '20080209', '20080210', '20080211', '20080217', '20080506', '20080507', '20080508',
    #               '20080510', '20080511', '20080512', '20080730']

    # test partial cloud day KSK15S
    # daystrList = ['20080730']

    # 2018 clear sky days for LUMA network (missing cases between doy 142 and 190)
    daystrList = ['20180418','20180419','20180420','20180505','20180506','20180507','20180514','20180515','20180519',
                  '20180520','20180805','20180806','20180902','20180216','20180406']

    days_iterate = eu.dateList_to_datetime(daystrList)
    # [i.strftime('%Y%j') for i in days_iterate]

    site_bsc = {'CL31-A_KSK15S': 40.5 - 31.4,
                'CL31-B_KSK15S': 40.5 - 31.4}

    sites = site_bsc.keys()

    # max height to cut off backscatter (avoice clouds above BL)
    max_height = 2000.0

    # correlate in height or in time?
    corr_type = 'height'

    # which axis of backscatter to slice into based on corr_type
    print 'corr_type = ' + corr_type
    if corr_type == 'time':
        axis = 1
    elif corr_type == 'height':
        axis = 1

    # number of range gate to lower CBH height by, as Vaisala algorithm puts CBH inside the cloud
    cbh_lower_gates = 4

    # minimum number of pairs to have in a correlation
    min_corr_pairs = 6

    # ==============================================================================
    # Read data
    # ==============================================================================

    # set up statistics dictionary
    statistics = setup_statistics(corr_type)

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d')

        # times to match to, so the time between days will line up
        start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
        end = start + dt.timedelta(days=1)
        time_match = eu.date_range(start, end, 15, 'seconds')

        # read all bsc obs for this day
        bsc_obs = ceil.read_all_ceils_BSC(day, site_bsc, ceilDatadir, calib=False, timeMatch=time_match)

        # read in CLD
        cld_obs = ceil.read_all_ceils(day, site_bsc, ceilCLDDatadir, 'CLD', timeMatch=time_match)

        # ==============================================================================
        # Process
        # ==============================================================================

        # process data before creating statistics
        for site in bsc_obs.iterkeys():

            # make all backscatter above the max_height allowed nan (e.g. all above 1500 m)
            _, max_height_idx, _ = eu.nearest(bsc_obs[site]['height'], max_height)
            bsc_obs[site]['backscatter'][:, max_height_idx+1:] = np.nan


            # nan cloud effected backscatter points
            bsc_obs[site]['backscatter'] = \
                remove_cloud_effected_backscatter(cld_obs[site], 'CLD_Height_L1',  bsc_obs[site]['backscatter'],
                                                  cbh_lower_gates, max_height)

        if corr_type == 'time':

            np.take(arr, indices, axis=3)

            # carry out statistics on the profiles
            for t, _ in enumerate(bsc_obs[sites[0]]['time']):

                # extract out 1d arrays from each
                x = bsc_obs[sites[0]]['backscatter'][t, :]
                y = bsc_obs[sites[1]]['backscatter'][t, :]

                # fast to do this, and index data, than use nan_policy='omi' in spearmanr function
                finite_bool = np.isfinite(x) & np.isfinite(y)

                # do stats
                stats_func = 1

                # if the number of pairs to correlate is high enough ... correlate
                if np.sum(finite_bool) >= min_corr_pairs:

                    # spearman correlation
                    statistics['corr_rs'][t, d], statistics['corr_ps'][t, d] = \
                        spearmanr(x[finite_bool], y[finite_bool])

        elif corr_type == 'height':

            # # find max height
            # _, max_height_idx, _ = eu.nearest(bsc_obs[site]['height'], max_height)

            # carry out statistics on each range gate
            for h, _ in enumerate(bsc_obs[sites[0]]['height']):

                x = bsc_obs[sites[0]]['backscatter'][:, h]
                y = bsc_obs[sites[1]]['backscatter'][:, h]

                # fast to do this, and index data, than use nan_policy='omi' in spearmanr function
                finite_bool = np.isfinite(x) &  np.isfinite(y)

                stats_func = 1
    # save statistics in numpy array


    # ==============================================================================
    # Plotting
    # ==============================================================================

    # plt.plot(bsc_obs[sites[0]]['backscatter'][finite_bool, h], label=sites[0])
    # plt.plot(bsc_obs[sites[1]]['backscatter'][finite_bool, h], label=sites[1])
    # plt.legend()
    # plt.suptitle(bsc_obs[sites[0]]['time'][0].strftime('%d/%m/%Y') + ' - DOY:' + bsc_obs[sites[0]]['time'][0].strftime('%j') +'; h='+str(bsc_obs[sites[0]]['height'][h])+'m')
    #
    # idx = np.array([all(np.isfinite(row)) for row in corr_rs])
    med_rs = np.nanmedian(corr_rs, axis=1)
    pct25_rs = np.nanpercentile(corr_rs, 25, axis=1)
    pct75_rs = np.nanpercentile(corr_rs, 75, axis=1)


    if corr_type == 'time':

        fig = plt.figure()
        ax = plt.gca()
        plt.plot_date(time_match, med_rs, '-',label='mean')
        ax.fill_between(time_match, pct25_rs, pct75_rs, alpha=0.2)
        plt.axhline(1.0, linestyle='--', color='black')

        plt.xlim([time_match[0], time_match[-1]])
        plt.ylabel('Spearman r')
        plt.xlabel('time [HH:MM]')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.suptitle(sites[0] + '_' + sites[1] +'; '+str(len(daystrList))+' days; 15sec')

        plt.savefig(savedir + sites[0] + '_' + sites[1] + '_spearmanr_time_cldexclude.png')

    elif corr_type == 'height':

        idx = np.array([all(np.isfinite(row)) for row in corr_rs])

        fig = plt.figure()
        ax = plt.gca()
        height = bsc_obs[site]['height']
        plt.plot(height[idx], med_rs[idx], '-', label='mean')
        ax.fill_between(height[idx], pct25_rs[idx], pct75_rs[idx], alpha=0.2)
        # plt.axhline(1.0, linestyle='--', color='black')

        # plt.xlim([time_match[0], time_match[-1]])
        plt.ylabel('Spearman r')
        plt.xlabel('height [m]')
        # ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.suptitle(sites[0] + '_' + sites[1] + '; ' + str(len(daystrList)) + ' days; 15sec')

        plt.savefig(savedir + sites[0] + '_' + sites[1] + '_spearmanr_height_cldexclude.png')


    print 'END PROGRAM'