"""
Compare the observed ceilometer backscatter from multiple instruments against each other.

Created by Elliott Tues 23rd Oct '18
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import datetime as dt
from copy import deepcopy

from ellUtils import ellUtils as eu
from ceilUtils import ceilUtils as ceil


def setup_statistics(corr_type, paired_sites, daystrList, data_type):
    """
    Set up the statistics dictionary. Arrays of predefined size, based on the corr_type
    :param corr_type:
    :return:
    """

    # which variable will the statistics be carried out in (time or height?) - make array correct shape
    if corr_type == 'time':
        if data_type == 'ceil':
            dim_length = 5761
        elif data_type == 'ukv':
            dim_length = 25
    elif corr_type == 'height':
        if data_type == 'ceil':
            dim_length = 770
        elif data_type == 'ukv':
            dim_length = 70
    else:
        raise ValueError('corr_type not set as time or height!')

    # what statistics to include?
    stat_list = ['corr_rs', 'corr_ps', # Spearman correlation r and p values
                 'mean_diff',
                 'median_diff',
                 'n'] # sample size for each statistic

    # set up empty arrays filled with nans
    # statistics dict -> site -> actual statistics e.g. correlation
    statistics = {}
    for site_i in paired_sites:
        statistics[site_i] = {}
        for var in stat_list:
            statistics[site_i][var] = np.empty((dim_length, len(daystrList)))
            statistics[site_i][var][:] = np.nan

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

def nearest_heights(x_height, y_height, max_height):

    """
    Get an idx array of the nearest y height to each x height. Makes sure each x_i in x height, has a single unique
    y_i pair from y height. Additional duplicate y pairings to a single x_i are removed. Works with height arrays
    of any length, with any relative start and end heights.

    Trims off any pairs where x_i was above a defined 'max_height'

    :param x_height (array):
    :param y_height:
    :param corr_max_height (float): maximum height of x_i in any x_i pair. Pairs over this height are removed.
    :return:
    """

    def unique_pairs(y_idx, diff):

        """
        Find range that excludes duplicate occurances. Keeps the pair with the smallest height difference and removes
        the rest.

        :param y_idx:
        :param diff:
        :return: unique_pairs_range

        At this point, the two arrays are like:
        y_idx = [0, 0, 0, 1, 3, 5, .... 769, 769, 769]
        x_idx = [0, 1, 2, 3, 4, 4, .... 67,  68,  69 ]
        By finding the unique pairs index array for y_idx, the same array can be used
        on the x_idx, as they are already paired up and of equal lengths. E.g. from above
        0-0, 0-1, ..., 3-4, 5-4 etc.
        """

        # 1. remove start duplicates
        # -------------------------------
        # find start idx to remove duplicate pairs
        duplicates = np.where(y_idx == y_idx[0])[0]  # find duplicates

        if len(duplicates) > 1:
            lowest_diff = np.argmin(abs(diff[duplicates]))  # find which has smallest difference
            pairs_idx_start = duplicates[lowest_diff]  # set start position for pairing at this point
        else:
            pairs_idx_start = 0

        # 2. remove end duplicates
        # -------------------------------
        # find end idx to remove duplicate pairs
        duplicates = np.where(y_idx == y_idx[-1])[0]  # find duplicates
        if len(duplicates) > 1:
            lowest_diff = np.argmin(abs(diff[duplicates]))  # find which has smallest difference
            pairs_idx_end = duplicates[lowest_diff]  # set start position for pairing at this point
        else:
            pairs_idx_end = len(y_idx)

        # create range in order to extract the unique pairs
        # unique_pairs_range = np.arange(pairs_idx_start, pairs_idx_end + 1)
        unique_pairs_range = np.arange(pairs_idx_start, pairs_idx_end)

        return unique_pairs_range

    a = np.array([eu.nearest(y_height, i) for i in x_height])
    values = a[:, 0]
    y_idx = np.array(a[:, 1], dtype=int)
    diff = a[:, 2]
    x_idx = np.arange(len(x_height))  # x_idx should be paired with y_idx spots.

    # Trim off the ends of y_idx, as UKV and y z0 and zmax are different, leading to the same gate matching multiple ukvs
    # assumes no duplicates in the middle of the arrays, just at the end

    # At this point, variables are like:
    # y_idx = [0, 0, 0, 1, 3, 5, .... 769, 769, 769]
    # x_idx = [0, 1, 2, 3, 4, 4, .... 67,  68,  69 ]
    unique_pairs_range = unique_pairs(y_idx, diff)

    # ALL unique pairs
    # Use these to plot correlations for all possible pairs, regardless of height
    y_unique_pairs = y_idx[unique_pairs_range]
    x_unique_pairs = x_idx[unique_pairs_range]
    values_unique_pairs = values[unique_pairs_range]
    diff_unique_pairs = diff[unique_pairs_range]

    # ~~~~~~~~~~~~~~~~~~~~ #

    # Remove pairs where y is above the max allowed height.
    # hc = height cut
    hc_unique_pairs_range = np.where(values_unique_pairs <= max_height)[0]

    # trim off unique pairs that are above the maximum height
    y_hc_unique_pairs = y_unique_pairs[hc_unique_pairs_range] # final idx for y
    x_hc_unique_pairs = x_unique_pairs[hc_unique_pairs_range] # final idx for x
    pairs_hc_unique_values = values_unique_pairs[hc_unique_pairs_range]
    pairs_hc_unique_diff = diff_unique_pairs[hc_unique_pairs_range]

    # get actual unique heights of each
    y_unique_pairs_heights = pairs_hc_unique_values
    x_unique_pairs_heights = y_unique_pairs_heights - pairs_hc_unique_diff


    return x_hc_unique_pairs, y_hc_unique_pairs, \
           x_unique_pairs_heights, y_unique_pairs_heights

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # correlate in 'height' or in 'time'?
    corr_type = 'time'

    main_ceil_name = 'CL31-A_IMU'
    #main_ceil_name = 'CL31-B_RGS'
    #main_ceil_name = 'CL31-C_MR'
    #main_ceil_name = 'CL31-D_SWT'
    #main_ceil_name = 'CL31-E_NK'

    # min and max height to cut off backscatter (avoice clouds above BL, make sure all ceils start fairly from bottom)
    min_height = 0.0
    max_height = 2000.0

    # save?
    numpy_save = True

    # ------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/'
    ceilCLDDatadir = datadir + 'L0/'
    savedir = maindir + 'figures/obs_intercomparison/paired/'
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    npysavedir = datadir + 'npy/'

    # # 12 clear days for the 2008 KSK15S pair
    # daystrList = ['20080208', '20080209', '20080210', '20080211', '20080217', '20080506', '20080507', '20080508',
    #               '20080510', '20080511', '20080512', '20080730']

    # test partial cloud day KSK15S
    # daystrList = ['20080730']

    # # 2018 clear sky days for LUMA network (missing cases between doy 142 and 190)
    daystrList = ['20180216', '20180406', '20180418', '20180419', '20180420',
        '20180505', '20180506', '20180507', '20180514', '20180515',
        '20180519', '20180520', '20180805', '20180806', '20180902']

    # # BIGGER list with more cloud that could be screened. (Sept, Oct and Nov + more summer cases)
    # # 2018 clear sky days for LUMA network (missing cases between doy 142 and 190)
    # daystrList = ['20180216', '20180406', '20180418', '20180419', '20180420',
    #               '20180505', '20180506', '20180507', '20180514', '20180515',
    #               '20180519', '20180520', '20180622', '20180623', '20180624',
    #               '20180625', '20180626', '20180802', '20180803', '20180804',
    #               '20180805', '20180806', '20180901', '20180902', '20180903',
    #               '20181007', '20181010', '20181020', '20181023',
    #               '20181102']


    days_iterate = eu.dateList_to_datetime(daystrList)
    [i.strftime('%Y%j') for i in days_iterate]

    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']

    # all ceilometers excluding the main_ceilometer. These will be paired with the main ceilometer, one at a time.
    paired_sites = deepcopy(all_sites)
    paired_sites.remove(main_ceil_name)

    # get height information for all the sites
    site_bsc = ceil.extract_sites(all_sites, height_type='agl')

    # KSK15S pair
    # site_bsc = {'CL31-A_KSK15S': 40.5 - 31.4,
    #             'CL31-B_KSK15S': 40.5 - 31.4}

    # which axis of backscatter to slice into based on corr_type
    print 'corr_type = ' + corr_type
    if corr_type == 'time':
        axis = 0
    elif corr_type == 'height':
        axis = 1

    # number of range gate to lower CBH height by for the cloud removal, as Vaisala algorithm puts CBH inside the cloud
    cbh_lower_gates = 4

    # minimum number of pairs to have in a correlation
    min_corr_pairs = 6

    # save info?
    savestr = main_ceil_name + '_' + corr_type + '_statistics.npy'

    print 'main ceilometer: ' + corr_type + '_' +main_ceil_name

    # ==============================================================================
    # Read data
    # ==============================================================================

    # set up statistics dictionary
    statistics = setup_statistics(corr_type, paired_sites, daystrList, 'ceil')

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d (%j)')

        # times to match to, so the time between days will line up
        start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
        end = start + dt.timedelta(days=1)
        time_match = eu.date_range(start, end, 15, 'seconds')

        # read all bsc obs for this day
        bsc_obs = ceil.read_all_ceils_BSC(day, site_bsc, ceilDatadir, calib=True, timeMatch=time_match, var_type='beta_tR')

        # read in CLD
        cld_obs = ceil.read_all_ceils(day, site_bsc, ceilCLDDatadir, 'CLD', timeMatch=time_match)

        # read in MLH
        #mlh_obs = ceil.read_all_ceils(day, site_bsc, ceilDatadir, 'MLH', timeMatch=time_match)

        # get list of all sites present this day, excluding the main site
        paired_sites_today = deepcopy(bsc_obs.keys())
        paired_sites_today.remove(main_ceil_name)

        # if the main ceil is present, then create statistics for it
        if main_ceil_name in bsc_obs.keys():

            # ==============================================================================
            # Process
            # ==============================================================================

            # process data before creating statistics
            for site in bsc_obs.iterkeys():

                # 1. make all backscatter above the max_height allowed nan (e.g. all above 1500 m)
                _, max_height_idx, _ = eu.nearest(bsc_obs[site]['height'], max_height)
                bsc_obs[site]['backscatter'][:, max_height_idx+1:] = np.nan

                # 2. make all backscatter below a min_height allowed nan (e.g. all below 70 m so all
                #   ceils start in the same place)
                if min_height != 0.0:
                    _, min_height_idx, _ = eu.nearest(bsc_obs[site]['height'], min_height)
                    bsc_obs[site]['backscatter'][:, :min_height_idx+1] = np.nan

                # 3. nan cloud effected backscatter points
                bsc_obs[site]['backscatter'] = \
                    remove_cloud_effected_backscatter(cld_obs[site], 'CLD_Height_L1',  bsc_obs[site]['backscatter'],
                                                      cbh_lower_gates, max_height)

                # # 4. nan above the mixing layer height using the MLH data
                # # ToDo change this to be a mixing height OR residual layer. Not just the mixing height.
                # for i, mlh_i in enumerate(mlh_obs[site]['MH']):
                #     _, mlh_height_idx, _ = eu.nearest(bsc_obs[site]['height'], mlh_i)
                #     # nan all backscatter above this level
                #     bsc_obs[site]['backscatter'][i, mlh_height_idx:] = np.nan


            for paired_site_i in paired_sites:

                # match heights from each ceilometer
                # get unique pairs of paired ceilometer to the main ceilometer
                x_hc_unique_pairs, y_hc_unique_pairs, \
                x_unique_pairs_heights, y_unique_pairs_heights = \
                    nearest_heights(bsc_obs[main_ceil_name]['height'], bsc_obs[paired_site_i]['height'], max_height)

                # extract out all unique pairs below the upper height limit
                # these are time and height matched now
                main_ceil_backscatter = bsc_obs[main_ceil_name]['backscatter'][:, x_hc_unique_pairs]
                pair_ceil_backscatter = bsc_obs[paired_site_i]['backscatter'][:, y_hc_unique_pairs]
                x_height = bsc_obs[main_ceil_name]['height'][x_hc_unique_pairs]

                # idx position of where to store the statistic.
                #   time idx will simply be all the idx positions ([0, 1, ... len(time)])
                #   height will be the extracted heights based on the height matching above
                if corr_type == 'time':
                    stat_store_idx = np.arange(len(bsc_obs[main_ceil_name]['time']))
                elif corr_type == 'height':
                    stat_store_idx = x_hc_unique_pairs
                else:
                    raise ValueError('corr_type not time or height!')

                # carry out statistics on the profiles
                # for i, _ in enumerate(dimension):
                for i, stat_store_idx_i in enumerate(stat_store_idx):

                    # extract out 1d arrays from each, either in time or height
                    x = np.take(main_ceil_backscatter, i, axis=axis)
                    y = np.take(pair_ceil_backscatter, i, axis=axis)

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
        save_dict = {'statistics': statistics, 'cases': days_iterate, 'site_bsc': site_bsc, 'main_ceil_name': main_ceil_name,
                     'height': bsc_obs[main_ceil_name]['height']}
        np.save(npysavedir + savestr, save_dict)
        print 'data saved!: ' + npysavedir + savestr

    # ==============================================================================
    # Plotting
    # ==============================================================================

    # # plotting details
    # if corr_type == 'time':
    #     x_axis = time_match
    #     x_label = 'time [HH:MM]'
    # elif corr_type == 'height':
    #     x_axis = bsc_obs[main_ceil_name]['height']
    #     x_label = 'height [m]'
    #
    # fig = plt.figure()
    # ax = plt.gca()
    #
    # for paired_site_i in paired_sites:
    #
    #     # line colour to match ceilometer
    #     split = paired_site_i.split('_')[-1]
    #     colour = ceil.site_bsc_colours[split]
    #
    #     corr_rs = statistics[paired_site_i]['corr_rs']
    #
    #     idx = np.array([all(np.isfinite(row)) for row in corr_rs])
    #     med_rs = np.nanmedian(corr_rs, axis=1)
    #     pct25_rs = np.nanpercentile(corr_rs, 25, axis=1)
    #     pct75_rs = np.nanpercentile(corr_rs, 75, axis=1)
    #
    #     plt.plot(x_axis[idx], med_rs[idx], '-', color=colour, label=paired_site_i)
    #     ax.fill_between(x_axis[idx], pct25_rs[idx], pct75_rs[idx], facecolor=colour, alpha=0.2)
    #     # plt.axhline(1.0, linestyle='--', color='black')
    #
    #
    # # plt.xlim([time_match[0], time_match[-1]])
    # plt.ylabel('Spearman r')
    # plt.xlabel(x_label)
    # plt.ylim([0.0, 1.05])
    # plt.legend()
    # if corr_type == 'time':
    #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    # plt.suptitle(main_ceil_name+ '; ' + str(len(daystrList)) + ' days; 15sec')
    # plt.savefig(savedir + main_ceil_name +'_spearmanr_'+corr_type+'.png')


    # ----------------------------------

    # # KSK15S pair
    #
    # # idx = np.array([all(np.isfinite(row)) for row in corr_rs])
    # med_rs = np.nanmedian(corr_rs, axis=1)
    # pct25_rs = np.nanpercentile(corr_rs, 25, axis=1)
    # pct75_rs = np.nanpercentile(corr_rs, 75, axis=1)
    #
    # if corr_type == 'time':
    #
    #     fig = plt.figure()
    #     ax = plt.gca()
    #     plt.plot_date(time_match, med_rs, '-',label='mean')
    #     ax.fill_between(time_match, pct25_rs, pct75_rs, alpha=0.2)
    #     plt.axhline(1.0, linestyle='--', color='black')
    #
    #     plt.xlim([time_match[0], time_match[-1]])
    #     plt.ylabel('Spearman r')
    #     plt.xlabel('time [HH:MM]')
    #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #     plt.suptitle(sites[0] + '_' + sites[1] +'; '+str(len(daystrList))+' days; 15sec')
    #
    #     plt.savefig(savedir + sites[0] + '_' + sites[1] + '_spearmanr_time_cldexclude.png')
    #
    # elif corr_type == 'height':
    #
    #     idx = np.array([all(np.isfinite(row)) for row in corr_rs])
    #
    #     fig = plt.figure()
    #     ax = plt.gca()
    #     height = bsc_obs[site]['height']
    #     plt.plot(height[idx], med_rs[idx], '-', label='mean')
    #     ax.fill_between(height[idx], pct25_rs[idx], pct75_rs[idx], alpha=0.2)
    #     # plt.axhline(1.0, linestyle='--', color='black')
    #
    #     # plt.xlim([time_match[0], time_match[-1]])
    #     plt.ylabel('Spearman r')
    #     plt.xlabel('height [m]')
    #     # ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #     plt.suptitle(sites[0] + '_' + sites[1] + '; ' + str(len(daystrList)) + ' days; 15sec')
    #
    #     plt.savefig(savedir + sites[0] + '_' + sites[1] + '_spearmanr_height_cldexclude.png')


    print 'END PROGRAM'