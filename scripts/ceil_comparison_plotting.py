"""
Plotting to help out with the ceil_comparison script. Reads in the statistics (.npy files) and plots the data

Created by Elliott Warren Thurs 08 Nov 2018
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime as dt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
from copy import deepcopy

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

def intercomparison_line_shade_plotting(savedir, ind_var, data, heights, time_match, stat_type, main_name, paired_sites):

    """
    Plot median and IQR of a prescribed statistic, within the [statistic] dictionary

    :param savedir:
    :param ind_var:
    :param data:
    :param heights:
    :param time_match:
    :param stat_type:
    :param main_name:
    :param paired_sites
    :return: fig
    """

    # extract out vars from data
    statistics = data['statistics']
    cases = data['cases']

    # plotting details
    if ind_var == 'time':
        x_axis = time_match
        x_label = 'time [HH:MM]'
    elif ind_var == 'height':
        x_axis = heights[main_name]
        x_label = 'height [m]'
        x_lim = [0.0, 2000.0]

    fig = plt.figure()
    ax = plt.gca()

    for paired_site_i in paired_sites:

        if stat_type == 'median_diff':
            y_label = main_name + ' - ' + paired_site_i
            # y_lim = [np.nanmin(statistics[paired_site_i][stat_type]),
            #          np.nanmax(statistics[paired_site_i][stat_type])]
        elif stat_type == 'corr_rs':
            y_lim = [0.6, 1.05]
            y_label = 'Spearman r'

        # line colour to match ceilometer
        split = paired_site_i.split('_')[-1]
        colour = ceil.site_bsc_colours[split]

        stat = statistics[paired_site_i][stat_type]

        idx = np.array([any(np.isfinite(row)) for row in stat])
        med_rs = np.nanmedian(stat, axis=1)
        pct25_rs = np.nanpercentile(stat, 25, axis=1)
        pct75_rs = np.nanpercentile(stat, 75, axis=1)

        plt.plot(x_axis[idx], med_rs[idx], '-', color=colour, label=paired_site_i)
        ax.fill_between(x_axis[idx], pct25_rs[idx], pct75_rs[idx], facecolor=colour, alpha=0.2)

    # plt.xlim([time_match[0], time_match[-1]])
    # plt.xlim([0.0, 2000.0])
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.ylim([0.0, 1.05])
    plt.ylim(y_lim)
    plt.legend()
    if ind_var == 'time':
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.suptitle(main_name + '; ' + str(len(cases)) + ' days')
    plt.savefig(savedir + stat_type + '_' + ind_var + '_' +  main_name + '.png')

    return fig

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # independent variable height or in time?
    ind_var = 'time'

    ceil_list = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']
    #ceil_list = ['CL31-A_IMU']
    #ceil_list = ['CL31-B_RGS']
    #ceil_list = ['CL31-C_MR']
    #ceil_list = ['CL31-D_SWT']
    #ceil_list = ['CL31-E_NK']

    for main_ceil_name in ceil_list:

        # ------------------

        # directories
        maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
        datadir = maindir + 'data/'
        savedir = maindir + 'figures/obs_intercomparison/'
        npysavedir = datadir + 'npy/'

        # # 12 clear days for the 2008 KSK15S pair
        # daystrList = ['20080208', '20080209', '20080210', '20080211', '20080217', '20080506', '20080507', '20080508',
        #               '20080510', '20080511', '20080512', '20080730']

        # test partial cloud day KSK15S
        # daystrList = ['20080730']

        # # 2018 clear sky days for LUMA network (missing cases between doy 142 and 190)
        # daystrList = ['20180216', '20180406', '20180418', '20180419', '20180420',
        #    '20180505', '20180506', '20180507', '20180514', '20180515',
        #    '20180519', '20180520', '20180805', '20180806', '20180902']
        #
        # days_iterate = eu.dateList_to_datetime(daystrList)
        # # [i.strftime('%Y%j') for i in days_iterate]

        # import all site names and heights
        all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']

        paired_sites = deepcopy(all_sites)
        paired_sites.remove(main_ceil_name)

        # KSK15S pair
        # site_bsc = {'CL31-A_KSK15S': 40.5 - 31.4,
        #             'CL31-B_KSK15S': 40.5 - 31.4}

        # save info?
        filename = main_ceil_name + '_' + ind_var + '_statistics.npy'

        print 'main ceilometer: ' + ind_var + '_' + main_ceil_name

        # ==============================================================================
        # Read data
        # ==============================================================================

        data = np.load(npysavedir + filename).flat[0]
        statistics = data['statistics']
        site_bsc = data['site_bsc']
        cases = data['cases']

        # temporarily use these at the height of the ceilometers until the height is saved with each .npy file
        heights = {site: np.arange(site_bsc[site] + 10.0, site_bsc[site] + 10.0 + (10.0 * 770), 10.0) for site in site_bsc}


        # array of times that would align with the time statistics
        #   can use ANY year, month and day - as we just want the seconds and hours to be right
        start = dt.datetime(2000, 2, 1, 0, 0, 0)
        end = start + dt.timedelta(days=1)
        time_match = eu.date_range(start, end, 15, 'seconds')

        # ==============================================================================
        # plot data
        # ==============================================================================

        # stat to plot (what is in the statistics dictionary) # corr_rs, median_diff,...
        stat_type = 'corr_rs'

        # ---------------------------------
        # median and IQR corr plot (all cases together)
        fig = intercomparison_line_shade_plotting(ind_var, data, heights, stat_type, main_ceil_name)

        # # plotting details
        # if ind_var == 'time':
        #     x_axis = time_match
        #     x_label = 'time [HH:MM]'
        # elif ind_var == 'height':
        #     x_axis = heights[main_ceil_name]
        #     x_label = 'height [m]'
        #     x_lim = [0.0, 2000.0]
        #
        # # stat to plot (what is in the statistics dictionary) # corr_rs, median_diff,...
        # stat_type = 'corr_rs'
        #
        # fig = plt.figure()
        # ax = plt.gca()
        #
        # for paired_site_i in paired_sites:
        #
        #     if stat_type == 'median_diff':
        #         y_label = main_ceil_name + ' - ' + paired_site_i
        #         # y_lim = [np.nanmin(statistics[paired_site_i][stat_type]),
        #         #          np.nanmax(statistics[paired_site_i][stat_type])]
        #     elif stat_type == 'corr_rs':
        #         y_lim = [-0.2, 1.05]
        #         y_label = 'Spearman r'
        #
        #     # line colour to match ceilometer
        #     split = paired_site_i.split('_')[-1]
        #     colour = ceil.site_bsc_colours[split]
        #
        #     corr_rs = statistics[paired_site_i][stat_type]
        #
        #     idx = np.array([any(np.isfinite(row)) for row in corr_rs])
        #     med_rs = np.nanmedian(corr_rs, axis=1)
        #     pct25_rs = np.nanpercentile(corr_rs, 25, axis=1)
        #     pct75_rs = np.nanpercentile(corr_rs, 75, axis=1)
        #
        #     plt.plot(x_axis[idx], med_rs[idx], '-', color=colour, label=paired_site_i)
        #     ax.fill_between(x_axis[idx], pct25_rs[idx], pct75_rs[idx], facecolor=colour, alpha=0.2)
        #
        #
        # # plt.xlim([time_match[0], time_match[-1]])
        # #plt.xlim([0.0, 2000.0])
        # plt.ylabel(y_label)
        # plt.xlabel(x_label)
        # #plt.ylim([0.0, 1.05])
        # plt.ylim(y_lim)
        # plt.legend()
        # if ind_var == 'time':
        #     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        # plt.suptitle(main_ceil_name+ '; ' + str(len(cases)) + ' days; 15sec')
        # plt.savefig(savedir + stat_type +'_'+main_ceil_name +'_'+ind_var+'.png')

        # # ---------------------------------
        # # corr in height - split up days. One plot per pairing. Each case as a seperate line
        #
        # # plotting details
        # if corr_type == 'time':
        #     x_axis = time_match
        #     x_label = 'time [HH:MM]'
        # elif corr_type == 'height':
        #     x_axis = heights[main_ceil_name]
        #     x_label = 'height [m]'
        #
        # case_num = len(days_iterate) # number of days
        # # colours = cm.coolwarm(np.linspace(0, 1, case_num))
        # colours = cm.rainbow(np.linspace(0, 1, case_num))
        # c_range = np.linspace(0, case_num, 1)
        # norm = mpl.colors.Normalize(vmin=0, vmax=case_num)
        #
        # for paired_site_i in paired_sites:
        #
        #     fig = plt.figure()
        #     ax = plt.gca()
        #
        #     corr_rs = statistics[paired_site_i]['corr_rs']
        #
        #     for i in range(statistics[paired_site_i]['corr_rs'].shape[1]):
        #         idx = np.array([any(np.isfinite(row)) for row in corr_rs])
        #
        #         plt.plot(x_axis[idx], corr_rs[idx, i], '-', color=colours[i], label=days_iterate[i].strftime('%d/%m/%Y'))
        #
        #     # plt.xlim([time_match[0], time_match[-1]])
        #     plt.ylabel('Spearman r')
        #     plt.xlabel(x_label)
        #     plt.ylim([0.0, 1.05])
        #     plt.legend(loc=1)
        #     if corr_type == 'time':
        #         ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        #     plt.suptitle(main_ceil_name + ' & ' + paired_site_i +'; ' + str(len(daystrList)) + ' days; 15sec')
        #     plt.savefig(savedir + 'sep_paired/' + main_ceil_name + '_' + paired_site_i +'_spearmanr_' + corr_type + '.png')
        #     plt.close()