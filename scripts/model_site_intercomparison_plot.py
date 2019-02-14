"""
Plotting to help out with the model intercomparison script. Copied heavily from the ceil comparison script.
 Reads in the statistics (.npy files) and plots the data

Created by Elliott Warren Mon 3 Dec 2018
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

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')
from ceil_comparison_plotting import intercomparison_line_shade_plotting

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # independent variable height or in time?
    ind_var = 'height'

    ceil_list = ['IMU', 'RGS', 'MR', 'SWT', 'NK']
    # ceil_list = ['RGS']

    for main_site in ceil_list:

        # ------------------

        # directories
        maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
        datadir = maindir + 'data/'
        savedir = maindir + 'figures/model_runs/intercomparison/'
        npysavedir = datadir + 'npy/model_runs/'

        # import all site names and heights
        all_sites = ['IMU', 'RGS', 'MR', 'SWT', 'NK']

        paired_sites = deepcopy(all_sites)
        paired_sites.remove(main_site)

        # save info?
        filename = main_site + '_aerFO_' + ind_var + '_statistics.npy'

        print 'main ceilometer: ' + ind_var + '_' + main_site

        # ==============================================================================
        # Read data
        # ==============================================================================

        data = np.load(npysavedir + filename).flat[0]
        statistics = data['statistics']
        cases = data['cases']

        # temporarily use these heights with each .npy file
        heights = {site: data['height'] for site in all_sites}


        # array of times that would align with the time statistics
        #   can use ANY year, month and day - as we just want the seconds and hours to be right
        start = dt.datetime(2000, 2, 1, 0, 0, 0)
        end = start + dt.timedelta(days=1)
        time_match = eu.date_range(start, end, 1, 'hour')

        # ==============================================================================
        # plot data
        # ==============================================================================

        stat_type = 'corr_rs'

        fig = intercomparison_line_shade_plotting(savedir, ind_var, data, heights, time_match,
                                                  stat_type, main_site, paired_sites)

        # # ---------------------------------
        # # median and IQR corr plot (all cases together)
        #
        # # plotting details
        # if ind_var == 'time':
        #     x_axis = time_match
        #     x_label = 'time [HH:MM]'
        # elif ind_var == 'height':
        #     x_axis = heights[main_site]
        #     x_label = 'height [m]'
        #     x_lim = [0.0, 2000.0]
        #
        # # stat to plot (what is in the statistics dictionary) # corr_rs
        # stat_type = 'corr_rs'
        #
        # fig = plt.figure()
        # ax = plt.gca()
        #
        # for paired_site_i in paired_sites:
        #
        #     if stat_type == 'median_diff':
        #         y_label = main_site + ' - ' + paired_site_i
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
        # plt.suptitle(main_site+ '; ' + str(len(cases)) + ' days; 15sec')
        # plt.savefig(savedir + stat_type +'_'+main_site +'_'+ind_var+'.png')

        # # ---------------------------------
        # # corr in height - split up days. One plot per pairing. Each case as a seperate line
        #
        # # plotting details
        # if corr_type == 'time':
        #     x_axis = time_match
        #     x_label = 'time [HH:MM]'
        # elif corr_type == 'height':
        #     x_axis = heights[main_site]
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
        #     plt.suptitle(main_site + ' & ' + paired_site_i +'; ' + str(len(daystrList)) + ' days; 15sec')
        #     plt.savefig(savedir + 'sep_paired/' + main_site + '_' + paired_site_i +'_spearmanr_' + corr_type + '.png')
        #     plt.close()