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

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/KSK15S/'
    savedir = maindir + 'figures/obs_intercomparison/paired/'
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'

    # wavelength (int) [nm]
    ceil_lambda_nm = 905

    # 12 very clear days. Just compare \beta_o below 1500 m to be safe.
    daystrList = ['20080208', '20080209', '20080210', '20080211', '20080217', '20080506', '20080507', '20080508',
                  '20080510', '20080511', '20080512', '20080730']
    # daystrList = ['20080506']
    days_iterate = eu.dateList_to_datetime(daystrList)

    site_bsc = {'CL31-A_KSK15S': 40.5 - 31.4,
                'CL31-B_KSK15S': 40.5 - 31.4}

    sites = site_bsc.keys()

    # max height to cut off backscatter (avoice clouds above BL)
    max_height = 1500.0

    # ==============================================================================
    # Read data
    # ==============================================================================

    # set up stats array (5761 is time length of backscatter data
    # Needs to be spearman as data is not calibrated
    corr_rs = np.empty((5761, len(daystrList)))
    corr_rs[:] = np.nan

    corr_ps = np.empty((5761, len(daystrList)))
    corr_ps[:] = np.nan

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d')

        # times to match to, so the time between days will line up
        start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
        end = start + dt.timedelta(days=1)
        time_match = eu.date_range(start, end, 15, 'seconds')

        # read all bsc obs for this day
        bsc_obs = ceil.read_all_ceils_BSC(day, site_bsc, ceilDatadir, calib=False, timeMatch=time_match)

        # ==============================================================================
        # Process
        # ==============================================================================

        # process data before creating statistics
        for site in bsc_obs.iterkeys():

            # make all backscatter above the max_height allowed nan (e.g. all above 1500 m)
            _, max_height_idx, _ = eu.nearest(bsc_obs[site]['height'], max_height)
            bsc_obs[site]['backscatter'][:, max_height_idx+1:] = np.nan


        # carry out statistics on the profiles
        for t, _ in enumerate(bsc_obs[sites[0]]['backscatter']):

            # spearman correlation
            corr_rs[t, d], corr_ps[t, d] = \
                spearmanr(bsc_obs[sites[0]]['backscatter'][t, :], bsc_obs[sites[1]]['backscatter'][t, :])

    # ==============================================================================
    # Plotting
    # ==============================================================================

    med_rs = np.nanmedian(corr_rs, axis=1)
    pct25_rs = np.nanpercentile(corr_rs, 25, axis=1)
    pct75_rs = np.nanpercentile(corr_rs, 75, axis=1)

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

    plt.savefig(savedir + sites[0] + '_' + sites[1] + '_spearmanr.png')

























    print 'END PROGRAM'