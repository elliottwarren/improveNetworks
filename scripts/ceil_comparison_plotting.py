"""
Plotting to help out with the ceil_comparison script. Reads in the statistics (.npy files) and plots the data

Created by Elliott Warren Thurs 08 Nov 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import datetime as dt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
from copy import deepcopy

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon
import ceilUtils as ceil


if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # correlate in height or in time?
    corr_type = 'time'

    #main_ceil_name = 'CL31-A_IMU'
    #main_ceil_name = 'CL31-B_RGS'
    main_ceil_name = 'CL31-C_MR'
    #main_ceil_name = 'CL31-D_SWT'
    #main_ceil_name = 'CL31-E_NK'

    # ------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    savedir = maindir + 'figures/obs_intercomparison/paired/'
    npysavedir = datadir + 'npy/'

    # # 12 clear days for the 2008 KSK15S pair
    # daystrList = ['20080208', '20080209', '20080210', '20080211', '20080217', '20080506', '20080507', '20080508',
    #               '20080510', '20080511', '20080512', '20080730']

    # test partial cloud day KSK15S
    # daystrList = ['20080730']

    # 2018 clear sky days for LUMA network (missing cases between doy 142 and 190)
    daystrList = ['20180216', '20180406', '20180418', '20180419', '20180420',
       '20180505', '20180506', '20180507', '20180514', '20180515',
       '20180519', '20180520', '20180805', '20180806', '20180902']

    days_iterate = eu.dateList_to_datetime(daystrList)
    # [i.strftime('%Y%j') for i in days_iterate]

    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']

    paired_sites = deepcopy(all_sites)
    paired_sites.remove(main_ceil_name)

    # main_ceil = {main_ceil_name: ceil.site_bsc[main_ceil_name]}
    site_bsc = ceil.extract_sites(all_sites)

    # KSK15S pair
    # site_bsc = {'CL31-A_KSK15S': 40.5 - 31.4,
    #             'CL31-B_KSK15S': 40.5 - 31.4}

    # save info?
    savestr = main_ceil_name + '_statistics.npy'

    print 'main ceilometer: ' + corr_type + '_' +main_ceil_name

    # ==============================================================================
    # Read data
    # ==============================================================================

    data = np.load(npysavedir + savestr).flat[0]
    statistics = data['statistics']
    site_bsc = data['site_bsc']

    # ==============================================================================
    # plot data
    # ==============================================================================

