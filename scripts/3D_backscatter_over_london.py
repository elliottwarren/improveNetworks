"""
Script to create and plot 3D modelled backscatter over London. Saves the 3D field in a numpy array for post processing
statistics.

Created by Elliott Warren Fri 23 Nov 2018
"""

import numpy as np
from scipy.stats import spearmanr
import datetime as dt
from copy import deepcopy

import ellUtils as eu
import ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # min and max height to cut off backscatter (avoice clouds above BL, make sure all ceils start fairly from bottom)
    min_height = 0.0
    max_height = 2000.0

    # save?
    numpy_save = True

    # ------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    savedir = maindir + 'figures/obs_intercomparison/paired/'
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    npysavedir = datadir + 'npy/'

    # test case from unused paper 2 UKV data
    daystr = ['20190902']
    day = eu.dateList_to_datetime(daystr)[0]
    #[i.strftime('%Y%j') for i in days_iterate]

    # save name
    savestr = day.strftime('%Y%m%d') + '_3Dbackscatter.npy'

    # ==============================================================================
    # Read data
    # ==============================================================================

    print 'day = ' + day.strftime('%Y-%m-%d')

    # calculate the 3D backscatter field across London
    mod_data = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=version,
                                        allvars=True)





























    print 'END PROGRAM'
