"""
Script to find the ideal case study days
"""

import numpy as np
from ellUtils import ellUtils as eu

import datetime as dt


if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = maindir + 'data/L1/'

    # check against these dates in 2016
    daystr = ['20160504','20160505','20160506','20160823','20160911','20161102','20161124','20161129','20161130',
             '20161204','20161205','20161227','20161229']
    # daystr=['20180903'] slow wind case from 2018, found using UKV winds.
    days_iterate = eu.dateList_to_datetime(daystr)


    # ==============================================================================
    # Read and process data
    # ==============================================================================

    file = 'Davis_BGH_2016_15min.nc'
    filepath = datadir+file

    metdata = eu.netCDF_read(filepath)
    time = metdata['time']
    wind = metdata['WS']

    wind_avg = np.empty(len(days_iterate))
    wind_avg[:] = np.nan
    wind_med = np.empty(len(days_iterate))
    wind_med[:] = np.nan

    for d, day in enumerate(days_iterate):

        print 'day = ' + day.strftime('%Y-%m-%d')

        # find time range
        _, idx_s, _ = eu.nearest(time, day)
        _, idx_e, _ = eu.nearest(time, day + dt.timedelta(days=1))
        day_range = np.arange(idx_s, idx_e+1)

        day_wind = wind[day_range]

        wind_avg[d] = np.nanmean(day_wind)
        wind_med[d] = np.median(day_wind)



    print 'END PROGRAM'