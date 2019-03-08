"""
Script to carry out PCA for the backscatter over London

Created by Elliott Warren Thurs 14 Feb 2019

Uses following website as a guide for PCA:
https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import math
import datetime as dt
from sklearn.decomposition import PCA

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

def read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx):
    """
    Read and compile mod_data across all the days in (days_iterate), through time, for a single height.
    :param days_iterate:
    :param modDatadir:
    :param model_type:
    :param Z:
    :param height_idx:
    :return:
    """

    # d = 0; day = days_iterate[0]
    for d, day in enumerate(days_iterate):

        print str(day)

        # read in all the data, across all hours and days, for one height
        mod_data_day = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, Z=Z, allvars=False,
                                                   height_extract_idx=height_idx)

        if day == days_iterate[0]:
            mod_data = mod_data_day

            # remove the last hour of time (hr 24, or hr 0 of the next day) as it may overlap across different cases
            # [:-1, ...] works with 1D arrays and up
            for key in ['aerosol_for_visibility', 'time', 'RH', 'backscatter']:
                mod_data[key] = mod_data[key][:-1, ...]

        # works if time is the first dimension in all arrays (1D, 4D or otherwise)
        for var in ['time', 'aerosol_for_visibility', 'RH', 'backscatter']:
            mod_data[var] = np.append(mod_data[var], mod_data_day[var][:-1, ...], axis=0)

    # get rid of the height dimension where present (only 1 height in the data)
    mod_data = {key: np.squeeze(item) for (key, item) in mod_data.items()}

    return mod_data

if __name__ == '__main__':

    # https://www.researchgate.net/profile/Jafar_Al-Badarneh/post/Does_anyboby_can_help_me_to_download_
    # a_PPT/attachment/5a2bc215b53d2f0bba42f44a/AS%3A569610665955328%401512817173845/download/Visualizing
    # +Data_EOFs.+Part+I_+Python_Matplotlib_Basemap+Part+II_+Empirical+orthogonal+func%3Bons.pdf
    # slide 25

    # SLP[time,lat,lon] # current shape
    # slp = np.reshape(slp, (time,nlat*nlon)) # flatten to two dimensions
    # np.transpose(slp) or slp = slp.T # time across and location down
    # try eigenvalue decomposi;on method over SVD as SVD is way more memory intensive

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # data variable to plot
    data_var = 'backscatter'
    # data_var = 'RH'

    height_range = np.arange(0,30) # only first set of heights
    lon_range = np.arange(30, 65) # only London area (right hand side of larger domain -35:

    # save?
    numpy_save = True

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]
    Z='21'

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    variogramsavedir = maindir + 'figures/model_runs/variograms/'
    twodrangedir = maindir + 'figures/model_runs/2D_range/'
    twodsilledir = maindir + 'figures/model_runs/2D_sill/'
    twodRangeCompositeDir = twodrangedir + 'composite/'
    npysavedir = datadir + 'npy/'

    # intial test case
    # daystr = ['20180406']
    # daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # current set (missing 20180215 and 20181101) # 08-03
    daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507',
              '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
              '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
              '20180901','20180902','20180903','20181007','20181010','20181020','20181023']
    days_iterate = eu.dateList_to_datetime(daystr)

    # save name
    # savestr = day.strftime('%Y%m%d') + '_3Dbackscatter.npy'

    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']

    # get height information for all the sites
    site_bsc = ceil.extract_sites(all_sites, height_type='agl')

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # print data variable to screen
    print data_var

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    height_idx = 0

    # 4/6/18 seems to be missing hr 24 (hr 23 gets removed by accident as a result...)
    mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx)

    # extract out the data (just over London for the UKV)
    data = np.log10(mod_data['backscatter'][:, 1:, lon_range])

    # get shape dimensions
    lat_shape = int(data.shape[1])
    lon_shape = int(data.shape[-1])
    X_shape = int(lat_shape * lon_shape)

    # reshape data so location is a single dimension (time, lat, lon) -> (time, X), where X is location
    # Stacked latitudinally, as we are cutting into the longitude for each loop of i
    data = np.hstack(([data[:, :, i] for i in np.arange(lon_shape)]))

    # (695L, 35L, 35L) - NOTE! reshape here is different to above! just use the syntatically messier version
    #   as it is actually clearer that the reshaping is correct!
    # data2 = np.reshape(data, (695, 1225))

    # mean center each column
    M = np.mean(data, axis=0)
    data_norm = data - M

    # calculate covariance matrix of centered matrix
    V = np.cov(data_norm.T)

    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(V)

    # project data
    P = vectors.T.dot(data_norm.T)
    print(P.T)




    A = data
    # new method using a package
    pca = PCA(2)
    # fit on data
    pca.fit(A)
    # access values and vectors
    print(pca.components_) # vectors
    print(pca.explained_variance_) # values
    # pca.explained_variance_ratio_ # actual percentage explained by each principal comopnent and EOF
    # transform data
    B = pca.transform(A)
    #print(B)

    # reshape components back to 2D grid and plot
    pca1 = pca.components_[0,:]

    # NEED to check if this is rotated back correctly (it might need tranposing)
    # as it was stacked row/latitude wise above (row1, row2, row3)
    # transpose turns shape into (lat, lon) (seems a little odd but needs to be plotted that
    #   way by plt.pcolormesh() to get the axis right...
    back = np.transpose(np.vstack([pca1[n:n+lat_shape] for n in np.arange(0, X_shape, lat_shape)])) # 1225

    plt.pcolormesh(back)