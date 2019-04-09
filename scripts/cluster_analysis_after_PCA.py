"""
Script to carry out cluster analysis following the principal component analysis.

Created by Elliott Warren Tues 09 Mar 2019

Clustering in python:
https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

Scaling the variables can act to weight the variables
One way to assign a weight to a variable is by changing its scale. The trick works for the clustering algorithms you
mention, viz. k-means, weighted-average linkage and average-linkage.

Kaufman, Leonard, and Peter J. Rousseeuw. "Finding groups in data: An introduction to cluster analysis." (2005) - page 11:

The choice of measurement units gives rise to relative weights of the variables. Expressing a variable in smaller units
will lead to a larger range for that variable, which will then have a large effect on the resulting structure. On the
other hand, by standardizing one attempts to give all variables an equal weight, in the hope of achieving objectivity.
As such, it may be used by a practitioner who possesses no prior knowledge. However, it may well be that some variables
are intrinsically more important than others in a particular application, and then the assignment of weights should be
based on subject-matter knowledge (see, e.g., Abrahamowicz, 1985).

On the other hand, there have been attempts to devise clustering techniques that are independent of the scale of the
variables (Friedman and Rubin, 1967). The proposal of Hardy and Rasson (1982) is to search for a partition that
minimizes the total volume of the convex hulls of the clusters. In principle such a method is invariant with respect
to linear transformations of the data, but unfortunately no algorithm exists for its implementation (except for an
approximation that is restricted to two dimensions). Therefore, the dilemma of standardization appears unavoidable at
present and the programs described in this book leave the choice up to the user
"""

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')

import numpy as np

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os
import datetime as dt
import pandas as pd
from scipy import stats
from copy import deepcopy
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon


if __name__ == '__main__':


    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # data variable to plot
    data_var = 'backscatter'
    #data_var = 'air_temperature'
    #data_var = 'RH'

    # subsampled?
    #pcsubsample = 'full'
    pcsubsample = '11-18_hr_range'

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    #res = FOcon.model_resolution[model_type]
    #Z='21'

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    savedir = pcsubsampledir + data_var+'/'
    expvarsavedir = savedir + 'explained_variance/'
    rotexpvarsavedir = savedir + 'rot_explained_variance/'
    barsavedir = savedir + 'barcharts/'
    corrmatsavedir = savedir + 'corrMatrix/'
    npysavedir = datadir + 'npy/PCA/'

    # # intial test case
    # # daystr = ['20180406']
    # # daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # # current set (missing 20180215 and 20181101) # 08-03
    # daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507',
    #           '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
    #           '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
    #           '20180901','20180902','20180903','20181007','20181010','20181020','20181023']
    # days_iterate = eu.dateList_to_datetime(daystr)
    # # a = [i.strftime('%Y%j') for i in days_iterate]
    # # '\' \''.join(a)

    # # import all site names and heights
    # all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']
    # site_bsc = ceil.extract_sites(all_sites, height_type='agl')


    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # read in the unrotated loadings
    filename = npysavedir + 'backscatter_11-18_hr_range_unrotLoadings_test.npy'
    raw = np.load(filename).flat[0]
    data = np.hstack(raw['loadings'].values())  # .shape(Xi, all_loadings)
    lons = raw['longitude']
    lats = raw['latitude']

    #a = cluster.ward_tree(data, n_clusters=5)
    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    cluster_groups = cluster.labels_

    # plot
    lat_shape = int(lats.shape[0])
    lon_shape = int(lons.shape[1])
    X_shape = int(lat_shape * lon_shape)
    aspectRatio = float(lons.shape[0]) / float(lats.shape[0])

    groups_reshape = np.transpose(
        np.vstack([cluster_groups[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

    fig, ax = plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
    plt.pcolormesh(lons, lats, groups_reshape)
    plt.colorbar()
    ax.set_xlabel(r'$Longitude$')
    ax.set_ylabel(r'$Latitude$')

    # highlight highest value across EOF
    # eof_i_max_idx = np.where(eof_i_reshape == np.max(eof_i_reshape))
    # plt.scatter(lons[eof_i_max_idx][0], lats[eof_i_max_idx][0], facecolors='none', edgecolors='black')
    # plt.annotate('max', (lons[eof_i_max_idx][0], lats[eof_i_max_idx][0]))

    # plot each ceilometer location
    for site, loc in ceil_metadata.iteritems():
        # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
        plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
        plt.annotate(site, (loc[0], loc[1]))

    plt.suptitle(matrix_type + str(m_idx + 1) + '; height=' + height_i_label + str(
        len(days_iterate)) + ' cases')
    savename = height_i_label + '_' + matrix_type + str(m_idx + 1) + '_' + data_var + '.png'
    plt.savefig(matrixsavedir + savename)
    plt.close(fig)




