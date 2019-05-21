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
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/Utils') #aerFO
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ellUtils') # general utils
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ceilUtils') # ceil utils

import numpy as np
import iris
import matplotlib as mpl
#mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy import stats

#from ellUtils import ellUtils as eu
#import ceilUtils.ceilUtils as ceil

import ellUtils as eu
import ceilUtils as ceil

import os
from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':


    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes
    # data variable to plot
    data_var = 'backscatter'
    #data_var = 'air_temperature'
    #data_var = 'RH'
    #data_var = 'aerosol_for_visibility'

    # subsampled?
    #pcsubsample = 'full'
    #pcsubsample = '11-18_hr_range'
    pcsubsample = 'daytime'
    #pcsubsample = 'nighttime'

    # cluster type - to match the AgglomerativeClustering function and used in savename
    linkage_type = 'ward'

    # number of clusters
    n_clusters = 7

    # ------------------

    # which modelled data to read in
    #model_type = 'UKV'
    model_type = 'LM'

    # ancillary type to read, calc and plot
    ancil_type = 'murk_aer'

    # Laptop directories
    # maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    # datadir = maindir + 'data/'
    # npydatadir = datadir + 'npy/'
    # ukvdatadir = maindir + 'data/UKV/'
    # ceilDatadir = datadir + 'L1/'
    # modDatadir = datadir + model_type + '/'
    # pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    # savedir = pcsubsampledir + data_var+'/'
    # clustersavedir = savedir + 'cluster_analysis/'
    # histsavedir = clustersavedir + 'histograms/'
    # npysavedir = datadir + 'npy/PCA/'

    # MO directories
    maindir = '/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/improveNetworks/'
    datadir = '/data/jcmm1/ewarren/'
    orogdatadir = '/data/jcmm1/ewarren/ancillaries/'
    murkdatadir = '/data/jcmm1/ewarren/ancillaries/murk_aer/'+model_type+'/'
    npydatadir = datadir + 'npy/'
    metadatadir = datadir + 'metadata/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    savedir = pcsubsampledir + data_var+'/'
    clustersavedir = savedir + 'cluster_analysis/'
    histsavedir = clustersavedir + 'histograms/'
    npysavedir = datadir + 'npy/PCA/'

    # ==============================================================================
    # Read
    # ==============================================================================

    # make directory paths for the output figures
    for dir_i in [clustersavedir, histsavedir]:
        if os.path.exists(dir_i) == False:
            os.mkdir(dir_i)

    # 1. ceilometer metadata
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(metadatadir, ceilsitefile)

    # 2. Loadings
    # Read in the unrotated loadings and extract out the loadings (data), longitude and latitude (WGS84 space)
    if model_type == 'UKV':
        filename = npysavedir + data_var +'_'+pcsubsample+'_unrotLoadings.npy'
        raw = np.load(filename).flat[0]
        data = np.hstack(raw['loadings'].values())  # .shape(Xi, all_loadings)
        # just get a few heights
        #data = np.hstack([raw['loadings'][str(i)] for i in np.arange(7,20)])
        lons = raw['longitude']
        lats = raw['latitude']

    elif model_type == 'LM':
        raw=[]
        for i in np.arange(24):
            filename = npydatadir + model_type + '_' + data_var + '_' + pcsubsample + '_heightidx' + '{}'.format(i) + \
            '_unrotLoadings.npy'
            raw += [np.load(filename).flat[0]]
        # double hstack required
        data = np.hstack([np.hstack(height_i['loadings'].values()) for height_i in raw])
        lons = raw[0]['longitude']
        lats = raw[0]['latitude']



    # 3. Orography
    # manually checked and lined up against UKV data used in PCA.
    # slight mismatch due to different number precision used in both (hence why spacing is not used to reduce lower
    #   longitude limit below.
    if model_type == 'UKV':
        spacing = 0.0135 # spacing between lons and lats in rotated space
        UK_lat_constraint = iris.Constraint(grid_latitude=lambda cell: -1.2326999-spacing <= cell <= -0.7737+spacing)
        UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 361.21997 <= cell <= 361.73297+spacing)
        orog = iris.load_cube(ukvdatadir + 'UKV_orography.nc', constraint=UK_lat_constraint & UK_lon_constraint)

        # 4. murk ancillaries (shape = month, height, lat, lon)
        murk_aer = np.load(npydatadir + model_type+'_murk_ancillaries.npy').flatten()[0]

    elif model_type == 'LM':
        # orography
        spacing = 0.003 # checked
        # checked that it perfectly matches LM data extract (setup is different to UKV orog_con due to
        #    number precision issues.
        orog_con = iris.Constraint(name='surface_altitude',
                                   coord_values={
                                       'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
                                       'grid_longitude': lambda cell: 1.21 < cell < 1.732 + spacing})
        orog = iris.load_cube(orogdatadir + '20181022T2100Z_London_charts', orog_con)

        # MURK - lon and lats here are in unrotated space but do match the same domain as orog (manualy checked).
        con = iris.Constraint(coord_values={
            'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
            'grid_longitude': lambda cell: 361.21 < cell < 361.732 + spacing})
        murk_aer_all = iris.load_cube(murkdatadir + 'qrclim.murk_L70')
        murk_aer = murk_aer_all.extract(con)

    # ==============================================================================
    # Process data
    # ==============================================================================

    #a = cluster.ward_tree(data, n_clusters=5)
    #linkage_type = ''
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage_type)
    cluster.fit_predict(data)
    # +1 so first group = 1, not 0. Also it produces a 1D array, flattened Fortran style (column-wise)
    cluster_groups = cluster.labels_ +1
    group_numbers = np.unique(cluster_groups)

    # # split orography into groups based on clusters
    # # Flatten in fortran style to match the clustering code above(column wise instead of default row-wise('C'))
    # # Kruskal-Wallis test (non-parametric equivalent of ANOVA)
    # kw_s, kw_p = stats.kruskal(*orog_groups)
    if ancil_type == 'murk_aer':
        height = 0
        month_idx = 0
        ancil_data = murk_aer.data[month_idx, height, :, :]
    elif ancil_type == 'orog':
        ancil_data = orog.data


    # split the ancillary into groups based on the cluster groups
    ancil_groups = [ancil_data.flatten('F')[cluster_groups == i] for i in group_numbers]

    # Kruskal-Wallis test (non-parametric equivalent of ANOVA)
    kw_s, kw_p = stats.kruskal(*ancil_groups)

    # plot
    lat_shape = int(lats.shape[0])
    lon_shape = int(lons.shape[1])
    X_shape = int(lat_shape * lon_shape)
    aspectRatio = float(lons.shape[0]) / float(lats.shape[1])

    groups_reshape = np.transpose(
        np.vstack([cluster_groups[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

    # ==============================================================================
    # Plotting
    # ==============================================================================

    # Looks complicated... tries to overlap histograms
    # n, x, rectangles = plt.hist(orog_groups, bins=20, histtype='stepfilled',alpha=0.8)
    # colors = [i[0].get_facecolor() for i in rectangles] # copy colours from first histogram plot
    # _, _, rectangles2 = plt.hist(orog_groups, bins=20, histtype='step', color=colors, alpha=1)
    # [i[0].set_linewidth(2) for i in rectangles2]

    vmin=np.percentile(np.hstack(ancil_groups), 5)
    vmax=np.percentile(np.hstack(ancil_groups), 95)
    step=(vmax-vmin)/20.0

    fig, axs = plt.subplots(n_clusters, 1, sharex=True, sharey=True)
    # hist plot each group of orography onto each axis
    for i, ax_i in enumerate(axs):

        # ax_i.hist(ancil_groups[i], bins=np.arange(0, 220, 10),
        #              histtype='stepfilled',alpha=0.8)
        ax_i.hist(ancil_groups[i], bins=np.arange(vmin, vmax, step),
                     histtype='stepfilled',alpha=0.8)
        eu.add_at(ax_i, str(i+1), loc=5) # +1 to have groups start from 1 not 0.

    plt.suptitle('K-W test p=%3.2f' % kw_p)

    savename = data_var+'_'+pcsubsample+'_'+ancil_type+'_'+str(n_clusters)+'clusters.png'
    plt.savefig(histsavedir + savename)
    plt.close()

    #2. Plot cluster analysis groups
    #fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))  # * aspectRatio
    fig = plt.figure(figsize=(6.0, 6.0*0.7*aspectRatio))  # * aspectRatio
    ax = fig.add_subplot(111, aspect=aspectRatio)

    #cmap, norm = eu.discrete_colour_map(0, n_clusters, 1)
    cmap = plt.get_cmap('jet', n_clusters)
    norm = mpl.colors.BoundaryNorm(np.arange(1,n_clusters+2), cmap.N)

    pcmesh = ax.pcolormesh(lons, lats, groups_reshape, cmap=cmap, norm=norm)
    ax.set_xlabel(r'$Longitude$')
    ax.set_ylabel(r'$Latitude$')
    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar=plt.colorbar(pcmesh, ticks=group_numbers, cax=color_axis)
    #cbar=plt.colorbar(pcmesh, cax=color_axis)
    plt.tight_layout()

    # plot each ceilometer location
    for site, loc in ceil_metadata.iteritems():
        plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
        plt.annotate(site, (loc[0], loc[1]))

    plt.suptitle(data_var+': '+pcsubsample+'; '+linkage_type)
    savename = data_var+'_'+pcsubsample+'_CA_'+str(n_clusters)+'clusters.png'
    plt.savefig(clustersavedir + savename)
    plt.close(fig)




