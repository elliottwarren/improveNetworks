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
import os
from sklearn.cluster import AgglomerativeClustering

#from ellUtils import ellUtils as eu
#from ceilUtils import ceilUtils as ceil

import ellUtils as eu
import ceilUtils as ceil


def read_murk_aer(datadir, model_type):

    # load data
    if model_type == 'UKV':
        # checked it matches other UKV output: slight differences in domain constraint number due to different
        #   number precision error in the saved files...
        murk_con = iris.Constraint(coord_values=
                                   {'grid_latitude': lambda cell: -1.2327999 <= cell <= -0.7738,
                                    'grid_longitude': lambda cell: 361.21997 <= cell <= 361.733})
        murk_aer = iris.load_cube(datadir + 'UKV_murk_surface.nc', murk_con)
        murk_aer = murk_aer[6, :, :]  # get July

    elif model_type == 'LM':
        spacing = 0.003  # checked
        # checked that it perfectly matches LM data extract (setup is different to UKV orog_con due to
        #    number precision issues.
        con = iris.Constraint(coord_values={
            'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
            'grid_longitude': lambda cell: 361.21 < cell < 361.732 + spacing})

        murk_aer_all = iris.load_cube(datadir + 'qrclim.murk_L70')
        murk_aer = murk_aer_all.extract(con)

    return murk_aer


def read_orography(model_type):

    """
    Load in orography from NWP models
    :param model_type:
    :return: orog (cube)

    Spacing and ranges need slightly changing as orography lat and lons are not precisely equal to those saved
    from the UKV elsewhere, because of numerical precision. Each defined orog lat and lon range, with the
    subsequent output was checked against the UKV to ensure they match.

    """

    if model_type == 'UKV':
        ukvdatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter' \
                     '/improveNetworks/data/UKV/ancillaries/'
        spacing = 0.0135 # spacing between lons and lats in rotated space
        UK_lat_constraint = iris.Constraint(grid_latitude=lambda cell: -1.2326999-spacing <= cell <= -0.7872) # +spacing
        UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 361.21997 <= cell <= 361.73297+spacing)
        orog = iris.load_cube(ukvdatadir + 'UKV_orography.nc', constraint=UK_lat_constraint & UK_lon_constraint)

    elif model_type == 'LM':
        orogdatadir = '/data/jcmm1/ewarren/ancillaries/'
        spacing = 0.003  # checked
        orog_con = iris.Constraint(name='surface_altitude',
                                   coord_values={
                                       'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
                                       'grid_longitude': lambda cell: 1.21 < cell < 1.732 + spacing})
        orog = iris.load_cube(orogdatadir + '20181022T2100Z_London_charts', orog_con)

    return orog


def load_loadings_lon_lats(npydatadir, data_var, pcsubsample, model_type):

    """
    Load in the PCA loadings, longitude and latitudes, given the model_type
    :param model_type:
    :return: data, lons, lats
    """

    # if model_type == 'UKV':
    #     filename = npysavedir + data_var + '_' + pcsubsample + '_unrotLoadings.npy'
    #     raw = np.load(filename).flat[0]
    #     data = np.hstack(raw['loadings'].values())  # .shape(Xi, all_loadings)
    #     # just get a few heights
    #     # data = np.hstack([raw['loadings'][str(i)] for i in np.arange(7,20)])
    #     lons = raw['longitude']
    #     lats = raw['latitude']
    #
    raw = []
    for i in np.arange(24):
        filename = npydatadir + model_type + '_' + data_var + '_' + pcsubsample + '_heightidx' + '{}'.format(i) + \
                   '_unrotLoadings.npy'
        raw += [np.load(filename).flat[0]['loadings']]

    # data = np.hstack([np.hstack(height_i['loadings'].values()) for height_i in raw]) # old LM
    data = np.hstack(raw)
    # use last filename in loop as long and lats are the same through all the numpy files.
    lons = np.load(filename).flat[0]['longitude']
    lats = np.load(filename).flat[0]['latitude']

    return data, lons, lats

def load_urban_loadings_lon_lats(npydatadir, data_var, pcsubsample, model_type):

    """
    Load in the PCA loadings, longitude and latitudes, given the model_type
    :param model_type:
    :return: data, lons, lats
    """

    # if model_type == 'UKV':
    #     filename = npysavedir + data_var + '_' + pcsubsample + '_unrotLoadings.npy'
    #     raw = np.load(filename).flat[0]
    #     data = np.hstack(raw['loadings'].values())  # .shape(Xi, all_loadings)
    #     # just get a few heights
    #     # data = np.hstack([raw['loadings'][str(i)] for i in np.arange(7,20)])
    #     lons = raw['longitude']
    #     lats = raw['latitude']
    #
    raw = []
    for i in range(24):

        filename = npydatadir + model_type + '_' + data_var + '_' + pcsubsample + '_heightidx' + '{}'.format(i) + \
                   '_unrotLoadings.npy'

        # extract out just the manually identified urban EOFs
        # Note: EOFs at 1461, 1605 and 1755 were mixed with other processes (inconsistent wind speeds, PC pattern to
        #   urban EOFs)
        # loadings extracted to keep the dimension e.g. [:, [3]] = shape of (n, 1) instead of (n)
        if i in range(4):
            raw += [np.load(filename).flat[0]['loadings'][:, [3]]] # extract EOF4 (idx = 3)
        elif i in range(4, 19):
            raw += [np.load(filename).flat[0]['loadings'][:, [4]]] # extract EOF5 (idx = 4)
        elif i in range(22, 24):
            raw += [np.load(filename).flat[0]['loadings'][:, [3]]]

        # if i in range(4):
        #     raw_i = np.load(filename).flat[0]['loadings'][:, 3] # extract EOF4 (idx = 3)
        #     raw += [raw_i[:, np.newaxis]]
        # elif i in range(4, 19):
        #     raw_i = np.load(filename).flat[0]['loadings'][:, 4] # extract EOF5 (idx = 4)
        #     raw += [raw_i[:, np.newaxis]]
        # elif i in range(22, 24):
        #     raw_i = np.load(filename).flat[0]['loadings'][:, 3]
        #     raw += [raw_i[:, np.newaxis]]  # extract EOF5 (idx = 4)
    # data = np.hstack([np.hstack(height_i['loadings'].values()) for height_i in raw]) # old LM
    data = np.hstack(raw)
    # use last filename in loop as long and lats are the same through all the numpy files.
    lons = np.load(filename).flat[0]['longitude']
    lats = np.load(filename).flat[0]['latitude']

    return data, lons, lats

def load_loading_explained_variance_and_height(npydatadir, data_var, pcsubsample, model_type):
    unrot_exp_var = []
    height = []
    for i in np.arange(24):
        filename = npydatadir + model_type + '_' + data_var + '_' + pcsubsample + '_heightidx' + '{}'.format(i) + \
                   '_statistics.npy'
        unrot_exp_var_i = np.sum(np.load(filename).flat[0]['unrot_exp_variance'])
        height_i = np.around(np.load(filename).flat[0]['level_height'], decimals=1)

        unrot_exp_var += [unrot_exp_var_i]
        height += [height_i]

    return np.array(unrot_exp_var), np.array(height)

def load_urban_loading_explained_variance_and_height(npydatadir, data_var, pcsubsample, model_type):
    unrot_exp_var = []
    height = []
    for i in np.arange(24):
        filename = npydatadir + model_type + '_' + data_var + '_' + pcsubsample + '_heightidx' + '{}'.format(i) + \
                   '_statistics.npy'

        # manually identified urban EOFs only
        if i in range(4):
            unrot_exp_var += [np.sum(np.load(filename).flat[0]['unrot_exp_variance'][3])]
            height += [np.around(np.load(filename).flat[0]['level_height'], decimals=1)]
        elif i in range(4, 19):
            unrot_exp_var += [np.sum(np.load(filename).flat[0]['unrot_exp_variance'][4])]
            height += [np.around(np.load(filename).flat[0]['level_height'], decimals=1)]
        elif i in range(22, 24):
            unrot_exp_var += [np.sum(np.load(filename).flat[0]['unrot_exp_variance'][3])]
            height += [np.around(np.load(filename).flat[0]['level_height'], decimals=1)]

    return np.array(unrot_exp_var), np.array(height)



def plot_cluster_analysis_groups(groups_reshape, lons, lats, orog, ceil_metadata, unrot_exp_var, n_clusters,
                                 group_numbers, data_var, pcsubsample, linkage_type, clustersavedir):

    """
    Plot the cluster analysis data with an overlaying orography contour map. Ratio and size fixed to match EOF maps.
    Return cmap so it can be used to colour the histogram later...
    :param groups_reshape:
    :param lons:
    :param lats:
    :param orog:
    :param ceil_metadata:
    :param unrot_exp_var:
    :param n_clusters:
    :param group_numbers:
    :param data_var:
    :param pcsubsample:
    :param linkage_type:
    :param clustersavedir:
    :return: cmap
    """

    aspectRatio = float(lons.shape[1]) / float(lons.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(6 * aspectRatio, 5))

    # get cmap and plot clusters and
    cmap = plt.get_cmap('jet', n_clusters)
    norm = mpl.colors.BoundaryNorm(np.arange(1, n_clusters + 2), cmap.N)
    pcmesh = ax.pcolormesh(lons, lats, groups_reshape, cmap=cmap, norm=norm)
    plt.tick_params(direction='out', top=False, right=False, labelsize=13)
    plt.setp(ax.get_xticklabels(), rotation=35, fontsize=13)

    # colorbar
    divider = make_axes_locatable(ax)
    color_axis = divider.append_axes("right", size="5%", pad=-0.1)
    cbar = plt.colorbar(pcmesh, ticks=group_numbers + 0.5, cax=color_axis)
    cbar.ax.set_yticklabels([str(i) for i in group_numbers])

    # plot orography
    levels=np.arange(30,300,30)
    cont = ax.contour(lons, lats, orog.data, cmap='OrRd', levels=levels)  # cmap='OrRd' # colors='black'
    ax.clabel(cont, fmt='%1d')  # , color='black'
    # dash the lowest orographic contour
    zc = cont.collections[0]
    plt.setp(zc, linestyle='--')

    # plot each ceilometer location
    for site, loc in ceil_metadata.iteritems():
        plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
        plt.annotate(site, (loc[0], loc[1]))

    # add % variance explained
    eu.add_at(ax, '%2.1f' % np.mean(unrot_exp_var) + ' %', loc=1, frameon=True, size=13)

    # prettify
    ax.set_xlabel('Longitude [degrees]', fontsize=13)
    ax.set_ylabel('Latitude [degrees]', fontsize=13)
    ax.set_aspect(aspectRatio, adjustable=None)
    plt.suptitle(data_var + ': ' + pcsubsample + '; ' + linkage_type)
    # plt.tight_layout()

    # save
    savename = data_var + '_' + pcsubsample + '_CA_' + str(n_clusters) + 'clusters.png'
    plt.savefig(clustersavedir + savename)
    plt.close(fig)

    return cmap

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
    #pcsubsample = 'daytime'
    pcsubsample = 'nighttime'

    # cluster type - to match the AgglomerativeClustering function and used in savename
    linkage_type = 'ward'

    # number of clusters
    n_clusters = 7

    # ------------------

    # which modelled data to read in
    #model_type = 'UKV'
    model_type = 'LM'

    # # Laptop directories
    # maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    # datadir = maindir + 'data/'
    # metadatadir = datadir
    # npydatadir = datadir + 'npy/PCA/'
    # ukvdatadir = maindir + 'data/UKV/'
    # ceilDatadir = datadir + 'L1/'
    # modDatadir = datadir + model_type + '/'
    # pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    # savedir = pcsubsampledir + data_var+'/'
    # clustersavedir = savedir + 'cluster_analysis/'
    # histsavedir = clustersavedir + 'histograms/'
    # #npysavedir = datadir + 'npy/PCA/'

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
    # clustersavedir = savedir + 'urban_cluster_analysis/'
    # histsavedir = clustersavedir + 'urban_histograms/'
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
    data, lons, lats = load_loadings_lon_lats(npydatadir, data_var, pcsubsample, model_type) # All EOFs
    # data, lons, lats = load_urban_loadings_lon_lats(npydatadir, data_var, pcsubsample, model_type) #ID'd as urban

    # 3. Orography
    # manually checked and lined up against UKV data used in PCA.
    # slight mismatch due to different number precision used in both (hence why spacing is not used to reduce lower
    #   longitude limit below.
    orog = read_orography(model_type)

    # 4. MURK aerosol
    # murk_aer = read_murk_aer(datadir, model_type)

    # EOF statistics
    unrot_exp_var, height = load_loading_explained_variance_and_height(npydatadir, data_var, pcsubsample, model_type)
    # unrot_exp_var, height = load_urban_loading_explained_variance_and_height(npydatadir, data_var, pcsubsample, model_type)

    # ==============================================================================
    # Process data
    # ==============================================================================

    #a = cluster.ward_tree(data, n_clusters=5)
    #linkage_type = ''
    print 'start clustering......'
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage_type)
    cluster.fit_predict(data)
    # +1 so first group = 1, not 0. Also it produces a 1D array, flattened Fortran style (column-wise)
    cluster_groups = cluster.labels_ +1
    group_numbers = np.unique(cluster_groups)

    # adjust numbering so largest group is 1, smallest is n

    # current idx for each group
    group_curr_idx = [np.where(cluster_groups == i)[0] for i in group_numbers]
    # get length of each group
    group_sizes = np.array([len(i) for i in group_curr_idx])
    # find what numbers they should be (1 = smallest, n_clusters = largest)
    a = np.argsort(group_sizes) # indicies to sort array
    aa = np.argsort(a) # actual rank of array
    b = n_clusters-aa # what the numbers should be for each group, if numbered by size, in ascending order (1=biggest)
    # change numbers
    # cluster_groups_new = deepcopy(cluster_groups)
    for i in np.arange(n_clusters):
        group_i_idx = group_curr_idx[i] # idx positions for this group
        cluster_groups[group_i_idx] = b[i] # change number to their new ordered number (e.g. largest = 1

    #cluster_groups.argsort()

    # # check
    for i in np.arange(1,n_clusters+1):
       print str(i) + ': ' + str(len(np.where(cluster_groups == i)[0]))

    for ancil_type in ['orog']: # ['murk_aer', 'orog']

        # # split orography into groups based on clusters
        # # Flatten in fortran style to match the clustering code above(column wise instead of default row-wise('C'))
        # # Kruskal-Wallis test (non-parametric equivalent of ANOVA)
        # kw_s, kw_p = stats.kruskal(*orog_groups)
        if ancil_type == 'orog':
            ancil_data = orog.data
        # elif ancil_type == 'murk_aer':
        #     ancil_data = murk_aer

        # split the ancillary into groups based on the cluster groups
        ancil_groups = [ancil_data.flatten('F')[cluster_groups == i] for i in group_numbers]

        # Kruskal-Wallis test (non-parametric equivalent of ANOVA)
        kw_s, kw_p = stats.kruskal(*ancil_groups)

        # ==============================================================================
        # Plotting
        # ==============================================================================

        # plot
        lat_shape = int(lats.shape[0])
        lon_shape = int(lons.shape[1])
        X_shape = int(lat_shape * lon_shape)
        aspectRatio = float(lons.shape[1]) / float(lons.shape[0])

        groups_reshape = np.transpose(
            np.vstack([cluster_groups[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

        #2. Plot cluster analysis groups
        cmap = plot_cluster_analysis_groups(groups_reshape, lons, lats, orog, ceil_metadata, unrot_exp_var, n_clusters,
                                     group_numbers, data_var, pcsubsample, linkage_type, clustersavedir)

        # Looks complicated... tries to overlap histograms
        # n, x, rectangles = plt.hist(orog_groups, bins=20, histtype='stepfilled',alpha=0.8)
        # colors = [i[0].get_facecolor() for i in rectangles] # copy colours from first histogram plot
        # _, _, rectangles2 = plt.hist(orog_groups, bins=20, histtype='step', color=colors, alpha=1)
        # [i[0].set_linewidth(2) for i in rectangles2]

        # 1. Plot ancillary split histograms

        # vmin=np.percentile(np.hstack(ancil_groups), 5)
        # vmax=np.percentile(np.hstack(ancil_groups), 95)
        # step=(vmax-vmin)/20.0
        vmin = 0.0
        vmax = 160.0
        step = 10.0

        fig, axs = plt.subplots(n_clusters, 1, sharex=True, figsize=(5,5))
        # hist plot each group of orography onto each axis
        for i, ax_i in enumerate(axs):

            # extract the right colour from cmap, for plotting. cmap() arg needs to be a fraction of 1.
            #   e.g. 5th colour out of 7 = 5.0/7.0 ~= 0.714... Also use float(i) to prevent rounding to 0 or 1.
            colour_i = cmap(float(i)/len(axs))

            freq, x, patches = ax_i.hist(ancil_groups[i], bins=np.arange(vmin, vmax, step),
                         histtype='stepfilled',alpha=0.8, color=colour_i, edgecolor='black', linewidth=1.0)
            eu.add_at(ax_i, str(i+1), loc=5, size=13) # +1 to have groups start from 1 not 0.

            ax_i.tick_params(direction='in', top=False, right=False, labelsize=13, pad=0)
            plt.setp(ax_i.get_xticklabels(), fontsize=13)
            plt.setp(ax_i.get_yticklabels(), fontsize=13)

            # find good label ticks, to prevent label overlap. Cycle through options and pick best one.
            steps = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
            for step_i in steps:
                pos_range = np.arange(0, np.max(freq)+step_i, step_i)
                if len(pos_range) <= 3:
                    break
            ax_i.yaxis.set_ticks(pos_range)

        # ax_i.tick_params(axis='y', which='major', pad=10)
        #ax_i.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        #plt.suptitle('K-W test p=%3.2f' % kw_p)

        ax_0 = eu.fig_majorAxis(fig)
        ax_0.set_ylabel('Frequency', fontsize=13)
        plt.xlabel('Height [m]', fontsize=13)
        plt.tight_layout(h_pad=0.0)

        savename = data_var+'_'+pcsubsample+'_'+ancil_type+'_'+str(n_clusters)+'clusters.png'
        plt.savefig(histsavedir + savename)
        plt.close()

        # # 4. thin vertical plot of explained variance per height
        # fig, ax = plt.subplots(1, 1, figsize=(2, 6.0 * 0.7 * aspectRatio))
        # plt.plot(unrot_exp_var, height/1000.0)
        # plt.xlabel('explained\nvariance')
        # plt.ylabel('height', rotation=90)
        # plt.tight_layout()

