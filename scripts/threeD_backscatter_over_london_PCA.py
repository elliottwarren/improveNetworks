"""
Script to carry out PCA for the backscatter over London

Created by Elliott Warren Thurs 14 Feb 2019

Uses following website as a guide for PCA:
https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
"""

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import math
import datetime as dt
from sklearn.decomposition import PCA
import pandas as pd

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

from threeD_backscatter_over_london import rotate_lon_lat_2D

def read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx, **kwargs):
    """
    Read and compile mod_data across all the days in (days_iterate), through time, for a single height.
    :param days_iterate:
    :param modDatadir:
    :param model_type:
    :param Z:
    :param height_idx:

    kwargs
    :param hr_range (2 element list): start and end hour of time range to extract from data

    :return:
    """

    def extract_hour_range(mod_data_day, day, **kwargs):

        """Extract out data for this hour range"""

        start = day + dt.timedelta(hours=kwargs['hr_range'][0])
        end = day + dt.timedelta(hours=kwargs['hr_range'][1])
        # time_range = eu.date_range(start, end, 1, 'hours')

        idx_start = eu.binary_search_nearest(mod_data_day['time'], start)
        idx_end = eu.binary_search_nearest(mod_data_day['time'], end)
        idx_range = np.arange(idx_start, idx_end + 1)

        for key in ['aerosol_for_visibility', 'time', 'RH', 'backscatter']:
            mod_data_day[key] = mod_data_day[key][idx_range, ...]

        return mod_data_day

    # d = 0; day = days_iterate[0]
    for d, day in enumerate(days_iterate):

        print str(day)

        # read in all the data, across all hours and days, for one height
        mod_data_day = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, Z=Z, metvars=True,
                                                   height_extract_idx=height_idx)

        # extract out variables with a time dimension
        met_vars = mod_data_day.keys()
        for none_met_var in ['time', 'longitude', 'latitude', 'level_height']:
            if none_met_var in met_vars: met_vars.remove(none_met_var)

        # remove the last hour of time (hr 24, or hr 0 of the next day) as it may overlap across different cases
        # [:-1, ...] works with 1D arrays and up
        if mod_data_day['time'][-1].hour == 0: # if its actually hour 0 of the next day....
            # for key in ['aerosol_for_visibility', 'time', 'RH', 'backscatter']:
            for key in met_vars:
                mod_data_day[key] = mod_data_day[key][:-1, ...]

        # extract out a time range from data?
        if 'hr_range' in kwargs:
            mod_data_day = extract_hour_range(mod_data_day, day, **kwargs)

        if day == days_iterate[0]:
            mod_data = mod_data_day

        else:
            # append day's data onto (mod_data) which is all the data combined
            # works if time is the first dimension in all arrays (1D, 4D or otherwise)
            for var in met_vars:
                mod_data[var] = np.append(mod_data[var], mod_data_day[var], axis=0)

    # get rid of the height dimension where present (only 1 height in the data)
    mod_data = {key: np.squeeze(item) for (key, item) in mod_data.items()}

    # calculate grid centre u and v winds from grid edges
    # will be needed for London model data :(
    if model_type == 'LM':
        raise ValueError('need to interpolate u, v and w winds onto the B-grid :(')

    return mod_data

def varimax(loadings, normalize=True, max_iter=500, tolerance=1e-5, output_type='numpy.array'):

    """
    Perform varimax (orthogonal) rotation, with optional
    Kaiser normalization.

    Taken from Factor_analyzer, created by Jeremy Biggs (jbiggs@ets.org)

    Added keyword arg (output_type) to change output to a numpy.array by default
        (or keep as pandas.DataFrame)


    Parameters
    ----------
    loadings : pd.DataFrame
        The loadings matrix to rotate.
    normalize : bool, optional
        Whether to perform Kaiser normalization
        and de-normalization prior to and following
        rotation.
        Defaults to True.
    max_iter : int, optional
        Maximum number of iterations.
        Defaults to 500.
    tolerance : float, optional
        The tolerance for convergence.
        Defaults to 1e-5.
    output_type : bool, optional
        return the rotated matrix as a numpy array or pandas dataframe
        Defaults to numpy.array

    Return
    ------
    loadings : pd.DataFrame
        The loadings matrix
        (n_cols, n_factors)
    rotation_mtx : np.array
        The rotation matrix
        (n_factors, n_factors)
    """
    df = loadings.copy()

    # since we're transposing the matrix
    # later, we want to reverse the column
    # names and index names from the original
    # factor loading matrix at this point
    column_names = df.index.values
    index_names = df.columns.values

    n_rows, n_cols = df.shape

    if n_cols < 2:
        return df

    X = df.values

    # normalize the loadings matrix
    # using sqrt of the sum of squares (Kaiser)
    if normalize:
        normalized_mtx = df.apply(lambda x: np.sqrt(sum(x**2)),
                                  axis=1).values

        X = (X.T / normalized_mtx).T

    # initialize the rotation matrix
    # to N x N identity matrix
    rotation_mtx = np.eye(n_cols)

    d = 0
    for _ in range(max_iter):

        old_d = d

        # take inner product of loading matrix
        # and rotation matrix
        basis = np.dot(X, rotation_mtx)

        # transform data for singular value decomposition
        transformed = np.dot(X.T, basis**3 - (1.0 / n_rows) *
                             np.dot(basis, np.diag(np.diag(np.dot(basis.T, basis)))))

        # perform SVD on
        # the transformed matrix
        U, S, V = np.linalg.svd(transformed)

        # take inner product of U and V, and sum of S
        rotation_mtx = np.dot(U, V)
        d = np.sum(S)

        # check convergence
        if old_d != 0 and d / old_d < 1 + tolerance:
            break

    # take inner product of loading matrix
    # and rotation matrix
    X = np.dot(X, rotation_mtx)

    # de-normalize the data
    if normalize:
        X = X.T * normalized_mtx

    else:
        X = X.T

    # if output_type == 'numpy.array':
    #     loadings = X.T
    # elif output_type == 'pandas.DataFrame':
    #     # convert loadings matrix to data frame
    #     loadings = pd.DataFrame(X,
    #                             columns=column_names,
    #                             index=index_names).T
    # else:
    #     raise ValueError('output_type keyword is not recognised as an option. Choose numpy.array or pandas.DataFrame')

    loadings = pd.DataFrame(X,
                            columns=column_names,
                            index=index_names).T

    return loadings, rotation_mtx

# Plotting
def plot_EOFs_height_i(pca, ceil_metadata, lons, lats, eofsavedir,
                       days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var):

    """Plot all EOFs for this height - save in eofsavedir (should be a subdirectory based on subsampled input data)"""

    for eof_idx in np.arange(pca.components_.shape[0]):

        # etract out eof_i
        eof_i = pca.components_[eof_idx, :]

        # NEED to check if this is rotated back correctly (it might need tranposing)
        # as it was stacked row/latitude wise above (row1, row2, row3)
        # transpose turns shape into (lat, lon) (seems a little odd but needs to be plotted that
        #   way by plt.pcolormesh() to get the axis right...
        eof_i_reshape = np.transpose(
            np.vstack([eof_i[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

        fig, ax = plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
        plt.pcolormesh(lons, lats, eof_i_reshape)
        plt.colorbar()
        ax.set_xlabel(r'$Longitude$')
        ax.set_ylabel(r'$Latitude$')

        # highlight highest value across EOF
        eof_i_max_idx = np.where(eof_i_reshape == np.max(eof_i_reshape))
        plt.scatter(lons[eof_i_max_idx][0], lats[eof_i_max_idx][0], facecolors='none', edgecolors='black')
        plt.annotate('max', (lons[eof_i_max_idx][0], lats[eof_i_max_idx][0]))

        # plot each ceilometer location
        for site, loc in ceil_metadata.iteritems():
            # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
            plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
            plt.annotate(site, (loc[0], loc[1]))

        plt.suptitle('EOF' + str(eof_idx + 1) + '; height=' + height_i_label + str(
            len(days_iterate)) + ' cases')
        savename = height_i_label +'_EOF' + str(eof_idx + 1) + '_' + data_var + '.png'
        plt.savefig(eofsavedir + savename)
        plt.close(fig)

    return

def plot_spatial_output_height_i(matrix, ceil_metadata, lons, lats, matrixsavedir,
                       days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var, matrix_type):

    """Plot all EOFs for this height - save in eofsavedir (should be a subdirectory based on subsampled input data)"""

    for m_idx in np.arange(matrix[0]):

        # etract out eof_i
        m_i = matrix[m_idx, :]

        # NEED to check if this is rotated back correctly (it might need tranposing)
        # as it was stacked row/latitude wise above (row1, row2, row3)
        # transpose turns shape into (lat, lon) (seems a little odd but needs to be plotted that
        #   way by plt.pcolormesh() to get the axis right...
        eof_i_reshape = np.transpose(
            np.vstack([m_i[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

        fig, ax = plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
        plt.pcolormesh(lons, lats, eof_i_reshape)
        plt.colorbar()
        ax.set_xlabel(r'$Longitude$')
        ax.set_ylabel(r'$Latitude$')

        # highlight highest value across EOF
        eof_i_max_idx = np.where(eof_i_reshape == np.max(eof_i_reshape))
        plt.scatter(lons[eof_i_max_idx][0], lats[eof_i_max_idx][0], facecolors='none', edgecolors='black')
        plt.annotate('max', (lons[eof_i_max_idx][0], lats[eof_i_max_idx][0]))

        # plot each ceilometer location
        for site, loc in ceil_metadata.iteritems():
            # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
            plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
            plt.annotate(site, (loc[0], loc[1]))

        plt.suptitle(matrix_type + str(m_idx + 1) + '; height=' + height_i_label + str(
            len(days_iterate)) + ' cases')
        savename = height_i_label +'_'+matrix_type + str(m_idx + 1) + '_' + data_var + '.png'
        plt.savefig(matrixsavedir + savename)
        plt.close(fig)

    return

def line_plot_exp_var_vs_EOF(perc_explained, height_i_label, days_iterate, expvarsavedir):
    """Plot the accumulated explained variance across the kept EOFs"""

    perc_explained_cumsum = np.cumsum(perc_explained)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.plot(np.arange(1, len(perc_explained_cumsum) + 1), perc_explained_cumsum * 100)
    plt.xticks(np.arange(1, len(perc_explained_cumsum) + 1, 1.0))
    fig.suptitle('height=' + height_i_label + '; ' + str(len(days_iterate)) + ' cases')
    plt.ylabel('explained variance [%]')
    plt.xlabel('EOF')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(expvarsavedir + 'exp_var_' + height_i_label + '.png')
    plt.close(fig)

    return

def line_plot_PCs_vs_days_iterate(pc_scores, days_iterate, pcsavedir):

    """Plot the EOFs paired PCs, against the days_iterate (x axis)"""

    for pc_idx in np.arange(pc_scores.shape[-1]):

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # normalise PC so it is clear, then plot
        # pc_norm = (pcs[:, pc_idx] - np.mean(pcs[:, pc_idx])/np.std(pcs[:, pc_idx]))
        pc_norm = pc_scores[:, pc_idx]  # / np.mean(pcs[:, pc_idx])
        plt.plot(pc_norm, label='PC' + str(pc_idx + 1))

        # set tick locations - 1 for each day
        step = pc_norm.shape[0] / len(days_iterate)
        ticks = np.arange(0, pc_norm.shape[0], step)  # +(step/2.0) to centre the label over the day
        plt.xticks(ticks)
        # get the days in a nice string format, again to plot 1 for each day
        labels = [i.strftime('%Y/%m/%d') for i in days_iterate]
        ax.set_xticklabels(labels)

        for label in ax.get_xticklabels():
            label.set_rotation(90)

        plt.subplots_adjust(bottom=0.3)

        plt.ylabel('PC')
        plt.xlabel('date')
        plt.suptitle('PC' + str(pc_idx + 1) + '; height=' + height_i_label)
        plt.savefig(pcsavedir + 'PC' + str(pc_idx + 1) + '; height=' + height_i_label + '.png')

        plt.close(fig)

    return

if __name__ == '__main__':

    # https://www.researchgate.net/profile/Jafar_Al-Badarneh/post/Does_anyboby_can_help_me_to_download_
    # a_PPT/attachment/5a2bc215b53d2f0bba42f44a/AS%3A569610665955328%401512817173845/download/Visualizing
    # +Data_EOFs.+Part+I_+Python_Matplotlib_Basemap+Part+II_+Empirical+orthogonal+func%3Bons.pdf
    # slide 25

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

    # subsampled?
    # pcasubdir = 'full'
    pcsubsample = '11-18_hr_range'

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
    savedir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    eofsavedir = savedir + 'EOFs/'
    pcsavedir = savedir + 'PCs/'
    expvarsavedir = savedir + 'explained_variance/'
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

    # make directory paths for the output figures
    # savedir needs to be checked first as it is parent dir
    for dir_i in [savedir, eofsavedir, pcsavedir, expvarsavedir]:
        if os.path.exists(dir_i) == False:
            os.mkdir(dir_i)

    # make small text file to make sure the figures used the right subsample technique
    with open(savedir + 'subsample.txt', 'w') as file_check:
        file_check.write(pcsubsample)

    # print data variable to screen
    print data_var

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    height_idx = 0

    #for height_idx in [height_idx]:
    for height_idx in np.arange(0,30): # np.arange(26,30): # max 30 -> ~ 3.1km

        # 4/6/18 seems to be missing hr 24 (hr 23 gets removed by accident as a result...)
        mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx, hr_range=[11,18])

        # extract out the height for this level
        height_i = mod_data['level_height']
        height_i_label = '%.1fm' % mod_data['level_height'] # add m on the end

        # rotate lon and lat into normal geodetic grid (WGS84)
        lons, lats = rotate_lon_lat_2D(mod_data['longitude'][lon_range], mod_data['latitude'], model_type)

        # extract out the data (just over London for the UKV)
        if data_var == 'backscatter':
            data = np.log10(mod_data[data_var][:, :, lon_range])
        else:
            raise ValueError('need to specify how to extract data if not backsatter')

        # get shape dimensions
        lat_shape = int(data.shape[1])
        lon_shape = int(data.shape[-1])
        X_shape = int(lat_shape * lon_shape)

        # reshape data so location is a single dimension (time, lat, lon) -> (time, X), where X is location
        # Stacked latitudinally, as we are cutting into the longitude for each loop of i
        data = np.hstack(([data[:, :, i] for i in np.arange(lon_shape)]))

        # # (695L, 35L, 35L) - NOTE! reshape here is different to above! just use the syntatically messier version
        # #   as it is actually clearer that the reshaping is correct!
        # # data2 = np.reshape(data, (695, 1225))
        # # mean center each column
        # M = np.mean(data, axis=0)
        # data_norm = data - M
        # # calculate covariance matrix of centered matrix
        # V = np.cov(data_norm.T)
        # # eigendecomposition of covariance matrix
        # values, vectors = np.linalg.eig(V)
        # # project data
        # P = vectors.T.dot(data_norm.T)
        # print(P.T)


        # method using a package
        # create PCA instance
        pca = PCA(20)
        # fit on data (creates covariance matrix etc, inside the package)
        pca.fit(data)

        # access values and vectors

        var_explained_unrot = pca.explained_variance_ # variance explained by each eigenvector
        var_explained_ratio_unrot = pca.explained_variance_ratio_
        perc_var_explained_ratio_unrot = pca.explained_variance_ratio_*100.0 # actual percentage explained by each principal comopnent and EOF
        # transform data
        pc_scores = pca.transform(data) # PC scores (composed of vectors)
        # reshape components back to 2D grid and plot
        eig_vecs = pca.components_.T # eigenvectors (composed of vectors)
        eig_vals = pca.explained_variance_ # eigenvalues (composed of scalers)
        # use .T to keep the same shape as eig_vecs
        loadings = np.array([eig_vecs[:, i] * np.sqrt(eig_vals[i]) for i in range(len(eig_vals))]).T # loadings

        # rotate the loadings to spread out the eplained variance between all the kept vectors
        loadings_pd = pd.DataFrame(loadings)
        rot_loadings, rot_matrix = varimax(loadings_pd, output_type='numpy.array')
        rot_loadings = np.array(rot_loadings)
        # reorder rotated loadings as the order might have changed after the rotation

        def rotated_loading_reorder(rot_loadings):
            # Taken from Lee
            # adapted to get ratio

            # Actual variance explained of orig dataset by each eigenvector (e.g. sum() = 79...)
            # var_explained_rot.sum() ~= var_explained_unrot.sum() (just spread out values)
            # shape is (loading_i, X_i) where X_i is spatial location, therefore axis to sum along is 0
            var_explained_rot = np.sum(rot_loadings ** 2.0, axis=0) #

            # percentage each eigenvector explains of the current set of eigenvectors (sum = 1.0)
            # this is only a part of the orig though so each %value here will be a little too high!
            # perc_current_rot_set = (var_explained_rot / var_explained_rot.sum())
            # Hence, use approach below this!

            # actual percentage each eigenvector explains of the ORIGINAL total set of eigenvectors (sum != 1.0).
            # Same calc as above, but scale the %values DOWN based on (total var % explained in the unrotated subset
            #   e.g. var_explained_ratio_unrot.sum() = 0.98 of original !KEY unrotated ratio)
            #   i.e. if original subset had 0.98 of variance, scale these percentages down using 0.98 as the coefficient
            subset_total_var = var_explained_rot.sum() # total variance in current subset
            remaining_ratio_var_original = var_explained_ratio_unrot.sum() # how much of the variance is left from original e.g. 0.98
            var_explained_ratio_rot = (var_explained_rot / subset_total_var) * remaining_ratio_var_original
            perc_var_explained_ratio_rot = var_explained_ratio_rot * 100.0
            # var_explained_ratio_rot = (var_explained_rot / var_explained_rot.sum()) * var_explained_ratio_unrot.sum()


            #ToDo! Now reorder the data like in Lee! The positions have changed!


        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------

        # aspect ratio for the map plots
        aspectRatio = float(mod_data['longitude'].shape[0]) / float(mod_data['latitude'].shape[0])

        # 1. colormesh() plot the EOFs for this height
        plot_EOFs_height_i(pca, ceil_metadata, lons, lats, eofsavedir,
                           days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var)

        plot_spatial_output_height_i(matrix, ceil_metadata, lons, lats, matrixsavedir,
                                     days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var,
                                     matrix_type)

        plot_loadings_height_i(pca, ceil_metadata, lons, lats, eofsavedir,
                           days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var, rotated=True)

        # 2. Explain variance vs EOF number
        line_plot_exp_var_vs_EOF(perc_explained, height_i_label, days_iterate, expvarsavedir)

        # 3. PC timeseries
        line_plot_PCs_vs_days_iterate(pc_scores, days_iterate, pcsavedir)




