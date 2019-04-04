"""
Script to carry out PCA for the backscatter over London

Created by Elliott Warren Thurs 14 Feb 2019

Uses following website as a guide for PCA:
https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
"""

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/scripts')

import numpy as np

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import math
import datetime as dt
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
#from scipy.linalg import eigh
#from scipy.linalg import inv
from scipy.stats import ttest_ind

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

        # extract out met variables with a time dimension
        met_vars = mod_data_day.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time']:
            if none_met_var in met_vars: met_vars.remove(none_met_var)

        for key in met_vars:
            mod_data_day[key] = mod_data_day[key][idx_range, ...]
        mod_data_day['time'] = mod_data_day['time'][idx_range, ...]

        # for key in ['aerosol_for_visibility', 'time', 'RH', 'backscatter']:
        #     mod_data_day[key] = mod_data_day[key][idx_range, ...]

        return mod_data_day

    # d = 0; day = days_iterate[0]
    for d, day in enumerate(days_iterate):

        print str(day)

        # read in all the data, across all hours and days, for one height
        mod_data_day = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, Z=Z, metvars=True,
                                                   height_extract_idx=height_idx)

        # extract out met variables with a time dimension
        met_vars = mod_data_day.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time']:
            if none_met_var in met_vars:
                met_vars.remove(none_met_var)

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
            # same for time
            mod_data['time'] = np.append(mod_data['time'], mod_data_day['time'], axis=0)

    # get rid of the height dimension where present (only 1 height in the data)
    mod_data = {key: np.squeeze(item) for (key, item) in mod_data.items()}

    # calculate grid centre u and v winds from grid edges
    # will be needed for London model data :(
    if model_type == 'LM':
        raise ValueError('need to interpolate u, v and w winds onto the B-grid :(')

    return mod_data

def flip_vector_sign(matrix):
    for i in range(matrix.shape[1]):
        if matrix[:, i].sum() < 0:
            matrix[:, i] *= -1.0
    return matrix

def varimax(matrix, normalize=True, max_iter=500, tolerance=1e-5, output_type='numpy.array'):

    """
    Perform varimax (orthogonal) rotation, with optional
    Kaiser normalization.

    Taken from Factor_analyzer, created by Jeremy Biggs (jbiggs@ets.org)

    Added keyword arg (output_type) to change output to a numpy.array by default
        (or keep as pandas.DataFrame)


    Parameters
    ----------
    matrix : pd.DataFrame
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
    df = matrix.copy()

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

    # The rotated matrix
    rotated_matrix = pd.DataFrame(X,
                            columns=column_names,
                            index=index_names).T

    return rotated_matrix, rotation_mtx

# rot_loadings, var_explained_ratio_unrot
def rotated_matrix_explained_and_reorder(rot_loadings, rot_eig_vals, eig_vals):

    """
    Resort the eigenvectors in decending order as the VARIMAX rotation means the largest eigenvectors may not be first.
    Also find % explained from each eigenvector

    :param rot_loadings:
    :param var_explained_ratio_unrot: ratio of the amount of variance explained from subsampled n UNROTATED matrix
        vectors to the total variance across all matrix vectors.
    :return: reordered_rot_matrix: reordered rot_matrix
    return: reorder_idx: index for reordering the data from most explained variance, to least explained, of the rotated
        matrix.
    """

    # calculate R2 from eq 12.4 of Wilks 2011
    # different to below calculation using Lee. Rotated eigenvectors are scaled by sqrt(eigenvalue) (Wilks  p529, Table 12.3)
    #    therefore loadings **2 should be new eigenvalues...

    # Actual variance explained of orig dataset by each eigenvector (e.g. sum() = 79...)
    # var_explained_rot.sum() ~= var_explained_unrot.sum() (just spread out values)
    # shape is (loading_i, X_i) where X_i is spatial location, therefore axis to sum along is 0
    # as ||loading|| = sqrt(eigenvalue), then below calculates that ||loading||**2 = eigenvalue

    # percentage each eigenvector explains of the current set of eigenvectors (sum = 1.0)
    # this is only a part of the orig though so each %value here will be a little too high!
    # perc_current_rot_set = (var_explained_rot / var_explained_rot.sum())
    # Hence, use approach below this!

    # # actual percentage each eigenvector explains of the ORIGINAL total set of eigenvectors (sum != 1.0).
    # # Same calc as above, but scale the %values DOWN based on (total var % explained in the unrotated subset
    # #   e.g. var_explained_ratio_unrot.sum() = 0.98 of original !KEY unrotated ratio)
    # #   i.e. if original subset had 0.98 of variance, scale these percentages down using 0.98 as the coefficient
    # subset_total_var = var_explained_rot.sum()  # total variance in current subset
    # remaining_ratio_var_original = var_explained_ratio_unrot.sum()  # how much of the variance is left from original e.g. 0.98
    # var_explained_ratio_rot = (var_explained_rot / subset_total_var) * remaining_ratio_var_original
    # perc_var_explained_ratio_rot = var_explained_ratio_rot * 100.0
    # # var_explained_ratio_rot = (var_explained_rot / var_explained_rot.sum()) * var_explained_ratio_unrot.sum()

    # fraction of total variance explained by ALL the ORIGINAL eigenvalues (matches SPSS)
    var_explained = np.array([i/np.sum(eig_vals) for i in rot_eig_vals])

    # get and apply reorder idx
    # reorder_idx = perc_var_explained_ratio_rot.argsort()[::-1]
    reorder_idx = var_explained.argsort()[::-1]

    reordered_matrix = rot_loadings[:, reorder_idx]
    perc_var_explained_ratio_rot = var_explained[reorder_idx] * 100.0

    return reordered_matrix, perc_var_explained_ratio_rot, reorder_idx

def pinv(a, rcond=1e-15):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    a : (M, N) array_like
      Matrix to be pseudo-inverted.
    rcond : float
      Cutoff for small singular values.
      Singular values smaller (in modulus) than
      `rcond` * largest_singular_value (again, in modulus)
      are set to zero.

    Returns
    -------
    B : (N, M) ndarray
      The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so
      is `B`.

    Raises
    ------
    LinAlgError
      If the SVD computation does not converge.

    Notes
    -----
    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
    value decomposition of A, then
    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
    of A's so-called singular values, (followed, typically, by
    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
    consisting of the reciprocals of A's singular values
    (again, followed by zeros). [1]_

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pp. 139-142.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+``:

    #>>> a = np.random.randn(9, 6)
    #>>> B = np.linalg.pinv(a)
    #>>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    #>>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    """
    from numpy.linalg.linalg import _makearray, _assertNoEmpty2d, svd, maximum, dot, transpose, multiply, newaxis

    a, wrap = _makearray(a)
    _assertNoEmpty2d(a)
    a = a.conjugate()
    u, s, vt = svd(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    #cutoff = rcond*maximum.reduce(s)
    #count=0
    # for i in range(min(n, m)):
    #     if s[i] > cutoff:
    #         s[i] = 1./s[i]
    #     else:
    #         s[i] = 0.
    #         count +=1
    for i in range(min(n, m)):
        s[i] = 1./s[i]

    res = dot(transpose(vt), multiply(s[:, newaxis], transpose(u)))
    #print('tot s='+str(len(s))+'; count='+str(count))
    return wrap(res)

# Plotting
def plot_EOFs_height_i(eig_vecs_keep, ceil_metadata, lons, lats, eofsavedir,
                       days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var):

    """Plot all EOFs for this height - save in eofsavedir (should be a subdirectory based on subsampled input data)"""

    for eof_idx in np.arange(eig_vecs_keep.shape[0]):

        # etract out eof_i
        eof_i = eig_vecs_keep[eof_idx, :]

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

    for m_idx in np.arange(matrix.shape[1]):

        # etract out eof_i
        m_i = matrix[:, m_idx]

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

# reordered_rot_pcScores, days_iterate, rotPCscoresdir, 'rotPC'
def line_plot_PCs_vs_days_iterate(scores, days_iterate, pcsavedir, pctype):

    """Plot the EOFs paired PCs, against the days_iterate (x axis)"""

    for pc_idx in np.arange(scores.shape[-1]):

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # extract out this PC
        pc_i = scores[:, pc_idx]  # / np.mean(pcs[:, pc_idx])

        # plot each day separately so it doesn't spike between days
        #days =
        # plt.plot(pc_norm, label='PC' + str(pc_idx + 1))

        # set tick locations - 1 for each day
        step = pc_i.shape[0] / len(days_iterate)
        ticks = np.arange(0, pc_i.shape[0], step)  # +(step/2.0) to centre the label over the day

        # plot each day separately
        for dplt in ticks[:-1]:
            x_range = np.arange(dplt, dplt+step)
            plt.plot(x_range, pc_i[x_range], label='PC' + str(pc_idx + 1), color='blue')

        plt.xticks(ticks)
        # get the days in a nice string format, again to plot 1 for each day
        labels = [i.strftime('%Y/%m/%d') for i in days_iterate]
        ax.set_xticklabels(labels)

        for label in ax.get_xticklabels():
            label.set_rotation(90)

        plt.subplots_adjust(bottom=0.3)

        plt.ylabel('score')
        plt.xlabel('date')
        plt.suptitle(pctype + str(pc_idx + 1) + '; height=' + height_i_label)
        plt.savefig(pcsavedir + pctype + str(pc_idx + 1) + '; height=' + height_i_label + '.png')

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
    # data_var = 'backscatter'
    data_var = 'air_temperature'
    # data_var = 'RH'

    height_range = np.arange(0, 30) # only first set of heights
    lon_range = np.arange(30, 65) # only London area (right hand side of larger domain -35:

    # save?
    numpy_save = True

    # subsampled?
    #pcsubsample = 'full'
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
    pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    savedir = pcsubsampledir + data_var+'/'
    eofsavedir = savedir + 'EOFs/'
    rotEOFsavedir = savedir + 'rotEOFs/'
    rotPCscoresdir = savedir + 'rotPCs/'
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
    #a = [i.strftime('%Y%j') for i in days_iterate]
    #'\' \''.join(a)

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
    # pcsubsampledir, then savedir needs to be checked first as they are parent dirs
    for dir_i in [pcsubsampledir, savedir,
                  eofsavedir, pcsavedir, expvarsavedir, rotEOFsavedir, rotPCscoresdir]:
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

    height_idx = 12

    #for height_idx in [height_idx]:
    for height_idx in [18]: #np.arange(10,30): # np.arange(26,30): # max 30 -> ~ 3.1km

        # 4/6/18 seems to be missing hr 24 (hr 23 gets removed by accident as a result...) - first day in days_iterate
        mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx, hr_range=[11,18])
        # mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx)

        # extract out the height for this level
        height_i = mod_data['level_height']
        height_i_label = '%.1fm' % mod_data['level_height'] # add m on the end

        print 'height_idx = '+str(height_idx)
        print 'height_i = '+str(height_i)

        # rotate lon and lat into normal geodetic grid (WGS84)
        lons, lats = rotate_lon_lat_2D(mod_data['longitude'][lon_range], mod_data['latitude'], model_type)

        # extract out the data (just over London for the UKV)
        if data_var == 'backscatter':
            data = np.log10(mod_data[data_var][:, :, lon_range])
        elif data_var == 'air_temperature':
            data = mod_data[data_var][:, :, lon_range]
        else:
            raise ValueError('need to specify how to extract data if not backsatter')

        # ==============================================================================
        # PCA
        # ==============================================================================

        # get shape dimensions
        lat_shape = int(data.shape[1])
        lon_shape = int(data.shape[-1])
        X_shape = int(lat_shape * lon_shape)

        # reshape data so location is a single dimension (time, lat, lon) -> (time, X), where X is location
        # Stacked latitudinally, as we are cutting into the longitude for each loop of i
        data = np.hstack(([data[:, :, i] for i in np.arange(lon_shape)]))

        # 1. ------ numpy.eig or scipy.eigh (the latter is apparently more numerically stable wrt square matricies)
        # (695L, 35L, 35L) - NOTE! reshape here is different to above! just use the syntatically messier version
        #   as it is actually clearer that the reshaping is correct!
        # data2 = np.reshape(data, (695, 1225))
        # mean center each column

        # data prep
        M = np.mean(data.T, axis=1)
        data_m = data - M
        data_m_norm = data_m / np.std(data.T, axis=1)
        # data_m_norm = data_m / np.std(data_m)
        # cov_data = np.cov(data_m.T)
        # cov_data_normed = np.cov(data_m_norm.T)
        # cov_data = np.cov(data_m.T)
        cov_data = np.cov(data_m_norm.T) # mathces SPSS this way...
        corr_data = np.corrcoef(data_m.T)
        corr_norm_data = np.corrcoef(data_m_norm.T)

        # WHAT GETS TAKEN FORWARD
        # U, S, V = np.linalg.svd(corr_data)
        U, S, V = np.linalg.svd(cov_data)
        eig_vals = S # first eigenvalue at height_idx=12, height_i=645m is 18695 if covariance matrix used...
        eig_vecs = flip_vector_sign(V.T)

        # lee's alt version for explained var
        # matches another example online specifically for covarance matrix (near the bottom:
        #  https://towardsdatascience.com/let-us-understand-the-correlation-matrix-and-covariance-matrix-d42e6b643c22
        var_explained_unrot = eig_vals * 100 / np.sum(eig_vals)

        # keep first n components that have eig_vals >= 1
        bool_components_keep = (eig_vals >= 1.0)
        n_components_keep = sum(bool_components_keep)
        n_components_keep = 5

        # calculate loadings for the kept PCs
        # same loading values as if the pca package was used
        eig_vecs_keep = eig_vecs[:, :n_components_keep]
        eig_vals_keep = eig_vals[:n_components_keep]
        loadings = eig_vecs[:, :n_components_keep] * np.sqrt(eig_vals[:n_components_keep]) #.shape(Xi, PCs)

        # make vector directions all positive (sum of vectors > 0) for consistency.
        # As eigenvector * constant = another eigenvector, mmultiply vectors with -1 if sum() < 0.

        # rotate the loadings to spread out the eplained variance between all the kept vectors
        #loadings = np.array([eig_vecs[:, i] * np.sqrt(eig_vals[i]) for i in range(len(eig_vals))]).T # loadings
        loadings_pd = pd.DataFrame(loadings)
        rot_loadings, rot_matrix = varimax(loadings_pd)
        rot_loadings = np.array(rot_loadings)
        rot_loadings = flip_vector_sign(rot_loadings)
        # Rotated eigenvalues
        rot_eig_vals = np.sum(rot_loadings ** 2, axis=0)

        # Weighted average (does not work) # large values, very spiky, highly correlated.
        rot_scores = np.dot(rot_loadings.T, data_m.T) # Wilks, 2011
        rot_scores2 = np.dot(data_m, rot_loadings)  # pca package - in base.py:  transform(self, X)

        # # Get rotated PC scores (do not behave quite the same as unrotated PCs)
        # # Transpose to get the columns being different PCs
        # # Just doesn't seem to work...
        # rot_pcScores = (rot_loadings.T).dot(data_m.T).T#  data_m.dot(rot_loadings) # eq 12.2 (Wilks 2011)
        # rot_pcScores_i = rot_loadings[:,2].dot(data_m_norm.T)
        # a=(rot_pcScores_i - np.mean(rot_pcScores_i))/np.std(rot_pcScores_i)
        # rot_eig_vals = np.sum(rot_loadings**2, axis=0)
        # #rot_pcScores.T / np.sqrt(rot_eig_vals)

        # SPSS rescaled loadings... why/how they use this?!!?! \Delta_m,R = ([diag \Sigma]^-1/2 )*\Delta_m
        #   \Delta_m = mth rotated loading; \Sigma = covariance matrix (page 409 of SPSS statistics algorithms manual)
        # Matches 'Rotated Component Matrix' Rescaled component (right table)
        rescaled_loadings=np.vstack([(1.0/np.sqrt(np.diag(cov_data)))*rot_loadings[:,i] for i in range(rot_loadings.shape[-1])]).T

        # attempt at what was in SPSS manual - doesn't work
        #a = np.linalg.inv(rescaled_loadings.T.dot(rescaled_loadings))
        #W=rescaled_loadings.dot(a)
        # a = 1.0/(rot_loadings.T.dot(rot_loadings))
        # W=rot_loadings.dot(a)
        #a=rescaled_loadings*(1.0/(rot_eig_vals)) # tried unrotated version... normal and rescaled - also doesn't work

        # # deal with ill-conditioned matrix by reducing corr a little bit
        # # this eq. to calculate the PC scores is HIGHLY sensitive to the upper cut off used here. Too high and it is
        # #   unstable - too low and it no longer matches the data well enough and is highly noisy.
        # zscore_corr[zscore_corr >= 0.99] = 0.99
        # for i in range(zscore_corr.shape[0]):
        #     zscore_corr[i,i] = 1.0

        # Matches 'Regression' approach in Lee (explained in Field 2005)
        # Closely matches test SPSS output on air temp; hours 11-18; height idx=12; height i = 645; using correlation matrix
        #   shape and relative magnitudes are very close, and given we want to subsample original dataset in time correctly,
        #   this is fine. Output is also weakly correlated, as expected.
        # Justification from Field (p786) The resulting factor score matrix [done this way] represents the relationship
        #   between each variable and each factor, adjusting for the original relationships between pairs of variables.
        #   This matrix represents a purer measure of the \i[unique]\i relationship between pairs of variables and factors.
        #   The above aproach is the 'regression' approach using the correlation matrix is better than the weighted average.

        # NOTE! probably why the normal way of just e_m^T x x' (a weighted average) doesn't work, because the relationships between the
        #   factor and original variables are not unique! Also if the variance of some variables is far higher than others
        #   the scores will be be unfairly calculated, as some variables will lead to large contributions

        # Need to condition the correlation matrix better to calculate the score coefficients better.
        # V is given already transposed here, as normally equation is A = U x S x V^T
        # S is just the diagonal elements. Need to use np.diag(S) to recreate the square diagonal matrix
        # Lee
        df_data = pd.DataFrame(data)
        #df_data += np.random.rand(224, 1225)*1e1 # attempt to reduce correlation slightly to help computation but limit impact on results
        zscore = (df_data - df_data.mean() ) / df_data.std()
        zscore_cov = np.cov(data_m_norm.T)
        zscore_corr = np.array(zscore.corr()) # symetric matrix
        # Pseudo-inverse (uses SVD to calculate corr_inv without a dense matrix inversion)
        # Trial and error used to get 1e-4, output is very close to SPSS (most importantly, the peaks and trough
        #   locations and relative magnitudes). 1e-4 and 1e-3 best..., 1e5 is ~ok low as possible but retain stability
        # Set the limit to be so small that it isn't actually used (saftey meausre).
        corr_inv = pinv(zscore_corr, rcond=1e-55)
        #corr_inv = np.linalg.inv(zscore_corr) # doesn't invert the matrix well! use above SVD approach
        pcScoreCoeff = corr_inv.dot(rot_loadings)
        rot_pcScores=np.array(zscore).dot(pcScoreCoeff) # rot_pcScores_keep
        #plt.plot(rot_pcScores[:, 0])

        cov_inv = pinv(cov_data, rcond=1e-55)
        #corr_inv = np.linalg.inv(zscore_corr) # doesn't invert the matrix well! use above SVD approach
        pcScoreCoeff = cov_inv.dot(rot_loadings)
        rot_pcScores=np.array(zscore).dot(pcScoreCoeff) # rot_pcScores_keep

        # Order ot leading loadings may have changed, so ensure order is still the most explained variance to the least.
        reordered_rot_loadings, perc_var_explained_ratio_rot, reorder_idx = \
            rotated_matrix_explained_and_reorder(rot_loadings, rot_eig_vals, eig_vals)

        # reorder PC scores to match loadings
        reordered_rot_pcScores = rot_pcScores[:, reorder_idx]

        # check correlation between PCs should be != 0 but not really high (near -1 or 1)
        # for i in range(rot_pcScores.shape[-1]):
        #     r, p = pearsonr(rot_pcScores[:, 0], rot_pcScores[:, i])
        #     print 'i='+str(i)+'; r='+str(r)
        # plt.plot(rot_pcScores[:,0])

        # student-t test (upper/lower 10 percentile PC extracted and compared)
        met_vars = mod_data.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time', data_var]:
            if none_met_var in met_vars: met_vars.remove(none_met_var)

        # approach - no need now if the SVD approach works ok.
        # 1. count number of s elements turned to 0
        # 1. check if correlation matrix really is symmetric (i,j/j,i =1) need a tolerance?
        #   1.1 enforce symmetry using a tolerence
        # 2. invert corr_matrix using SVD, then check against identity
        # 3. check inversion diff = (I - A-1A) and (I - AA-1)
        #   3.1 check norm of difference (sqrt.sum(squares(diag elements diff))) / (sqrt.sum(squares(diag elements corr matrix)))

        # ==============================================================================
        # Calculate and save statistics
        # ==============================================================================

        t_test={}
        met_avg={}
        met_med={}
        met_std={}
        met_iqr={}
        # extract data (10% upper and lower limits)
        perc_limit = 10 # [%]
        for i in range(reordered_rot_pcScores.shape[-1]):
            pc_i = reordered_rot_pcScores[:, i]
            pc_i_name = 'rotPC'+str(i+1) # do not start at 0...
            t_test[pc_i_name] = {}

            for var in met_vars:
                # find upper and lower percentiles
                up_perc = np.percentile(reordered_rot_pcScores[:,i], 100-perc_limit)
                lower_perc = np.percentile(reordered_rot_pcScores[:, i], perc_limit)
                # idx positions for all data above or below each percentile
                top_scores_idx = np.where(reordered_rot_pcScores[:, i] >= up_perc)
                lower_scores_idx = np.where(reordered_rot_pcScores[:, i] <= lower_perc)
                #student t-test on original data
                x = mod_data[var][top_scores_idx, :, :].flatten()
                y = mod_data[var][lower_scores_idx, :, :].flatten()

                # equal_var=False means ttst_ind is the Welch's t-test (not student t-test)
                #   https://en.wikipedia.org/wiki/Welch%27s_t-test
                # This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values
                #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
                t_test[pc_i_name][var]=ttest_ind(x, y, equal_var=False)
                r = []
                p = []
                for i in range(mod_data[var].shape[1]):
                    for j in range(mod_data[var].shape[2]):
                        r_i, p_i =  stats.pearsonr(mod_data[var][:,i,j], reordered_rot_pcScores[:,0])
                        r += [r_i]
                        p += [p_i]


        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------

        # aspect ratio for the map plots
        aspectRatio = float(mod_data['longitude'].shape[0]) / float(mod_data['latitude'].shape[0])

        # 1. colormesh() plot the EOFs for this height
        # unrotated
        plot_EOFs_height_i(eig_vecs_keep, ceil_metadata, lons, lats, eofsavedir,
                           days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var)

        # rotated EOFs
        plot_spatial_output_height_i(reordered_rot_loadings, ceil_metadata, lons, lats, rotEOFsavedir,
                                     days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var,
                                     'rotEOFs')

        # 2. Explain variance vs EOF number
        #line_plot_exp_var_vs_EOF(perc_explained, height_i_label, days_iterate, expvarsavedir)

        # 3. PC timeseries
        #line_plot_PCs_vs_days_iterate(pc_scores, days_iterate, pcsavedir, 'PC')

        # rot PC
        line_plot_PCs_vs_days_iterate(reordered_rot_pcScores, days_iterate, rotPCscoresdir, 'rotPC')



# ----------------------------------------------------
# Trash code with making the eigenvectors, values, loadings and PCs
# ----------------------------------------------------

# # 2. ------ package
# # method using a package
# # create PCA instance
# pca = PCA(5)
# # # fit on data (creates covariance matrix etc, inside the package)
# pca.fit(data)
# # # access values and vectors
# # var_explained_unrot = pca.explained_variance_ # variance explained by each eigenvector
# var_explained_ratio_unrot = pca.explained_variance_ratio_
# # perc_var_explained_ratio_unrot = pca.explained_variance_ratio_*100.0 # actual percentage explained by each principal comopnent and EOF
# # # transform data
# pc_scores = pca.transform(data) # PC scores (composed of vectors) # eq 12.1 Wilk 2011
# # # reshape components back to 2D grid and plot
# eig_vecs_pca = pca.components_.T # eigenvectors (composed of vectors)
# # # make eigenvectors positive
# # eig_vecs = flip_vector_sign(eig_vecs) # will have no effect if package is used
# eig_vals_pca = pca.explained_variance_ # eigenvalues (composed of scalers)
# # # use .T to keep the same shape as eig_vecs
# # pca_cov = pca.get_covariance()
# #
# # # Note pca.get_covariance() and np.cov() give different covariance matricies! Therefore use pca func if using
# # #   it's outputs.
# # # no need to sort afterward, already in order with highest eig_val to lowest
# # #    also, no need to flip signs as vectors are already in the positive directions
