"""
Script to carry out PCA for the backscatter over London

Created by Elliott Warren Thurs 14 Feb 2019

Uses following website as a guide for PCA:
https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
"""

# workaround while PYTHONPATH plays up on MO machine
import sys
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/Utils') #aerFO
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ellUtils') # general utils
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ceilUtils') # ceil utils

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import datetime as dt
import pandas as pd
from scipy import stats
from copy import deepcopy
import sunrise
import iris

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

# import ellUtils as eu
# import ceilUtils as ceil
# import FOUtils as FO
# import FOconstants as FOcon
# # from Utils import FOconstants as FOcon

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

    def extract_daytime(mod_data_day, day):

        """
        Extract out data for day time (+2 after sunrise, -2 before sunset)
        :param mod_data_day:
        :param day:
        :return:
        """

        # get sunrise for this day (central London)
        s = sunrise.sun(lat=51.5, long=-0.1)
        d_sunrise = s.sunrise(when=day)  # (hour, minute, second)
        d_sunset = s.sunset(when=day)

        # create start and end times to be 2 hours after and before sunrise and sunset respectively
        start = day + dt.timedelta(hours=d_sunrise.hour+2, minutes=d_sunrise.minute, seconds = d_sunrise.second)
        end = day + dt.timedelta(hours=d_sunset.hour-2, minutes=d_sunset.minute, seconds = d_sunset.second)

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

        return mod_data_day

    def extract_nighttime(mod_data_day, day):

        """
        Extract out data for either night time (-2 before sunrise, +2 after sunset)
        :param mod_data_day:
        :param day:
        :return:
        """

        # get sunrise for this day (central London)
        s = sunrise.sun(lat=51.5, long=-0.1)
        d_sunrise = s.sunrise(when=day)  # (hour, minute, second)
        d_sunset = s.sunset(when=day)

        # create start and end times to be 2 hours after and before sunrise and sunset respectively
        pre_morn = day + dt.timedelta(hours=d_sunrise.hour-2, minutes=d_sunrise.minute, seconds=d_sunrise.second)
        start_even = day + dt.timedelta(hours=d_sunset.hour+2, minutes=d_sunset.minute, seconds=d_sunset.second)

        idx_morn = eu.binary_search_nearest(mod_data_day['time'], pre_morn)
        idx_even = eu.binary_search_nearest(mod_data_day['time'], start_even)
        # idx positions for (midnight - pre_morning) and (post_evening - following midnight), concatonated
        #   gives the nighttime idx positions
        # data inclues midnight of following day (a 25th hour), so leave that out and just go to 24
        idx_range = np.hstack([range(idx_morn+1), range(idx_even, 24)])

        # extract out met variables with a time dimension
        met_vars = mod_data_day.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time']:
            if none_met_var in met_vars: met_vars.remove(none_met_var)

        for key in met_vars:
            mod_data_day[key] = mod_data_day[key][idx_range, ...]
        mod_data_day['time'] = mod_data_day['time'][idx_range, ...]

        return mod_data_day

    # d = 0; day = days_iterate[0]
    
    for d, day in enumerate(days_iterate):

        #statement= 'extracting ' + str(d) + ' ' + day.strftime('%Y%m%d')
        #os.system('echo '+statement)

        # read in all the data, across all hours and days, for one height
        mod_data_day = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, Z=Z, metvars=True,
                                                   height_extract_idx=height_idx)

        # extract out met variables with a time dimension
        # met_vars = mod_data_day.keys()
        # for none_met_var in ['longitude', 'latitude', 'level_height', 'time']:
        #     if none_met_var in met_vars:
        #         met_vars.remove(none_met_var)

        met_vars = mod_data_day.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time']:
            if none_met_var in met_vars:
                met_vars.remove(none_met_var)

        # remove the last hour of time (hr 24, or hr 0 of the next day) as it may overlap across different cases
        # [:-1, ...] works with 1D arrays and up
        if mod_data_day['time'][-1].hour == 0: # if its actually hour 0 of the next day....
            # for key in ['aerosol_for_visibility', 'time', 'RH', 'backscatter']:
            for key in met_vars:
                #print key
                mod_data_day[key] = mod_data_day[key][:-1, ...]

        # extract out a time range from data?
        # daytime and nighttime extract fixed to +/2 hours around sunrise and sunset
        if 'hr_range' in kwargs:
            mod_data_day = extract_hour_range(mod_data_day, day, **kwargs)
        elif 'subsample' in kwargs:
            if kwargs['subsample'] == 'daytime':
                mod_data_day = extract_daytime(mod_data_day, day)
            elif kwargs['subsample'] == 'nighttime':
                mod_data_day = extract_nighttime(mod_data_day, day)
        else:
            if d == 0:
                print 'Extracting data for the full day'

        if day == days_iterate[0]:
            mod_data = mod_data_day

        else:
            # append day's data onto (mod_data) which is all the data combined
            # works if time is the first dimension in all arrays (1D, 4D or otherwise)
            for var in met_vars + ['time']:
                mod_data[var] = np.append(mod_data[var], mod_data_day[var], axis=0)
            # # same for time
            # mod_data['time'] = np.append(mod_data['time'], mod_data_day['time'], axis=0)

    # get rid of the height dimension where present (only 1 height in the data)
    mod_data = {key: np.squeeze(item) for (key, item) in mod_data.items()}

    ### Remove times with low aerosol (75th perc. < 1 microgram)
    # # remove all instances of time where the aerosol for this height was too low (below 1 microgram kg-1)
    # # checked slicing with t_idx and lon_range at the same time works. Output array is transposed to (lon, lat) but
    # #   as a median/percentile is used from it, order of elements doesn't matter.
    # if model_type == 'UKV':
    #     idx = np.array([t_idx for t_idx in range(len(mod_data['time']))
    #                     if np.percentile(mod_data['aerosol_for_visibility'][t_idx, :, kwargs['lon_range']], 75) < 1.0])
    #
    #                     # if np.median(mod_data['aerosol_for_visibility'][t_idx, :, :]) < 1.0])
    #     for key in met_vars + ['time']:
    #         mod_data[key] = np.delete(mod_data[key], idx, axis=0)
    # else:
    #     raise ValueError('Need to change lon_range for the (if aerosol < 1microgram) testing')

    print 'data read in...'

    # calculate grid centre u and v winds from grid edges
    # will be needed for London model data :(
    #if model_type == 'LM':
        
        #lin_interpolate_u_v_to_bgrid(mod_data)
    #     raise ValueError('need to interpolate u, v and w winds onto the B-grid :(')

    return mod_data

def rotate_lon_lat_2D(longitude, latitude, model_type, corner_locs=False):
    """
    Create 2D array of lon and lats in rotated space (WGS84)

    :param mod_data:
    :keyword corner_locs (bool): return lon and lat corner locations so (0,0) is bottom left corner
            of bottom left box
    :return: rotlon2D, rotlat2D (2D arrays): 2D arrays of rotate lon and latitudes (rot. to WSG84)
    """

    import cartopy.crs as ccrs
    import iris

    # test corners with a plot if necessary
    def test_create_corners(rotlat2D, rotlon2D, corner_lats, corner_lons):
        """ quick plotting to see whether the functions within the script have created the corner lat and lon properly.
        Original and extrap/interp data together."""

        # copy and pasted out of main functions. Put back if this function is to be used.
        # # 3D array with a slice with nothing but 0s...
        # normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
        # lons_orig = normalgrid[:, :, 0]
        # lats_orig = normalgrid[:, :, 1]
        # # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...
        #
        # # 3D array with a slice with nothing but 0s...
        # normalgrid = ll.transform_points(rotpole, corner_lons, corner_lats)
        # lons = normalgrid[:, :, 0]
        # lats = normalgrid[:, :, 1]

        plt.figure()
        plt.scatter(rotlat2D, rotlon2D, color='red')
        plt.scatter(corner_lats, corner_lons, color='blue')
        # plt.scatter(a, d, color='green')
        # plt.scatter(lats, lons, color='red')
        # plt.scatter(lats_orig, lons_orig, color='blue')
        plt.figure()
        plt.pcolormesh(corner_lats, vmin=np.nanmin(corner_lats), vmax=np.nanmax(corner_lats))
        plt.colorbar()

        return

    # duplicate the single column array but number of rows in latitude
    # rotlat2D needs to be transposed as rows need to be latitude, and columns longitude.
    # np.transpose() does transpose in the correct direction:
    # compare mod_all_data['latitude'][n] with rotlat2D[n,:]

    rotlon2D = np.array([longitude] * latitude.shape[0])
    rotlat2D = np.transpose(np.array([latitude] * longitude.shape[0]))

    # Get model transformation object (ll) to unrotate the model data later
    # Have checked rotation is appropriate for all models below
    if (model_type == 'UKV') | (model_type == '55m') | (model_type == 'LM'):
        rotpole = (iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
            6371229.0))).as_cartopy_crs()  # rot grid
        rotpole2 = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        ll = ccrs.Geodetic()  # to normal grid

    # 3D array with a slice with nothing but 0s...
    normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
    lons = normalgrid[:, :, 0]
    lats = normalgrid[:, :, 1]
    # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...

    # test to see if the functions above have made the corner lat and lons properly
    # test_create_corners(rotlat2D, rotlon2D, corner_lats, corner_lons)

    return lons, lats

def flip_vector_sign(matrix):
    """If ||eig_vector|| = -1, make it +1 instead"""

    # eof_t = np.transpose(
    #     np.vstack([matrix[:,3][n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))
    # plt.pcolormesh(eof_t)

    for i in range(matrix.shape[1]):  # columnwise
        if matrix[:, i].sum() < 0:
            print 'EOF'+str(i+1) + ' is being flipped'
            matrix[:, i] *= -1.0
    return matrix

def pinv(a):
    """
    Copied and adapted from numpy.pinv (numpy version 1.11.3)

    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    a : (M, N) array_like
      Matrix to be pseudo-inverted.

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
    for i in range(min(n, m)):
        s[i] = 1./s[i]

    res = dot(transpose(vt), multiply(s[:, newaxis], transpose(u)))
    return wrap(res)

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
    var_explained = np.array([i / np.sum(eig_vals) for i in rot_eig_vals])

    # get and apply reorder idx
    # reorder_idx = perc_var_explained_ratio_rot.argsort()[::-1]
    reorder_idx = var_explained.argsort()[::-1]

    reordered_matrix = rot_loadings[:, reorder_idx]
    perc_var_explained_ratio_rot = var_explained[reorder_idx] * 100.0

    return reordered_matrix, perc_var_explained_ratio_rot, reorder_idx

def pca_analysis(data_m, cov_data, cov_inv):

    """
    Carry out Principal Component Analysis (PCA) (no rotation here... that comes later)
    Requires data inputs and covariance matricies to be calculated before entering the function
    :param data_m:
    :param cov_data:
    :param cov_inv:
    :return: pcScores: the new variables (time series)
    :return loadings: here defined as -> eigenvectors * sqrt(eigenvalues)
    """

    # WHAT GETS TAKEN FORWARD
    # U, S, V = np.linalg.svd(corr_data)
    U, S, V = np.linalg.svd(cov_data)
    eig_vals = S
    eig_vecs = flip_vector_sign(V.T)  # make sure sign of each eig vec is positive, i.e. ||eig_vec_i|| = 1 (not -1)

    # lee's alt version for explained var
    # matches another example online specifically for covarance matrix (near the bottom:
    #  https://towardsdatascience.com/let-us-understand-the-correlation-matrix-and-covariance-matrix-d42e6b643c22
    var_explained_unrot = eig_vals * 100 / np.sum(eig_vals)

    # keep first n components that have eig_vals >= 1 if correlation matrix used
    # bool_components_keep = (eig_vals >= 1.0)
    # bool_components_keep = (eig_vals >= np.mean(eig_vals)) # Kaisers rule eq 12.13, page 540 of Wilks 2011
    bool_components_keep = (var_explained_unrot >= 1.0)  # keep EOF/PC pairs that explain more than 1% of underlying data
    n_components_keep = sum(bool_components_keep)
    # n_components_keep = 5 choose to fix number of EOF and PC pairs

    # calculate loadings for the kept PCs
    # same loading values as if the pca package was used
    eig_vecs_keep = eig_vecs[:, :n_components_keep]
    eig_vals_keep = eig_vals[:n_components_keep]
    perc_var_explained_unrot_keep = var_explained_unrot[:n_components_keep]
    loadings = eig_vecs[:, :n_components_keep] * np.sqrt(eig_vals[:n_components_keep])  # .shape(Xi, PCs)

    # get pc scores for unrotated EOFs
    pcScoreCoeff = cov_inv.dot(loadings)
    pcScores = data_m.dot(pcScoreCoeff)  # rot_pcScores_keep

    return eig_vecs_keep, eig_vals, pcScores, loadings, perc_var_explained_unrot_keep

def rotate_loadings_and_calc_scores(loadings, cov_data, eig_vals):
    """
    Rotate the loadings, calculate PC scores for the new loadings and variance explained of new rotated loadings.
    Rotation is VARIMAX.
    :param loadings: ordered unrotated loadings
    :param cov_data:  covariance data matrix of mean centered data
    :param eig_vals: eigenvalues of ALL the originally calculated, unrotated loadings
    :return: reordered_rot_loadings: rotated, reordered loadings so first loadings explain the most variance
    :return: reordered_rot_pcScores: rotated, reordered PC scores
    :return: perc_var_explained_ratio_rot: percentage explained by each set of loadings.
    """

    def varimax(matrix, normalize=True, max_iter=500, tolerance=1e-5):

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
            normalized_mtx = df.apply(lambda x: np.sqrt(sum(x ** 2)),
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
            transformed = np.dot(X.T, basis ** 3 - (1.0 / n_rows) *
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

    # rotate loadings
    loadings_pd = pd.DataFrame(loadings)
    rot_loadings, rot_matrix = varimax(loadings_pd)
    rot_loadings = np.array(rot_loadings)
    rot_loadings = flip_vector_sign(rot_loadings)  # make sure ||rot_loadings|| = +ve

    # for i in range(a.shape[1]):  # columnwise
    #     if a[:, i].sum() < 0:
    #         print i

    # Caculate rotated eigenvalues from rotated loadings
    # as ||loadings_i|| = sqrt(values), ||loadings_i||^2  = values
    rot_eig_vals = np.sum(rot_loadings ** 2, axis=0)

    # Matches 'Regression' approach in Lee (explained in Field 2005)
    # Closely matches test SPSS output on air temp; hours 11-18; height idx=12; height i = 645; using correlation
    #   or covariance matrix. Output is weakly correlated, as expected.
    # Justification from Field (p786) The resulting factor score matrix [done this way] represents the relationship
    #   between each variable and each factor, adjusting for the original relationships between pairs of variables.
    #   This matrix represents a purer measure of the \i[unique]\i relationship between pairs of variables and factors.
    #   The above aproach is the 'regression' approach using the correlation matrix is better than the weighted average.
    # pseudo-inverse covariance matrix as it is ill-conditioned. Use SVD approach.
    cov_pinv = pinv(cov_data)
    # corr_inv = np.linalg.inv(zscore_corr) # doesn't invert the matrix well! use above SVD approach
    pcScoreCoeff = cov_pinv.dot(rot_loadings)
    rot_pcScores = data_m.dot(pcScoreCoeff)  # rot_pcScores_keep
    # plt.plot(rot_pcScores[:, 0]) check it looks sensible

    # Order of the leading loadings may have changed, so ensure order is still the most explained
    #   variance to the least.
    reordered_rot_loadings, perc_var_explained_ratio_rot, reorder_idx = \
        rotated_matrix_explained_and_reorder(rot_loadings, rot_eig_vals, eig_vals)

    # reorder PC scores to match loadings
    reordered_rot_pcScores = rot_pcScores[:, reorder_idx]

    return reordered_rot_loadings, reordered_rot_pcScores, perc_var_explained_ratio_rot

# statistics

def pcScore_subsample_statistics(reordered_rot_pcScores, mod_data, met_vars, ceil_metadata, height_i_label,
                                 topmeddir, botmeddir):

    """
    Calculate statistics for each variable, for each PC, for this height
    Also calculate and export statistics for box plotting later (without needing to export the entire distribution)
    :param reordered_rot_pcScores: used to find index positions to subsample original data
    :param mod_data: original data
    :param met_vars: variables to extract and calclulate statistics for
    :return: statistics_height (dict([var][pc_i_name])): statistics for this height
    """

    def plot_medians(med, lons, lats, ceil_metadata, pc_i_name, height_i_label, var, days_iterate, meddir,
                     med_type, wpvalue):

        """
        Plot the top or bottom median
        :param med:
        :param lons:
        :param lats:
        :param ceil_metadata:
        :param pc_i_name:
        :param height_i_label:
        :param var:
        :param days_iterate:
        :param meddir:
        :param med_type:
        :param wpvalue:
        :return:
        """

        def forceAspect(ax, aspect=1):
            im = ax.get_images()
            extent = im[0].get_extent()
            ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

        vmin = np.percentile(med, 2)
        vmax = np.percentile(med, 98)

        aspectRatio = float(lons.shape[0]) / float(lats.shape[1])
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5*0.8))
        # fig, ax = plt.subplots(1, 1, figsize=(4.5 * 1.0, 4.5))
        mesh=plt.pcolormesh(lons, lats, med, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
        #plt.colorbar()
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(mesh, cax=color_axis)
        ax.set_xlabel(r'$Longitude$')
        ax.set_ylabel(r'$Latitude$')

        #ax.axes.set_aspect(1)#(aspectRatio)


        # plot each ceilometer location
        for site, loc in ceil_metadata.iteritems():
            # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
            plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
            plt.annotate(site, (loc[0], loc[1]))
        plt.tight_layout()
        plt.subplots_adjust(top=0.8, bottom=0.2)
        #forceAspect(ax,aspect=aspectRatio)

        plt.suptitle(med_type + '; ' + pc_i_name + '; height=' + height_i_label + '; ' + var + ';\n ' + str(
            len(days_iterate)) + ' cases; p=%5.2f' % wpvalue)
        savename = height_i_label + '_' + med_type + '_' + var + pc_i_name + '_' + '.png'
        plt.savefig(meddir + savename)
        plt.close(fig)

        return

        # derive and store the boxplot statistics for each met. variable in boxplot_stats

    boxplot_stats_top = {}
    boxplot_stats_bot = {}
    statistics_height = {}
    # extract data (5% upper and lower limits)
    perc_limit = 10  # [%]

    # for each meteorological variable: carry out the statistics
    for var in met_vars:

        print var

        # add dictionary for this var
        # statistics[height_idx_str][var] = {}
        boxplot_stats_top[var] = []  # list so it keeps its order (PC1, PC2...)
        boxplot_stats_bot[var] = []
        statistics_height[var] = {}  # dict

        # for each PC...
        for i in range(reordered_rot_pcScores.shape[-1]):

            pc_i = reordered_rot_pcScores[:, i]
            pc_i_name = 'rotPC' + str(i + 1)  # do not start at 0...

            # create stats_i to store all the statistics in, and be later copied over to the full statistics dict
            statistics_i = {}

            # 2.0 Data prep and extraction
            # find upper and lower percentiles
            up_perc = np.percentile(pc_i, 100 - perc_limit)
            lower_perc = np.percentile(pc_i, perc_limit)
            # idx positions for all data above or below each percentile
            top_scores_idx = np.where(pc_i >= up_perc)[0]
            lower_scores_idx = np.where(pc_i <= lower_perc)[0]
            # extract out data subsample based on pc scores

            # top_x = mod_data[var][top_scores_idx[:,np.newaxis], :, lon_range[:,np.newaxis]].flatten()
            top_x = mod_data[var][top_scores_idx, :, :]
            bot_y = mod_data[var][lower_scores_idx, :, :]
            #else:
            #    raise ValueError('Need to define how to subsample top_x and bot_y from different model ([scores_idx, :, :])?')

            if var == 'backscatter':
                top_x = np.log10(top_x)
                bot_y = np.log10(bot_y)

            # median stats
            top_x_med = np.median(top_x, axis=0)
            bot_y_med = np.median(bot_y, axis=0)

            # flatten top_x and bot_y
            top_x = top_x.flatten()
            bot_y = bot_y.flatten()

            # Wilcoxon signed-rank test FOR MEDIANS - comparison between two dependent samples
            wstat, wpvalue = stats.wilcoxon(top_x_med.flatten(), bot_y_med.flatten())
            statistics_i['wilcoxon_signed_rank_w_medians'] = wstat
            statistics_i['wilcoxon_signed_rank_p_medians'] = wpvalue

            # plot and save, top and bottom medians
            plot_medians(top_x_med, lons, lats, ceil_metadata, pc_i_name, height_i_label, var, days_iterate,
                         topmeddir, 'top_median', wpvalue)
            plot_medians(bot_y_med, lons, lats, ceil_metadata, pc_i_name, height_i_label, var, days_iterate,
                         botmeddir, 'bot_median', wpvalue)



            # repeat array to match dimensions of original data.
            #y = np.squeeze([[[reordered_rot_pcScores[:, i]]*35]*39]).T
            #plt.figure() # (326L, 35L, 39L) = data.
            #stats.pearsonr(mod_data[var].flatten(), y.flatten())
            #plt.scatter(mod_data[var].flatten(), y.flatten())
            # plt.figure()
            # plt.hist(top_x, label='top', bins=500, alpha=0.5, color='blue')
            # plt.hist(bot_y, label='bot', bins=500, alpha=0.5, color='red')

            # 2.0 get boxplot stats
            boxplot_stats_top[var] += cbook.boxplot_stats(top_x, whis=[5, 95])
            boxplot_stats_bot[var] += cbook.boxplot_stats(bot_y, whis=[5, 95])

            # 2.1. Descriptive stats
            statistics_i['mean_top'] = np.mean(top_x)
            statistics_i['std_top'] = np.std(top_x)
            statistics_i['median_top'] = np.median(top_x)
            statistics_i['IQR_top'] = np.percentile(top_x, 75) - np.percentile(top_x, 25)
            statistics_i['75thpct_top'] = np.percentile(top_x, 75)
            statistics_i['25thpct_top'] = np.percentile(top_x, 25)
            statistics_i['skewness_top'] = stats.skew(top_x)
            statistics_i['n_top'] = len(top_x)

            statistics_i['mean_bot'] = np.mean(bot_y)
            statistics_i['std_bot'] = np.std(bot_y)
            statistics_i['median_bot'] = np.median(bot_y)
            statistics_i['IQR_bot'] = np.percentile(bot_y, 75) - np.percentile(bot_y, 25)
            statistics_i['75thpct_bot'] = np.percentile(bot_y, 75)
            statistics_i['25thpct_bot'] = np.percentile(bot_y, 25)
            statistics_i['skewness_bot'] = stats.skew(bot_y)
            statistics_i['n_bot'] = len(bot_y)

            # 2.2. Welch's t test (parametric) equal means (do not need equal variance or sample size between distributions)
            # equal_var=False means ttst_ind is the Welch's t-test (not student t-test)
            #   https://en.wikipedia.org/wiki/Welch%27s_t-test
            # This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values
            #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
            tstat, pvalue = stats.ttest_ind(top_x, bot_y, equal_var=False)
            statistics_i['Welchs_t'] = tstat
            statistics_i['Welchs_p'] = pvalue

            # # fast compare
            # top_x_trim = top_x[(top_x <= np.percentile(top_x, 90)) & (top_x >= np.percentile(top_x, 10))]
            # bot_y_trim = bot_y[(bot_y <= np.percentile(bot_y, 90)) & (bot_y >= np.percentile(bot_y, 10))]
            # mwstat_trim, mwpvalue_trim = stats.mannwhitneyu(top_x_trim, bot_y_trim, alternative='two-sided')

            # 2.3 Mann-Whitney U test (two-sided, non-parametric) test for different distribution shapes
            mwstat, mwpvalue = stats.mannwhitneyu(top_x, bot_y, alternative='two-sided')
            statistics_i['Mann-Whitney-U_stat'] = mwstat
            statistics_i['Mann-Whitney-U_p'] = mwpvalue

            # 2.4 Kolmogorov-Smirnov test for goodness of fit
            _, kspvalue_t = stats.kstest(top_x, 'norm')
            statistics_i['kstest_top_p'] = kspvalue_t
            _, kspvalue_b = stats.kstest(bot_y, 'norm')
            statistics_i['kstest_bot_p'] = kspvalue_b

            # 2.4 Wilcoxon signed-rank test - comparison between two dependent samples
            wstat, wpvalue = stats.wilcoxon(top_x, bot_y)
            statistics_i['wilcoxon_signed_rank_w'] = wstat
            statistics_i['wilcoxon_signed_rank_p'] = wpvalue

            # copy statistics for this var into the main statistics dictionary
            #   use deepcopy to ensure there isn't any shallow copying
            # statistics[height_idx_str][var][pc_i_name] = deepcopy(statistics_i)

            statistics_height[var][pc_i_name] = deepcopy(statistics_i)

    return statistics_height, boxplot_stats_top, boxplot_stats_bot

# Plotting

def plot_corr_matrix_table(matrix, mattype, data_var, height_i_label):
    """
    Plot and save the data talbe for a correlation matrix
    :param matrix:
    :param mattype:
    :param data_var:
    :param height_i_label:
    :return:
    """
    if mattype == 'loadingsCorrMatrix':
        labels = ['loadings' + str(i) for i in np.arange(matrix.shape[0]) + 1]
    elif mattype == 'pcCorrMatrix':
        labels = ['PC' + str(i) for i in np.arange(matrix.shape[0]) + 1]
    else:
        raise ValueError('Labels not set: Need to define how to make the labels with current mattype')
    plt.figure()
    plt.table(cellText=matrix,
              rowLabels=labels,
              colLabels=labels,
              loc='center')
    plt.axis('off')
    plt.grid('off')
    plt.tight_layout()
    plt.savefig(corrmatsavedir + data_var + '_' + mattype + '_' + height_i_label+'.png')
    plt.close()
    return

def plot_spatial_output_height_i(matrix, ceil_metadata, lons, lats, matrixsavedir,
                       days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var,
                       perc_var_explained, matrix_type, model_type):

    """Plot all EOFs for this height - save in eofsavedir (should be a subdirectory based on subsampled input data)"""

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
            orogdatadir = ''
            orog_con = iris.Constraint(name='surface_altitude',
                                       coord_values={
                                           'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
                                           'grid_longitude': lambda cell: 1.21 < cell < 1.732 + spacing})
            orog = iris.load_cube(orogdatadir + '20181022T2100Z_London_charts', orog_con)

        return orog

    # read in orography data to plot underneath EOFs
    orog = read_orography(model_type)

    for m_idx in np.arange(matrix.shape[1]):

        # etract out eof_i
        m_i = matrix[:, m_idx]

        # var explained for this eof
        var_exp_i = perc_var_explained[m_idx]

        # NEED to check if this is rotated back correctly (it might need tranposing)
        # as it was stacked row/latitude wise above (row1, row2, row3)
        # transpose turns shape into (lat, lon) (seems a little odd but needs to be plotted that
        #   way by plt.pcolormesh() to get the axis right...
        eof_i_reshape = np.transpose(
            np.vstack([m_i[n:n + lat_shape] for n in np.arange(0, X_shape, lat_shape)]))  # 1225

        fig, ax = plt.subplots(1, 1, figsize=(6 * aspectRatio, 5))
        cmap_i = plt.get_cmap('viridis')
        # cmap_i = plt.get_cmap('Blues')
        im = plt.pcolormesh(lons, lats, eof_i_reshape, cmap=cmap_i)
        plt.tick_params(direction='out', top=False, right=False, labelsize=13)
        plt.setp(ax.get_xticklabels(), rotation=35, fontsize=13)

        # ax.set_xlabel(r'$Longitude$')
        # ax.set_ylabel(r'$Latitude$')
        ax.set_xlabel('Longitude [degrees]', fontsize=13)
        ax.set_ylabel('Latitude [degrees]', fontsize=13)
        ax.axis('tight')

        # highlight highest value across EOF
        #eof_i_max_idx = np.where(eof_i_reshape == np.max(eof_i_reshape))
        #plt.scatter(lons[eof_i_max_idx][0], lats[eof_i_max_idx][0], facecolors='none', edgecolors='black')
        #plt.annotate('max', (lons[eof_i_max_idx][0], lats[eof_i_max_idx][0]))

        # plot orography
        #levels = np.arange(60, 270, 30)
        cont = ax.contour(lons, lats, orog.data, cmap='OrRd') # cmap='YlOrRd'
        ax.clabel(cont, fmt='%1d') # , color='black'

        # dash the lowest orographic contour
        zc = cont.collections[0]
        plt.setp(zc, linestyle='--')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=-0.1)
        plt.colorbar(im, cax=cax, format='%1.3f')
        # plt.colorbar(cont, cax=cax)

        # plot each ceilometer location
        for site, loc in ceil_metadata.iteritems():
            # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
            ax.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
            ax.annotate(site, (loc[0], loc[1]))
            # txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

        # add % variance explained
        eu.add_at(ax, '%2.1f' % var_exp_i + ' %', loc=1, frameon=True, size=13)

        plt.suptitle(matrix_type + str(m_idx + 1) + '; height=' + height_i_label + '; ' + '%3.2f' % var_exp_i +'%; '+
                     str(len(days_iterate)) + ' cases')
        # make sure the domain proportions are correct
        ax.set_aspect(aspectRatio, adjustable=None)
        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1)
        savename = height_i_label +'_'+matrix_type + str(m_idx + 1) + '_' + data_var + '.png'
        plt.savefig(matrixsavedir + savename)
        plt.close(fig)

    return

def line_plot_exp_var_vs_EOF(perc_explained, height_i_label, expvarsavedir, matrix_type):
    """Plot the accumulated explained variance across the kept EOFs"""

    #perc_explained_cumsum = np.cumsum(perc_explained)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.plot(np.arange(1, len(perc_explained) + 1), perc_explained)
    plt.xticks(np.arange(1, len(perc_explained) + 1, 1.0))
    fig.suptitle('height = ' + height_i_label)
    plt.ylabel('Explained Variance [%]')
    plt.xlabel(matrix_type) # e.g. 'EOF'
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(expvarsavedir + 'exp_var_' + height_i_label + '.png')
    plt.close(fig)

    return

def line_plot_PCs_vs_days_iterate(scores, time, pcsavedir, pctype):

    """Plot the EOFs paired PCs, against the days_iterate (x axis)"""

    for pc_idx in np.arange(scores.shape[-1]):

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # extract out this PC
        pc_i = scores[:, pc_idx]  # / np.mean(pcs[:, pc_idx])

        # plot each day separately so it doesn't spike between days
        #days =
        # plt.plot(pc_norm, label='PC' + str(pc_idx + 1))

        # get idx for the start of each day
        sep_days = np.array([dt.datetime(i.year, i.month, i.day) for i in time])
        uniq_days = np.unique(sep_days)
        idx = np.array([np.where(sep_days == i)[0][0] for i in uniq_days]) # [0][0] = first instance


        # # set tick locations - 1 for each day
        # step = pc_i.shape[0] / len(days_iterate)
        # ticks = np.arange(0, pc_i.shape[0], step)  # +(step/2.0) to centre the label over the day

        # plot each day separately (varying hours so use the idx to identify which data belongs to this day.
        for i, idx_i in enumerate(idx):

            # find range for the day
            if idx_i != idx[-1]:
                x_range = np.arange(idx_i, idx[i+1])
            else: # if at the end of idx, the final bit of the PC will be for this day
                x_range  = np.arange(idx_i, len(sep_days))

            plt.plot(x_range, pc_i[x_range], label='PC' + str(pc_idx + 1), color='blue')

        plt.xticks(idx)
        # get the days in a nice string format, again to plot 1 for each day
        labels = [sep_days[i].strftime('%Y/%m/%d') for i in idx]
        ax.set_xticklabels(labels)

        # # set tick locations - 1 for each day
        # step = pc_i.shape[0] / len(days_iterate)
        # ticks = np.arange(0, pc_i.shape[0], step)  # +(step/2.0) to centre the label over the day
        #
        # # plot each day separately
        # for dplt in ticks[:-1]:
        #     x_range = np.arange(dplt, dplt+step)
        #     plt.plot(x_range, pc_i[x_range], label='PC' + str(pc_idx + 1), color='blue')
        #
        # plt.xticks(ticks)
        # # get the days in a nice string format, again to plot 1 for each day
        # labels = [i.strftime('%Y/%m/%d') for i in days_iterate]
        # ax.set_xticklabels(labels)

        for label in ax.get_xticklabels():
            label.set_rotation(90)

        plt.subplots_adjust(bottom=0.3)

        plt.ylabel('score')
        plt.xlabel('date')
        plt.suptitle(pctype + str(pc_idx + 1) + '; height=' + height_i_label)
        plt.savefig(pcsavedir + pctype + str(pc_idx + 1) + '; height=' + height_i_label + '.png')

        plt.close(fig)

    return

def boxplots_vars(met_vars, mod_data, boxplot_stats_top, boxplot_stats_bot, stats_height, barsavedir,
                  height_i_label):

    """
    Create the boxplots for each variable using pre-calculated statistics. Boxplots show the 'top' and 'bottom'
    subsample of the original data and the significance of the Mann-Whitney-U test ** = 99%, * = 95 %.
    :param met_vars:
    :param mod_data:
    :param boxplot_stats_top:
    :param boxplot_stats_bot:
    :param stats_height:
    :param barsavedir:
    :param height_i_label:
    :return:
    """

    def create_stats_significant_stars(boxplot_stats_bot, var, stats_height):

        """
        Use the statistics results of the Mann-Whitney-U test to create stars and show the extent to which
        the test was significant. ** = 99% sig. * = 95%
        :param boxplot_stats_bot:
        :param var:
        :param stats_height:
        :return: mw_p (list): the stars for each test instance
        """

        mw_p = []
        for i in range(len(boxplot_stats_bot[var])):
            pc_i_name = 'rotPC' + str(i + 1)  # do not start at 0...
            # stats for this iteration
            stats_j = stats_height[var][pc_i_name]

            # Wilcoxon sined ranked test (paired)
            if stats_j['wilcoxon_signed_rank_p'] < 0.01:  # 99.0 %
                sig = '**'
            elif stats_j['wilcoxon_signed_rank_p'] < 0.05:  # 95.0 %
                sig = '*'
            else:
                sig = ''
            mw_p += [sig]



            # # Welch t test
            # if stats_j['Mann-Whitney-U_p'] < 0.01:  # 99.0 %
            #     sig = '**'
            # elif stats_j['Mann-Whitney-U_p'] < 0.05:  # 95.0 %
            #     sig = '*'
            # else:
            #     sig = ''
            # mw_p += [sig]

        return mw_p

    def set_box_color(bp, color):
        # set colour of the boxplots
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

        return

    # boxplot for each variable
    
    for var in met_vars:

        # create stars to signifiy how statistically significant each test was
        # ** = 99 %, * = 95 %, no star = not significant
        mw_p = create_stats_significant_stars(boxplot_stats_bot, var, stats_height)

        fig = plt.figure()
        ax = plt.gca()
        # boxplot from premade stats
        width = 0.2
        # posotion of boxplots.
        # Adjust the central position of boxplots so pairs do not overlap
        pos = np.array(range(len(boxplot_stats_top[var])))+1  # +1 so it starts with xlabel starts PC1, not PC0...
        pos_adjust = (0.6*width)
        # plot
        bptop = ax.bxp(boxplot_stats_top[var], positions=pos-pos_adjust, widths=width, showfliers=False)
        set_box_color(bptop, 'blue')
        bpbot = ax.bxp(boxplot_stats_bot[var], positions=pos+pos_adjust, widths=width, showfliers=False)
        set_box_color(bpbot, 'red')
        # median and IQR of all data on this level
        plt.axhline(np.median(mod_data[var]), linestyle='--', linewidth=0.7, color='black', alpha=0.5)
        if model_type == 'UKV':
            plt.axhline(np.percentile(mod_data[var], 75), linestyle='-.', linewidth=0.7, color='black', alpha=0.5)
            plt.axhline(np.percentile(mod_data[var], 25), linestyle='-.', linewidth=0.7, color='black', alpha=0.5)
  
        # prettify, and correct xtick label and position
        plt.xticks(pos, pos)  # sets position and label
        plt.ylabel(var)
        plt.xlabel('PC')
        # n samples equal across PCs, vars and between top and bottom distributions. Therefore just use this var rotPC1
        plt.suptitle('median')#n_per_dist='+boxplot_stats_top[var]['rotPC1']['n_top'])
        plt.axis('tight')

        # add sample size at the top of plot for each box and whiskers
        top = ax.get_ylim()[1]
        for tick, label in zip(range(len(pos)), ax.get_xticklabels()):
            k = tick % 2
            # ax.text(pos[tick], top - (top * 0.08), upperLabels[tick], # not AE
            ax.text(pos[tick], top - (top * 0.1), mw_p[tick],  # AE
                    horizontalalignment='center')#size='x-small'

        savename = barsavedir + 'median_' + var + '_' + height_i_label + '_rotPCs.png'
        plt.savefig(savename)
        plt.close(fig)

    return


if __name__ == '__main__':

    # time main script was ran
    script_start = dt.datetime.now()

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # data variable to plot
    data_var = 'backscatter'
    #data_var = 'air_temperature'
    #data_var = 'RH'
    #data_var = 'aerosol_for_visibility'

    # save?
    numpy_save = True

    # subsampled?
    #pcsubsample = 'full'
    #pcsubsample = '11-18_hr_range'
    pcsubsample = 'daytime'
    # pcsubsample = 'nighttime'

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    #model_type = 'LM'
    #res = FOcon.model_resolution[model_type]
    Z='21'

    #laptop directories - list needs filtering of MO machine directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    # ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    metadatadir = datadir
    #metadatadir = '/data/jcmm1/ewarren/metadata/'
    #modDatadir = datadir
    pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    savedir = pcsubsampledir + data_var+'/'
    topmeddir = savedir + 'top_median/'
    botmeddir = savedir + 'bot_median/'
    eofsavedir = savedir + 'EOFs/'
    rotEOFsavedir = savedir + 'rotEOFs/'
    rotPCscoresdir = savedir + 'rotPCs/'
    pcsavedir = savedir + 'PCs/'
    expvarsavedir = savedir + 'explained_variance/'
    rotexpvarsavedir = savedir + 'rot_explained_variance/'
    boxsavedir = savedir + 'boxplots/'
    corrmatsavedir = savedir + 'corrMatrix/'
    npysavedir = maindir + '/data/npy/'
    windrosedir = savedir + 'windrose/'

    # # MO directories
    # maindir = '/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/improveNetworks/'
    # datadir = '/data/jcmm1/ewarren//full_forecasts/'+model_type+'/'
    # # ceilDatadir = datadir + 'L1/'
    # modDatadir = datadir + '/London/'
    # metadatadir = '/data/jcmm1/ewarren/metadata/'
    # pcsubsampledir = maindir + 'figures/model_runs/PCA/'+pcsubsample+'/'
    # savedir = pcsubsampledir + data_var+'/'
    # topmeddir = savedir + 'top_median/'
    # botmeddir = savedir + 'bot_median/'
    # eofsavedir = savedir + 'EOFs/'
    # rotEOFsavedir = savedir + 'rotEOFs/'
    # rotPCscoresdir = savedir + 'rotPCs/'
    # pcsavedir = savedir + 'PCs/'
    # expvarsavedir = savedir + 'explained_variance/'
    # rotexpvarsavedir = savedir + 'rot_explained_variance/'
    # boxsavedir = savedir + 'boxplots/'
    # corrmatsavedir = savedir + 'corrMatrix/'
    # npysavedir = '/data/jcmm1/ewarren/npy/'
    # windrosedir = savedir + 'windrose/'

    # intial test case
    # daystr = ['20180406']
    # daystr = ['20180903'] # low wind speed day (2.62 m/s)
    # current set (missing 20180215 and 20181101) # 08-03
    daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507', # all days
              '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
              '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
              '20180901','20180902','20180903','20181007','20181010','20181020','20181023']

    days_iterate = eu.dateList_to_datetime(daystr)
    # a = [i.strftime('%Y%j') for i in days_iterate]
    # '\' \''.join(a)

    # import all site names and heights
    all_sites = ['CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_SWT', 'CL31-E_NK']
    site_bsc = ceil.extract_sites(all_sites, height_type='agl')

    # define dictionary to contain all the statistics drawn from the EOFs and PCs
    # prepare level_height array to store each level height, as they are made
    statistics = {'level_height': np.array([])}

    # keep unrotated PCs for cluster anaylsis in another script
    unrot_loadings_for_cluster = {}

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # make directory paths for the output figures
    # pcsubsampledir, then savedir needs to be checked first as they are parent dirs
    for dir_i in [pcsubsampledir, savedir,
                  eofsavedir, pcsavedir, expvarsavedir, rotexpvarsavedir, rotEOFsavedir, rotPCscoresdir,
                  boxsavedir, corrmatsavedir,
                  topmeddir, botmeddir, windrosedir]:
        if os.path.exists(dir_i) == False:
            os.mkdir(dir_i)

    # make small text file to make sure the figures used the right subsampled data
    with open(savedir + 'subsample.txt', 'w') as file_check:
        file_check.write(pcsubsample)

    # print data variable to screen
    print data_var
    print pcsubsample

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(metadatadir, ceilsitefile)

    # 10=471.7m # np.arange(24) # 4 = 111.7m
    height_idx = 4
    #height_idx = int(sys.argv[1])
    
    #for height_idx in [int(sys.argv[1])]: #np.arange(24):# [0]: #np.arange(24): # max 30 -> ~ 3.1km = too high! v. low aerosol; [8] = 325 m; [23] = 2075 m
    os.system('echo height idx '+str(height_idx)+' being processed')
    os.system('echo '+str(dt.datetime.now() - script_start))

    #print 'Reading in data...'

    # read in model data and subsample using different **kwargs
    # Can accept lon_range=lon_range as an argument if >1microgram cut off is being used
    if pcsubsample == '11-18_hr_range':
        mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx,
                                                     hr_range=[11,18])
    elif (pcsubsample == 'daytime') | (pcsubsample == 'nighttime'):
        mod_data = read_and_compile_mod_data_in_time(days_iterate, modDatadir, model_type, Z, height_idx,
                                                     subsample=pcsubsample)

    # extract out the height for this level
    height_idx_str = str(height_idx)
    height_i = mod_data['level_height']
    height_i_label = '%.1fm' % mod_data['level_height'] # add m on the end

    print 'height_idx = '+str(height_idx)
    print 'height_i = '+str(height_i)

    # if sample size is too small, skip PCA and stat creation (can't remember where but 200 was deemed ~ok)
    if mod_data['time'].shape < 200:
        os.system('echo '+'skipping '+str(height_i)+' as too few samples: '+str(len(mod_data['time'])))
        exit()

    # rotate lon and lat into normal geodetic grid (WGS84)
    lons, lats = rotate_lon_lat_2D(mod_data['longitude'], mod_data['latitude'], model_type)

    # extract out the data (just over London for the UKV)
    if data_var == 'backscatter':
        data = np.log10(mod_data[data_var])
    else:# (data_var == 'air_temperature') | (data_var == 'RH'):
        data = mod_data[data_var]

    # ==============================================================================
    # PCA
    # ==============================================================================
    os.system('echo beginning data processing...')
    os.system('echo '+str(dt.datetime.now() - script_start))

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
    os.system('echo start calc cov matrix...')
    os.system('echo '+str(dt.datetime.now() - script_start))
    
    # set up cov_data array (takes ~ 3 hours for ~27000^2 array)
    cov_data = np.cov(data_m.T) # comaprison shows this, rot loadings and PCs matches SPSS ...   
    os.system('echo finished calc cov matrix...')
    os.system('echo '+str(dt.datetime.now() - script_start))
    
    
    # pseduo inverse as the cov_matrix is too ill-conditioned for a normal inv.
    os.system('echo start invert cov matrix...')
    os.system('echo '+str(dt.datetime.now() - script_start))
    cov_inv = pinv(cov_data)
    os.system('echo finished invert cov matrix...')
    os.system('echo '+str(dt.datetime.now() - script_start))

    # cov_data_normed = np.cov(data_m_norm.T)
    #corr_data = np.corrcoef(data_m.T)

    # carry out Principal Component Analaysis
    # loadings = eig_vecs * sqrt(eig_values)
    os.system('echo beginning PCA...')
    os.system('echo '+str(dt.datetime.now() - script_start))

    eig_vecs_keep, eig_vals, pcScores, loadings, perc_var_explained_unrot_keep = pca_analysis(data_m, cov_data, cov_inv)

    # store the kept loadings, for this height for later saving, and subsequent cluster analysis in another script
    unrot_loadings_for_cluster[height_idx_str] = loadings

    # If there is more than 1 set of EOFs, PCs and loadings - VARIMAX rotate
    # Else, set the 'rotated' component equal to the unrotated component.
    os.system('echo beginning VARIMAX rotation...')
    os.system('echo '+str(dt.datetime.now() - script_start))
    if loadings.shape[-1] > 1:
        # rotate the loadings to spread out the eplained variance between all the kept vectors
        reordered_rot_loadings, reordered_rot_pcScores, perc_var_explained_ratio_rot = \
            rotate_loadings_and_calc_scores(loadings, cov_data, eig_vals)
    else:
        reordered_rot_loadings = loadings
        reordered_rot_pcScores = pcScores
        perc_var_explained_ratio_rot = perc_var_explained_unrot_keep

    # ==============================================================================
    # Calculate and save statistics
    # ==============================================================================

    os.system('echo beginning statistic calculations...')
    os.system('echo '+str(dt.datetime.now() - script_start))

    # met variables to calculate statistics with
    met_vars = mod_data.keys()
    print 'removing Q_H for now as it is on different height levels'
    for none_met_var in ['longitude', 'latitude', 'level_height', 'time', 'Q_H']:
        if none_met_var in met_vars: met_vars.remove(none_met_var)

    # add height dictionary within statistics
    statistics[height_idx_str] = {}

    # keep a list of heights for plotting later
    statistics['level_height'] = np.append(statistics['level_height'], height_i)
    # statistics->height->EOF->met_var->stat
    # plots to make...
    #   bar chart... up vs lower, for each met var, for each height

    # keep explained variances for unrotated and rotated EOFs
    statistics[height_idx_str]['unrot_exp_variance'] = perc_var_explained_unrot_keep
    statistics[height_idx_str]['rot_exp_variance'] = perc_var_explained_ratio_rot

    # Pearson (product moment) correlation between PCs and between EOFs
    if loadings.shape[-1] > 1:
        statistics[height_idx_str]['pcCorrMatrix'] = np.corrcoef(reordered_rot_pcScores.T)
        statistics[height_idx_str]['loadingsCorrMatrix'] = np.corrcoef(reordered_rot_loadings.T)

        # plot and save correlation matrix
        plot_corr_matrix_table(statistics[height_idx_str]['pcCorrMatrix'], 'pcCorrMatrix', data_var, height_i_label)
        plot_corr_matrix_table(statistics[height_idx_str]['loadingsCorrMatrix'], 'loadingsCorrMatrix', data_var, height_i_label)

    # Calculate statistics for each var, for each PC.
    # Includes creating a dictionary of statistics for box plotting, without needing to export the whole
    #   distribution
    statistics_height, boxplot_stats_top, boxplot_stats_bot = \
        pcScore_subsample_statistics(reordered_rot_pcScores, mod_data, met_vars, ceil_metadata, height_i_label,
                             topmeddir, botmeddir)
    # copy statistics for this height into the full statistics dicionary for all heights
    statistics[height_idx_str] = deepcopy(statistics_height)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    os.system('echo beginning plotting...')
    os.system('echo '+str(dt.datetime.now() - script_start))

    # aspect ratio for the map plots
    aspectRatio = float(mod_data['longitude'].shape[0]) / float(mod_data['latitude'].shape[0])
    #aspectRatio = 1.857142 # match UKV plots

    # 1. colormesh() plot the EOFs for this height
    # unrotated
    plot_spatial_output_height_i(eig_vecs_keep, ceil_metadata, lons, lats, eofsavedir,
                                 days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var,
                                 perc_var_explained_unrot_keep, 'EOFs', model_type)
    # rotated EOFs
    plot_spatial_output_height_i(reordered_rot_loadings, ceil_metadata, lons, lats, rotEOFsavedir,
                                 days_iterate, height_i_label, X_shape, lat_shape, aspectRatio, data_var,
                                 perc_var_explained_ratio_rot, 'rotEOFs', model_type)

    # 2. Explain variance vs EOF number
    # unrot
    line_plot_exp_var_vs_EOF(perc_var_explained_unrot_keep, height_i_label, expvarsavedir, 'EOFs')
    # rotated
    line_plot_exp_var_vs_EOF(perc_var_explained_ratio_rot, height_i_label, rotexpvarsavedir, 'rotEOFs')

    # 3. PC timeseries
    # unrotated
    line_plot_PCs_vs_days_iterate(pcScores,  mod_data['time'], pcsavedir, 'PC')
    # rot PC
    line_plot_PCs_vs_days_iterate(reordered_rot_pcScores, mod_data['time'], rotPCscoresdir, 'rotPC')

    # 4. Boxplot the statistics for each var and PC combination
    # Create boxplots for each variable subsampled using each PC (better than the bar chart plottng below)
    stats_height = statistics[height_idx_str]
    boxplots_vars(met_vars, mod_data, boxplot_stats_top, boxplot_stats_bot, stats_height, boxsavedir,
                  height_i_label)

    # 5. wind rose
    from windrose import WindroseAxes
    U = np.sqrt((mod_data['u_wind']**2.0) + (mod_data['v_wind']**2.0))
    # arctan2 needs v then u as arguments! Tricksy numpies!
    U_dir = 180 + ((180 / np.pi) * np.arctan2(mod_data['v_wind'], mod_data['u_wind']))
    # u_wind_i = np.mean(mod_data['u_wind'], axis=(1,2))
    # v_wind_i = np.mean(mod_data['v_wind'], axis=(1,2))
    # U = np.sqrt((u_wind_i ** 2.0) + (v_wind_i ** 2.0))
    # U_dir = 180 + ((180 / np.pi) * np.arctan2(v_wind_i, u_wind_i))
    # https://github.com/python-windrose/windrose/issues/43
    plt.hist([0, 1])
    plt.close()
    fig = plt.figure(figsize=(10, 5))
    rectangle = [0.1, 0.1, 0.8, 0.75]  # [left, bottom, width, height]
    ax = WindroseAxes(fig, rectangle)
    fig.add_axes(ax)
    # bin_range = np.arange(0, 15, 2.5) # higher winds
    bin_range = np.arange(0.0, 12.0, 2.0)
    ax.bar(U_dir.flatten(), U.flatten(), normed=True, opening=0.8, edgecolor='white', bins=bin_range)
    ax.set_title(pcsubsample, position=(0.5, 1.1))
    ax.set_legend()
    ax.legend(title="wind speed (m/s)", loc=(1.1, 0), fontsize=12)
    savename = windrosedir + 'windrose_' + height_i_label + '.png'
    plt.savefig(savename)
    plt.close(fig)


    # ---------------------------------------------------------
    # Save stats
    # ---------------------------------------------------------

    os.system('echo saving statistics to numpy array')

    # save statistics 
    npysavedir_statistics_fullpath = npysavedir+model_type+'_'+data_var+'_'+pcsubsample+'_heightidx'+height_idx_str+'_statistics.npy'
    np.save(npysavedir_statistics_fullpath, statistics)

    # save clusters
    npysavedir_loadings_fullpath = npysavedir +model_type+'_'+data_var + '_' + pcsubsample + '_heightidx'+height_idx_str+'_unrotLoadings.npy'
    save_dict = {'loadings': unrot_loadings_for_cluster,
                 'longitude': lons, 'latitude': lats}
    np.save(npysavedir_loadings_fullpath, save_dict)

    os.system('echo END PROGRAM')
    print 'END PROGRAM'