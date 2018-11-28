"""
Script to create and plot 3D modelled backscatter over London. Saves the 3D field in a numpy array for post processing
statistics.

Created by Elliott Warren Fri 23 Nov 2018
"""

import numpy as np
from scipy.stats import spearmanr
import datetime as dt
import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

import ellUtils.ellUtils as eu
import ceilUtils.ceilUtils as ceil

from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon


def calculate_corner_locations(coord):

    """
    Calculate the corner coordinates to match coord. corner[0,0] would be the bottom left corner for coord[0,0].

    :param coord (2D array): either longitude or latitude coordinates
    :return: corners (2D array with +1 size than coord, in each dimension): corner values for coord.

    Bottom left corner is [0,0], top right is [-1,-1] ([row, col])

    Examples ('.' are grid centre points, '-' and '|' are grid edges).

    red outline (bot left)             blue outline of grid (top right
                       [-1,-1]
       ---------------------           ----b----------------
       | . | . | . | . | . |           | . b . | . | . | . |
       r r r r r r r r r ---           ----b----------------
       | . | . | . | . r . |           | . b . | . | . | . |
       ----------------r----           ----b----------------
       | . | . | . | . r . |           | . b . | . | . | . |
       ----------------r----           ----b-b-b-b-b-b-b-b-b
       | . | . | . | . r . |           | . | . | . | . | . |
       ----------------r----           ---------------------
    [0,0]

    """

    # corner
    corner = np.empty(np.array(coord.shape) + 1)
    corner[:] = np.nan

    # quadrant (sub sampled areas that exclude a row or column)
    blue = coord[1:, 1:]  # top right
    red = coord[:-1, :-1]  # bot left
    black = coord[1:, :-1]  # top left
    green = coord[:-1, 1:]  # bot right

    # calculate bottom left corner values to match what is required by plt.pcolourmesh() X,Y parameters
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html

    # 1. centre boxes (most of the corner values)
    corner[1:-1, 1:-1] = red + ((blue - red) / 2.0)  # red +((blue - red)/2)

    # 2. deal with outer edge cases...
    # calculate the 2D arrays, then extract out of it the edge cases Avoid overwriting values
    #  as the original 'centre box' calculation above is more accurate for the centre boxes.

    a = blue + ((blue - red) / 2.0)  # Side and corner
    corner[-1, 2:] = a[-1, :]  # top centre and top right corner
    corner[2:, -1] = a[:, -1]  # centre right side and rest of top right corner

    b = red - ((blue - red) / 2.0)  # Side and corner
    corner[:-2, 0] = b[:, 0]  # centre left side and bottom left corner
    corner[0, :-2] = b[0, :]  # centre bottom side and rest of bottom left corner

    c = green - ((black - green) / 2.0)  # corner only
    corner[0, -2:] = c[0, -2:]  # just bottom right corner
    corner[1, -1] = c[1, -1]  # rest of bottom right corner

    c = black + ((black - green) / 2.0)  # corner only
    corner[-2:, 0] = c[-2:, 0]  # just top left corner
    corner[-1, 1] = c[-1, 1]  # rest of top left corner

    return corner

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

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    modDatadir = datadir + model_type + '/'
    savedir = maindir + 'figures/model_runs/'
    npysavedir = datadir + 'npy/'

    # test case from unused paper 2 UKV data
    daystr = ['20170902']
    day = eu.dateList_to_datetime(daystr)[0]
    #[i.strftime('%Y%j') for i in days_iterate]

    # save name
    savestr = day.strftime('%Y%m%d') + '_3Dbackscatter.npy'

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    # ceilometer list to use
    ceilsitefile = 'improveNetworksCeils.csv'
    ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

    print 'day = ' + day.strftime('%Y-%m-%d')

    # calculate the 3D backscatter field across London
    mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, res, 905, allvars=True)

    # create 2D array of lon and lats in rotated space
    # duplicate the single column array but number of rows in latitude
    # rotlat2D needs to be transposed as rows need to be latitude, and columns longitude.
    # np.transpose() does transpose in the correct direction:
    # compare mod_all_data['latitude'][n] with rotlat2D[n,:]
    rotlon2D = np.array([mod_data['longitude']] * mod_data['latitude'].shape[0])
    rotlat2D = np.transpose(np.array([mod_data['latitude']] * mod_data['longitude'].shape[0]))

    #spec_mesh = np.meshgrid(mod_data['latitude'], mod_data['longitude'])

    # calculate corners of grid, from their centre positions

    # ---------------------

    corner_lats = calculate_corner_locations(rotlat2D)
    corner_lons = calculate_corner_locations(rotlon2D)

    # unrotate the model data
    if model_type == 'UKV':
        rotpole = (iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
            6371229.0))).as_cartopy_crs()  # rot grid
        rotpole2 = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        ll = ccrs.Geodetic()  # normal grid
    else:
        raise ValueError('model_type not set as UKV!')

    # 3D array with a slice with nothing but 0s...
    normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
    lons_orig = normalgrid[:, :, 0]
    lats_orig = normalgrid[:, :, 1]
    # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...

    # 3D array with a slice with nothing but 0s...
    normalgrid = ll.transform_points(rotpole, corner_lons, corner_lats)
    lons = normalgrid[:, :, 0]
    lats = normalgrid[:, :, 1]

    plt.figure()
    plt.scatter(rotlat2D, rotlon2D, color='red')
    plt.scatter(corner_lats, corner_lons, color='blue')
    #plt.scatter(a, d, color='green')
    #plt.scatter(lats, lons, color='red')
    # plt.scatter(lats_orig, lons_orig, color='blue')
    plt.figure()
    plt.pcolormesh(corner_lats, vmin=np.nanmin(corner_lats), vmax=np.nanmax(corner_lats))
    plt.colorbar()

    # ==============================================================================
    # Plotting
    # ==============================================================================

    # fast plot - need to convert lon and lats from centre points to corners for pcolormesh()
    for hr_idx, hr in enumerate(mod_data['time']):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.4))

        mesh = plt.pcolormesh(lons, lats, mod_data['bsc_attenuated'][hr_idx, 6, :, :],
                       norm=LogNorm(vmin=1e-7, vmax=1e-5), cmap=cm.get_cmap('jet'))

        # plot each ceilometer location
        for site, loc in ceil_metadata.iteritems():
            # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
            plt.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
            plt.annotate(site, (loc[0], loc[1]))

        ax.set_xlabel(r'$Longitude$')
        ax.set_ylabel(r'$Latitude$')
        plt.colorbar(mesh)
        plt.suptitle(hr.strftime('%Y-%m-%d_%H') + ' backscatter')
        savename = hr.strftime('%Y-%m-%d_%H') + '_beta.png'
        plt.savefig(savedir + savename)
        plt.close(fig)




























    print 'END PROGRAM'
