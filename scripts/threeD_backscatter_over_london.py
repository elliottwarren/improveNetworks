"""
Script to create and plot 3D modelled backscatter over London. Saves the 3D field in a numpy array for post processing
statistics.

Created by Elliott Warren Fri 23 Nov 2018
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

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
    if (model_type == 'UKV') | (model_type == '55m'):
        rotpole = (iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
            6371229.0))).as_cartopy_crs()  # rot grid
        rotpole2 = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        ll = ccrs.Geodetic() # to normal grid
    else:
        raise ValueError('model_type not set as UKV!')

    # calculate corners of grid, from their centre positions, of unrotated data
    # ---------------------
    if corner_locs == True:
        rotlat2D = calculate_corner_locations(rotlat2D)
        rotlon2D = calculate_corner_locations(rotlon2D)

    # 3D array with a slice with nothing but 0s...
    normalgrid = ll.transform_points(rotpole, rotlon2D, rotlat2D)
    lons = normalgrid[:, :, 0]
    lats = normalgrid[:, :, 1]
    # [:, :, 2] always seems to be 0.0 ... not sure what it is meant to be... land mask maybe...

    # test to see if the functions above have made the corner lat and lons properly
    # test_create_corners(rotlat2D, rotlon2D, corner_lats, corner_lons)

    return lons, lats

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # --- User changes

    # save?
    numpy_save = True

    # ------------------

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]
    lon_range = np.arange(26, 65)

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/'
    modDatadir = datadir + model_type + '/'
    savedir = maindir + 'figures/model_runs/cross_sections/'
    npysavedir = datadir + 'npy/'

    # # test case from unused paper 2 UKV data
    # daystr = ['20180903']`
    # current set (missing 20180215 and 20181101) ## start again at 15-05-2018
    # daystr = ['20180406','20180418','20180419','20180420','20180505','20180506','20180507',
    #           '20180514','20180515','20180519','20180520','20180622','20180623','20180624',
    #           '20180625','20180626','20180802','20180803','20180804','20180805','20180806',
    #           '20180901','20180902','20180903','20181007','20181010','20181020','20181023']
    daystr = ['20180902','20180903','20181007','20181010','20181020','20181023']
    days_iterate = eu.dateList_to_datetime(daystr)
    #[i.strftime('%Y%j') for i in days_iterate]

    # ==============================================================================
    # Read and process data
    # ==============================================================================

    for d, day in enumerate(days_iterate):

        # ceilometer list to use
        ceilsitefile = 'improveNetworksCeils.csv'
        ceil_metadata = ceil.read_ceil_metadata(datadir, ceilsitefile)

        print 'day = ' + day.strftime('%Y-%m-%d')

        # calculate the 3D backscatter field across London
        mod_data = FO.mod_site_extract_calc_3D(day, modDatadir, model_type, 905, metvars=True)

        # rotate the lon and lats onto a normal geodetic grid (WGS84)
        lons, lats = rotate_lon_lat_2D(mod_data['longitude'][lon_range], mod_data['latitude'], model_type)

        # extract out met variables with a time dimension
        met_vars = mod_data.keys()
        for none_met_var in ['longitude', 'latitude', 'level_height', 'time', 'Q_H']:
            if none_met_var in met_vars:
                met_vars.remove(none_met_var)

        # ==============================================================================
        # Plotting
        # ==============================================================================

        # find best aspect ratio for the plot so to portray the UKV grid more accurately
        if model_type == 'UKV':
            aspectRatio = float(mod_data['longitude'][lon_range].shape[0]) / float(mod_data['latitude'].shape[0])
        else:
            aspectRatio = float(mod_data['latitude'].shape[0]) / float(mod_data['longitude'].shape[0])

        # each variable in the dataset
        for var in met_vars:

            print 'working on ' + var + '...'

            # make directory paths for the output figures
            # pcsubsampledir, then savedir needs to be checked first as they are parent dirs
            crosssavedir = savedir + var + '/'
            if os.path.exists(crosssavedir) == False:
                os.mkdir(crosssavedir)

            # fig = plt.figure(figsize=(6.5, 3.5))
            # ax = fig.add_subplot(111, aspect=aspectRatio)

            # fast plot - need to convert lon and lats from centre points to corners for pcolormesh()
            for height_idx, height_i in enumerate(mod_data['level_height'][:24]):

                # plotting limits for this height
                # Extract is transposed when indexed like this but
                vmin = np.percentile(mod_data[var][:, height_idx, :, lon_range], 2)
                vmax = np.percentile(mod_data[var][:, height_idx, :, lon_range], 98)

                for hr_idx, hr in enumerate(mod_data['time'][:-1]):  # miss out midnight of the next day...
                    fig = plt.figure(figsize=(6.5, 3.5))
                    ax = fig.add_subplot(111, aspect=aspectRatio)


                    data = mod_data[var][hr_idx, height_idx, :, :]
                    # subsample further if UKV, to trim off some of the domain
                    if model_type == 'UKV':
                        data = data[:, lon_range]

                    # fixed colourbar
                    # mesh = plt.pcolormesh(lons, lats, mod_data['bsc_attenuated'][hr_idx, height_idx, :, :],
                    #                       norm=LogNorm(), cmap=cm.get_cmap('jet'))

                    # if (hr_idx > 0) & (height_idx > 0):
                    #     mesh.set_array(data.ravel())
                    #     fig.canvas.draw()
                    #     fig.canvas.flush_events()


                    if var == 'backscatter':
                        mesh = ax.pcolormesh(lons, lats, data, vmin=vmin, vmax=vmax,
                                              norm=LogNorm(), cmap=cm.get_cmap('jet'))
                    else:
                        mesh = ax.pcolormesh(lons, lats, data, vmin=vmin, vmax=vmax,
                                              cmap=cm.get_cmap('jet'))
                    ax.set_aspect(aspectRatio)

                    the_divider = make_axes_locatable(ax)
                    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
                    cbar = plt.colorbar(mesh, cax=color_axis)
                    # cbar.set_label('$col bar$', fontsize=21, labelpad=-2)

                    # plot each ceilometer location
                    for site, loc in ceil_metadata.iteritems():
                        # idx_lon, idx_lat, glon, glat = FO.get_site_loc_idx_in_mod(mod_all_data, loc, model_type, res)
                        ax.scatter(loc[0], loc[1], facecolors='none', edgecolors='black')
                        ax.annotate(site, (loc[0], loc[1]))

                    ax.set_xlabel(r'$Longitude$')
                    ax.set_ylabel(r'$Latitude$')
                    #plt.colorbar(mesh)
                    plt.suptitle(hr.strftime('%Y-%m-%d_%H') + '; height='+str(mod_data['level_height'][height_idx])+'m')
                    savesubdir = crosssavedir + hr.strftime('%Y-%m-%d') + '/' # sub dir within the savedir
                    savename = '{:04.0f}'.format(mod_data['level_height'][height_idx]) + 'm_'+hr.strftime('%Y-%m-%d_%H')+'.png'

                    plt.tight_layout()

                    if os.path.exists(savesubdir) == False:
                        os.mkdir(savesubdir)
                    plt.savefig(savesubdir + savename)
                    plt.close(fig)

    print 'END PROGRAM'
