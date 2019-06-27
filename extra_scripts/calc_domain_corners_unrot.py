"""
Calculate the grid corners for a domain, given the corner grid box centre positions and the grid length.

Find corners in model space - then rotate to normal

Created by Elliott Warren Thurs 27/06/19
"""

# read in rot lons and lats
# get grid spacing
# extent to get corners
# rotate to normal


import sys

sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/Utils')  # aerFO
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ellUtils')  # general utils
sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ceilUtils')  # ceil utils

import numpy as np

import cartopy.crs as ccrs
import iris

# import ellUtils.ellUtils as eu

import ellUtils as eu

if __name__ == '__main__':

    model_type = 'LM'

    # read in lons and lats
    filename = '/data/jcmm1/ewarren/full_forecasts/LM/London/extract_London_prodm_op_LM_20180405_21_full.nc'
    data = eu.netCDF_read(filename, vars=['longitude', 'latitude'])
    lats = data['latitude']
    lons = data['longitude']

    # half grid spcaing
    hgs = lons[1] - lons[0]

    # find corners coordinates in the rotated model space ([0] = smallest, [-1] largest in lons and lats)
    # lower limits [0] - hgs, upper limits [-1] + hgs
    rot_top_left = [lons[0] - hgs, lats[-1] + hgs]  # good
    rot_top_right = [lons[-1] + hgs, lats[-1] + hgs]  #
    rot_bot_left = [lons[0] - hgs, lats[0] - hgs]
    rot_bot_right = [lons[-1] + hgs, lats[0] - hgs]

    # Get model transformation object (ll) to unrotate the model data later
    # Have checked rotation is appropriate for all models below
    if (model_type == 'UKV') | (model_type == '55m') | (model_type == 'LM'):
        rotpole = (iris.coord_systems.RotatedGeogCS(37.5, 177.5, ellipsoid=iris.coord_systems.GeogCS(
            6371229.0))).as_cartopy_crs()  # rot grid
        ll = ccrs.Geodetic()  # to normal grid
    else:
        raise ValueError('model_type not set as UKV/55m/LondonModel! - Check rotation type!')

    # from rotated into unrotated
    unrot_top_left = ll.transform_point(rot_top_left[0], rot_top_left[1], rotpole)
    unrot_top_right = ll.transform_point(rot_top_right[0], rot_top_right[1], rotpole)
    unrot_bot_left = ll.transform_point(rot_bot_left[0], rot_bot_left[1], rotpole)
    unrot_bot_right = ll.transform_point(rot_bot_right[0], rot_bot_right[1], rotpole)

    print 'unrot_top_left:' + str(unrot_top_left)
    print 'unrot_top_right:' + str(unrot_top_right)
    print 'unrot_bot_left:' + str(unrot_bot_left)
    print 'unrot_bot_right:' + str(unrot_bot_right)
