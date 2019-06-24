"""
Read and plot the UKV surface types
Created on 4 April 2019

@originalauthor: frpz
Edited by Elliott Warren: Thurs 25 Apr 2018
"""

# workaround while PYTHONPATH plays up on MO machine
import sys
#sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts')
sys.path.append('/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/improveNetworks/scripts')

import iris
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import iris.plot as iplt

import iris.palette
import numpy as np
import os
#import komodo.util.timekeeper as kt

from threeD_backscatter_over_london import rotate_lon_lat_2D

if __name__ == '__main__':

    # what to plot? Surface types of orography? (Come in different files)
    # data_to_plot = 'surface type'
    #data_to_plot = 'orography'
    data_to_plot = 'murk_aer'

    model_type = 'UKV'
    #model_type = 'LM'

    # veg_pseudo_levels=1,2,3,4,5,7,8,9,601,602
    SurfTypeD={1:'Broadleaf trees', 2:'Needleleaf trees', 3:'C3 (temperate) grass',
               4:'C4 (tropical) grass', 5:'shrubs',
               7:'Inland water', 8:'Bare Soil', 9:'Ice',
               601:'Urban canyon', 602: 'Urban roof' }


    llabels=('Broadleaf trees','Needleleaf trees','C3 (temperate) grass',
             'C4 (tropical) grass','Shrubs', 
             'Inland water', 'Bare Soil', 'Ice', 
             'Urban canyon',  'Urban roof')

    if model_type == 'UKV':
        maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
        datadir = maindir + 'data/UKV/ancillaries/'
        npydatadir = maindir + '/data/npy/'
        savedir = maindir + 'figures/model_ancillaries/'+model_type+'/'
    elif model_type == 'LM': # on MO machine
        maindir = '/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/improveNetworks/'
        datadir = '/data/jcmm1/ewarren/ancillaries/'+data_to_plot+'/'+model_type+'/'
        #npydatadir = maindir + '/data/npy/'
        savedir = maindir + 'figures/ancillaries/'+model_type+'/'

    # Extract domain constraints from UKV and LM the domain over London
    if model_type == 'UKV':
        orog_con = iris.Constraint(name='surface_altitude',
                                   coord_values= {'grid_latitude': lambda cell: -1.0326999 <= cell <= -0.2738,
                                                  'grid_longitude': lambda cell: 360.41997 <= cell <= 361.733})
    elif model_type == 'LM':
        spacing = 0.003 # checked
        # checked that it perfectly matches LM data extract (setup is different to UKV orog_con due to
        #    number precision issues.
        orog_con = iris.Constraint(name='surface_altitude',
                                   coord_values={
                                       'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
                                       'grid_longitude': lambda cell: 1.21 < cell < 1.732 + spacing})
    # name='surface_altitude',
    # UK_lon_constraint =
    # aspect ratio for plotting
    #aspectRatio = 1.8571428571428572  # from my other plots

    if data_to_plot == 'surface type':

        temp_cmap = plt.get_cmap('BuPu')

        # Pick color map
        surf_frac=iris.load_cube(datadir + 'ukv_surf_frac_types.nc',constraint=UK_lat_constraint & UK_lon_constraint)
        for i,iid in enumerate(surf_frac.coord('pseudo_level').points):
            if iid not in [4, 9]: # ignore ice and tropical grass
                plt.subplots(1,1, figsize=(5, 4.5))
                im=iplt.pcolormesh(surf_frac[i],cmap=temp_cmap)
                plt.colorbar(im,orientation='vertical')
                plt.gca().coastlines('10m')
                plt.title(SurfTypeD[iid])
                plt.savefig(savedir + model_type+'_'+SurfTypeD[iid]+'.png')
                plt.close()

    elif data_to_plot == 'orography': # orography

        if model_type == 'UKV':
            orog = iris.load_cube(datadir + 'UKV_orography.nc', orog_con)
        elif model_type == 'LM':
            # new_data = iris.load_cube(datadir + '20181022T2100Z_London_charts', 'surface_altitude', constraint=orog_con)
            orog = iris.load_cube(datadir + '20181022T2100Z_London_charts', orog_con) # orog_con

        # rotate lon and lat back to normal
        orog_rot_lats = orog.coord('grid_latitude').points
        orog_rot_lons = orog.coord('grid_longitude').points
        lons, lats = rotate_lon_lat_2D(orog_rot_lons, orog_rot_lats, model_type)

        temp_cmap = plt.get_cmap('terrain')
        plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
        # im = iplt.pcolormesh(orog, cmap=temp_cmap)
        im = plt.pcolormesh(lons, lats, orog.data, cmap=temp_cmap)
        plt.colorbar(im, orientation='vertical')
        #plt.gca().coastlines('10m')
        plt.title(model_type +' orography')
        plt.savefig(savedir + model_type+'_orography.png')
        plt.close()

    elif data_to_plot == 'murk_aer':

        murksavedir = savedir + 'murk_aer/'
        if os.path.exists(murksavedir) == False:
            os.mkdir(murksavedir)

        temp_cmap = plt.get_cmap('jet')

        # load data
        if model_type == 'UKV':
            # checked it matches other UKV output: slight differences in domain constraint number due to different
            #   number precision error in the saved files...
            murk_con = iris.Constraint(coord_values=
                                       {'grid_latitude': lambda cell: -1.2327999 <= cell <= -0.7738,
                                        'grid_longitude': lambda cell: 361.21997 <= cell <= 361.733})
            murk_aer = iris.load_cube(datadir + 'UKV_murk_surface.nc', murk_con)
            murk_aer = murk_aer[6, :, :] # get July
        elif model_type == 'LM':

            spacing = 0.003  # checked
            # checked that it perfectly matches LM data extract (setup is different to UKV orog_con due to
            #    number precision issues.
            con = iris.Constraint(coord_values={
                                      'grid_latitude': lambda cell: -1.214 - spacing < cell < -0.776,
                                      'grid_longitude': lambda cell: 361.21 < cell < 361.732 + spacing})

            murk_aer_all = iris.load_cube(datadir + 'qrclim.murk_L70')
            murk_aer = murk_aer_all.extract(con)

        rot_lats = murk_aer.coord('grid_latitude').points
        rot_lons = murk_aer.coord('grid_longitude').points# - 360.0 - removing 360 has no real effect on rotation
        lons, lats = rotate_lon_lat_2D(rot_lons, rot_lats, model_type)
        # aspectRatio = float(lons.shape[0]) / float(lats.shape[1])
        aspectRatio = float(lons.shape[1]) / float(lats.shape[0])

        #height_idx = 4# 10
        #height = murk_aer.coord('level_height').points[height_idx]
        month_idx = 0
        # murk_data = murk_aer.data[month_idx, height_idx, :, :]
        murk_data = murk_aer.data
        # plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
        fig, ax = plt.subplots(1, 1, figsize=(6 * aspectRatio, 5))
        temp_cmap = plt.get_cmap('jet')
        im = plt.pcolormesh(lons, lats, murk_data, cmap=temp_cmap, vmin=0, vmax=0.18)
        plt.tick_params(direction='out', top=False, right=False, labelsize=13)
        plt.setp(ax.get_xticklabels(), rotation=35, fontsize=13)
        ax.set_xlabel('Longitude [degrees]', fontsize=13)
        ax.set_ylabel('Latitude [degrees]', fontsize=13)
        ax.axis('tight')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=-0.1)
        plt.colorbar(im, cax=cax, format='%1.3f')

        ax.set_aspect(aspectRatio, adjustable=None)
        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1)

        #plt.gca().coastlines('10m')
        #plt.title('murk_aer')
        # savename = 'UKV_murk_aer_'+str(month_idx+1)+'_'+str(height)+'m.png'
        savename = model_type+'_murk_aer_July_5m.png'
        plt.savefig(murksavedir + savename)
        plt.close()

