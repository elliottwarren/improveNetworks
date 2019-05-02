"""
Read and plot the UKV surface types
Created on 4 April 2019

@originalauthor: frpz
Edited by Elliott Warren: Thurs 25 Apr 2018
"""

import iris
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcols
import iris.plot as iplt
from matplotlib import cm
import matplotlib.cm as mpl_cm

import iris.palette
import iris.quickplot as qplt
from matplotlib.colors import from_levels_and_colors
import numpy as np
import iris.coords as icoords
import iris.coord_systems as icoord_systems
import os
from os.path import join, exists
#import komodo.util.timekeeper as kt

def main():

    # what to plot? Surface types of orography? (Come in different files)
    # data_to_plot = 'surface type'
    # data_to_plot = 'orography'
    data_to_plot = 'murk_aer'

    model_type = 'UKV'

    # veg_pseudo_levels=1,2,3,4,5,7,8,9,601,602
    SurfTypeD={1:'Broadleaf trees', 2:'Needleleaf trees', 3:'C3 (temperate) grass',
               4:'C4 (tropical) grass', 5:'shrubs',
               7:'Inland water', 8:'Bare Soil', 9:'Ice',
               601:'Urban canyon', 602: 'Urban roof' }


    llabels=('Broadleaf trees','Needleleaf trees','C3 (temperate) grass',
             'C4 (tropical) grass','Shrubs', 
             'Inland water', 'Bare Soil', 'Ice', 
             'Urban canyon',  'Urban roof')

    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/UKV/'
    npydatadir = maindir + '/data/npy/'
    savedir = maindir + 'figures/model_ancillaries/'

    # # Cristina's
    # UK_lat_constraint = iris.Constraint(grid_latitude=lambda cell: -2.0 < cell < -0.5)
    # UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 360.0 < cell < 362.5)
    # UKV London domain

    aspectRatio = 1.8571428571428572  # from my other plots

    if data_to_plot == 'surface type':

        temp_cmap = plt.get_cmap('BuPu')

        # Pick color map
        surf_frac=iris.load_cube(datadir + 'ukv_surf_frac_types.nc',constraint=UK_lat_constraint & UK_lon_constraint)
        for i,iid in enumerate(surf_frac.coord('pseudo_level').points):
            if iid not in [4, 9]: # ignore ice and tropical grass
                plt.subplots(1,1, figsize=(4.5 * aspectRatio, 4.5))
                im=iplt.pcolormesh(surf_frac[i],cmap=temp_cmap)
                plt.colorbar(im,orientation='vertical')
                plt.gca().coastlines('10m')
                plt.title(SurfTypeD[iid])
                plt.savefig(savedir + 'UKV_'+SurfTypeD[iid]+'.png')
                plt.close()

    elif data_to_plot == 'orography': # orography

        if model_type == 'UKV':
            spacing = 0.0135  # spacing between lons and lats in rotated space
            UK_lat_constraint = iris.Constraint(
                grid_latitude=lambda cell: -1.2326999 - spacing <= cell <= -0.7737 + spacing)
            UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 361.21997 <= cell <= 361.73297 + spacing)
            orog = iris.load_cube(datadir + 'UKV_orography.nc', constraint=UK_lat_constraint & UK_lon_constraint)

        temp_cmap = plt.get_cmap('terrain')
        aspectRatio = 1.8571428571428572  # from my other plots

        # Pick color map
        orog = iris.load_cube(datadir + 'UKV_orography.nc', constraint=UK_lat_constraint & UK_lon_constraint)
        plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
        im = iplt.pcolormesh(orog, cmap=temp_cmap)
        plt.colorbar(im, orientation='vertical')
        plt.gca().coastlines('10m')
        plt.title('orography')
        plt.savefig(savedir + 'UKV_orography.png')
        plt.close()

    elif data_to_plot == 'murk_aer':

        murksavedir = savedir + 'murk_aer/'
        if os.path.exists(murksavedir) == False:
            os.mkdir(murksavedir)

        temp_cmap = plt.get_cmap('jet')

        # load data
        murk_aer = np.load(npydatadir + model_type + '_murk_ancillaries.npy').flatten()[0]
        for height_idx in np.arange(8):
            #height_idx = 10
            height = murk_aer.coord('level_height').points[height_idx]
            month_idx = 6
            murk_data = murk_aer.data[month_idx, height_idx, :, :]
            plt.subplots(1, 1, figsize=(4.5 * aspectRatio, 4.5))
            im = plt.pcolormesh(murk_data, cmap=temp_cmap, vmin=0, vmax=0.18)
            plt.colorbar(im, orientation='vertical')
            #plt.gca().coastlines('10m')
            plt.title('murk_aer')
            savename = 'UKV_murk_aer_'+str(month_idx+1)+'_'+str(height)+'m.png'
            plt.savefig(murksavedir + savename)
            plt.close()


    

if __name__ == '__main__':
    main()
