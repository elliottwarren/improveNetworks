"""
Read and plot the UKV surface types
Created on 4 April 2019

@originalauthor: frpz
Edited by Elliott Warren: Thurs 25 Apr 2018
"""

import iris
import matplotlib
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
    # veg_pseudo_levels=1,2,3,4,5,7,8,9,601,602
    SurfTypeD={1:'Broadleaf trees', 2:'Needleleaf trees', 3:'C3 (temperate) grass',
               4:'C4 (tropical) grass', 5:'shrubs',
               7:'Inland water', 8:'Bare Soil', 9:'Ice',
               601:'Urban canyon', 602: 'Urban roof' }


    llabels=('Broadleaf trees','Needleleaf trees','C3 (temperate) grass',
             'C4 (tropical) grass','Shrubs', 
             'Inland water', 'Bare Soil', 'Ice', 
             'Urban canyon',  'Urban roof' )

    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/improveNetworks/'
    datadir = maindir + 'data/UKV/'
    savedir = maindir + 'figures/model_ancillaries/'

    # # Cristina's
    # UK_lat_constraint = iris.Constraint(grid_latitude=lambda cell: -2.0 < cell < -0.5)
    # UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 360.0 < cell < 362.5)
    # UKV London domain
    UK_lat_constraint = iris.Constraint(grid_latitude=lambda cell: -1.2326999 < cell < -0.7737)
    UK_lon_constraint = iris.Constraint(grid_longitude=lambda cell: 361.21997 < cell < 361.73297)

    temp_cmap = plt.get_cmap('BuPu')
    aspectRatio = 1.8571428571428572 # from my other plots

    # Pick color map
    surf_frac=iris.load_cube(datadir + 'ukv_surf_frac_types.nc',constraint=UK_lat_constraint & UK_lon_constraint)
    for i,iid in enumerate(surf_frac.coord('pseudo_level').points):
        if iid not in [4,9]: # ignore ice and tropical grass
            plt.subplots(1,1, figsize=(4.5 * aspectRatio, 4.5))
            im=iplt.pcolormesh(surf_frac[i],cmap=temp_cmap)
            plt.colorbar(im,orientation='vertical')
            plt.gca().coastlines('10m')
            plt.title(SurfTypeD[iid])
            plt.savefig(savedir + 'UKV_'+SurfTypeD[iid]+'.png')
            plt.close()

    plt.show()
    

if __name__ == '__main__':
    main()
