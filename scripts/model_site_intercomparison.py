"""
Script to forward model backscatter for several sites within London and intercompare them in the same
way as the ceilometer observations.

Created by Elliott Warren Mon 3 Dec 2018
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


