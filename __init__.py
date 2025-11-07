#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:20:00 2024

@author: Matthew Bogumil
"""

#######################################################################
###################### ExoCcycle Module Imports #######################
#######################################################################

# Import modules used in the slab model module.
import copy as cp
from tqdm.auto import tqdm # used for progress bar
import sys
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy import stats
import matplotlib.pyplot as plt
import time



#######################################################################
######################### Import helper modules #######################
#######################################################################
from ExoCcycle import utils                     # type: ignore
from ExoCcycle import Bathymetry                # type: ignore
from ExoCcycle import functionClassBuilding     # type: ignore



