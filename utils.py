#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:20:00 2024

@author: bogumilmatt-21
"""


#######################################################################
############################### Imports ###############################
#######################################################################
import os
import numpy as np
import copy as cp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib as mpl
import networkx as nx # type: ignore
from netCDF4 import Dataset 
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import itertools

# For progress bars
from tqdm.auto import tqdm # used for progress bar

from ExoCcycle import Bathymetry # type: ignore

################################################################
###################### Functions to Sort #######################
################################################################

def combine_lists(*lists):
    """
    Combine N input lists into all possible unique combinations.
    Returns a numpy array of shape (number_of_combinations, N).

    lists : Arbitrary number of positional argument lists
        Lists need not be equal in length, but must contain
        only float or int elements. String elements do not
        work in this function

    Return : NUMPY ARRAY
        All possible unique combinations from input list.

    Example:
    fileNames = ["file1.nc", "file2.nc"]
    fileNameIDs = [0, 1]
    ensembleSizes = [10, 50]

    A = combine_lists(fileNameIDs, ensembleSizes)

    for fileNameID, ensembleSize in A:
        print("values:", fileNameID, ensembleSize, fileNames[fileNameID])

    Output:
    values: 0 10 "file1.nc"
    values: 0 50 "file1.nc"
    values: 1 10 "file2.nc"
    values: 1 50 "file2.nc"
    """
    # Generate all combinations (cartesian product)
    combinations = list(itertools.product(*lists))
    
    # Convert to numpy array
    A = np.array(combinations)
    
    return A

def filterNc(options={"inputFile":None, "outputFile":None, "threshold":None, "lr":None, "gt":None},
            keepVars = ['lat', 'lon', 'bathymetry']):  
    """
    filterNc is a function used for filtering a NetCDF4 file.
    Options for filtering can be expanded, but the main use
    of this function is to filter data above or below some
    threshold value.
    
    Parameters
    -----------
    options : DICTIONARY
        A set of options used to modify the input netcdf4.
    
    Returns
    --------
    None
    """
    # Check options have been defined, if not then assign a default
    optionsList = ["inputFile", "outputFile", "threshold", "le", "gt"];
    for option in optionsList:        
        try:
            #print(option, options[option])
            options[option]
        except:
            options[option] = None
            #print(option,options[option], "changed")
    
    
    # Open original file
    src = Dataset(options["inputFile"], 'r')
    
    # Determine dimensions needed by variables we want to keep
    needed_dims = set()
    for name, variable in src.variables.items():
        if name in keepVars:
            needed_dims.update(variable.dimensions)
    
    # Create new file
    dst = Dataset(options["outputFile"], 'w', format='NETCDF4_CLASSIC')
    
    # Copy only needed dimensions
    for name, dimension in src.dimensions.items():
        if name in needed_dims:
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        
    # Copy variables we want to keep
    for name, variable in src.variables.items():
        if name in keepVars:
            # Create variable in new file
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            # Copy variable attributes
            x.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})

            if name == options["varName"]:
                var = variable[:];
                if options["le"]:
                    var.data[var.data<=options["threshold"]] = np.nan;
                if options["gt"]:
                    var.data[var.data>options["threshold"]] = np.nan;

                # Copy modified variable data
                x[:] = var;
            else:
                # Copy variable data
                x[:] = variable[:];
    
    # Copy global attributes
    dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()});
    
    # Close files
    src.close();
    dst.close();

#######################################################################
###################### ExoCcycle Module Imports #######################
#######################################################################

def create_file_structure(list_of_directories, root=False, verbose=True):
    """
    create_file_structure function creates new directories from a provided
    list of directories.

    
    Parameters
    ----------
    list_of_directories : LIST
        A list of directories to be created. For example :
            list_of_directories = [ "/data",
                                    "/data/folder1", "/data/folder1/folder1",
                                    "/data/folder2", "/data/folder2/folder1", "/data/folder2/folder1",
                                    "/data/folder3"];
    root : BOOLEAN, optional
        An option to choose whether directories to be made include a root
        directory path or a local path. If False is chosen then all input
        directories will be preceded by os.getcwd(). The default is False.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Returns
    -------
    None.
    """
    # List to hold directories that already exist.
    list_of_existing_directories = [];
    
    if root == True:
        cwd = ""
    else:
        cwd = os.getcwd();

    # Create directories
    for directory in list_of_directories:
        if not os.path.exists(cwd+directory):
            os.mkdir(cwd+directory);
        else:
            list_of_existing_directories.append(directory);
    if verbose:
        if len(list_of_existing_directories) != 0:
            print("\nThe following folder(s) exists within current directory:\n{}".format(", ".join(list_of_existing_directories) ));


def makeFolderSeries(fldBase='fldBase', maxFolders=1000):
    """
    makeFolderSeries function is used to make a
    one folder in a series of folders with a
    standardized name.
    
    Parameters
    -----------
    fldBase : STRING
        The base name of the folder series.
    maxFolders : INT
        The maximum number of folders allowed in the series

    Return
    -------
    fldName : String
        The path of the folder created.
    """
    # initialize counter
    i=0
    
    # Loop over maxFolders
    for i in range(maxFolders):
        # Try to make folder and break if folder is created.
        try:
            os.mkdir('{0}_{1}'.format(fldBase, i))
            # folder was created successfully, return path/name
            return '{0}_{1}'.format(fldBase, i)
        except:
            # folder was not created successfully
            pass

def downloadSolarSystemBodies(data_dir):
    """
    downloadSolarSystemBodies is a function used to download all
    topography models integrated into ExoCcycle. 
    

    Parameters
    ----------
    data_dir : STRING
        A directory which you store local data within. Note that this
        function will download directories [data_dir]/topographies

    Returns
    -------
    None.
    
    """

    # Define the set of bodies to be downloaded.
    bodies = ["venus", "earth", "mars", "moon"];

    # Iterate over solar system bodies.
    for bodyi in bodies:
        # Create object for topography model.
        bodyBathymetry = Bathymetry.BathyMeasured(body=bodyi);

        # Download raw topography model.
        bodyBathymetry.getTopo(data_dir)

        # Create body topography netCDF with standard resolution.
        bodyBathymetry.readTopo(data_dir, new_resolution = 1, verbose=False);

        # Run function again: will create a gmt post script of topography 
        # since the netCDF with standard resolution has already been created.
        bodyBathymetry.readTopo(data_dir, new_resolution = 1, verbose=False);

def weightedAvgAndStd(values, weights):
    """
    Return the weighted average and standard deviation.

    The weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).
    
    Parameters
    -----------
    values : NUMPY ndarray 
        Values to calculate the average and variance with.
    weights : NUMPY ndarray
        Weights to be used for average and variance calculation.
        numpy array must ahve same shape as 'values.'
    """
    # Create masked values and weights
    masked_values  = np.ma.masked_array(values, np.isnan(values))
    masked_weights = np.ma.masked_array(weights, np.isnan(values))
    # Normalize weights so they sum to 1.
    masked_weights /= np.nansum(masked_weights)
    # Find weighted average
    average = np.average(masked_values, weights=masked_weights)
    # Fast and numerically precise:
    variance = np.average((masked_values-average)**2, weights=masked_weights)
    return (average, np.sqrt(variance))

def areaWeights(resolution = 1, radius = 6371e3, LonStEd = [-180,180], LatStEd = [-90,90], verbose=True):
    """
    areaWeights function is used to make an array of global degree to area
    weights based on an input resolution. Note that this function calculates
    area weights by assuming bodies are spherical.


    Parameters
    ----------
    resolution : FLOAT
        Resolution of array, in degrees. The default is 1.
    radius : FLOAT
        Radius represents the radius of the body, in m. For earth (6371e3),
        venus (6051e3), mars (3389.5e3), moon (1737.4km), one might use these
        provied values. The default is 6371e3, in m2.
    LonStEd : LIST
        2 element list with entries marking the starting and ending longitudes.
    LatStEd : LIST
        2 element list with entries marking the starting and ending latitudes.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Returns
    -------
    areaWeights : NUMPY ARRAY
        An array of global degree to area weights. The size is dependent on
        input resolution. The sum of the array equals 4 pi radius^2 for 
        sufficiently high resolution, in m2.
    longitudes : NUMPY ARRAY
        An array of longitudes corresponding to areaWeights.
    latitudes : NUMPY ARRAY
        An array of latitudes corresponding to areaWeights.
    totalArea : FLOAT
        Total area, in m3, calculated using the area of surface formula.
    totalAreaCalculated : FLOAT
        Total area, in m3, calculated using the area weights.

    """

    # Create vectors throughout domains and along dimensions
    Y = np.arange(LatStEd[1]-resolution/2, LatStEd[0]-resolution/2, -resolution);
    X = np.arange(LonStEd[0]+resolution/2, LonStEd[1]+resolution/2, resolution);
    if (Y[0] != -Y[-1]) & (LatStEd[1] == -1*LatStEd[0]):
        Y = np.array([resolution/2, -resolution/2])
        while (Y[0]+resolution)<90:
            Y = np.append( Y[0]+resolution, Y)
        while -90 < (Y[-1]-resolution):
            Y = np.append( Y, Y[-1]-resolution)
    if (X[0] != -X[-1]) & (LonStEd[1] == -1*LonStEd[0]):
        X = np.array([-resolution/2, resolution/2])
        while -180<(X[0]-resolution):
            X = np.append( X[0]-resolution, X)
        while (X[-1]+resolution)<180:
            X = np.append( X, X[-1]+resolution)

    # Create meshgrid of latitude and longitudes.
    longitudes, latitudes = np.meshgrid(X,Y);

    # Total area
    totalArea = 4*np.pi*(radius**2);
    
    # Calculate the area weights for this resolution
    areaWeights = np.zeros(np.shape(longitudes));
    for i in range(len(latitudes)):
        areaWeights[i,:] = cellAreaOnSphere(latitudes[i], resolution=resolution, radius=radius);
    totalAreaCalculated = np.sum(np.sum(areaWeights));

    if verbose:
        import matplotlib.pyplot as plt
        #FIXME: Add plotGlobal Function
        ...

    return areaWeights, longitudes, latitudes, totalArea, totalAreaCalculated 


# Define function to calculate the area of a cell on a sphere.
def cellAreaOnSphere(clat, resolution = 1., radius=6371e3):
    """
    Calculate the area of a polygon on a sphere using spherical excess formula.
    
    Parameters
    ----------
    clat : FLOAT
        Center latitude value, in degrees.
    resolution : FLOATlatitudes
        Resolution of cell, in degrees.
    radius : FLOAT
        Radius of the sphere.
        
    Returns
    -------
    float: Area of the polygon on the sphere in square kilometers.
    """
    deltaLat = np.deg2rad(resolution);
    detlaLon = np.deg2rad(resolution);

    area = radius*np.cos(np.deg2rad(clat))*(detlaLon)*(radius*deltaLat);

    return area

#######################################################################
################ Plotting function, might not be used #################
#######################################################################


import copy as cp
import cartopy.crs as ccrs # type: ignore

def plotGlobal(lat, lon, values,
               outputDir = os.getcwd(),
               fidName = "plotGlobal.png",
               cmapOpts={"cmap":"viridis",
                         "cbar-title":"cbar-title",
                         "cbar-range":[0,1],
                         "cbar-levels":[0,0.5,1]},
               pltOpts={"valueType": "Bathymetry",
                        "valueUnits": "m",
                        "plotTitle":"",
                        "plotZeroContour":False,
                        "plotIntegerContours":False,
                        "transparent":False,
                        "region":[-180,180,-90,90],
                        "projection":ccrs.Mollweide(),
                        "regionalZBoundaries":False},
               saveSVG=False,
               savePNG=False):
    """
    plotGlobal function is used to plot global ranging datasets that
    are represented with evenly spaced latitude and longitude values.

    Parameters
    ----------
    lat : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    lon : NUMPY ARRAY
        nx2n array representing cell registered longitudes, in deg,
        ranging from [-180, 180]. Longitudes change from column to column.
    Values : NUMPY ARRAY
        nx2n array representing cell registered geographic data, in [-] units.
    cmapOpts : DICTIONARY
        A set of options to format the color map and bar for the plot
    pltOpts : DICTIONARY
        A set of options to format the plot
    saveSVG : BOOLEAN
        An option to save an SVG output. The default is False.
    savePNG : BOOLEAN
        An option to save an PNG output. The default is False.

    Returns
    -------
    None.
    """
    # Copy values such that the arguments are not changed, if
    # say they are from a class attribute.
    values = cp.deepcopy(values)

    # Start making figure
    ## Create a figure
    fig = plt.figure(figsize=(10, 5))
    
    ## Set projection and extent
    projection = pltOpts.get("projection", ccrs.Mollweide())
    region = pltOpts.get('region', [-180, 180, -90, 90])

    ## Set up the Mollweide projection
    ax = plt.axes(projection=projection)
    if (region[0]==-180)&(region[1]==180)&(region[2]==-90)&(region[3]==90):
        pass
    else:
        ax.set_extent(region, crs=ccrs.PlateCarree())
    
    ## Set default for option to use regional Zvalue boundaries
    regionalZBoundaries = pltOpts.get('regionalZBoundaries', False)
    
    if regionalZBoundaries:
        # Set values outside of region to nan
        values[ ~((lon>=pltOpts["region"][0])&(lon<=pltOpts["region"][1])&(lat>=pltOpts["region"][2])&(lat<=pltOpts["region"][3]))] = np.nan
        # Reset indexing
        cnt = 0;
        for idx in np.unique(values):
            if idx != np.nan:
                values[values==idx] = cnt
                cnt+=1;
        
        cmapOpts["cbar-range"] = [0,np.nanmax(values)]
        
    ## Set if the mesh should be plotted
    meshOpt = pltOpts.get('mesh', True)
        
    ## Set if solid polygons should be plotted for nan values.
    nanSolidPoly        = pltOpts.get("nanSolidPoly", False)
    nanSolidPolyOutline = pltOpts.get("nanSolidPolyOutline", False)

    ## Set if the coastline should be plotted
    coastlinesOpt = pltOpts.get("coastlines", False);

    ## Set option to add contour for zero value
    plotZeroContour = pltOpts.get("plotZeroContour", False);
    
    ## Set default option for plotIntegerContours
    plotIntegerContours = pltOpts.get("plotIntegerContours", False);

    ## Add the plot using pcolormesh
    if meshOpt:
        try:
            mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1],
                                levels = cmapOpts["cbar-levels"],
                                zorder=0)
        except:
            mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1],
                                zorder=0)
    

    ## Add zero value contour
    if plotZeroContour:
        # Set any np.nan values to 0.ccrs
        values[np.isnan(values)] = 0;
        zeroContour = ax.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())

    ## Add solid polygons for nan values
    if nanSolidPoly:
        valuesNan                   = cp.deepcopy(values)
        valuesNan[:]                = np.nan;
        valuesNan[np.isnan(values)] = 1;
        mesh2 = ax.pcolormesh(lon, lat, valuesNan,
                              transform=ccrs.PlateCarree(),
                              cmap='YlOrRd',
                              vmin=0, vmax=3, zorder=1)
    
    ## Add Line around clusters of nan values
    if nanSolidPolyOutline:
        valuesNan                   = cp.deepcopy(values)
        valuesNan[:]                = 0;
        valuesNan[np.isnan(values)] = 1;
        nanContour = ax.contour(lon, lat, valuesNan,
                                levels=[1/2],
                                colors='blue',
                                linewidths=1.1,
                                transform=ccrs.PlateCarree(),
                                zorder=2)

    ## Add contours in integer steps (useful for dividing catagorical data)
    valuesContour = cp.deepcopy(values)
    if plotIntegerContours:
        # Set any np.nan values to 0.
        for i in range(len(np.unique(values))):
            valuesContour[values==i] = 1;
            valuesContour[values!=i] = 0;
            ax.contour(lon, lat, valuesContour,
                       levels=[1/2],
                       colors='black',
                       linewidths=1,
                       transform=ccrs.PlateCarree(),
                       zorder=0)

    ## Add coastlines
    if coastlinesOpt:
        ax.coastlines(color='blue', linewidth=1, zorder=10)

    ## Add a colorbar
    if meshOpt:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
        
        ## Set four tick marks with integer values unless
        sep = 4
        if np.diff(cmapOpts["cbar-range"])[0]<sep:
            c = np.diff(cmapOpts["cbar-range"])/sep
        else:
            c = np.diff(cmapOpts["cbar-range"])//sep
        a = cmapOpts["cbar-range"][0]
        b = cmapOpts["cbar-range"][1]+c/2
        
        cbar.set_ticks(np.arange(a,b,c))
        
        ## Set cbar name
        if regionalZBoundaries:
            cbar.set_label(label="Regional {} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
        else:
            cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
        cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    if (region[0]==-180)&(region[1]==180)&(region[2]==-90)&(region[3]==90):
        gl = ax.gridlines(draw_labels=False,
                          crs=ccrs.PlateCarree(),
                          xlocs=np.arange(region[0], region[1]+1, 30),
                          ylocs=np.arange(region[2], region[3]+1, 30)
                         )
    else:
        gl = ax.gridlines(draw_labels=True,
                          crs=ccrs.PlateCarree(),
                          xlocs=np.arange(region[0], region[1]+1, 20),
                          ylocs=np.arange(region[2], region[3]+1, 20)
                         )

        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_left = True
        gl.ylabels_right = False

        # Set the tick label color here
        gl.xlabel_style = {"color": "red", "size": 10, "rotation": 0}
        gl.ylabel_style = {"color": "red", "size": 10, "rotation": 0}

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    ## Set transparency value
    try:
        pltOpts["transparent"];
    except:
        pltOpts["transparent"] = False;

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600, transparent=pltOpts["transparent"])
    if saveSVG:
        plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))


def plotGlobalSilhouette(lat, lon, values, BasinIDA,
               outputDir = os.getcwd(),
               fidName = "plotGlobal_silhouette.png",
               cmapOpts={"cmap":"jet",
                         "cbar-title":"cbar-title",
                         "cbar-range":[0,1],
                         "cbar-levels":np.array([0,.25, .5, .7, 1]),
                         "cbar-intervals":np.array([0,.25, .5, .7, 1]),
                         "cbar-level-names":["No", "Weak", "Medium", "Strong"]},
               pltOpts={"valueType": "Silhouette Structure",
                        "valueUnits": "-",
                        "plotTitle":"",
                        "plotZeroContour":False,
                        "nanSolidPoly":True,
                        "nanSolidPolyOutline":True,
                        "plotIntegerContours":True,
                        "regionalZBoundaries":False,
                        "region":[-180,180,-90,90],
                        "projection":ccrs.Mollweide(),
                        "transparent":True},
               saveSVG=False,
               savePNG=False):
    """
    plotGlobalSilhouette function is used to plot global silhouette values
    for each node that is represented with evenly spaced latitude and
    longitude values.

    Parameters
    ----------
    lat : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    lon : NUMPY ARRAY
        nx2n array representing cell registered longitudes, in deg,
        ranging from [-180, 180]. Longitudes change from column to column.
    Values : NUMPY ARRAY
        nx2n array representing cell registered geographic silhouette values, in [-] units.
    BasinIDA : NUMPY ARRAY
        nx2n array representing cell registered geographic basinIDs, in [-] units.
    cmapOpts : DICTIONARY
        A set of options to format the color map and bar for the plot
    pltOpts : DICTIONARY
        A set of options to format the plot
    saveSVG : BOOLEAN
        An option to save an SVG output. The default is False.
    savePNG : BOOLEAN
        An option to save an PNG output. The default is False.

    Returns
    -------
    None.
    """
    # Copy values such that the arguments are not changed, if
    # say they are from a class attribute.
    values = cp.deepcopy(values)

    # Start making figure
    ## Create a figure
    fig = plt.figure(figsize=(10, 5))
    
    ## Set projection and extent
    projection = pltOpts.get("projection", ccrs.Mollweide())
    region = pltOpts.get('region', [-180, 180, -90, 90])

    ## Set up the Mollweide projection
    ax = plt.axes(projection=projection)
    if (region[0]==-180)&(region[1]==180)&(region[2]==-90)&(region[3]==90):
        pass
    else:
        ax.set_extent(region, crs=ccrs.PlateCarree())
    
    ## Set default for option to use regional Zvalue boundaries
    regionalZBoundaries = pltOpts.get('regionalZBoundaries', False)
    
    if regionalZBoundaries:
        # Set values outside of region to nan
        values[ ~((lon>=pltOpts["region"][0])&(lon<=pltOpts["region"][1])&(lat>=pltOpts["region"][2])&(lat<=pltOpts["region"][3]))] = np.nan
        # Reset indexing
        cnt = 0;
        for idx in np.unique(values):
            if idx != np.nan:
                values[values==idx] = cnt
                cnt+=1;
        
        cmapOpts["cbar-range"] = [0,np.nanmax(values)]
        
    ## Set if the mesh should be plotted
    meshOpt = pltOpts.get('mesh', True)
        
    ## Set if solid polygons should be plotted for nan values.
    nanSolidPoly        = pltOpts.get("nanSolidPoly", False)
    nanSolidPolyOutline = pltOpts.get("nanSolidPolyOutline", False)

    ## Set if the coastline should be plotted
    coastlinesOpt = pltOpts.get("coastlines", False);

    ## Set option to add contour for zero value
    plotZeroContour = pltOpts.get("plotZeroContour", False);
    
    ## Set default option for plotIntegerContours
    plotIntegerContours = pltOpts.get("plotIntegerContours", False);
    
    ## Get cmap at 'cbar-intervals'
    cmapOpts["cmap"] =  plt.cm.get_cmap(cmapOpts["cmap"])
    
    ## Add the plot using pcolormesh
    if meshOpt:
        try:
            mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                levels = cmapOpts["cbar-levels"],
                                norm = mpl.colors.BoundaryNorm(cmapOpts["cbar-levels"], ncolors=cmapOpts["cmap"].N, clip=False),
                                zorder=0)
                                # vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1],
        except:
            mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                norm= mpl.colors.BoundaryNorm(cmapOpts["cbar-levels"], ncolors=cmapOpts["cmap"].N, clip=False),
                                zorder=0)
                                #vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1],
    

    ## Add zero value contour
    if plotZeroContour:
        # Set any np.nan values to 0.ccrs
        values[np.isnan(values)] = 0;
        zeroContour = ax.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())

    ## Add solid polygons for nan values
    if nanSolidPoly:
        valuesNan                   = cp.deepcopy(BasinIDA)
        valuesNan[:]                = np.nan;
        valuesNan[np.isnan(BasinIDA)] = 1;
        mesh2 = ax.pcolormesh(lon, lat, valuesNan,
                              transform=ccrs.PlateCarree(),
                              cmap='YlOrRd',
                              vmin=0, vmax=3, zorder=1)
    
    ## Add Line around clusters of nan values
    if nanSolidPolyOutline:
        valuesNan                   = cp.deepcopy(BasinIDA)
        valuesNan[:]                = 0;
        valuesNan[np.isnan(BasinIDA)] = 1;
        nanContour = ax.contour(lon, lat, valuesNan,
                                levels=[1/2],
                                colors='blue',
                                linewidths=1.1,
                                transform=ccrs.PlateCarree(),
                                zorder=2)

    ## Add contours in integer steps (useful for dividing catagorical data)
    valuesContour = cp.deepcopy(BasinIDA)
    if plotIntegerContours:
        # Set any np.nan values to 0.
        for i in range(len(np.unique(BasinIDA))):
            valuesContour[BasinIDA==i] = 1;
            valuesContour[BasinIDA!=i] = 0;
            ax.contour(lon, lat, valuesContour,
                       levels=[1/2],
                       colors='black',
                       linewidths=1,
                       transform=ccrs.PlateCarree(),
                       zorder=1)

    ## Add coastlines
    if coastlinesOpt:
        ax.coastlines(color='blue', linewidth=1, zorder=10)

    ## Add a colorbar
    if meshOpt:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
        
        # Define silhouette values that are 
        a = cmapOpts["cbar-levels"]
        cbarTicks = np.diff(a)/2+a[:-1]
        cbarTickNames = cmapOpts["cbar-level-names"]
        
        # Set tick marks and names
        cbar.set_ticks(cbarTicks)
        cbar.set_ticklabels(cbarTickNames)

        ## Set cbar name
        if regionalZBoundaries:
            cbar.set_label(label="Regional {} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
        else:
            cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
        cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    if (region[0]==-180)&(region[1]==180)&(region[2]==-90)&(region[3]==90):
        gl = ax.gridlines(draw_labels=False,
                          crs=ccrs.PlateCarree(),
                          xlocs=np.arange(region[0], region[1]+1, 30),
                          ylocs=np.arange(region[2], region[3]+1, 30)
                         )
    else:
        gl = ax.gridlines(draw_labels=True,
                          crs=ccrs.PlateCarree(),
                          xlocs=np.arange(region[0], region[1]+1, 20),
                          ylocs=np.arange(region[2], region[3]+1, 20)
                         )

        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_left = True
        gl.ylabels_right = False

        # Set the tick label color here
        gl.xlabel_style = {"color": "red", "size": 10, "rotation": 0}
        gl.ylabel_style = {"color": "red", "size": 10, "rotation": 0}

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    ## Set transparency value
    try:
        pltOpts["transparent"];
    except:
        pltOpts["transparent"] = False;

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600, transparent=pltOpts["transparent"])
    if saveSVG:
        plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))





















def plotGlobalOld(lat, lon, values,
               outputDir = os.getcwd(),
               fidName = "plotGlobal.png",
               cmapOpts={"cmap":"viridis",
                         "cbar-title":"cbar-title",
                         "cbar-range":[0,1]},
               pltOpts={"valueType": "Bathymetry",
                        "valueUnits": "m",
                        "plotTitle":"",
                        "plotZeroContour":False,
                        "plotIntegerContours":False,
                        "transparent":False,
                        "Region":[-90,90,-180,180]},
               saveSVG=False,
               savePNG=False):
    """
    plotGlobal function is used to plot global ranging datasets that
    are represented with evenly spaced latitude and longitude values.

    Parameters
    ----------
    lat : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    lon : NUMPY ARRAY
        nx2n array representing cell registered longitudes, in deg,
        ranging from [-180, 180]. Longitudes change from column to column.
    Values : NUMPY ARRAY
        nx2n array representing cell registered geographic data, in [-] units.
    cmapOpts : DICTIONARY
        A set of options to format the color map and bar for the plot
    pltOpts : DICTIONARY
        A set of options to format the plot
    saveSVG : BOOLEAN
        An option to save an SVG output. The default is False.
    savePNG : BOOLEAN
        An option to save an PNG output. The default is False.

    Returns
    -------
    None.
    """
    # Copy values such that the arguments are not changed, if
    # say they are from a class attribute.
    values = cp.deepcopy(values)

    # Start making figure
    ## Create a figure
    fig = plt.figure(figsize=(10, 5))

    ## Set up the Mollweide projection
    ax = plt.axes(projection=ccrs.Mollweide())

    ## Set if the mesh should be plotted
    try: 
        pltOpts["mesh"];
    except:
        pltOpts["mesh"] = True;
        
    ## Set if solid polygons should be plotted for nan values.
    try:
        pltOpts["nanSolidPoly"];
    except:
        pltOpts["nanSolidPoly"] = False;
    try:
        pltOpts["nanSolidPolyOutline"];
    except:
        pltOpts["nanSolidPolyOutline"] = False;

    ## Set if the coastline should be plotted
    try: 
        pltOpts["coastlines"];
    except:
        pltOpts["coastlines"] = False;

    ## Add the plot using pcolormesh
    if pltOpts["mesh"]:
        mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                            vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1],
                            zorder=0)

    ## Add zero contour value
    try:
        if pltOpts["plotZeroContour"]:
            # Set any np.nan values to 0.ccrs
            values[np.isnan(values)] = 0;
            zeroContour = ax.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())
    except:
        # Case where pltOpts["plotZeroContour"] was not defined
        pass

    ## Add solid polygons for nan values
    try:
        if pltOpts["nanSolidPoly"]:
            valuesNan                   = cp.deepcopy(values)
            valuesNan[:]                = np.nan;
            valuesNan[np.isnan(values)] = 1;
            mesh2 = ax.pcolormesh(lon, lat, valuesNan,
                                  transform=ccrs.PlateCarree(),
                                  cmap='YlOrRd',
                                  vmin=0, vmax=3, zorder=1)
    except:
        # Case where pltOpts["nanSolidPoly"] was not defined
        pass
    
    ## Add Line around clusters of nan values
    try:
        if pltOpts["nanSolidPolyOutline"]:
            valuesNan                   = cp.deepcopy(values)
            valuesNan[:]                = 0;
            valuesNan[np.isnan(values)] = 1;
            nanContour = ax.contour(lon, lat, valuesNan,
                                    levels=[1/2],
                                    colors='blue',
                                    linewidths=1.1,
                                    transform=ccrs.PlateCarree(),
                                    zorder=2)
    except:
        # Case where pltOpts["nanSolidPoly"] was not defined
        pass

    ## Add contours in integer steps (useful for dividing catagorical data)
#     try:
#         valuesContour = cp.deepcopy(values)
#         if pltOpts["plotIntegerContours"]:
#             # Set any np.nan values to 0.
#             values[np.isnan(values)] = -1;
#             zeroContour = ax.contour(lon, lat, values,
#                                      levels=np.arange(len(np.unique(values)))-1/2,
#                                      colors='black',
#                                      linewidths=1,
#                                      transform=ccrs.PlateCarree(),
#                                      zorder=0)
#     except:
#         # Case where pltOpts["plotIntegerContours"] was not defined
#         pass
    try:
        valuesContour = cp.deepcopy(values)
        if pltOpts["plotIntegerContours"]:
            # Set any np.nan values to 0.
            for i in range(len(np.unique(values))):
                valuesContour[values==i] = 1;
                valuesContour[values!=i] = 0;
                ax.contour(lon, lat, valuesContour,
                           levels=[1/2],
                           colors='black',
                           linewidths=1,
                           transform=ccrs.PlateCarree(),
                           zorder=0)
    except:
        # Case where pltOpts["plotIntegerContours"] was not defined
        pass

    ## Add coastlines
    if pltOpts["coastlines"]:
        ax.coastlines(color='blue', linewidth=1, zorder=10)

    ## Add a colorbar
    if pltOpts["mesh"]:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
        cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
        cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    ax.gridlines()

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    ## Set transparency value
    try:
        pltOpts["transparent"];
    except:
        pltOpts["transparent"] = False;

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600, transparent=pltOpts["transparent"])
    if saveSVG:
        plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))



def plotRegional(lat, lon, values,
               outputDir = os.getcwd(),
               fidName = "plotRegional.png",
               cmapOpts={"cmap":"viridis",
                         "cbar-title":"cbar-title",
                         "cbar-range":[0,1]},
               pltOpts={"valueType": "Bathymetry",
                        "valueUnits": "m",
                        "plotTitle":"",
                        "plotZeroContour":False,
                        "plotIntegerContours":False,
                        "transparent":False,
                        "Region":[-180,180,-90,90]},
               saveSVG=False,
               savePNG=False):
    """
    plotRegional function is used to plot global ranging datasets that
    are represented with evenly spaced latitude and longitude values on
    a regional map.

    Parameters
    ----------
    lat : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    lon : NUMPY ARRAY
        nx2n array representing cell registered longitudes, in deg,
        ranging from [-180, 180]. Longitudes change from column to column.
    Values : NUMPY ARRAY
        nx2n array representing cell registered geographic data, in [-] units.
    cmapOpts : DICTIONARY
        A set of options to format the color map and bar for the plot
    pltOpts : DICTIONARY
        A set of options to format the plot. Defaults to plotting a
        global map.
    saveSVG : BOOLEAN
        An option to save an SVG output. The default is False.
    savePNG : BOOLEAN
        An option to save an PNG output. The default is False.

    Returns
    -------
    None.
    """
    # FIXME: 
    print("working progress")
    return
    # Copy values such that the arguments are not changed, if
    # say they are from a class attribute.
    values = cp.deepcopy(values)

    # Start making figure
    ## Create a figure
    fig = plt.figure(figsize=(10, 5))

    ## Set up the Mollweide projection
    ax = plt.axes(projection=ccrs.Mollweide())

    ## Add the plot using pcolormesh
    mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                         vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1])

    ## Add zero contour value
    try:
        if pltOpts["plotZeroContour"]:
            # Set any np.nan values to 0.ccrs
            values[np.isnan(values)] = 0;
            zeroContour = ax.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())
    except:
        # Case where pltOpts["plotZeroContour"] was not defined
        pass

    ## Add contours in integer steps (useful for dividing catagorical data)
    try:
        if pltOpts["plotIntegerContours"]:
            # Set any np.nan values to 0.
            values[np.isnan(values)] = 0;
            zeroContour = ax.contour(lon, lat, values, levels=np.arange(len(np.unique(values)))+1/2, colors='black', transform=ccrs.PlateCarree())
    except:
        # Case where pltOpts["plotIntegerContours"] was not defined
        pass

    ## Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
    cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
    cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    ax.gridlines()

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    ## Set transparency value
    try:
        pltOpts["transparent"];
    except:
        pltOpts["transparent"] = False;

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600, transparent=pltOpts["transparent"])
    if saveSVG:
        plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))


def plotGlobalwHist(lat, lon, values,
                    binEdges, bathymetryAreaDist_wHighlat, bathymetryAreaDist, highlatlat,
                    outputDir = os.getcwd(),
                    fidName = "plotGlobal.png",
                    cmapOpts={"cmap":"viridis",
                                "cbar-title":"cbar-title",
                                "cbar-range":[0,1]},
                    pltOpts={"valueType": "Bathymetry",
                                "valueUnits": "m",
                                "plotTitle":"",
                                "plotZeroContour":False},
                    saveSVG=False,
                    savePNG=False):
    """
    plotGlobal function is used to plot global ranging datasets that
    are represented with evenly spaced latitude and longitude values.

    Parameters
    ----------
    lat : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    lon : NUMPY ARRAY
        nx2n array representing cell registered longitudes, in deg,
        ranging from [-180, 180]. Longitudes change from column to column.
    Values : NUMPY ARRAY
        nx2n array representing cell registered geographic data, in [-] units.
    cmapOpts : DICTIONARY
        A set of options to format the color map and bar for the plot
    pltOpts : DICTIONARY
        A set of options to format the plot
    saveSVG : BOOLEAN
        An option to save an SVG output. The default is False.
    savePNG : BOOLEAN
        An option to save an PNG output. The default is False.

    Returns
    -------
    None.
    """
    # Start making figure
    ## Create a figure
    
    ## Set up the Mollweide projection
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1]);  # 2 rows, 1 column, with the first row 3 times taller

    ax1 = fig.add_subplot(gs[0], projection=ccrs.Mollweide());

    ## Add the plot using pcolormesh
    mesh = ax1.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                         vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1])
    if pltOpts["plotZeroContour"]:
        # Set any np.nan values to 0.
        values[np.isnan(values)] = 0;
        zeroContour = ax1.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())

    ## Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax1, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
    cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
    cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    ax1.gridlines()

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    ## Make histogram plot
    ax2 = fig.add_subplot(gs[1]);
    
    factor1 = .2
    factor2 = .25
    plt.bar(x=binEdges[1:]-(factor2/2)*np.diff(binEdges),
            height=bathymetryAreaDist_wHighlat,
            width=factor1*np.diff(binEdges),
            label= "All bathymetry")
    plt.bar(x=binEdges[1:]+(factor2/2)*np.diff(binEdges),
            height=bathymetryAreaDist,
            width=factor1*np.diff(binEdges),
            label= "Removed high latitude bathymetry: {:2.1f} degrees".format(highlatlat))
    # ticks
    plt.xticks(binEdges[1:]);
    plt.yticks(np.arange(0,35,5));

    # Labels
    plt.legend();
    plt.title("Planet's Bathymetry Distribution")
    plt.xlabel("Bathymetry Bins [km]");
    plt.ylabel("Seafloor Area [%]");

    # figure format
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600)
    if saveSVG:
        plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))


def plotLargeBasins(basins, fieldNum="Field1", percentage=0.05, fldName=os.getcwd(), savePNG=False, verbose=True):
    """
    Function is use to plot basins larger than X% of surface area of
    where data is represented.
    
    Parameters
    -----------
    basins : EXOCCYCLE OBJECT
        Includes lat, lon, BasinIDA, and Fields attributes.
    fieldNum : STRING
        Name of field to plot. Information of the
        field is stored in basins.Fields[{FieldNum}].
        The default is Field 1.
    percentage : percentage
        Decimal percentage of area that the community
        must cover such that it is plotted. The default
        is 5 %.
    fldName : STRING
        Directory to store plotted png image within.
        The default is the current working directory.
    savePNG : BOOLEAN
        Option to save png of plotted large basins.
        The default is False
    verbose : BOOLEAN
        Outputs additional information on communities to be
        plotted. The default is True.
        
    Returns
    --------
    None.
    
    """
    BasinIDAMod = cp.deepcopy(basins.BasinIDA)
    # Make weights mask
    areaWeights = cp.deepcopy(basins.areaWeights)
    areaWeights[np.isnan(basins.BasinIDA)] = 0;
    TotalArea = np.nansum(areaWeights)
    if verbose:
        print(f"TotalArea: {TotalArea}")
    
    # Iterate over basins (communitieslen()
    keepBasin = np.ones(len(np.unique(BasinIDAMod)))
    for basinID in np.unique(BasinIDAMod):
        if np.isnan(basinID):
            continue;
        if np.sum(areaWeights[BasinIDAMod==basinID]) < TotalArea*percentage:
            # If total basin area is smaller than the percentage threshold. 
            keepBasin[int(basinID)] = 0;
            if verbose:
                print(f"basinID: {basinID}")
    if verbose:
        print(f"Smallest community plotted {TotalArea*percentage} km^2" )
        print(f"Basins to be plotted (>100 will take more than a minute): {np.sum(keepBasin)}")
        
    # Remove small basins
    print(f"len(np.unique(BasinIDAMod)): {len(np.unique(BasinIDAMod))}")
    for basinID in range(len(keepBasin)):
        if not keepBasin[basinID]:
            BasinIDAMod[BasinIDAMod==basinID] = np.nan;

    # Reindex large basins basins
    cnt=0;
    for basinID in np.unique(BasinIDAMod):
        BasinIDAMod[BasinIDAMod==basinID] = cnt
        cnt+=1
        
    # Plot
    plotGlobal(basins.lat, basins.lon, BasinIDAMod,
               outputDir = fldName,
               fidName = "plotGlobal_LargeBasins.png",
               cmapOpts={"cmap":"jet",
                         "cbar-title":"cbar-title",
                         "cbar-range":[0,np.nanmax(BasinIDAMod)]},
               pltOpts={"valueType": "BasinID divided by {}".format(basins.Fields[fieldNum]['parameterName']),
                        "valueUnits": "-",
                        "plotTitle":"",
                        "plotZeroContour":False,
                        "plotIntegerContours":True,
                        "transparent":True},
               savePNG=savePNG,
               saveSVG=False)


#######################################################################
###################### Basin definition functions #####################
#######################################################################
#import numpy as np
#import plotly.graph_objects as go
class eaNodes():

    def __init__(self, inputs={"undefined":True}, precalculated=False, precalculate=False):
        '''
        Initiation of eaNodes class that is used for creating a grid of
        nodes that represent centroids of equal area diamond shape regions
        on the surface of a sphere.

        Parameters
        -----------
        inputs : DICTIONARY
            inputs["resolution"] : FLOAT
                Spatial resolution in degrees (i.e., node spacing)
            inputs["dataGrid"] : STRING
                Path to input data grid
        precalculated : BOOLEAN
            An option to read a precalculated set of node locations, connections,
            and other eaNodes object defined values. The default is False.
        precalculate : BOOLEAN
            An option to write a precalculated set of node locations, connections,
            and other eaNodes object defined values. The default is False.

        Relevant Class Attributes
        --------------------------
        self.ealat : NUMPY VECTOR
            Equal area node locations, latitude in degrees. 
        self.ealon : NUMPY VECTOR
            Equal area node locations, longitude in degrees.
        self.connectionNodeDis : NUMPY ARRAY
            nodeCnt x 5 array with columns indicating 1) (nodei) with
            2) connection 1 distance 3) connection 2 distance 4)
            connection 3 distance, and 5) connection 4 distance.
        self.connectionNodeIDs : NUMPY ARRAY
            nodeCnt x 5 array with columns indicating 1) (nodei) with
            2) connection 1 3) connection 2 4) connection 3, and 5)
            connection 4.

        '''
        # Assign inputs
        self.inputs = inputs

        # Assign class attributes
        try:
            if inputs["undefined"]:
                self.resolution = 5;
                self.dataGrid   = "/home/bogumil/Documents/data/Muller_etal_2019_Tectonics_v2.0_netCDF/Muller_etal_2019_Tectonics_v2.0_AgeGrid-0.nc";
                self.interpGrid = "EA_Nodes_{}_LatLon.txt".format(self.resolution);
                self.output     = "Muller_etal_2019_Tectonics_v2.0_AgeGrid-0_EASampled.nc";
        except:
            self.resolution = self.inputs["resolution"];
            self.dataGrid   = self.inputs["dataGrid"];

        # Create directory to store precalculated files within
        if not os.path.exists(os.getcwd()+"/eaNodes"):
            os.mkdir(os.getcwd()+"/eaNodes")

        # Assign more class attributes (filepaths)
        self.filename1  = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'
        self.filename2  = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt'
        self.interpGrid = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt';
        self.output     = os.getcwd()+"/eaNodes/EASampled.txt";


        # Define all attributes assigned to object.
        self.color  = None; # Holds the colors used to distinguish between regions equal-area points on a sphere. 
        self.hist   = None; #

        # Equal area node locations
        self.ealon = None;  # Equal area node longitude
        self.ealat = None;  # Equal area node latitude
        
        # Set read/write options
        self.precalculated = precalculated;
        self.precalculate  = precalculate;
        
        # Equal area node interpolated data
        # This dictionary can be added to with self.interp2IrregularGrid(...). 
        self.data = {};

        # Try to read precalculated node grid 
        if self.precalculated:
            self.precalculated = self.save_or_load_attributes(mode="r")


    def save_or_load_attributes(self, mode="w"):
        """
        Save or load all class attributes to/from a single .npz file.

        Parameter
        ----------
        mode : STRING
            'w' to wrtie attributes, 'r' to read attributes.

        File
        -----
            File will be named 'EA_Nodes_{self.resolution}.npz'
        """
        # Create filename
        filename = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'.replace("_xyz.txt", ".npz")

        if mode == "w":
            attr_dict = {
                'inputs': self.inputs,
                'resolution': self.resolution,
                'dataGrid': self.dataGrid,
                'interpGrid': self.interpGrid,
                'output': self.output,
                'filename1': self.filename1,
                'filename2': self.filename2,
                'color': self.color,
                'hist': self.hist,
                'ealon': self.ealon,
                'ealat': self.ealat,
                'data': self.data,
                'connectionNodeIDs': self.connectionNodeIDs,
                'connectionNodeDis': self.connectionNodeDis
            }
            # Save using numpy savez (allow_pickle needed for objects like dicts)
            np.savez_compressed(filename, **attr_dict)
            print(f"Attributes saved to {filename}")

            return True

        elif mode == "r":
            if not os.path.exists(filename):
                print(f"File {filename} does not exist. Cannot load. {filename} will be created for current use and written (precalculated) for future use at {self.resolution:0.1f} degree spatial resolution.")
                return False
            
            print(f"File {filename} does exist. loading. {filename} will be read for current use.")

            npzfile = np.load(filename, allow_pickle=True)

            # Assign attributes from file
            self.inputs = npzfile['inputs'].item()
            self.resolution = npzfile['resolution'].item()
            self.dataGrid = npzfile['dataGrid'].item()
            #self.interpGrid = npzfile['interpGrid'].item()
            #self.output = npzfile['output'].item()
            #self.filename1 = npzfile['filename1'].item()
            #self.filename2 = npzfile['filename2'].item()
            self.color = npzfile['color']
            self.hist = npzfile['hist'].item()
            self.ealon = npzfile['ealon']
            self.ealat = npzfile['ealat']
            self.data = npzfile['data'].item()
            self.connectionNodeIDs = npzfile['connectionNodeIDs']
            self.connectionNodeDis = npzfile['connectionNodeDis']

            # Assign more class attributes (filepaths)
            # These are not defined with the values stored in the
            # .npz file since the .npz file might have been constructed
            # on a different system.
            self.filename1  = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'
            self.filename2  = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt'
            self.interpGrid = os.getcwd()+"/eaNodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt';
            self.output     = os.getcwd()+"/eaNodes/EASampled.txt";

            print(f"Attributes loaded from {filename}")

            return True
        else:
            print("Mode must be 'w' or 'r'.")


    def lonlat2xyz(self, longitude, latitude, radius=1):
        """
        Convert latitude and longitude on a sphere to XYZ coordinates.

        Parameters:
        longitude (float): Longitude in degrees
        latitude (float): Latitude in degrees
        radius (float): Radius of the sphere (default is Earth's mean radius in meters)

        Returns:
        tuple: (X, Y, Z) coordinates
        """
        # Convert degrees to radians
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)

        # Compute XYZ coordinates
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)

        return x, y, z

    def xyz2lonlat(self, x, y, z, radius=1):
        """
        Convert XYZ coordinates on a sphere to latitude and longitude.

        Parameters:
        x (float): X coordinate
        y (float): Y coordinate
        z (float): Z coordinate
        radius (float): Radius of the sphere (default is Earth's mean radius in meters)

        Returns:
        tuple: (latitude in degrees, longitude in degrees)
        """

        # Compute longitude (lambda)
        longitude = np.degrees(np.arctan2(y, x))

        # Compute latitude (phi)
        hyp = np.sqrt(x**2 + y**2)  # Projection on the equatorial plane
        latitude = np.degrees(np.arctan2(z, hyp))

        return longitude, latitude


    def rotate_around_vec_by_a(self, A, x, y, z):
        L = np.sqrt(x**2 + y**2 + z**2)
        xU, yU, zU = x/L, y/L, z/L
        cos_A, sin_A = np.cos(A), np.sin(A)
        
        R = np.array([
            [xU**2 * (1 - cos_A) + cos_A, xU * yU * (1 - cos_A) - zU * sin_A, xU * zU * (1 - cos_A) + yU * sin_A],
            [xU * yU * (1 - cos_A) + zU * sin_A, yU**2 * (1 - cos_A) + cos_A, yU * zU * (1 - cos_A) - xU * sin_A],
            [xU * zU * (1 - cos_A) - yU * sin_A, yU * zU * (1 - cos_A) + xU * sin_A, zU**2 * (1 - cos_A) + cos_A]
        ])
        return R

    def rotate_vector(self, from_vec, to_vec, by_angle):
        axis_vec = np.cross(from_vec, to_vec)
        R = self.rotate_around_vec_by_a(by_angle, *axis_vec)
        return R @ from_vec


    def makegrid(self, plotq=0):
        '''
        makegrid creates a grid of nodes that represent centroids of
        equal area diamond shape regions on the surface of a sphere,
        and determines the edges (links) between adjacent nodes.
        
        Re(defined)
        ------------
        self.ealat : NUMPY VECTOR
            Equal area node locations, latitude in degrees. 
        self.ealon : NUMPY VECTOR
            Equal area node locations, longitude in degrees.
        self.connectionNodeDis : NUMPY ARRAY
            nodeCnt x 5 array with columns indicating 1) (nodei) with
            2) connection 1 distance 3) connection 2 distance 4)
            connection 3 distance, and 5) connection 4 distance.
        self.connectionNodeIDs : NUMPY ARRAY
            nodeCnt x 5 array with columns indicating 1) (nodei) with
            2) connection 1 3) connection 2 4) connection 3, and 5)
            connection 4.

        '''
        nelsedge = 2 * int(np.ceil(54 / (2 * self.resolution)))
        dtor = np.pi / 180.0
        radius = 1.0
        subtend = 54.735610

        ###################
        #### Section 1 ####
        ###################
        # Define the coordinates of the four corners of the diamond, and calculate
        # the internal points
        #    1
        #    /\  
        # 2 /  \ 4
        #   \  / 
        #    \/ 
        #    3
        
        cS = np.array([
            [radius, 90.0 * dtor,               0.00],
            [radius, (90.0 - subtend) * dtor,   0],
            [radius, 0 * dtor,                  45.0 * dtor],
            [radius, (90.0 - subtend) * dtor,   90.0 * dtor]
        ])
        
        c = np.column_stack((
            cS[:, 0] * np.cos(cS[:, 1]) * np.cos(cS[:, 2]),
            cS[:, 0] * np.cos(cS[:, 1]) * np.sin(cS[:, 2]),
            cS[:, 0] * np.sin(cS[:, 1])
        ))

        if plotq:

            # Create the unit sphere data
            r =.95;
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = r*np.outer(np.cos(u), np.sin(v))
            y = r*np.outer(np.sin(u), np.sin(v))
            z = r*np.outer(np.ones(np.size(u)), np.cos(v))

            # Create the figure
            fig = go.Figure()

            # Set the layout
            fig.update_layout(
                title='Unit Equal Area Spaced Node',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    xaxis=dict(range=[-1.2, 1.2], visible=False),
                    yaxis=dict(range=[-1.2, 1.2], visible=False),
                    zaxis=dict(range=[-1.2, 1.2], visible=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    
                )
            )
            # Add the sphere surface
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='greys', opacity=0.8, colorbar=dict(tickvals=[-1,0,1])))


        kcolor = ["rgb(0,0,255)", "rgb(0,255,0)", "rgb(0,0,255)", "rgb(0,255,0)",
                "rgb(255,0,0)", "rgb(0,255,255)", "rgb(255,0,0)", "rgb(0,255,255)",
                "rgb(255,0,255)", "rgb(255,255,0)", "rgb(255,0,255)", "rgb(255,255,0)"];
        
        node_num = 0
        EAgrid = np.zeros((12, nelsedge + 1, nelsedge + 1, 4))
        colorGrid = np.zeros((12, nelsedge + 1, nelsedge + 1));
        colorGrid = colorGrid.astype(str);

        
        vec_from, vec_to = c[0], c[1]
        A14 = np.arccos(np.dot(c[0], c[3]) / (np.linalg.norm(c[0]) * np.linalg.norm(c[3]))) / nelsedge
        A23 = np.arccos(np.dot(c[1], c[2]) / (np.linalg.norm(c[1]) * np.linalg.norm(c[2]))) / nelsedge
        
        for i in range(nelsedge + 1):
            angle_in = np.arccos(np.dot(vec_from, vec_to) / (np.linalg.norm(vec_from) * np.linalg.norm(vec_to)))
            A = angle_in / nelsedge
            new_vec = vec_from.copy()
            
            for j in range(nelsedge + 1):
                node_num += 1
                EAgrid[0, i, j] = [node_num, *new_vec]
                new_vec = self.rotate_vector(vec_from, vec_to, A * (j + 1))
                colorGrid[0, i, j] = kcolor[0];
            
            vec_to = self.rotate_vector(c[1], c[2], A23 * (i + 1))
            vec_from = self.rotate_vector(c[0], c[3], A14 * (i + 1))

        
        ###################
        #### Section 2 ####
        ###################
        # Rotate the grid to cover the sphere
        # |   |   |   |   |
        # | 3 | 4 | 1 | 2 |
        #  \ / \ / \ / \ /
        #   X 12X 9 X 10X 11
        #  / \ / \ / \ / \
        # | 7 | 8 | 5 | 6 | 
        # |   |   |   |   |
        #

        # Matrix of rotations to be used for rotating diamond shape
        # of nodes across a sphere.
        Z90 =   self.rotate_around_vec_by_a(90*dtor,     0, 0, 1);
        ktotal = 12;
        rotations = np.zeros((ktotal+1,np.shape(Z90)[0],np.shape(Z90)[1]),dtype=float)


        rotations[2,:,:]    = Z90;
        rotations[3,:,:]    = Z90;
        rotations[4,:,:]    = Z90;
        rotations[5,:,:]    = self.rotate_around_vec_by_a(180*dtor,    1, 1, 0);
        rotations[6,:,:]    = self.rotate_around_vec_by_a(180*dtor, -1, 1, 0);
        rotations[7,:,:]    = self.rotate_around_vec_by_a(180*dtor, -1, -1, 0);
        rotations[8,:,:]    = self.rotate_around_vec_by_a(180*dtor, 1, -1, 0);
        rotations[9,:,:]    = self.rotate_around_vec_by_a(90*dtor, 1, 1, 0);
        rotations[10,:,:]   = self.rotate_around_vec_by_a(90*dtor, -1, 1, 0);
        rotations[11,:,:]   = self.rotate_around_vec_by_a(90*dtor, -1, -1, 0);
        rotations[12,:,:]   = self.rotate_around_vec_by_a(90*dtor, 1, -1, 0);

        duplicateCondition = ["","",
                            "i > 0",
                            "i > 0",
                            "(i > 0) & (j > 0)",
                            "(i == nelsedge) & (j == nelsedge)",
                            "(j == 0) | ((i == nelsedge) & (j == nelsedge))",
                            "(j == 0) | ((i == nelsedge) & (j == nelsedge))",
                            "((i == 0) | (j == 0)) | ((i == nelsedge) & (j == nelsedge))",
                            "(i > 0) & (i < nelsedge) & (j > 0) & (j < nelsedge)",
                            "(i > 0) & (i < nelsedge) & (j > 0) & (j < nelsedge)",
                            "(i > 0) & (i < nelsedge) & (j > 0) & (j < nelsedge)",
                            "(i > 0) & (i < nelsedge) & (j > 0) & (j < nelsedge)"];


        # Rotate through 90degs 3 times for 2-4:
        for k in np.array([2,3,4]):
            for i in range(nelsedge+1):
                for j in range(nelsedge+1):
                    if eval(duplicateCondition[k]):
                        node_num += 1;
                        EAgrid[k-1,i,j,0] = node_num;
                        colorGrid[k-1,i,j] = kcolor[k-1];
                    else:
                        # Mark duplicate points
                        EAgrid[k-1,i,j,0] = -1;
                        colorGrid[k-1,i,j] = 'k'
                    
                    vec_from[0] = EAgrid[(k-2),i,j,1];
                    vec_from[1] = EAgrid[(k-2),i,j,2];
                    vec_from[2] = EAgrid[(k-2),i,j,3];

                    
                    vec_rotated = np.dot(rotations[k,:,:], vec_from.T);
                    
                    EAgrid[k-1,i,j,1] = vec_rotated[0];
                    EAgrid[k-1,i,j,2] = vec_rotated[1];
                    EAgrid[k-1,i,j,3] = vec_rotated[2];
                    


        # Rotate each diamond around it's equatorial corner by 180 degs for 5-8
        for k in np.array([5,6,7,8]):
            for i in range(nelsedge+1):
                for j in range(nelsedge+1):
                    if eval(duplicateCondition[k]):
                        # Mark duplicate points
                        EAgrid[k-1,i,j,0] = -1;  
                        colorGrid[k-1,i,j] = 'k';
                    else:
                        node_num += 1;
                        EAgrid[k-1,i,j,0] = node_num;  
                        colorGrid[k-1,i,j] = kcolor[k-1];
                        
                        vec_from[0] = EAgrid[(k-5),i,j,1];
                        vec_from[1] = EAgrid[(k-5),i,j,2];
                        vec_from[2] = EAgrid[(k-5),i,j,3];
                        
                        vec_rotated = np.dot(rotations[k,:,:], vec_from.T);

                        EAgrid[k-1,i,j,1] = vec_rotated[0];
                        EAgrid[k-1,i,j,2] = vec_rotated[1];
                        EAgrid[k-1,i,j,3] = vec_rotated[2];
                        
        # Rotate diamonds 1-4 around their equatorial corners to generate diamonds 9-12
        for k in np.array([9,10,11,12]):
            for i in range(nelsedge+1):
                for j in range(nelsedge+1):
                    if eval(duplicateCondition[k]):
                        node_num += 1;
                        EAgrid[k-1,i,j,0] = node_num;
                        colorGrid[k-1,i,j] = kcolor[k-1];
                    else:
                        # Mark duplicate points
                        EAgrid[k-1,i,j,0] = -1;    
                        colorGrid[k-1,i,j] = 'k';
                    
                    vec_from[0] = EAgrid[(k-9),i,j,1];
                    vec_from[1] = EAgrid[(k-9),i,j,2];
                    vec_from[2] = EAgrid[(k-9),i,j,3];


                    vec_rotated = np.dot(rotations[k,:,:], vec_from.T);


                    EAgrid[k-1,i,j,1] = vec_rotated[0];
                    EAgrid[k-1,i,j,2] = vec_rotated[1];
                    EAgrid[k-1,i,j,3] = vec_rotated[2];
                    

        ###################
        #### Section 3 ####
        ###################
        # Convert the grid back to spherical coords, and save the results
        # Saving the grid points (but don't save duplicates)
        duplicates = 0;
        with open(self.filename1, 'w') as f:
            for k in range(12):
                for i in range(nelsedge + 1):
                    for j in range(nelsedge + 1):
                        if EAgrid[k, i, j, 0] != -1:
                            x, y, z = EAgrid[k, i, j, 1:]
                            f.write(f'{int(EAgrid[k, i, j, 0])}, {x:.8e}, {y:.8e}, {z:.8e}\n');
                        else:
                            duplicates += 1;
        duplicates = 0;
        with open(self.filename2, 'w') as f:
            for k in range(12):
                for i in range(nelsedge + 1):
                    for j in range(nelsedge + 1):
                        if EAgrid[k, i, j, 0] != -1:
                            x, y, z = EAgrid[k, i, j, 1:]
                            lon, lat = self.xyz2lonlat(x, y, z);
                            f.write(f'{lon:.8e}, {lat:.8e}\n');
                        else:
                            duplicates += 1;

        ###################
        #### Section 4 ####
        ###################
        # Find connectable nodes
        maxEdgeConnections = 4;

        # Read nodes file
        ealocation = np.loadtxt(self.filename2, delimiter=',',usecols=[0,1])
        self.ealon, self.ealat = ealocation[:,0], ealocation[:,1]

        # Define array to hold node connection ids
        self.connectionNodeIDs = np.zeros(shape=(len(ealocation[:,0]), 1+maxEdgeConnections));
        self.connectionNodeIDs[:,0] = np.arange(len(ealocation[:,0]))
        
        # Define array to hold node connection distances
        self.connectionNodeDis = np.zeros(shape=(len(ealocation[:,0]), 1+maxEdgeConnections));
        self.connectionNodeDis[:,0] = np.arange(len(ealocation[:,0]))

        ## Calculate a threshold distance connections must be with at least this distance,
        ## otherwise they are discarded
        distanceThres = haversine_distance(lat1=0,lon1=1.2*self.resolution, lat2=0, lon2=0, radius=1) 

        for i in range(len(ealocation[:,0])):
            # Iterate over all nodes

            # Find the distance from node i to all other nodes
            distance = haversine_distance(lat1=ealocation[i,1],
                                          lon1=ealocation[i,0],
                                          lat2=self.ealat,
                                          lon2=self.ealon,
                                          radius=1);

            # Iterate over the closest maxEdgeConnections+1 nodes. 
            for j in range(maxEdgeConnections+1):
                # Find index of minimum distance node
                distancei = np.nanmin(distance);
                connectionNodeIDi = np.argwhere(distance == distancei)[0][0];

                if j == 1:
                    # Store current value to compare with later
                    # distancej is used to prevent edges being draw to
                    # 4 nodes for 6 nodes that that only have 3 vertices.
                    distancej = cp.deepcopy(distancei);
                
                if j == 0:
                    # Set value to far away (this is the node we are connecting to others)
                    distance[connectionNodeIDi] = np.nanmax(distance);
                    continue
                    #elif (distancei>(distanceThres*1.1)):
                    #    # Set value to far away (this is the node we are connecting to others)
                    #    distance[connectionNodeIDi] = np.nanmax(distance);
                    #    # Define connection ID
                    #    self.connectionNodeIDs[i,j]=None;
                    #    #print("testing: j={}".format(j))
                    #    continue
                elif distancei<distanceThres: 
                    # Define connection ID
                    self.connectionNodeIDs[i,j]=connectionNodeIDi;
                    # Set value to far away (this is the node we are connecting to others)
                    distance[connectionNodeIDi] = np.nanmax(distance);
                    # Store previous value to compare with later
                    self.connectionNodeDis[i,j]=distancei;
                    #distancej = cp.deepcopy(distancei);
                
                # Check if one of the node distances is much larger than others.
                # If so then this is node likely one of the 8 that shares a vertex
                # between 4 diamond regions (i.e., the node should only have 3 connections
                # with neighboring nodes)
                if j == 4:
                    distances = self.connectionNodeDis[i,1:]
                    self.connectionNodeIDs[i,1:][~(self.connectionNodeDis[i,1:]<np.mean(distances) + 1.5*np.std(distances))]=None;
                    self.connectionNodeDis[i,1:][~(self.connectionNodeDis[i,1:]<np.mean(distances) + 1.5*np.std(distances))]=None;
        
        ###################
        #### Section 5 ####
        ###################
        # Plot node locations
        print("Duplicates removed:", duplicates)
        print(f'Grid saved to: {self.filename1}')
        
        duplicateslogical = (EAgrid[:,:,:,0].flatten()==-1);
        if plotq==1:
            # Add the scatter plot
            fig.add_trace(go.Scatter3d(x=EAgrid[:,:,:,1].flatten()[~duplicateslogical],
                                    y=EAgrid[:,:,:,2].flatten()[~duplicateslogical],
                                    z=EAgrid[:,:,:,3].flatten()[~duplicateslogical],
                                    mode='markers',
                                    marker=dict(size=5, color=colorGrid[:,:,:].flatten()[~duplicateslogical])))
            '''
            fig.add_trace(go.Scatter3d(x=EAgrid[:,:,:,1].flatten()[~duplicateslogical],
                                    y=EAgrid[:,:,:,2].flatten()[~duplicateslogical],
                                    z=EAgrid[:,:,:,3].flatten()[~duplicateslogical],
                                    mode='markers',
                                    marker=dict(size=5, color='r')))
            '''

            #ax.scatter(EAgrid[:,:,:,1].flatten()[~duplicateslogical],
            #           EAgrid[:,:,:,2].flatten()[~duplicateslogical],
            #           EAgrid[:,:,:,3].flatten()[~duplicateslogical],
            #           c=colorGrid[:,:,:].flatten()[~duplicateslogical], marker='*');

        self.color = colorGrid[:,:,:].flatten()[~duplicateslogical];

        if plotq:
            # Show the plot
            fig.show()

        # Write precalculated node grid 
        if self.precalculate:
            self.save_or_load_attributes(mode="w")

    def interp2IrregularGrid(self, path, name, resolution=1):
        """
        interp2IrregularGrid method is used to interpolate data at
        an input path to the the equal area points.

        Parameters
        -----------
        path : STRING
            A path to some input netCDF4 file that will be used to
            find interpolated values at equal area nodes. 
        name : STRING
            The name of the data. This will be used to define a dictionary
            entry that points to the interpolated data.  

        (Re)define
        -----------
        self.data : DICTIONARY
            A dictionary that holds the data that has been interpolated to
            the equal area nodes. self.data[name] is the added entry.

        self.connectionNodeIDs : NUMPY ARRAY
            A len(self.data) x 5 array of values with column 1 corresponding
            to node index, and column 2-5 correspond to the first 4 closest
            connected node indices.
        """
        # Interpolate the values to node locations
        # Note that the region R must be set to -181/181 so that 
        # nodes at edges lon=-180=180 and lon=0 will have appropriately
        # interpolated values (i.e., not result in an nan value when
        # a value does exist).
        os.system("gmt grdtrack -R-181/181/-90/90 {0} -G{1} -N > {2} -Vq".format(self.filename2, path, 'temp.txt'))
        #os.system("gmt grdtrack -R-180/180/-90/90 {0} -G{1} -N > {2} -Vq".format(self.filename2, path, 'temp.txt'))
        
        # Nearest neighbor interpolation (needed to get values at north/south pole)
        #os.system("gmt grdtrack -R-181/181/-90/90 {0} -G{1} -nn -N > {2}".format(self.filename2, path, 'temp.txt'))

        # Read interpolated values
        self.data[name] = np.loadtxt('temp.txt', delimiter='\t',usecols=[2])

        # Delete temporary file
        os.system('temp.txt');
        
    def interp2IrregularGrid_New(self, path, name, resolution):
        """
        interp2IrregularGrid method is used to interpolate data at
        an input path to the the equal area points.

        Parameters
        -----------
        path : STRING
            A path to some input netCDF4 file that will be used to
            find interpolated values at equal area nodes. 
        name : STRING
            The name of the data. This will be used to define a dictionary
            entry that points to the interpolated data.  

        (Re)define
        -----------
        self.data : DICTIONARY
            A dictionary that holds the data that has been interpolated to
            the equal area nodes. self.data[name] is the added entry.

        self.connectionNodeIDs : NUMPY ARRAY
            A len(self.data) x 5 array of values with column 1 corresponding
            to node index, and column 2-5 correspond to the first 4 closest
            connected node indices.
        """
        # Interpolate the values to node locations
        # Note that the region R must be set to -181/181 so that 
        # nodes at edges lon=-180=180 and lon=0 will have appropriately
        # interpolated values (i.e., not result in an nan value when
        # a value does exist).
        cmd = "gmt grdtrack -R{3:0.2f}/{4:0.2f}/{5:0.2f}/{6:0.2f} {0} -G{1} -N > {2} -Vq".format(self.filename2, path, 'temp.txt',
                                                                             -180+resolution/2,
                                                                             180-resolution/2,
                                                                             -90+resolution/2,
                                                                             90-resolution/2)
        os.system(cmd)
                                                                             

        # Nearest neighbor interpolation (needed to get values at north/south pole)
        #os.system("gmt grdtrack -R-181/181/-90/90 {0} -G{1} -nn -N > {2}".format(self.filename2, path, 'temp.txt'))

        # Read interpolated values
        self.data[name] = np.loadtxt('temp.txt', delimiter='\t',usecols=[2])

        # Delete temporary file
        os.system('rm temp.txt');

    def sort_spherical_points(self, x, y, z):
        """
        Sorts points on a sphere from the North Pole to the South Pole.
        If points have the same latitude, they are sorted by longitude.
        
        Parameters:
        x, y, z : array-like
            Cartesian coordinates of points on the sphere.
        
        Returns:
        sorted_points : np.ndarray
            Sorted (x, y, z) points as a NumPy array.
        """

        print("Unused and old function: remove")

        return

        # Convert Cartesian to spherical coordinates
        lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))  # Latitude
        lon = np.degrees(np.arctan2(y, x))  # Longitude
        
        # Combine into a structured array for sorting
        points = np.array(list(zip(lat, lon, x, y, z)), dtype=[('lat', 'f8'), ('lon', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        
        # Sort by latitude (descending), then by longitude (ascending)
        sorted_points = np.sort(points, order=['lat', 'lon'])[::-1]  # Reverse latitude for North-South ordering
        
        # Extract sorted x, y, z
        return np.column_stack((sorted_points['x'], sorted_points['y'], sorted_points['z']))

    def interpIrregularGrid(self, plotq):

        print("Unused and old function: remove")

        return

        # Load the file with interpolated values
        location = np.loadtxt(self.filename1, delimiter=',',usecols=[1,2,3])
        x, y, z  = location[:,0], location[:,1], location[:,2];

        # Interpolate the values to node locations
        os.system("gmt grdtrack -Rg {0} -G{1} > {2}".format(self.interpGrid, self.dataGrid, self.output))

        
        #os.system("gmt grdtrack {0} -G{1} --FORMAT_FLOAT_OUT=%.6e> {2}".format(self.interpGrid, self.dataGrid, self.output))

        # plot string 1
        plotstringn = ("gmt begin scatter_plot png",
                    "\tgmt coast -Rg -JW4.5i -Baf -W1/0.5p -Glightgray -Slightblue",
                    "\tawk -F, '{print $1, $2}' EA_Nodes_20_LatLon.txt | gmt plot -Sc0.2c -Gred",
                    "gmt end show")
        
        plotstring = "\n".join(plotstringn)

        # plot string 2
        plotstringn = ( "# Set region based on data extent (auto-detected)",
                        "region=$(gmt info -I1 {0})".format(self.output),
                        "region=d",
                        "# Create color palette table (CPT) for contouring",
                        "gmt makecpt -T0/200/20 -Cturbo > colors.cpt",
                        "# Begin GMT session",
                        "gmt begin 'color_contour.png' png",
                        "\t# Plot base map with coastlines",
                        "\t gmt basemap -R$region -JW4.5i -Baf",
                        "\t# Interpolate scattered data into a grid using 'surface'",
                        "\tgmt surface {0} -R$region -I0.5 -Ggrid.nc".format(self.output),
                        "\t# Create color-filled contour plot",
                        "\tgmt grdimage grid.nc -Ccolors.cpt -Baf",
                        "\t# Overlay contour lines",
                        "\tgmt grdcontour grid.nc -Ccolors.cpt -A10",
                        "\t# Plot data points for reference",
                        "\tgmt plot {0} -Sc0.2c -Gblack".format(self.output),
                        "\tgmt coast -Rg -JW4.5i -Baf -W1/0.5p -Glightgray",
                        "gmt end show",
                        "# Cleanup temporary files",
                        "rm grid.nc colors.cpt")
        
        plotstringn = ( "# Set region based on data extent (auto-detected)",
                        "region=$(gmt info -I1 {0})".format(self.output),
                        "region=d",
                        "# Create color palette table (CPT) for contouring",
                        "gmt makecpt -T0/200/20 -Cturbo > colors.cpt",
                        "# Begin GMT session",
                        "gmt begin 'color_contour.png' png",
                        "\t# Plot base map with coastlines",
                        "\t gmt basemap -R$region -JW4.5i -Baf",
                        "\t# Interpolate scattered data into a grid using 'surface'",
                        "\tgmt surface {0} -R$region -I0.5 -Ggrid.nc".format(self.output),
                        "\t# Create color-filled contour plot",
                        "\tgmt grdimage grid.nc -Ccolors.cpt -Baf",
                        "\t# Overlay contour lines",
                        "\tgmt grdcontour grid.nc -Ccolors.cpt -A10",
                        "\t# Plot data points for reference",
                        "\tgmt coast -R$region -JW4.5i -Baf -W1/0.5p -Glightgray",
                        "gmt end show",
                        "# Cleanup temporary files",
                        "rm grid.nc colors.cpt")

        # Plot using gmt
        if plotq == 1:
            plotstring = "\n".join(plotstringn)
            os.system(plotstring)


        # Plot using plotly
        if plotq == 2:
            from plotly.subplots import make_subplots
            # Create the figure
            fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Nodes', 'Interpolated - Original Grid'],
                    )
            
            # Set the layout
            fig.update_layout(
                title='Unit Equal Area Spaced Node',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    xaxis=dict(range=[-1.2, 1.2], visible=True),
                    yaxis=dict(range=[-1.2, 1.2], visible=True),
                    zaxis=dict(range=[-1.2, 1.2], visible=True),
                    aspectratio=dict(x=1, y=1, z=1),
                    
                )
            )
            

            # Sort nodes
            #location = self.sort_spherical_points(x, y, z)
            #x, y, z  = location[:,0], location[:,1], location[:,2];
            # T:    Tension factor (see gmt surface documentation), 0.35 is useful for sharp boundaries
            # R:    Domain 'd' global [-180, 180]
            # rp:   Pixel registration
            # I:    Resolution [in degree]
            #os.system("gmt surface {0} -Rd -I{1} -T0.35 -rp -Ggrid.nc+n -V3".format(self.output, self.resolution))
            os.system("gmt grdsample {0} -Rd -I{2} -rp -G{1}".format(self.dataGrid, 'tempSimilarRes.nc', self.resolution))
            
            #os.system("awk '!/NaN/' {0} | gmt greenspline -Rg -T{1} -Sp -Z0 -I{2} -rp -Ggrid.nc+n".format(self.output, 'tempSimilarRes.nc', self.resolution))
            #os.system("gmt grdsample {0} -Rd -I{2} -rp -G{1}".format(self.dataGrid, 'tempSimilarRes.nc', self.resolution/2))
            #os.system("gmt greenspline {0} -Rd -T{1} -Sp -Z3 -I{2} -rp -fg -Ggrid.nc+n".format(self.output, 'tempSimilarRes.nc', self.resolution))
            #os.system("awk '!/NaN/' {0} | gmt greenspline -Rd -T{1} -Sq -Z3 -I{2} -rp -fg -Ggrid.nc+n".format('EASampled_rounded.txt', 'tempSimilarRes.nc', self.resolution/2))
            #os.system("gmt greenspline {0} -Rg -Sc -Z2 -I{1} -rp -fg -Ggrid.nc+n".format(self.output, self.resolution))

            # Green spherical spline option
            os.system("gmt greenspline {0} -Rd -Sp -Z4 -I{1} -Ggrid.nc".format(self.output, self.resolution))

            ds

            # Use nearest neighbor interpolation to go from equal area grid to evenly-spaced grid (in degrees).
            #os.system("gmt nearneighbor {0} -Rd -I{1} -Ggrid.nc -rp -S200k -N4".format(self.output, self.resolution))

            # Take the difference between new grid and the original grid values
            os.system("gmt grdmath grid.nc {0} SUB = diff.nc".format('tempSimilarRes.nc'))


            file = "diff.nc";
            #file = "grid.nc"

            if file == "grid.nc":
                self.ncfile = Dataset("grid.nc");
                XX, YY = np.meshgrid(self.ncfile['x'][:].data, self.ncfile['y'][:].data)

                x1, y1, z1 = self.lonlat2xyz(XX, YY, radius=1)
                surface_color = np.where(np.isnan(self.ncfile['z'][:].data), 0, self.ncfile['z'][:].data)
                plotly_colorscale = get_plotly_colorscale("viridis")

            elif file == "diff.nc":
                self.ncfile = Dataset("diff.nc");
                XX, YY = np.meshgrid(self.ncfile['lon'][:].data, self.ncfile['lat'][:].data)
                x1, y1, z1 = self.lonlat2xyz(XX, YY, radius=1)
                surface_color = np.where(np.isnan(self.ncfile['z'][:].data), 0, self.ncfile['z'][:].data)
                #surface_color[self.ncfile['z'][:].data<0.5] = 0;

                areaWeightsOut, longitudes, latitudes, totalArea, totalAreaCalculated = areaWeights(resolution = self.resolution, radius = 1)

                minmax = [np.nanmin(self.ncfile['z'][:].data),
                          np.nanmax(self.ncfile['z'][:].data)];
                spacing= 1; 
                minmax[0] = minmax[0]-minmax[0]%spacing
                minmax[1] = minmax[1]-minmax[1]%spacing+spacing;
                binEdges = np.arange(minmax[0]-spacing/2,minmax[1]+spacing,spacing);
                binMeans = np.arange(minmax[0],minmax[1]+spacing,spacing)

                plt.figure()
                self.hist = plt.hist(self.ncfile['z'][:].data.flatten(), weights=areaWeightsOut.flatten(), density=True, bins=binEdges)
                plt.xlabel("Seafloor Ages [Myr]")
                plt.ylabel("Area weighted Distribution\n(Interpolated - Original Grid)")

                value = spacing; contin = True;
                thres = .95


                while contin:
                    percent = np.sum(self.hist[0][ np.abs(binMeans) < value])
                    if (percent > thres) | (percent == np.sum(self.hist[0])):
                        contin = False;
                    else:
                        value+=spacing
                
                plt.vlines([np.min(binMeans[ np.abs(binMeans) < value]),
                            np.max(binMeans[ np.abs(binMeans) < value])],
                            ymin=0,ymax=np.max(self.hist[0]), colors='r',
                            label="95%")
                

                plt.figure()
                plt.hist(self.ncfile['z'][:].data.flatten(), weights=areaWeightsOut.flatten(), density=True, bins=binEdges)
                plt.xlabel("Seafloor Ages [Myr]")
                plt.ylabel("Area weighted Distribution\n(Interpolated - Original Grid)")
                plt.xlim([-10,10]);
                
                plt.vlines([np.min(binMeans[ np.abs(binMeans) < value]),
                            np.max(binMeans[ np.abs(binMeans) < value])],
                            ymin=0,ymax=np.max(self.hist[0]), colors='r',
                            label="95%")



                plotly_colorscale = get_plotly_colorscale("viridis")



            # Masked values where do data exist
            #os.system("gmt grdsample {0} -Rd -I{2} -rp -G{1}".format(self.dataGrid, 'tempSimilarRes.nc', self.resolution))
            #data = Dataset('tempSimilarRes.nc', 'r')
            #surface_color[data['z'][:].mask] = np.nan;
            #data.close()


            
            # Add nodes
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,
                                       mode='markers',
                                       marker=dict(size=1, color=self.color)), 1, 1)
            #fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='greys', opacity=1, colorbar=dict(tickvals=[-1,0,1])), 1, 1)

            # Add the sphere surface
            fig.add_trace(go.Surface(x=x1*.99,
                                     y=y1*.99,
                                     z=z1*.99,
                                     surfacecolor=np.sqrt((x1*.98)**2+(y1*.98)**2+(z1*.98)**2),
                                     colorscale='greys',
                                     opacity=1,
                                     colorbar=dict(tickvals=[-1,0,1])), 1, 2)
            fig.add_trace(go.Surface(x=x1,
                                     y=y1,
                                     z=z1,
                                     surfacecolor=surface_color,
                                     colorscale=plotly_colorscale,
                                     opacity=1,
                                     colorbar=dict(tickvals=np.arange(np.min(surface_color), np.max(surface_color), int((np.max(surface_color)-np.min(surface_color)-(np.max(surface_color)-np.min(surface_color))%8)/8)) )), 1, 2)
            print( "np.sum(np.isnan(self.ncfile['z'][:].data))", np.sum(np.isnan(self.ncfile['z'][:].data)) )
            
            self.ncfile.close()

            # Add coastlines



            fig.show()


def get_plotly_colorscale(cmap_name="viridis", nan_color="rgba(0,0,0,0)", n_colors=256):
    """
    Converts a Matplotlib colormap to a Plotly colorscale, adding transparency for NaN values.
    
    Parameters:
    - cmap_name (str): Name of the Matplotlib colormap (e.g., "viridis", "plasma").
    - nan_color (str): RGBA string for NaN values (default: fully transparent).
    - n_colors (int): Number of discrete color levels.
    
    Returns:
    - plotly_colorscale (list): A Plotly-compatible colorscale.
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name, n_colors)  # Get the colormap
    
    colorscale = []
    for i in range(n_colors):
        if i == 0:
            # Ensure NaNs map to a fully transparent color
            colorscale.insert(0, [0.0, nan_color])  # Optionally enforce transparency at the lowest value
        else:
            norm_val = i / (n_colors - 1)  # Normalize value between 0-1
            rgba = cmap(norm_val)  # Get RGBA color from Matplotlib colormap
            rgba_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
            colorscale.append([norm_val, rgba_str])  # Append to colorscale
    
    
    

    return colorscale



#######################################################################
###################### Basin definition functions #####################
#######################################################################
class synthField():
    """
    synthField is used to make synthetic bathymetry fields that can be
    applied to communitiy detection algorithms to evaluate the algorithms'
    efficacy.

    """
    def __init__(self, resolution=None, registration=None, outputFID=None):
        """
        Initialization of synthField.
        
        Parameters
        -----------
        resolution : FLOAT
            Resolution of the synthetic field, in degrees.
            The default is None.
        registration : STRING
            The registration of the synthetic field. Either
            'pixel' or 'gridline'. The default is None.
        outputFID : STRING
            Name of output netCDF4. Argument can include
            directory (e.g., '{Path}/mynetCDF4.nc'). The
            default is None.
        """
        
        # Run test 
        if resolution==None:
            runTest = True;
            resolution = 10;
            registration = 'pixel';
            outputFID = 'mynetCDF4';
        else:
            runTest = False;

        
        # Set class attributes
        self.resolution = resolution;
        self.registration = registration;
        self.outputFID = outputFID;
        
        if registration == 'pixel':
            offset = resolution/2;
        else:
            offset = 0;
        
        # Vectors
        self.latv = np.arange(-90+offset, 90+self.resolution-2*offset, self.resolution)
        self.lonv = np.arange(-180+offset, 180+self.resolution-2*offset, self.resolution)
        
        # Grids
        self.lon, self.lat = np.meshgrid(self.lonv, self.latv)
        self.z = np.ones(self.lat.shape)
        areaWeightsA, longitudes, latitudes, totalArea, totalAreaCalculated = areaWeights(resolution=self.resolution, radius=1, verbose=False)
        self.areaWeights = areaWeightsA;
        
        # netCDF4
        self.nc = None;
        
        # Run test
        if runTest:
            self.runTest();
        
    def makenetCDF(self, verbose=True):
        """
        makenetCDF method is used to save the user created
        grid to a netCDF4.
        """
        
        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(self.outputFID)

        # Make new .nc file
        ncfile = Dataset(self.outputFID, mode='w', format='NETCDF4_CLASSIC')
        
        # Format title
        ncfile.title='Synthetic bathymetry model.'

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ncfile.createDimension('lat', len(self.z[:,0]));     # latitude axis
        lon_dim = ncfile.createDimension('lon', len(self.z[0,:]));     # longitude axis

        # Define lat/lon with the same names as dimensions to make variables.
        lat = ncfile.createVariable('lat', np.float32, ('lat',));
        lat.units = 'degrees_north'; lat.long_name = 'latitude';
        lon = ncfile.createVariable('lon', np.float32, ('lon',));
        lon.units = 'degrees_east'; lon.long_name = 'longitude';
        
        # Define single values parameters (e.g., VOC, AOC, high latitude cutoff)
        highlatlat = ncfile.createVariable('highlatlat', None)
        highlatlat.units = 'degrees'
        highlatlat.standard_name = 'highlatlat'

        highlatA = ncfile.createVariable('highlatA', None)
        highlatA.units = 'meters sq'
        highlatA.standard_name = 'highlatA'

        VOC = ncfile.createVariable('VOC', None)
        VOC.units = 'meters cubed'
        VOC.standard_name = 'VOC'

        AOC = ncfile.createVariable('AOC', None)
        AOC.units = 'meters sq'
        AOC.standard_name = 'AOC'

        # Define a 2D variable to hold the elevation data
        z = ncfile.createVariable('bathymetry',np.float64,('lat','lon'))
        z.units = 'm'
        z.standard_name = 'z'
        
        # Define vector as function with longitude dependence
        areaWeights = ncfile.createVariable('areaWeights',np.float64,('lat',))
        areaWeights.units = 'meters sq'
        areaWeights.standard_name = 'areaWeights'

        # Add attributes
        highlatlat[:] = 90;
        highlatA[:] = 0;
        VOC[:] = 0;
        AOC[:] = 0;
        
        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        z[:] = self.z;
        areaWeights[:] = self.areaWeights[:,0];

        # Close the netcdf
        ncfile.close();
            
        # Report contents of the created netCDF4
        if verbose:
            # Open netCDF4
            ncfile = Dataset(self.outputFID, mode='r', format='NETCDF4_CLASSIC')

            # Report netCDF4 contents
            print("Variable\t\tDimensions\t\t\tShape")
            print("--------------------------------------------------------------------------------------")
            for variable in ncfile.variables:
                if len(variable) != 20: 
                    variablePrint = variable.ljust(24)
                print(variablePrint.ljust(24)+
                    str(ncfile[variable].dimensions).ljust(32)+
                    str(ncfile[variable].shape).ljust(32))

            # Close netCDF4
            ncfile.close();
            
        
    def addLine(self, startPos, endPos, magnitude, verbose=True):
        """
        addLine method is used to add a line of values
        from startPos to endPos and assign them a magnitude.
        Note that this method will override other non Nan
        values. Note that this method can only make vertical
        and horizonal lines and have the thickness of the
        class' resolution.
        
        Parameters
        -----------
        startPos : LIST
            Starting position of position of the line (lon, lat),
            in degrees.
        endPos : LIST
            Ending position of position of the line (lon, lat),
            in degrees.
        magnitude : FLOAT
            magnitude of line, represented as a multiple of
            1.
        
        """
        if endPos[1] == startPos[1]:
            # If longitudinal line (lats equal)
            # Make logical
            logical = (self.lat==endPos[1]) & (self.lon>startPos[0]) & (self.lon<endPos[0]);
            if verbose:
                print("longitudinal line")
        elif endPos[0] == startPos[0]:
            # If latitudinal line (lons equal)
            # Make logical
            logical = (self.lon==endPos[0]) & (self.lat>startPos[1]) & (self.lat<endPos[1]);
            if verbose:
                print("latitudinal line")
        else:
            print("Line is not longitudinal or latitudinal.")
            return

        # Apply logical
        self.z[logical] = magnitude;
        if verbose:
            print("logical", np.nansum(logical), logical)
                
    def addShape(self, shape, size, position, magnitude, verbose=True):
        """
        Add a cluster of magnitude values of a input shape
        and size and centered at some position. The size
        will be limited to the resolution of the synthField.
        
        Parameters
        -----------
        shape : STRING
            The shape of the continent. Either 'circle' or
            'square'.
        size : FLOAT
            Input radius of circle or half length square,
            in degrees.
        position : LIST
            Input center position of the shape (lon, lat),
            in degrees.
        magnitude : FLOAT
            magnitude of shape, represented as a multiple of
            1.
        
        """
        
        if shape == 'circle':
            # If shape is circle
            # Make logical
            logical = (np.sqrt((self.lon-position[0])**2 + (self.lat-position[1])**2) <= size);
        elif shape == 'square':
            # If shape is square
            # Make logical
            logical = (self.lat>(position[1]-size)) & (self.lat<(position[1]+size)) & (self.lon>(position[0]-size)) & (self.lon<(position[0]+size));
        else:
            print("Invalid shape inputed. No modification has been made.")
            return
        
        # Apply logical
        self.z[logical] = magnitude;
        if verbose:
            print("logical", np.nansum(logical), logical)
        
        
    def plot(self, verbose=True):
        """
        plot is a simple plot method for class development.
        """
        
        if verbose:
            print("self.latv", self.latv.shape, self.latv);
            print("self.lonv", self.lonv.shape, self.lonv);
            
            print("self.lat", self.lat.shape, self.lat);
            print("self.lon", self.lon.shape, self.lon);
            
        fig, ax= plt.subplots();
        
        plt.contourf(self.lon, self.lat, self.z);
        plt.axis('equal');
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.spines['left'].set_visible(False);
        
        ax.get_xaxis().set_ticks(np.arange(-180,181,30));
        ax.get_yaxis().set_ticks(np.arange(-90,91,30));
            
    def runTest(self):
        self.addLine(startPos=[-100, -55], endPos=[100,-55], magnitude=10, verbose=False)
        self.addLine(startPos=[5, -20], endPos=[5,10], magnitude=10, verbose=False)
        self.addShape(shape='square', size=20, position=[-100, 15], magnitude=10, verbose=False)
        self.addShape(shape='circle', size=20, position=[100, 15], magnitude=10, verbose=False)
        self.plot(verbose=False);


#######################################################################
######################## Basin helper functions #######################
#######################################################################


import numpy as np
import igraph as ig
import leidenalg
import louvain
import ctypes
import multiprocessing

# Global variables to pass to worker
_global_g = None
_global_coassoc_base = None
_global_n = None
_global_resolution_parameter = None
_global_weight_attr = None
_global_method = None
_global_partition_strategy = None

def _CReduction_init_worker(g, coassoc_base, n, resolution_parameter, method, partition_strategy):
    # Update global variable to initialize the _work
    global _global_g, _global_coassoc_base, _global_n, _global_resolution_parameter, _global_method, _global_partition_strategy
    _global_g = g
    _global_coassoc_base = coassoc_base
    _global_n = n
    _global_resolution_parameter = resolution_parameter
    _global_method = method
    _global_partition_strategy = partition_strategy

def _CReduction_worker(seed_i):
    # Assign global variables to local worker
    n = _global_n
    g = _global_g
    coassoc_base = _global_coassoc_base
    resolution_parameter = _global_resolution_parameter
    method = _global_method
    partition_strategy = _global_partition_strategy

    # Use one of two reduction methods
    if method == "leiden":
        part = leidenalg.find_partition(
            g,
            partition_strategy,
            resolution_parameter=resolution_parameter,
            weights=g.es["weight"],
            seed=seed_i
        )
    elif method == "louvain":
        part = louvain.find_partition(
            g,
            partition_strategy,
            resolution_parameter=resolution_parameter,
            weights=g.es["weight"],
            seed=seed_i
        )
    else:
        raise ValueError("Method must be 'leiden' or 'louvain'.")

    # Define local array to add to co-association matrix
    local = np.zeros((n, n), dtype=np.float64)
    for community in part:
        for u in community:
            for v in community:
                local[u, v] = 1.0

    # Sum local matrix to co-association matrix
    coassoc = np.ctypeslib.as_array(coassoc_base.get_obj()).reshape((n, n))
    with coassoc_base.get_lock():
        coassoc[:] += local


#######################################################################
###################### Basin definition functions #####################
#######################################################################


class BasinsEA():
    """
    Basins is a class meant to construct basins and bathymetry properties
    given a bathymetry model netCDF4.
    """

    def __init__(self, dataDir, filename, body):
        """
        Initialization of Basins class.

        Parameter
        ----------
        dataDir : STRING
            A directory which you store local data within. Note that this
            function will download directories [data_dir]/topographies
        filename : STRING
            Output file name 
        body : STRING
            Name of the input terrestial body. This is only used for
            visualization purposes, not naming convections when writing
            model files.
        
        Define
        ----------
        self.bathymetry : NUMPY ARRAY
        self.areaWeights : NUMPY ARRAY
        self.lat : NUMPY ARRAY
        self.lon : NUMPY ARRAY
        self.radius : FLOAT

        """

        # Read netCDF4 bathymetry file
        self.nc = Dataset("{}/{}".format(dataDir, filename));
        self.dataDir = dataDir;
        self.filename = filename;
        self.body = body;

        # Set latitude/longitude/elev
        self.lon, self.lat  = np.meshgrid( np.array(self.nc['lon'][:]), np.array(self.nc['lat'][:]) );
        self.bathymetry = np.array(self.nc['bathymetry'][:]);
        self.areaWeights = np.reshape(np.repeat(np.array(self.nc["areaWeights"][:]), np.shape(self.nc["bathymetry"][:])[1] ), np.shape(self.nc["bathymetry"]) );
        self.radius = np.sqrt( np.sum(np.sum(self.areaWeights))/(4*np.pi) );
        self.highlatlat = self.nc['highlatlat'][:].data;
        self.highlatA = self.nc['highlatA'][:].data;
        self.VOC = self.nc['VOC'][:].data;
        self.AOC = self.nc['AOC'][:].data;

        # Define inputs for the class that makes equal area
        EAinputs = {"resolution":np.diff(self.nc['lon'][:])[0], "dataGrid":"{}/{}".format(dataDir, filename), "parameter": "bathymetry", "parameterUnit":"m", "parameterName":"bathymetry" }

        # Define attribute to hold multiple scalar fields.
        # These will be used to determine edge weights.
        self.Fields = {};
        self.Fields["MultipleFields"] = False;
        self.Fields["FieldCnt"] = 1;
        self.Fields["Field1"] = EAinputs;

        # Set the fields to be used for edge weight calculations.
        self.useFields()

        # Define class attributes to be redefined throughout analysis
        ## Have basin connection been defined.
        self.basinConnectionDefined = False;
        ## Have basin bathymetry parameters been defined.
        self.BasinParametersDefined = False;

        # Define mask to use when interpolating from node values
        # to evenly spaced latitude-longitude points
        self.setFieldMask() 
        
        # Close file  
        self.nc.close();

    def addField(self, resolution, dataGrid, parameter, parameterUnit, parameterName):
        """
        addField method is used to set a field to calculate edge weights with.

        Parameters
        -----------
        resolution : FLOAT
            Resolution of input data set. This value needs to be the same for all set fields.
        dataGrid : STRING
            Directory to netCDF4 to be used in calculations.
        parameter : STRING
            Name of parameter in netCDF4 to be used for edge weight calculations.
        parameterUnit : STRING
            Unit of parameter.
        parameterName : STRING
            Common name for parameter (e.g., bathymetry, PSU, Temperature, organism1)
        
        Re(defines)
        ------------
        self.Fields : DICTIONARY
            Holds entries corresponding to fields

        
        """
        # Define EAinputs for added field
        EAinputs = {"resolution":resolution, "dataGrid":dataGrid, "parameter":parameter, "parameterUnit":parameterUnit, "parameterName":parameterName }

        # Add dictionary entry for added field.
        self.Fields["Field{}".format(self.Fields["FieldCnt"]+1)] = EAinputs;

        # Redefine the number of used fields.
        self.Fields["FieldCnt"] += 1;

    def getFields(self, usedFields = False):
        """
        getFields method is used to output the information of all fields stored
        within the BasinsEA object.

        Parameter
        ----------
        Option to only plot the fields being used in the calculation of weights
        and communities.

        Return
        -------
        None.
        
        """

        if not usedFields:
            # Show all fields in object
            print("\nAll fields\n---------------")
            for i in range(self.Fields["FieldCnt"]):
                # Define field number
                fieldNum = 1+i;
                # Plot field information
                print("Field{}".format(fieldNum))
                print("\tdataGrid: {}".format(self.Fields["Field{}".format(fieldNum)]["dataGrid"]))                
                print("\tparameter: {}".format(self.Fields["Field{}".format(fieldNum)]["parameter"]))
                print("\tparameterUnit: {}".format(self.Fields["Field{}".format(fieldNum)]["parameterUnit"]))
                print("\tparameterName: {}\n".format(self.Fields["Field{}".format(fieldNum)]["parameterName"]))
        else:
            # Show fields in object that are used for graph construction
            # and community detection. 
            print("\nUsed fields\n---------------")
            for fieldNum in self.Fields["usedFields"]:
                # Plot field information
                print("{}".format(fieldNum))
                print("\tdataGrid: {}".format(self.Fields[fieldNum]["dataGrid"]))                
                print("\tparameter: {}".format(self.Fields[fieldNum]["parameter"]))
                print("\tparameterUnit: {}".format(self.Fields[fieldNum]["parameterUnit"]))
                print("\tparameterName: {}\n".format(self.Fields[fieldNum]["parameterName"]))


    def useFields(self, fieldList=np.array(["Field1"])):
        """
        useFields method is used to define which fields will be used to
        calculate edge weights with.

        Parameters
        -----------
        fieldList : NUMPY LIST
            A list of strings. The default is ["Field1"].

        Re(defines)
        ------------
        self.Fields : DICTIONARY
            Holds entries corresponding to fields
        """

        # Define the fields used in the calculation of edge weights.
        self.Fields["usedFields"] = fieldList;

    def setFieldMask(self, fieldMaskParameter={"usedField":None}, Field='bathymetry'):
        '''
        setFieldMask is a method used to set a mask for the
        communities once they are interpolated back to an
        equal-spaced latitude-longitude array.

        Parameter
        ----------
        fieldMaskParameter : DICTIONARY
            A set of parameters used to determine the field
            mask used.
                "usedField" : INT, None
                    Set to None to use bathymetry map load
                    when initiating BasinsEA class. Set to
                    an integer 0-n to assign the values with
                    usedFields.
                "fliprl" : False
                    An option to flip the input mask along
                    longitude 0 values (prime meridian).
                "flipud" : True
                    An option to flip the input mask along
                    latitude 0 values (equator).
        usedField : INT
            An integer value associated with the field to be
            used for masking. Note that this argument should
            be assigned the index of the used field vector
            self.Fields['usedFields']
        Field : NUMPY ARRAY
            An array of values with np.nan corresponding to
            values to be masked out. The default is 'bathymetry',
            and forces the mask to be initial set to class'
            initialized array (e.g., etopo).
        fliprl : BOOLEAN

        flipud : BOOLEAN

        Re(define)
        -----------
        self.maskValue : NUMPY ARRAY
            Masked values represented with non-np.nan values.
        '''

        # An option to use an input field array
        # as a mask

        if Field == 'bathymetry':
            self.maskValue = self.bathymetry;
        else:
            self.maskValue = Field;
        
        if fieldMaskParameter['usedField'] is not None:
            # Set values
            usedField = fieldMaskParameter['usedField']

            # Input and output filenames
            input_grid = "tempSimp_{}.nc".format( self.Fields[ self.Fields['usedFields'][usedField] ]['parameterName'] )
            output_grid = "mask.nc"

            # GMT command: resample to 1x1 degree with pixel registration
            # -R specifies global extent (0 to 360 or -180 to 180, adjust as needed)
            # -I1d sets 1-degree spacing
            # -rp forces pixel registration

            # This is a very odd way to use a gmt grdsample call, this will create a 
            # pixel registered mask of input_grid using nearest neighbor interpolation.
            resolution = self.Fields[ self.Fields['usedFields'][usedField] ]['resolution']
            cmd = "gmt grdsample {0} -G{1} -I{2}d -rg -nn -R{3}/{4}/{5}/{6}".format(input_grid,
                                                                                            output_grid,
                                                                                            resolution,
                                                                                            -180+resolution/2,
                                                                                             180-resolution/2,
                                                                                             -90+resolution/2,
                                                                                              90-resolution/2)
            # cmd = "gmt grdsample {0} -G{1} -I{2}d -rp -R-181/181/-91/91".format(input_grid,
            #                                                                     output_grid,
            #                                                                     self.Fields[ self.Fields['usedFields'][usedField] ]['resolution'])
            # Execute the command
            os.system(cmd)


            # Read mask and set field
            ds = Dataset("mask.nc")
            field = ds['z'][:].data;
            mask = np.ones( np.shape(ds['z'][:].mask) )
            mask[ds['z'][:].mask] = np.nan
            ds.close();

            # Apply operations to field
            if fieldMaskParameter['fliprl']:
                mask    = np.fliplr(mask)
                field   = np.fliplr(field);
            if fieldMaskParameter['flipud']:
                mask    = np.flipud(mask)
                field   = np.flipud(field);
                
            # Assign mask and field to class attribute            
            self.maskValue = mask;
            self.bathymetry = field; # FIXME: This might not be the most appropriate way to redefine 'self.bathymetry'
        else:
            self.maskValue = cp.deepcopy(self.bathymetry);
            self.maskValue[~np.isnan(self.maskValue)] = 1;

    

    def simplifyNetCDF(self,
                       inputPath="path/file1.nc",
                       outputPath="~/file2.nc",
                       parameterIn="bathymetry",
                       parameterOut="z",):
        """
        simplifyNetCDF method reads a NetCDF4 file and writes a new NetCDF4 file
        with only lat, lon, and bathymetry variables.

        Parameters
        -----------
        input_path : STRING
            Path to the input NetCDF4 file. The default
            is "path/file1.nc".
        output_path : STRING
            Path to save the new NetCDF4 file. The default
            is "path/file2.nc"
        parameterIn : STRING
            Name of parameter in the netCDF4 that
            will be copied. The default is "bathymetry".
        parameterOut : STRING
            Name of new parameter in the copied netCDF4.
            This should be a standard name, so gmt netcdf
            operations are simple. The default is "z".
        """
        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(outputPath)

        # Open the original NetCDF file (file1.nc) in read mode
        with Dataset(inputPath, 'r') as src:
            # Create a new NetCDF file (file2.nc) in write mode
            with Dataset(outputPath, 'w', format="NETCDF4_CLASSIC") as dst:
                # Copy global attributes
                if "title" in src.ncattrs():
                    dst.title = src.title  # Preserve title attribute

                # Copy lat & lon dimensions
                for dim_name in ["lat", "lon"]:
                    if dim_name in src.dimensions:
                        dst.createDimension(dim_name, len(src.dimensions[dim_name]))

                # Copy lat & lon variables
                for var_name in ["lat", "lon"]:
                    if var_name in src.variables:
                        var = src.variables[var_name]
                        dst_var = dst.createVariable(var_name, var.datatype, var.dimensions)
                        dst_var[:] = var[:]  # Copy data
                        dst_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})  # Copy attributes

                # Copy bathymetry variable and rename it to 'z'
                if parameterIn in src.variables:
                    z_var = src.variables[parameterIn]
                    dst_z = dst.createVariable(parameterOut, z_var.datatype, z_var.dimensions, fill_value=np.nan)
                    dst_z[:] = z_var[:]  # Copy data
                    # Copy attributes
                    for attr in z_var.ncattrs():
                        if attr != "_FillValue":
                            try:
                                #print("attr",attr)
                                dst_z.setncatts({attr: z_var.getncattr(attr)})
                            except:
                                pass

    def getBasinSize(self, fraction=True, Threshold=None):
        """
        getBasinSize returns the surface area of each basin
        in either absolute m2 units or percentage of total
        surface area of graph network.
        
        Parameter
        ----------
        fraction : BOOLEAN
            Option to return percentage of total surface
            area covered by each basin.            
        Threshold : FLOAT
            Threshold value to define small vs large basins.
            If fraction is true, then this must be a fraction
            value as well. Otherwise set to absolute m2 value.
        """
        totalNodeCnt   = 0;
        areaAbsolutem2 = np.zeros( len(self.communitiesFinal) );
        areaperNode    = self.G.nodes[0]['areaWeightm2']

        # Iterate over each community
        for i in range(len(self.communitiesFinal)):
            totalNodeCnt += len(self.communitiesFinal[i])
            areaAbsolutem2[i] = len(self.communitiesFinal[i])*areaperNode

        areaFrac = 100* areaAbsolutem2/(np.sum(areaAbsolutem2));

        # Calculate the number of large and small basins based on input threshold
        if Threshold is None:
            # Assumes all basins are consider large
            LargeBasins = np.sum(areaFrac>0)
            SmallBasins = np.sum(areaFrac<=0)            
        else:
            # Uses user input to determine size of large vs. small basins.
            LargeBasins = np.sum(areaFrac>Threshold)
            SmallBasins = np.sum(areaFrac<=Threshold)

        if fraction:
            text = "\nLargeBasinThresholdPercentage: {}".format(Threshold)
        else:
            text = "\nLargeBasinThresholdm2: {}".format(Threshold)
        text += "\nTotalBasins: {}".format(len(areaFrac))
        text += "\nLargeBasins: {}".format(LargeBasins)
        text += "\nSmallBasins: {}".format(SmallBasins)

        # Define return dictionary
        returnDictionary = {"text": text};
        if fraction:
            returnDictionary["areaFrac"] = areaFrac;
        else:
            returnDictionary["areaAbsolutem2"] = areaAbsolutem2;
        
        return returnDictionary
        
        

    def defineBasins(self,
                     detectionMethod = {"method":"Louvain","resolution":1, "minBasinCnt":40, "minBasinLargerThanSmallMergers":True},
                     edgeWeightMethod = {"method":"useLogistic"},
                     fieldMaskParameter = {"usedField":None},
                     reducedRes={"on":False,"factor":15},
                     read=False,
                     write=False,
                     verbose=True,
                     initiation=True):
        """
        defineBasins method will define basins with network analysis
        using either the Girvan-Newman or Louvain algorithm to define
        communities.

        Parameter
        ----------
        detectionMethod : DICTIONARY
            Determines the implemented community detection algorithm and
            other properties to use for community detection. This dictionary
            has the following keys:
                method : STRING
                    The options are "Girvan-Newman", "Louvain", or
                    "Louvain-Girvan-Newman". The former is more
                    robust with low scalability and the latter are
                    practical but produces non-deterministic communities.
                    The default is "Louvain".
                resolution : FLOAT
                    The resolution value to be used with the Louvain
                    community detection algorithm. Values greater than 1,
                    makes the algorithm favor smaller communities (more
                    communities). Values less than 1, makes the algorithm
                    favor larger communities (less communities). The default
                    is 1.
                minBasinCnt : INT
                    The minimum amount of basins the user chooses to define
                    for the given bathymetry model input.
                ensembleSize : INT
                    Number of community detection runs to use in the community
                    reduction step of a composite community detection.
                minBasinLargerThanSmallMergers : BOOLEAN
                    An option that requires the minBasinCnt variable to equal
                    to the number of merged basins that are larger than the
                    small basins merger options defined in a mergerPackage.
        edgeWeightMethod : DICTIONARY
            Determines the implemented edge weight scheme. Options are:
                "useGlobalDifference"
                    ...FIXME: add user input parameters
                "useEdgeDifference"
                    ...FIXME: add user input parameters
                "useEdgeGravity"
                    ...FIXME: add user input parameters
                "useLogistic"
                    Choose lower and upper bound weights 'S_at_lower',
                    'S_at_upper' (between 0-1) and their correpsonding
                    data field values 'factor_at_lower', 'factor_at_upper'
                    (in units of standard deviation) that will be used
                    to construct the logistic-like weighting curve.
                "useNormPDFFittedSigmoid"
                    ...FIXME: add user input parameters
                "useQTGaussianSigmoid"
                    ...FIXME: add user input parameters
                "useQTGaussianShiftedGaussianWeightDistribution"
                    Choose 'shortenFactor' and 'shiftFactor' factor (in units
                    of standard deviation) that will be used to construct
                    the cumulative density function for this weighting scheme.


        reducedRes : DICTIONARY
            Option to reduce the resolution of the basin definition
            network calculation. Note that this should be turned
            off when doing analysis, and only kept on for testing
            purposes. The default is {"on":False,"factor":15}.
        read : BOOLEAN
            An option to read basin definitions for a given
            bathymetry model. The default is False. 
        write : BOOLEAN
            An option to write basin definitions for a given
            bathymetry model. The default is False. 
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.basinCnt : INT
            Number of basins in global bathymetry model.
        self.basinDis : NUMPY ARRAY
            binCnt x basinCnt array of bathymetry distributions.


        Structure:
        i. Determine to create or read graph
        Read graph
            1. read graph


        Create graph
            1. Reduce resolution and flatten: bathymetry, latitude, longitude
            2. Create EA grid of points (for interpolation): eaNodes(...)
                - Creates Class object
            3. Reduce resolution of EAinput data: simplifyNetCDF(...)
                - Creates temp tempSimp.nc
            4. Iterpolate temp file to EA grid points: eaPoint.interp2IrregularGrid(...)
                - Replaces temp tempSimp.nc
            5. Remove missing data from EA grid class object
                - Updates self.eaPoint
                - Changes: Need to make sure either 1) all data is represented each point or 2) have some locations represented with only some data  
            6. Create dictionary holding: lat, lon, parameter, area weight
                - Changes: can be made to hold additional parameters
            7. Create graph and add node attributes
                - Changes: add inner-loop for more attributes for additional fields
            8. Create array of node differences (at node edges)
                - Changes: add inner-loop for additional fields
            9. Calculate statistics difference array
                - Changes: add created attributes to self.Fields["Fieldi"]
            10. Iterate over nodes
                i. Iterate over node edges (defined with eaNodes object)
                    I. Calculate and assign weight
                        - Changes: calculate multiple weights combined them (add, subtract, product)


        Create a class for each of the methods
            - Initialize for variable
                - calculate statistics, QT, logistic function, parameters, ect
            - Method of function
                - weight = getwWight(value1, value2)
        
        
        self.Fields["MultipleFields"] = False;
        self.Fields["FieldCnt"] = 1;
        self.Fields["Field1"]
        self.Fields["usedFields"]
        
        """

        ##########################
        ### Write/Load network ###
        ##########################
        if read:
            # FIXME: Needs to be updated for EAnodes
            ####################
            ### Read network ###
            ####################
            self.G = nx.read_gml("{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), destringizer=float);
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
            
            # Define resolution
            self.resolution = self.reducedRes*np.diff(self.lon)[0][0];
        
        else:
            if initiation:
                # Only reduce resolution if option is set. Note that this must
                # be consistent with written network
                if not reducedRes['on']:
                    self.reducedRes = np.diff(self.lon)[0][0];
                    self.latf = self.lat.flatten();
                    self.lonf = self.lon.flatten();
                    bathymetryf = self.bathymetry.flatten();
                else:
                    self.reducedRes = reducedRes['factor'];
                    self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                    self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                    bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();

                # Readjust resolution if reduced resolution was used.
                if reducedRes['on']:
                    for field in self.Fields['usedFields']:
                        self.Fields[field]["resolution"] *= reducedRes['factor'];

                ######################
                ### Create network ###
                ######################
                # Define equal area points
                # eaPoint.lat, eaPoint.lon are created here
                # Note that only one eaNodes object is need for multiple fields.
                # Use the first used field to create the object
                # self.eaPoint.precalculated is set to False is the grid has not been
                # precalculated. Otherwise the precalculated grid is read in.
                self.eaPoint = eaNodes(inputs = self.Fields[self.Fields['usedFields'][0]],
                                    precalculate=True,
                                    precalculated=True);
                                    

                # Creates
                # 1) Set of nodes that represent equal area quadrangles.
                # 2) Define the connects between all nodes (even to nodes
                # with missing data)
                if not self.eaPoint.precalculated:
                    self.eaPoint.makegrid(plotq=0);

                # Rounding to about 1.1 km resolution
                Spresolution = np.round( np.abs(np.diff(self.lat[:,0])[0]), 5)

                # Loop over all used fields
                for field in self.Fields['usedFields']:
                    # Define parameter name
                    parameter = self.Fields[field]['parameter']
                    parameterName = self.Fields[field]['parameterName']
                    parameterOut = "z";

                    # Simplify netCDF4 for interpolation inputPath="path/file.nc", outputPath
                    self.simplifyNetCDF(inputPath=self.Fields[field]['dataGrid'],
                                        outputPath='tempSimp_{}.nc'.format(parameterName),
                                        parameterIn=parameter,
                                        parameterOut=parameterOut)
                    

                    
                    # Interpolate from grided nodes to equal area nodes
                    # Defines self.eaPoint.data with data at equal area nodes.
                    self.eaPoint.interp2IrregularGrid(path='tempSimp_{}.nc'.format(parameterName),
                                                    name=parameterOut,
                                                    resolution=Spresolution)


                    # Assign interpolated grid and connections to dictionary entry
                    self.Fields[field]['interpolatedData'] = self.eaPoint.data[parameterOut]

                    # Note that connectionNodeIDs are not the same for each field (i.e., there
                    # is a dependence on where np.nan values exist within
                    # self.Fields[field]['interpolatedData'].)
                    self.Fields[field]['connectionNodeIDs'] = self.eaPoint.connectionNodeIDs

                # Assign the field to be used for masking communities when converting
                # from node spacing to equally-spaced latitude and longitude values.
                self.setFieldMask(fieldMaskParameter=fieldMaskParameter);

                # Assign new-updated AOC value that reflects the data being fed into the graph network
                self.AOCMask = np.nansum( self.areaWeights[~np.isnan(self.maskValue)] )
                
                # Make a field that liberally defines nodes and connections across all fields.
                # I.e., even if only one field has data at a node then it will be represented
                # in all fields. 
                # However, weighting of node connections later will only be dependent on the
                # collection of fields that have edges between two nodes with valid data (non-NaNs).
                #for i
                # Loop over all used fields
                #for field in self.Fields['usedFields']:
                #    self.Fields['DataExist'] = 


                ## FIXME: Current area to work on

                ## Merge all fields into a single netCDF: "superImposedFields.nc"

                ### Create a string
                '''
                ### 1. Create field that contains the NaNs structure of all the input files.
                allFields = np.array([], dtype=str);
                for field in self.Fields['usedFields']:
                    # Define parameter name
                    parameterName = self.Fields[field]['parameterName']
                    # Append field name to string
                    allFields = np.append( allFields,
                                            'tempSimp_{}.nc'.format(parameterName)
                                        )
                #### i. Create NaN mask command: file1.nc ISNAN file2.nc ISNAN OR ...
                nan_mask_cmd = []
                for i, f in enumerate(allFields):
                    nan_mask_cmd.append(f"{f} ISNAN")
                    if i > 0:
                        nan_mask_cmd.append("OR")
                nan_mask_cmd.append("= nanmask.nc")
                nan_mask_str = " ".join(nan_mask_cmd)

                ##### ii. Create sum command: file1.nc file2.nc ADD file3.nc ADD ...
                sum_cmd = [allFields[0]]
                for f in allFields[1:]:
                    sum_cmd.append(f)
                    sum_cmd.append("ADD")
                sum_cmd.append("= summed.nc")
                sum_cmd_str = " ".join(sum_cmd)

                ##### iii. Use gmt grdmath to create nanmask.nc
                os.system("gmt grdmath {}".format(nan_mask_str))
                print("gmt grdmath {}".format(nan_mask_str))
                #os.system("gmt grdmath{}".format(sum_cmd_str))
                #gmt grdmath summed.nc nanmask.nc NAN = output.nc

                ##### iv. 
                # Interpolate from grided nodes to equal area nodes
                # Defines self.eaPoint.data with data at equal area nodes.
                self.eaPoint.interp2IrregularGrid(path='nanmask.nc',
                                                    name='z')
                '''

                # Assign interpolated grid and connections to dictionary entry
                #self.Fields[field]['interpolatedData'] = self.eaPoint.data[parameterOut]

                ## Run the interp2IrregularGrid method for eaPoints to get connectionNodeIDs
                # Note that connectionNodeIDs are not the same for each field (i.e., there
                # is a dependence on where np.nan values exist within
                # self.Fields[field]['interpolatedData'].)
                #self.connectionNodeIDs = self.eaPoint.connectionNodeIDs


                
                #self.Fields["MultipleFields"] = False;
                #self.Fields["FieldCnt"] = 1;
                #self.Fields["Field1"]
                # "resolution":
                # "dataGrid":"{}/{}".format(dataDir, filename)
                # "parameter": "bathymetry"
                # "parameterUnit":"m"
                # "parameterName":"bathymetry"
                #self.Fields["usedFields"]

                # Iterate over all fields and liberally select all nodes + connections 
                # Any missing values values. Where one field is represented, but the
                # other is not will be replaced with np.nan values. These values will
                # be dealt with in the later weight calculation.
                # 
                # Make logical to define which nodes represent at least 1 data-field.
                firstField = True;
                for field in self.Fields['usedFields']:
                    # First assign a base field to expand with other fields
                    if firstField:
                        logicalFields = ~np.isnan(self.Fields[field]['interpolatedData'])
                        firstField = False;
                    else:
                        logicalFields = ( logicalFields | ~np.isnan(self.Fields[field]['interpolatedData']) )
                
                # Area covered by a node m2. FIXME: might be able to define outside of field-loop.
                self.areaWeighti = (4*np.pi*(self.radius)**2)/len(self.eaPoint.data[parameterOut]);

                # Remove points with no data at any of the fields
                allNodes = False;
                if not allNodes:
                    self.eaPoint.ealat = self.eaPoint.ealat[logicalFields];
                    self.eaPoint.ealon = self.eaPoint.ealon[logicalFields];
                    self.eaPoint.connectionNodeIDs = self.eaPoint.connectionNodeIDs[logicalFields]
                    for field in self.Fields['usedFields']:
                        self.Fields[field]['interpolatedData'] = self.Fields[field]['interpolatedData'][logicalFields];
                
                # Define counter and point dictionary
                cnt = 0.
                points = {};
                # Create dictionary and array of bathymetry points
                pos = np.zeros( (2, len(~np.isnan(self.Fields[field]['interpolatedData']))) );
                for i in tqdm( range(len(self.eaPoint.ealon)) ):
                    # Create list of values to store in nodes: 
                    nodeAttributes = list([self.eaPoint.ealat[i], self.eaPoint.ealon[i], self.areaWeighti]);    # (latitude, longitude, areaWeight, Field1, Field2, ..., Fieldn) w/ units (deg, deg, m2, -, -, -)
                    # Add field properties: (latitude, longitude, areaWeight, Field1, Field2, ..., Fieldn) w/ units (deg, deg, m2, -, -, -)
                    for field in self.Fields['usedFields']:
                        nodeAttributes.append(self.Fields[field]['interpolatedData'][i])

                    #if (~np.isnan(bathymetryi)):
                    points[int(cnt)] = tuple(nodeAttributes);    # (latitude, longitude, areaWeight, Field1, Field2, ..., Fieldn) w/ units (deg, deg, m2, -, -, -)
                    pos[:,int(cnt)] = np.array( [self.eaPoint.ealat[i], self.eaPoint.ealon[i]] ); 
                    # Iterate node counter
                    cnt+=1;

                # Calculate the starting index of fields stored in nodes
                startOfFieldIdx = len(nodeAttributes)-len(self.Fields['usedFields']);

                # Create a graph
                G = nx.Graph()

                ## Add nodes (points)
                for node, values in points.items():
                    #if not np.isnan(values[2]):
                    G.add_node(node, pos=values[0:2], areaWeightm2=values[2]);
                    # Add field attributes
                    cnt = 0;
                    for field in self.Fields['usedFields']:
                        G.nodes[node][field] = values[startOfFieldIdx+cnt]
                        cnt+=1;

                
                ## Create a list of property difference between connected nodes
                ### Assign and empty vector to self.dataEdgeDiff 
                for field in self.Fields['usedFields']:
                    self.Fields[field]['dataEdgeDiff'] = np.array([], dtype=np.float64)

                ### Iterate through each node to add edges
                nodeCnt=0;
                for i in tqdm(np.arange(len(pos[0,:]))):
                    # Iterate over all nodes

                    # Assign all field values to values1 (evalued node)
                    valuesNode = points[int(nodeCnt)][startOfFieldIdx:];
                    #coordsNode = G.nodes[node1]['pos'];
                    
                    # Get connection node ids
                    connections = self.eaPoint.connectionNodeIDs[i,1:]

                    for connection in connections:
                        # Iterate over connections
                        if (connection==self.eaPoint.connectionNodeIDs[:,0]).any():
                            # Connection found between value1 and value2. This would not happen
                            # if the connection was to a node over land.
                            idxConnected = self.eaPoint.connectionNodeIDs[:,0][connection==self.eaPoint.connectionNodeIDs[:,0]][0]
                            nodeConnected = np.argwhere(self.eaPoint.connectionNodeIDs[:,0]==idxConnected)[0][0]

                            # Assign all field values to values2 (connected node)
                            valuesConnected = points[int(nodeConnected)][startOfFieldIdx:];

                            # Assign difference to dataEdgeDiff vector
                            cnt=0;
                            for field in self.Fields['usedFields']:
                                self.Fields[field]['dataEdgeDiff'] = np.append(self.Fields[field]['dataEdgeDiff'], np.abs(valuesNode[cnt]-valuesConnected[cnt]))
                                cnt+=1;

                    # Iterate node counter
                    nodeCnt+=1;
                
                ## Remove outliers from self.dataEdgeDiff and calculate std
                def remove_outliers_iqr(data):
                    """
                    remove_outliers_iqr function removes outliers from 
                    a Numpy array using the IQR method.

                    Parameters
                    -----------
                    data : NUMPY ARRAY
                        The input NumPy array.

                    Returns
                    --------
                    filtered_data : NUMPY ARRAY
                        A new NumPy array with outliers removed.
                    """
                    q1 = np.nanpercentile(data, 25)
                    q3 = np.nanpercentile(data, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                    return filtered_data

                for field in self.Fields['usedFields']:
                    ### Get outliers filtered dataEdgeDiff using the IQR method.
                    ## Mirror dataEdgeDiffIQRFiltered about zero when finding the std.
                    ## This is appropriate since each edge is bidirectional.
                    ## As a result, the mean should be zero.
                    self.Fields[field]['dataEdgeDiffIQRFiltered'] = remove_outliers_iqr(self.Fields[field]['dataEdgeDiff']);
                    #self.Fields[field]['dataEdgeDiffIQRFiltered'] = remove_outliers_iqr( np.append(self.Fields[field]['dataEdgeDiff'], -self.Fields[field]['dataEdgeDiff']) );

                    # Calculate standard deivation
                    self.Fields[field]['dataEdgeDiffSTD'] = np.nanstd( np.append(self.Fields[field]['dataEdgeDiffIQRFiltered'], -self.Fields[field]['dataEdgeDiffIQRFiltered']) );
                    #self.Fields[field]['dataEdgeDiffSTD'] = np.nanstd( self.Fields[field]['dataEdgeDiffIQRFiltered'] );

                    ## Define dataRange with dataEdgeDiffIQRFiltered
                    self.Fields[field]['dataEdgeDiffRange'] = np.max(self.Fields[field]['dataEdgeDiffIQRFiltered']) - np.min(self.Fields[field]['dataEdgeDiffIQRFiltered'])
                    
                    ## Define a dictionary to hold the weight parameters
                    self.Fields[field]['weightMethodPara'] = {};

                    ## Define the multi field merger method
                    try:
                        edgeWeightMethod['multiFieldMethod'];
                    except:
                        edgeWeightMethod['multiFieldMethod'] = "mean";

                    ## Set lower bound for difference value to influence connectivity
                    ## Assuming a normal distribution
                    ## factor = 5: Strength in node connection changes over 84% data with greater variation than the 16% with the lowest variation.
                    ## factor = 4: Strength in node connection changes over 80% data with greater variation than the 20% with the lowest variation.
                    ## factor = 3: Strength in node connection changes over 74% data with greater variation than the 26% with the lowest variation.
                    ## factor = 2: Strength in node connection changes over 57% data with greater variation than the 38% with the lowest variation.
                    ## factor = 1: Strength in node connection changes over 0% data with greater variation than the 68% with the lowest variation.
                    factor = 2;
                    #lowerbound = dataEdgeDiffSTD/factor;
                    self.Fields[field]['weightMethodPara']['lowerbound'] = 0;
                    self.Fields[field]['weightMethodPara']['upperbound'] = self.Fields[field]['dataEdgeDiffSTD']*factor;

                    ## Define the std and dataRange to be used in the following calculations of edge weights.
                    useGlobalDifference = False;
                    useEdgeDifference = False;
                    useEdgeGravity = False;
                    useLogistic = False;
                    useNormPDFFittedSigmoid = False;
                    useQTGaussianSigmoid = False;
                    useQTGaussianShiftedGaussianWeightDistribution = True;

                    if edgeWeightMethod['method'] == "useGlobalDifference":
                        # Set method
                        useGlobalDifference = True
                    elif edgeWeightMethod['method'] == "useEdgeDifference":
                        # Set method
                        useEdgeDifference = True;
                    elif edgeWeightMethod['method'] == "useEdgeGravity":
                        # Set method
                        useEdgeGravity=True;
                    elif edgeWeightMethod['method'] == "useLogistic":
                        # Set method
                        useLogistic = True;

                        # Set method parameters - if not user defined
                        if not (np.array(list(edgeWeightMethod.keys())) == "S_at_lower").any():
                            self.Fields[field]['weightMethodPara']['S_at_lower'] = 0.1;
                        if not (np.array(list(edgeWeightMethod.keys())) == "S_at_upper").any():
                            self.Fields[field]['weightMethodPara']['S_at_upper'] = 0.9;
                        if not (np.array(list(edgeWeightMethod.keys())) == "factor_at_lower").any():
                            self.Fields[field]['weightMethodPara']['lowerbound'] = self.Fields[field]['dataEdgeDiffSTD']*1
                        else:
                            self.Fields[field]['weightMethodPara']['lowerbound'] = self.Fields[field]['dataEdgeDiffSTD']*edgeWeightMethod['factor_at_lower']
                        if not (np.array(list(edgeWeightMethod.keys())) == "factor_at_upper").any():
                            self.Fields[field]['weightMethodPara']['upperbound'] = self.Fields[field]['dataEdgeDiffSTD']*2
                        else:
                            self.Fields[field]['weightMethodPara']['upperbound'] = self.Fields[field]['dataEdgeDiffSTD']*edgeWeightMethod['factor_at_upper']

                    elif edgeWeightMethod['method'] == "useNormPDFFittedSigmoid":
                        # Set method
                        useNormPDFFittedSigmoid = True;
                    elif edgeWeightMethod['method'] == "useQTGaussianSigmoid":
                        # Set method
                        useQTGaussianSigmoid = True;
                    elif edgeWeightMethod['method'] == "useQTGaussianShiftedGaussianWeightDistribution":
                        # Set method
                        useQTGaussianShiftedGaussianWeightDistribution = True;

                        # Set method parameters - if not user defined
                        ## The factor of standard deviations to shorten the CDF distribution by.
                        if not (np.array(list(edgeWeightMethod.keys())) == "shortenFactor").any():
                            edgeWeightMethod['shortenFactor'] = 3;
                        ## The factor of standard deviations to shift the CDF distribution by.
                        if not (np.array(list(edgeWeightMethod.keys())) == "shiftFactor").any():
                            edgeWeightMethod['shiftFactor'] = 1;
                        ## The minimum value used for edge weights
                        if not (np.array(list(edgeWeightMethod.keys())) == "minWeight").any():
                            edgeWeightMethod['minWeight'] = 0.01;

                    if useEdgeDifference and not (useGlobalDifference | useEdgeGravity):
                        self.Fields[field]['weightMethodPara']['dataSTD']   = self.Fields[field]['dataEdgeDiffSTD'];
                        self.Fields[field]['weightMethodPara']['dataRange'] = self.Fields[field]['dataEdgeDiffRange'];
                        ## Define the weight (S) at the upper bound of values1-values2 difference
                        self.Fields[field]['weightMethodPara']['S_at_upperbound'] = .05

                        ## Calculate the stretch factor for the exponential decay, such that
                        ## S(lowerbound) = 1 and S(upperbound) = S_at_upperbound.
                        self.Fields[field]['weightMethodPara']['stretchEdgeDifference'] = (self.Fields[field]['weightMethodPara']['lowerbound']-self.Fields[field]['weightMethodPara']['upperbound'])/np.log(self.Fields[field]['weightMethodPara']['S_at_upperbound'])/self.Fields[field]['weightMethodPara']['dataSTD']

                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -1;

                    elif useGlobalDifference and not (useEdgeDifference or useEdgeGravity):
                        ## Define the range of input node edge values
                        self.Fields[field]['weightMethodPara']['dataRange'] = np.nanmax(self.Fields[field]['interpolatedData'])-np.nanmin(self.Fields[field]['interpolatedData']);

                        ## Define the std of the input node edge values. Node should be
                        ## representing equal area, so no weights for the std need to be defined.
                        self.Fields[field]['weightMethodPara']['dataSTD'] = np.nanstd(self.Fields[field]['interpolatedData']);

                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -1;

                    elif useEdgeGravity and not (useEdgeDifference or useGlobalDifference):
                        ## Define the range of input node edge values
                        self.Fields[field]['weightMethodPara']['dataRange'] = np.nanmax(self.Fields[field]['interpolatedData'])-np.nanmin(self.Fields[field]['interpolatedData']);

                        ## Define the std of the input node edge values. Node should be
                        ## representing equal area, so no weights for the std need to be defined.
                        self.Fields[field]['weightMethodPara']['dataSTD'] = np.nanstd(self.Fields[field]['interpolatedData']);

                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -2;

                    elif useLogistic:
                        # Create attribute dictionary for logistic edge weight method
                        self.Fields[field]['weightMethodPara']['logisticAttributes'] = {};

                        # Define some attributes for the logistic edge weight method
                        # S(property_difference=lowerbound) = S_at_lower
                        # S(property_difference=upperbound) = S_at_upper
                        self.Fields[field]['weightMethodPara']['S_at_lower'] = 0.1;
                        self.Fields[field]['weightMethodPara']['S_at_upper'] = 0.9;
                        self.Fields[field]['weightMethodPara']['lowerbound'] = self.Fields[field]['dataEdgeDiffSTD']
                        self.Fields[field]['weightMethodPara']['upperbound'] = self.Fields[field]['dataEdgeDiffSTD']*factor

                        # Define attributes for the logistic edge weight method
                        # logisticAttributes["L"]       : Maximum value of logistic curve
                        # logisticAttributes["k"]       : Controls rate of change of curve 
                        # logisticAttributes["shift"]   : Controls the range of values with near logisticAttributes["L"] values.
                        self.Fields[field]['weightMethodPara']['logisticAttributes']["L"] = 1
                        
                        xl = np.log( (self.Fields[field]['weightMethodPara']['logisticAttributes']["L"]-self.Fields[field]['weightMethodPara']['S_at_upper'])/self.Fields[field]['weightMethodPara']['S_at_upper'] )
                        xu = np.log( (self.Fields[field]['weightMethodPara']['logisticAttributes']["L"]-self.Fields[field]['weightMethodPara']['S_at_lower'])/self.Fields[field]['weightMethodPara']['S_at_lower'] )
                        self.Fields[field]['weightMethodPara']['logisticAttributes']["k"] = -1*(xl-xu)/(self.Fields[field]['weightMethodPara']['lowerbound']-self.Fields[field]['weightMethodPara']['upperbound'])
                        self.Fields[field]['weightMethodPara']['logisticAttributes']["shift"] = self.Fields[field]['weightMethodPara']['logisticAttributes']["k"]*self.Fields[field]['weightMethodPara']['lowerbound'] + xl
                        
                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -1;
                        
                    elif useNormPDFFittedSigmoid:
                        # Import
                        from scipy.stats import norm
                        from scipy.optimize import curve_fit

                        # Define sigmoid function for fitting
                        def sigmoid(x, L, k, s):
                            '''
                            
                            '''
                            y = -L / (1 + np.exp(-k*np.abs(x)+s))
                            return y

                        # Calculate the PDF with data mirror at zero. Mirroring is
                        # done since dataEdgeDiffIQRFiltered is constructed as
                        # np.abs(value1-value2).
                        self.Fields[field]['weightMethodPara']['pdf'] = \
                            norm.pdf(np.append(self.Fields[field]['dataEdgeDiffIQRFiltered'],
                                            -self.Fields[field]['dataEdgeDiffIQRFiltered']),
                                    0,
                                    self.Fields[field]['dataEdgeDiffSTD'])

                        ## Initial guess for parameters
                        ## These need to be automatically chosen in an appropriate way: FIXME
                        self.Fields[field]['weightMethodPara']['p0'] = \
                            [max(self.Fields[field]['dataEdgeDiffIQRFiltered']),
                            self.Fields[field]['dataEdgeDiffSTD'],
                            self.Fields[field]['dataEdgeDiffSTD']]

                        ## Fit the curve for data mirror at zero. Mirroring is
                        # done since the pdf is constructed with mirrored
                        # data.
                        self.Fields[field]['weightMethodPara']['popt'], self.Fields[field]['weightMethodPara']['pcov'] = \
                            curve_fit(sigmoid,
                                    np.append(self.Fields[field]['dataEdgeDiffIQRFiltered'],
                                                -self.Fields[field]['dataEdgeDiffIQRFiltered']),
                                    self.Fields[field]['weightMethodPara']['pdf'],
                                    self.Fields[field]['weightMethodPara']['p0'],
                                    method='trf')
                        
                        verbose = True;
                        if verbose:
                            xValues = np.append(self.Fields[field]['dataEdgeDiffIQRFiltered'], -self.Fields[field]['dataEdgeDiffIQRFiltered'])
                            # Plot subplot
                            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True);
                            # Compare pdf w/ sigmoid function 
                            plt.sca(axes[0])
                            plt.plot(xValues, self.Fields[field]['weightMethodPara']['pdf']/np.max(self.Fields[field]['weightMethodPara']['pdf']), 'b.', alpha=1, label='pdf');
                            plt.plot(xValues, sigmoid(xValues, *self.Fields[field]['weightMethodPara']['popt'])/sigmoid(0, *self.Fields[field]['weightMethodPara']['popt']), 'r.', alpha=1, label='sigmoid');
                            plt.ylabel("Normalized weight")

                            # Compare pdf, sigmoid, and original distribution of values
                            plt.sca(axes[1])
                            plt.plot(xValues, self.Fields[field]['weightMethodPara']['pdf'], 'b.', alpha=1, label='pdf');
                            plt.hist(xValues, alpha=.4, bins=np.linspace(np.min(xValues), np.max(xValues), 30), label='Data', density=True);
                            plt.plot(xValues, sigmoid(xValues, *self.Fields[field]['weightMethodPara']['popt']), 'r.', alpha=1, label='sigmoid');
                            plt.ylabel("UnNormalized weight")
                            
                            # Plot formatting
                            plt.legend();
                            plt.xminBasinCntlabel("Difference in Node Property");
                            plt.show();
                        verbose = False;

                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -1;

                    elif useQTGaussianSigmoid or useQTGaussianShiftedGaussianWeightDistribution:
                        # Create difference data to Gaussian transform
                        from sklearn.preprocessing import QuantileTransformer
                        
                        xValues = np.append(self.Fields[field]['dataEdgeDiffIQRFiltered'], -self.Fields[field]['dataEdgeDiffIQRFiltered'])
                        #xValues = cp.deepcopy(self.Fields[field]['dataEdgeDiffIQRFiltered'])
                        self.Fields[field]['weightMethodPara']['qt'] = \
                            QuantileTransformer(n_quantiles=1000,
                                                random_state=0,
                                                output_distribution='normal')
                        qtDiss  = self.Fields[field]['weightMethodPara']['qt'].fit_transform(np.reshape(xValues, (len(xValues),1)))

                        verbose = True;
                        if verbose:
                            # Create a set of equal space values in the data domain
                            # These can be plotted on the gaussian domain to see the data stretching
                            bins   = np.linspace(np.min(xValues), np.max(xValues), 20);
                            binsqt = self.Fields[field]['weightMethodPara']['qt'].transform(np.reshape(bins, (len(bins),1)))

                            # Plot subplot
                            fig, axes = plt.subplots(nrows=2, ncols=1);
                            # QT distribution
                            plt.sca(axes[0])
                            plt.hist(qtDiss, alpha=1, bins=np.arange(-6, 6, .1), label='DataFilter', density=True)
                            plt.vlines(x=binsqt, ymin=0, ymax=0.4, colors='r', alpha=.2)

                            # Compare original vs QT 
                            plt.sca(axes[1])
                            hist = plt.hist(xValues, alpha=.4, bins=bins, label='DataFilter', density=True)
                            plt.vlines(x=bins, ymin=0, ymax=np.max(hist[0]), colors='r', alpha=.2)
                            qtxValues = self.Fields[field]['weightMethodPara']['qt'].inverse_transform(qtDiss)
                            plt.hist(qtxValues, alpha=.4, bins=bins, label='QT to data', density=True);
                            
                            plt.ylabel("UnNormalized weight")

                            # Plot formatting
                            plt.legend();
                            plt.xlabel("Difference in Node Property");
                            plt.show();
                        verbose = False;

                        # Create attribute dictionary for logistic edge weight method
                        logisticAttributes = {};

                        # Define some attributes for the logistic edge weight method
                        # S(property_difference=lowerbound) = S_at_lower
                        # S(property_difference=upperbound) = S_at_upper
                        S_at_lower = 0.1;
                        S_at_upper = 0.9;
                        # Note that this assumes of QT Gaussian transformed data
                        # have a stand deviation of 1. This should be approximately
                        # true if the transform is sucessful.
                        self.Fields[field]['weightMethodPara']['qtDissSTD'] = np.std(qtDiss)
                        #lowerbound = self.Fields[field]['weightMethodPara']['qtDissSTD']*1
                        #upperbound = self.Fields[field]['weightMethodPara']['qtDissSTD']*2

                        # Define attributes for the logistic edge weight method
                        # logisticAttributes["L"]       : Maximum value of logistic curve
                        # logisticAttributes["k"]       : Controls rate of change of curve 
                        # logisticAttributes["shift"]   : Controls the range of values with near logisticAttributes["L"] values.
                        #logisticAttributes["L"] = 1
                        #xl = np.log( (logisticAttributes["L"]-S_at_upper)/S_at_upper )
                        #xu = np.log( (logisticAttributes["L"]-S_at_lower)/S_at_lower )
                        #logisticAttributes["k"] = -1*(xl-xu)/(lowerbound-upperbound)
                        #logisticAttributes["shift"] = logisticAttributes["k"]*lowerbound + xl

                        # Set distance power
                        self.Fields[field]['weightMethodPara']['disPower'] = -1;

                        # Imports
                        from scipy import stats

                    else:
                        print("No method chosen.")

                ## Iterate through each node to add edges
                node1=0;
                for i in tqdm(np.arange(len(pos[0,:]))):
                    # Iterate over all nodes

                    # Assign bathymetryi 
                    values1 = points[int(node1)][startOfFieldIdx:];
                    coords1 = G.nodes[node1]['pos'];
                    
                    # Get connection node ids
                    connections = self.eaPoint.connectionNodeIDs[i,1:]

                    for connection in connections:
                        # Iterate over connections

                        if (connection==self.eaPoint.connectionNodeIDs[:,0]).any():
                            # Connection found between value1 and value2. This would not happen
                            # if the connection was to a node over land.
                            idx2 = self.eaPoint.connectionNodeIDs[:,0][connection==self.eaPoint.connectionNodeIDs[:,0]][0]
                            node2 = np.argwhere(self.eaPoint.connectionNodeIDs[:,0]==idx2)[0][0]

                            # Assign bathymetryj
                            values2 = points[int(node2)][startOfFieldIdx:];


                            if useGlobalDifference:
                                # FIXME: Update for multiple fields
                                # Determine average property (e.g., bathymetry) between two nodes.
                                # propertyAve= (values1+values2)/2;
                                # Determine the minimum property (e.g., bathymetry) between two nodes.
                                # propertyMin= np.min(np.array([values1, values2]))
                                # Determine the absolute value of the inverse difference between two
                                # node properties. Set propertyInv to range of global data for values
                                # that are the same. Then normalize by the range 
                                SInv = 1/np.abs((values1-values2))
                                if np.isinf(SInv):
                                    SInv = 1;

                                # Determine Exponential decaying property weight between two
                                stretch = .2
                                SExp = ( np.exp( (np.abs(dataRange) - np.abs(values1-values2))/(stretch*dataSTD) ) ) / np.exp( (np.abs(dataRange)/(stretch*dataSTD) ) )

                                # Calculate parabolic relationship for weight (AijExp and AijExpdata)
                                ## BreakPoint for exp-parabolic relationship change
                                breakPointWeight = stretch; # should be set to where curves are equal
                                breakPointWeight = 1
                                breakPointDiff = (dataRange - (stretch*dataSTD)*np.log(breakPointWeight*np.exp(dataRange/(stretch*dataSTD))) )

                                ## Recalse Inverse weight 
                                if not (breakPointWeight == 1):
                                    SInv     = SInv    * (breakPointWeight*breakPointDiff)

                                ## Use inverse weight if difference in values are large.
                                if SExp<breakPointWeight:
                                    S = SInv;
                                elif SExp>=breakPointWeight:
                                    S = SExp;

                                # Note that setting the breakPointWeight to the following values
                                # has the following affect
                                #
                                # breakPointWeight = 0       --> Uses only Exponential relationship
                                # breakPointWeight > 1       --> Uses only Inverse relationship
                                # 0 < breakPointWeight < 1   --> Uses both Exponential & Inverse relationship. Used relationship depends on how similar properties are.
                            elif useEdgeDifference:
                                # FIXME: Update for multiple fields
                                ## Define the upper and lower bound of the property difference (np.abs(values1-values2))
                                ## for an exponentially decaying of weight is calculated.
                                ## Value is calculated above.
                                # lowerbound = dataEdgeDiffSTD*factor;
                                # upperbound = dataEdgeDiffSTD/factor;
                                
                                ## Define the weight (S) at the upper bound of values1-values2 difference
                                ## Value is set above.
                                # S_at_upperbound = .1

                                ## Calculate the stretch factor for the exponential decay, such that
                                ## S(lowerbound) = 1 and S(upperbound) = S_at_upperbound.
                                ## Value is calculated above.
                                # stretchEdgeDifference = (lowerbound-upperbound)/np.log(S_at_upperbound)/dataSTD

                                ## Calculate the weight 
                                S  = ( np.exp( (np.abs(dataRange) - (0+np.abs(values1-values2)))/(stretchEdgeDifference*dataSTD) ) );

                                ## If value is within lowerbound distance then set connection strength to a value that
                                ## normalizes to 1.
                                if np.abs(values1-values2)<=lowerbound:
                                    S = np.exp( ((np.abs(dataRange)- (lowerbound))/(stretchEdgeDifference*dataSTD) ) );

                                ## Normalize weights to weight value at lowerbound
                                S /= np.exp( ((np.abs(dataRange)- (lowerbound))/(stretchEdgeDifference*dataSTD) ) );

                                ## Set minimum S value to S_at_upperbound
                                if S < S_at_upperbound:
                                    S = S_at_upperbound;

                            elif useEdgeGravity:
                                # FIXME: Update for multiple fields
                                # Note that the gravity model represents node edges weights
                                # with (property1*property2)/(distanceV**2).

                                # Calculate the product of the property weights:
                                # The inverse distance squared is added with the
                                # later calculated nodeSpacingNormalizer.
                                S = values1*values2;

                            elif useLogistic:

                                # Create an array to hold edge weights for each input field
                                Ss = np.ones(len(values1));
                                Ss[:] = np.nan;

                                # Iterate over all the data fields stored within nodes
                                cnt=0
                                for value1, value2 in zip(values1, values2):
                                    if np.isnan(value1) | np.isnan(value2):
                                        # If current node or connecting node do not have a data field value
                                        # (i.e. don't have a connecting edge).
                                        cnt+=1
                                        continue
                                    else:
                                        # If current node or connecting node both have a data field value
                                        # (i.e. have a connecting edge).
                                        
                                        # Define current working field
                                        field = self.Fields["usedFields"][cnt];

                                        # Difference in properties at nodes
                                        diff = np.abs(value1-value2)

                                        # Use the logistic function to calculated the edge weight component S.
                                        Ss[cnt] = self.Fields[field]['weightMethodPara']['logisticAttributes']["L"]*(1+np.exp(-self.Fields[field]['weightMethodPara']['logisticAttributes']["k"]*diff+self.Fields[field]['weightMethodPara']['logisticAttributes']["shift"]))**(-1)

                                        # Move data field index
                                        cnt+=1

                                # Merge field weights, default is mean
                                if edgeWeightMethod['multiFieldMethod'] == "prod":
                                    # Take the product of all fields
                                    S = np.nanprod(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "min":
                                    # Take the max of all fields
                                    S = np.nanmin(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "max":
                                    # Take the min of all fields
                                    S = np.nanmax(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "mean":
                                    # Take the mean of all fields
                                    S = np.nanmean(Ss)


                            elif useNormPDFFittedSigmoid:
                                # FIXME: Update for multiple fields
                                # Difference in properties at nodes
                                diff = np.abs(values1-values2)
                                # Use the pdf fitted sigmoid function
                                # to calculate the edge weight component S.
                                S = sigmoid(diff, *self.popt)

                            elif useQTGaussianSigmoid:
                                # FIXME: Update for multiple fields
                                # Difference in properties at nodes
                                diff = np.abs(values1-values2);
                                # Transform from diff-space to gaussian-space
                                QTGdiff = qt.transform( np.reshape( np.array(diff), (1,1) ) );

                                # Apply stretch factor after QTGdiff is defined with a Guassian Transformer.
                                # This will give less weight to tail values of the distribution
                                QTGdiffStretch = 0.1; # Decimal percentage to stretch the QTGdiff value.
                                QTGdiff *= (1 + QTGdiffStretch);

                                # Use the logistic function to calculated the edge weight component S.
                                S = logisticAttributes["L"]*(1+np.exp(-logisticAttributes["k"]*QTGdiff+logisticAttributes["shift"]))**(-1)

                            elif useQTGaussianShiftedGaussianWeightDistribution:
                                # This method does the following to calculate weights
                                # 1. Filter outliers from difference data (using IQR method)
                                # 2. Convert difference data into gaussian (using QT method)
                                # 3. Calculate z-score of difference data between nodei and nodej
                                # 4. Given the z-score from step 3) calculate a CDF (0-1) value on a
                                # new distribution centered at 1 sigma (from the first distribution)
                                # and with a std of sigma/2 (from the first distribution).
                                # 4. Define weight as S=(1-CDF) 


                                # Create an array to hold edge weights for each input field 
                                Ss = np.ones(len(values1));
                                Ss[:] = np.nan;

                                # Iterate over all the data fields stored within nodes
                                cnt=0
                                for value1, value2 in zip(values1, values2):
                                    if np.isnan(value1) | np.isnan(value2):
                                        # If current node or connecting node do not have a data field value
                                        # (i.e. don't have a connecting edge).
                                        cnt+=1
                                        continue
                                    else:
                                        # If current node or connecting node both have a data field value
                                        # (i.e. have a connecting edge).

                                        # The factor of standard deviations to shorten the CDF distribution by.
                                        shortenFactor = edgeWeightMethod['shortenFactor']
                                        # The factor of standard deviations to shift the CDF distribution by.
                                        shiftFactor = edgeWeightMethod['shiftFactor']

                                        # Difference in properties at nodes
                                        diff = np.abs(value1-value2);
                                        # Transform from diff-space to gaussian-space
                                        QTGdiff = self.Fields[field]['weightMethodPara']['qt'].transform( np.reshape( np.array(diff), (1,1) ) );
                                        # Get probablity in stretched distribution
                                        cdfCenter  = self.Fields[field]['weightMethodPara']['qtDissSTD']*shiftFactor
                                        cdfStretch = self.Fields[field]['weightMethodPara']['qtDissSTD']/shortenFactor
                                        CDF = stats.norm.cdf(QTGdiff, loc=cdfCenter, scale=cdfStretch)
                                        # Divide by probablity in normal distribution. This
                                        # scales probablility between 0-1.
                                        # Note that:
                                        #   S->1 for |value1 - value2|-> 0   and
                                        #   S->0 for |value1 - value2|-> inf
                                        Ss[cnt] = ( (1-CDF) + edgeWeightMethod['minWeight'] )/(edgeWeightMethod['minWeight']+1);

                                        # Move data field index
                                        cnt+=1

                                # Merge field weights, default is mean
                                if edgeWeightMethod['multiFieldMethod'] == "prod":
                                    # Take the product of all fields
                                    S = np.nanprod(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "min":
                                    # Take the max of all fields
                                    S = np.nanmin(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "max":
                                    # Take the min of all fields
                                    S = np.nanmax(Ss)
                                elif edgeWeightMethod['multiFieldMethod'] == "mean":
                                    # Take the mean of all fields
                                    S = np.nanmean(Ss)

                            # Note that this weight contains node spacing information
                            # (i.e., change in node density with latitude and increased \
                            # strength in with high latitude... )
                            coords2 = G.nodes[node2]['pos'];
                            distanceV = haversine_distance(coords1[0], coords1[1],
                                                        coords2[0], coords2[1],
                                                        1);
                            nodeSpacingNormalizer = distanceV**self.Fields[field]['weightMethodPara']['disPower'];
                            

                            # Set edge
                            G.add_edge(node1, node2, bathyAve=S*nodeSpacingNormalizer);

                    # Iterate node counter
                    node1+=1;
                
            
                # Set some class parameters for testing purposes.
                self.G = G;
            
            else:
                G = self.G

            # Look through all nodes and check for more than 4 connections
            if verbose:
                nodes = [self.G.degree[i] for i in range(len(self.G.degree))]
                edgeNode3 = np.argwhere(np.array(nodes)<4).T[0]
                edgeNode5 = np.argwhere(np.array(nodes)>4).T[0]
                print(edgeNode3, "nodes have only 3 edges shared with other nodes. This should occur for 8 nodes.")
                print(edgeNode5, "nodes have 5 edges shared with other nodes. This should not occur for any nodes.")
                del nodes, edgeNode5, edgeNode3


            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(detectionMethod = detectionMethod,
                                 method = detectionMethod["method"],
                                 minBasinCnt = detectionMethod['minBasinCnt'],
                                 minBasinLargerThanSmallMergers = detectionMethod['minBasinLargerThanSmallMergers'],
                                 resolution = detectionMethod["resolution"]);

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
            if write:
                nx.write_gml(G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
            

            ################
            ### Plotting ###
            ################
            if verbose:
                # Plot the network on a geographic map
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

                # Draw the nodes (points) on the map
                for node, data in G.nodes(data=True):
                    ax.plot(data['pos'][1], data['pos'][0], 'bo', markersize=1)  # longitude, latitude

                # Draw the edges (connections)
                for edge in G.edges(data=True):
                    node1, node2, weight = edge
                    lon1, lat1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
                    lon2, lat2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                    ax.plot([lon1, lon2], [lat1, lat2], 'k-', transform=ccrs.PlateCarree())

                # Add coastlines and gridlines
                if "Earth" in self.filename:
                    ax.coastlines()
                ax.gridlines()

                plt.title("Geographic Network of points")
                plt.show()

    def reportEvaluationMetrics(self, returnText=True, resolution="Not Recorded", ensembleSize="Not Recorded", distance_threshold="Not Recorded"):
        """
        reportEvaluationMetrics reports evaluation metrics
        for the quality of communities detected.
        
        Parameters
        -----------
        returnText : BOOLEAN
            Option to return a string of evaluation metrics
            that is useful for outputing to commandline or
            reporting to log text file.
        resolution : FLOAT
            Resolution of the field. The default is "Not Recorded".
        ensembleSize : INT
            The size of the ensemble used for the reduction step.
            The default is "Not Recorded".
        distance_threshold : FLOAT
            Distance threshold used for ensemble merge.
            The default is "Not Recorded".

        
        Return
        -------
        DICTIONARY
            A set of evaluation metrics 
        STRING
            A set of evaluation metrics
        
        """
        # Create nodeclustering object
        from cdlib import evaluation
        from cdlib import NodeClustering


        ###########################################
        ### Report community evaluation metrics ###
        ###########################################

        # Create node cluster
        # Note that the small basin mergers are not inlcuded
        # in the LGNClusters. Only large basin mergers such that
        # small basin mergers results in X chosen basins.
        ReducedClusters=NodeClustering(communities=self.Rcommunities,
                            graph=self.G,
                            method_name="consensus_ledien_fixed",
                            method_parameters={
                                "resolution_parameter": resolution,
                                "runs": ensembleSize,
                                "distance_threshold": distance_threshold}
                            )

        LGNClusters=NodeClustering(communities=self.communitiesFinal,
                            graph=self.G,
                            method_name="consensus_ledien_fixed",
                            method_parameters={
                                "resolution_parameter": resolution,
                                "runs": ensembleSize,
                                "distance_threshold": distance_threshold}
                            )
        
        # Define string
        readmetxt = "";
        
        # Define dictionary
        metrics = {};
        
        # Calculate community detection metrics
        for cluster, method in zip([ReducedClusters, LGNClusters], ["ReducedClusters", "LGNClusters"]):
            metrics[method+"-newman_girvan_modularity"] = evaluation.newman_girvan_modularity(self.G, cluster)
            metrics[method+"-internal_edge_density"] = evaluation.internal_edge_density(self.G, cluster)
            metrics[method+"-erdos_renyi_modularity"]= evaluation.erdos_renyi_modularity(self.G, cluster)
            metrics[method+"-modularity_density"]    = evaluation.modularity_density(self.G, cluster)
            metrics[method+"-avg_embeddedness"]      = evaluation.avg_embeddedness(self.G, cluster)
            metrics[method+"-conductance"]           = evaluation.conductance(self.G, cluster)
            metrics[method+"-surprise"]              = evaluation.surprise(self.G, cluster)

            # Add community evaluation metrics to output
            readmetxt += "Community evaluation metrics ({}):\n".format(method);
            readmetxt += "newman_girvan_modularity:\t {}\n".format(metrics[method+"-newman_girvan_modularity"].score)
            readmetxt += "erdos_renyi_modularity:\t\t {}\n".format(metrics[method+"-erdos_renyi_modularity"].score)
            readmetxt += "modularity_density:\t\t {}\n".format(metrics[method+"-modularity_density"].score)
            readmetxt += "internal_edge_density:\t\t {} +- {} (std)\n".format(metrics[method+"-internal_edge_density"].score, metrics[method+"-internal_edge_density"].std)
            readmetxt += "avg_embeddedness:\t\t {} +- {} (std)\n".format(metrics[method+"-avg_embeddedness"].score, metrics[method+"-avg_embeddedness"].std)
            readmetxt += "conductance:\t\t\t {} +- {} (std)\n".format(metrics[method+"-conductance"].score, metrics[method+"-conductance"].std)
            readmetxt += "surprise:\t\t\t {}\n".format(metrics[method+"-surprise"].score)
            readmetxt += "\n\n"
            
        if returnText:
            return metrics, readmetxt
        else:
            return metrics


    def defineBasinsSingleField(self, minBasinCnt = 3,
                     detectionMethod = {"method":"Louvain", "resolution":1},
                     reducedRes={"on":False,"factor":15},
                     read=False,
                     write=False,
                     verbose=True):
        """
        defineBasins method will define basins with network analysis
        using either the Girvan-Newman or Louvain algorithm to define
        communities.

        Parameter
        ----------
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.
        detectionMethod : DICTIONARY
            Determines the implemented community detection algorithm and
            other properties to use for community detection. This dictionary
            has the following keys:
                method : STRING
                    The options are "Girvan-Newman", "Louvain", or
                    "Louvain-Girvan-Newman". The former is more
                    robust with low scalability and the latter are
                    practical but produces non-deterministic communities.
                    The default is "Louvain".
                resolution : FLOAT
                    The resolution value to be used with the Louvain
                    community detection algorithm. Values greater than 1,
                    makes the algorithm favor smaller communities (more
                    communities). Values less than 1, makes the algorithm
                    favor larger communities (less communities). The default
                    is 1.
        reducedRes : DICTIONARY
            Option to reduce the resolution of the basin definition
            network calculation. Note that this should be turned
            off when doing analysis, and only kept on for testing
            purposes. The default is {"on":False,"factor":15}.
        read : BOOLEAN
            An option to read basin definitions for a given
            bathymetry model. The default is False. 
        write : BOOLEAN
            An option to write basin definitions for a given
            bathymetry model. The default is False. 
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.basinCnt : INT
            Number of basins in global bathymetry model.
        self.basinDis : NUMPY ARRAY
            binCnt x basinCnt array of bathymetry distributions.
        
        """

        ##########################
        ### Write/Load network ###
        ##########################
        if read:
            # FIXME: Needs to be updated for EAnodes
            ####################
            ### Read network ###
            ####################
            self.G = nx.read_gml("{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), destringizer=float);
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
            
            # Define resolution
            self.resolution = self.reducedRes*np.diff(self.lon)[0][0];
        
        else:
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();

            ######################
            ### Create network ###
            ######################
            # Define equal area points
            # eaPoint.lat, eaPoint.lon are created here
            if reducedRes['on']:
                self.EAinputs["resolution"] *= reducedRes['factor'];
            self.eaPoint = eaNodes(inputs=self.EAinputs);

            # Creates
            # 1) Set of nodes that represent equal area quadrangles.
            # 2) Define the connects between all nodes (even to nodes
            # with missing data)
            self.eaPoint.makegrid(plotq=0);

            # Simple netCDF4 for interpolation inputPath="path/file.nc", outputPath
            self.simplifyNetCDF(inputPath=self.EAinputs['dataGrid'],
                                outputPath='tempSimp.nc',
                                parameterIn=self.EAinputs['parameter'],
                                parameterOut=self.EAinputs['parameter'])

            # Interpolate from grided nodes to equal area nodes
            self.eaPoint.interp2IrregularGrid(path='tempSimp.nc',name='depth')
            self.areaWeighti = (4*np.pi*(self.radius)**2)/len(self.eaPoint.data['depth']);      # Area covered by a node m2

            # Remove points with no depth data
            allNodes = False;
            if allNodes:
                self.eaPoint.ealat = self.eaPoint.ealat;
                self.eaPoint.ealon = self.eaPoint.ealon;
                self.eaPoint.connectionNodeIDs = self.eaPoint.connectionNodeIDs
                self.eaPoint.depth = self.eaPoint.data['depth'];
            else:
                self.eaPoint.ealat = self.eaPoint.ealat[~np.isnan(self.eaPoint.data['depth'])];
                self.eaPoint.ealon = self.eaPoint.ealon[~np.isnan(self.eaPoint.data['depth'])];
                self.eaPoint.connectionNodeIDs = self.eaPoint.connectionNodeIDs[~np.isnan(self.eaPoint.data['depth'])]
                self.eaPoint.depth = self.eaPoint.data['depth'][~np.isnan(self.eaPoint.data['depth'])];

            # Define counter and point dictionary
            cnt = 0.
            points = {};

            # Create dictionary and array of bathymetry points
            pos = np.zeros( (2, len(~np.isnan(self.eaPoint.depth))) );
            for i in tqdm( range(len(self.eaPoint.ealon)) ):
                bathymetryi = self.eaPoint.depth[i];
                #if (~np.isnan(bathymetryi)):
                points[int(cnt)] = (self.eaPoint.ealat[i], self.eaPoint.ealon[i], bathymetryi, self.areaWeighti);    # (latitude, longitude, depth, areaWeight) w/ units (deg, deg, m, m2)
                pos[:,int(cnt)] = np.array( [self.eaPoint.ealat[i], self.eaPoint.ealon[i]] ); 
                # Iterate node counter
                cnt+=1;

            # Create a graph
            G = nx.Graph()

            ## Add nodes (points)
            for node, values in points.items():
                #if not np.isnan(values[2]):
                G.add_node(node, pos=values[0:2], depth=values[2], areaWeightm2=values[3]);

            
            ## Create a list of property difference between connected nodes
            ### Assign and empty vector to self.dataEdgeDiff 
            self.dataEdgeDiff = np.array([], dtype=np.float64)

            ### Iterate through each node to add edges
            node1=0;
            for i in tqdm(np.arange(len(pos[0,:]))):
                # Iterate over all nodes

                # Assign bathymetryi 
                values1 = points[int(node1)][2];
                coords1 = G.nodes[node1]['pos'];
                
                # Get connection node ids
                connections = self.eaPoint.connectionNodeIDs[i,1:]

                for connection in connections:
                    # Iterate over connections

                    if (connection==self.eaPoint.connectionNodeIDs[:,0]).any():
                        # Connection found between value1 and value2. This would not happen
                        # if the connection was to a node over land.
                        idx2 = self.eaPoint.connectionNodeIDs[:,0][connection==self.eaPoint.connectionNodeIDs[:,0]][0]
                        node2 = np.argwhere(self.eaPoint.connectionNodeIDs[:,0]==idx2)[0][0]

                        # Assign bathymetryj
                        values2 = points[int(node2)][2];

                        # Assign difference to dataEdgeDiff vector
                        self.dataEdgeDiff = np.append(self.dataEdgeDiff, np.abs(values1-values2))

                # Iterate node counter
                node1+=1;
            ## Remove outliers from self.dataEdgeDiff and calculate std
            def remove_outliers_iqr(data):
                """
                remove_outliers_iqr function removes outliers from 
                a Numpy array using the IQR method.

                Parameters
                -----------
                data : NUMPY ARRAY
                    The input NumPy array.

                Returns
                --------
                filtered_data : NUMPY ARRAY
                    A new NumPy array with outliers removed.
                """
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                return filtered_data

            ### Get outliers filtered dataEdgeDiff using the IQR method.
            self.dataEdgeDiffIQRFiltered = remove_outliers_iqr(self.dataEdgeDiff);

            ## Mirror dataEdgeDiffIQRFiltered about zero when finding the std.
            ## This is appropriate since each edge is bidirectional.
            ## As a result, the mean should be zero.
            dataEdgeDiffSTD = np.std( np.append(self.dataEdgeDiffIQRFiltered, -self.dataEdgeDiffIQRFiltered) );

            ## Define dataRange with self.dataEdgeDiffIQRFiltered
            dataEdgeDiffRange = np.max(self.dataEdgeDiffIQRFiltered) - np.min(self.dataEdgeDiffIQRFiltered)

            ## Set lower bound for difference value to influence connectivity
            ## Assuming a normal distribution
            ## factor = 5: Strength in node connection changes over 84% data with greater variation than the 16% with the lowest variation.
            ## factor = 4: Strength in node connection changes over 80% data with greater variation than the 20% with the lowest variation.
            ## factor = 3: Strength in node connection changes over 74% data with greater variation than the 26% with the lowest variation.
            ## factor = 2: Strength in node connection changes over 57% data with greater variation than the 38% with the lowest variation.
            ## factor = 1: Strength in node connection changes over 0% data with greater variation than the 68% with the lowest variation.
            factor = 2;
            #lowerbound = dataEdgeDiffSTD/factor;
            lowerbound = 0;
            upperbound = dataEdgeDiffSTD*factor;

            ## Define the std and dataRange to be used in the following calculations of edge weights.
            useGlobalDifference = False;
            useEdgeDifference = False;
            useEdgeGravity = False;
            useLogistic = False;
            useNormPDFFittedSigmoid = False;
            useQTGaussianSigmoid = False;
            useQTGaussianShiftedGaussianWeightDistribution = True;
            if useEdgeDifference and not (useGlobalDifference | useEdgeGravity):
                dataSTD   = dataEdgeDiffSTD;
                dataRange = dataEdgeDiffRange;
                ## Define the weight (S) at the upper bound of values1-values2 difference
                S_at_upperbound = .05

                ## Calculate the stretch factor for the exponential decay, such that
                ## S(lowerbound) = 1 and S(upperbound) = S_at_upperbound.
                stretchEdgeDifference = (lowerbound-upperbound)/np.log(S_at_upperbound)/dataSTD

                # Set distance power
                disPower = -1;

            elif useGlobalDifference and not (useEdgeDifference or useEdgeGravity):
                ## Define the range of input node edge values
                dataRange = np.max(self.eaPoint.depth)-np.min(self.eaPoint.depth);

                ## Define the std of the input node edge values. Node should be
                ## representing equal area, so no weights for the std need to be defined.
                dataSTD = np.std(self.eaPoint.depth);

                # Set distance power
                disPower = -1;

            elif useEdgeGravity and not (useEdgeDifference or useGlobalDifference):
                ## Define the range of input node edge values
                dataRange = np.max(self.eaPoint.depth)-np.min(self.eaPoint.depth);

                ## Define the std of the input node edge values. Node should be
                ## representing equal area, so no weights for the std need to be defined.
                dataSTD = np.std(self.eaPoint.depth);

                # Set distance power
                disPower = -2;
            elif useLogistic:
                # Create attribute dictionary for logistic edge weight method
                logisticAttributes = {};

                # Define some attributes for the logistic edge weight method
                # S(property_difference=lowerbound) = S_at_lower
                # S(property_difference=upperbound) = S_at_upper
                S_at_lower = 0.1;
                S_at_upper = 0.9;
                lowerbound = dataEdgeDiffSTD
                upperbound = dataEdgeDiffSTD*factor

                # Define attributes for the logistic edge weight method
                # logisticAttributes["L"]       : Maximum value of logistic curve
                # logisticAttributes["k"]       : Controls rate of change of curve 
                # logisticAttributes["shift"]   : Controls the range of values with near logisticAttributes["L"] values.
                logisticAttributes["L"] = 1
                xl = np.log( (logisticAttributes["L"]-S_at_upper)/S_at_upper )
                xu = np.log( (logisticAttributes["L"]-S_at_lower)/S_at_lower )
                logisticAttributes["k"] = -1*(xl-xu)/(lowerbound-upperbound)
                logisticAttributes["shift"] = logisticAttributes["k"]*lowerbound + xl
                
                # Set distance power
                disPower = -1;
            elif useNormPDFFittedSigmoid:
                # Import
                from scipy.stats import norm
                from scipy.optimize import curve_fit

                # Define sigmoid function for fitting
                def sigmoid(x, L, k, s):
                    '''
                    
                    '''
                    y = -L / (1 + np.exp(-k*np.abs(x)+s))
                    return y

                # Calculate the PDF with data mirror at zero. Mirroring is
                # done since dataEdgeDiffIQRFiltered is constructed as
                # np.abs(value1-value2).
                self.pdf = norm.pdf(np.append(self.dataEdgeDiffIQRFiltered, -self.dataEdgeDiffIQRFiltered),
                                0,
                                dataEdgeDiffSTD)

                ## Initial guess for parameters
                ## These need to be automatically chosen in an appropriate way: FIXME
                p0 = [max(self.dataEdgeDiffIQRFiltered),
                        dataEdgeDiffSTD,
                        dataEdgeDiffSTD]

                ## Fit the curve for data mirror at zero. Mirroring is
                # done since the pdf is constructed with mirrored
                # data.
                self.popt, self.pcov = curve_fit(sigmoid,
                                                    np.append(self.dataEdgeDiffIQRFiltered,
                                                            -self.dataEdgeDiffIQRFiltered),
                                                    self.pdf,
                                                    p0,
                                                    method='trf')
                
                verbose = True;
                if verbose:
                    xValues = np.append(self.dataEdgeDiffIQRFiltered, -self.dataEdgeDiffIQRFiltered)
                    # Plot subplot
                    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True);
                    # Compare pdf w/ sigmoid function 
                    plt.sca(axes[0])
                    plt.plot(xValues, self.pdf/np.max(self.pdf), 'b.', alpha=1, label='pdf');
                    plt.plot(xValues, sigmoid(xValues, *self.popt)/sigmoid(0, *self.popt), 'r.', alpha=1, label='sigmoid');
                    plt.ylabel("Normalized weight")

                    # Compare pdf, sigmoid, and original distribution of values
                    plt.sca(axes[1])
                    plt.plot(xValues, self.pdf, 'b.', alpha=1, label='pdf');
                    plt.hist(xValues, alpha=.4, bins=np.linspace(np.min(xValues), np.max(xValues), 30), label='Data', density=True);
                    plt.plot(xValues, sigmoid(xValues, *self.popt), 'r.', alpha=1, label='sigmoid');
                    plt.ylabel("UnNormalized weight")
                    
                    # Plot formatting
                    plt.legend();
                    plt.xlabel("Difference in Node Property");
                    plt.show();
                verbose = False;

                # Set distance power
                disPower = -1;
            elif useQTGaussianSigmoid or useQTGaussianShiftedGaussianWeightDistribution:
                # Create difference data to Gaussian transform
                from sklearn.preprocessing import QuantileTransformer
                xValues = np.append(self.dataEdgeDiffIQRFiltered, -self.dataEdgeDiffIQRFiltered)
                qt = QuantileTransformer(n_quantiles=1000, random_state=0,  output_distribution='normal')
                qtDiss  = qt.fit_transform(np.reshape(xValues, (len(xValues),1)))

                verbose = True;
                if verbose:
                    # Create a set of equal space values in the data domain
                    # These can be plotted on the gaussian domain to see the data stretching
                    bins   = np.linspace(np.min(xValues), np.max(xValues), 20);
                    binsqt = qt.transform(np.reshape(bins, (len(bins),1)))

                    # Plot subplot
                    fig, axes = plt.subplots(nrows=2, ncols=1);
                    # QT distribution
                    plt.sca(axes[0])
                    plt.hist(qtDiss, alpha=1, bins=np.arange(-6, 6, .1), label='DataFilter', density=True)
                    plt.vlines(x=binsqt, ymin=0, ymax=0.4, colors='r', alpha=.2)

                    # Compare original vs QT 
                    plt.sca(axes[1])
                    hist = plt.hist(xValues, alpha=.4, bins=bins, label='DataFilter', density=True)
                    plt.vlines(x=bins, ymin=0, ymax=np.max(hist[0]), colors='r', alpha=.2)
                    qtxValues = qt.inverse_transform(qtDiss)
                    plt.hist(qtxValues, alpha=.4, bins=bins, label='QT to data', density=True);
                    
                    plt.ylabel("UnNormalized weight")

                    # Plot formatting
                    plt.legend();
                    plt.xlabel("Difference in Node Property");
                    plt.show();
                verbose = False;

                # Create attribute dictionary for logistic edge weight method
                logisticAttributes = {};

                # Define some attributes for the logistic edge weight method
                # S(property_difference=lowerbound) = S_at_lower
                # S(property_difference=upperbound) = S_at_upper
                S_at_lower = 0.1;
                S_at_upper = 0.9;
                # Note that this assumes of QT Gaussian transformed data
                # have a stand deviation of 1. This should be approximately
                # true if the transform is sucessful.
                qtDissSTD = np.std(qtDiss)
                lowerbound = qtDissSTD*1
                upperbound = qtDissSTD*2

                # Define attributes for the logistic edge weight method
                # logisticAttributes["L"]       : Maximum value of logistic curve
                # logisticAttributes["k"]       : Controls rate of change of curve 
                # logisticAttributes["shift"]   : Controls the range of values with near logisticAttributes["L"] values.
                logisticAttributes["L"] = 1
                xl = np.log( (logisticAttributes["L"]-S_at_upper)/S_at_upper )
                xu = np.log( (logisticAttributes["L"]-S_at_lower)/S_at_lower )
                logisticAttributes["k"] = -1*(xl-xu)/(lowerbound-upperbound)
                logisticAttributes["shift"] = logisticAttributes["k"]*lowerbound + xl


                # Set distance power
                disPower = -1;

                from scipy import stats
            else:
                print("No method chosen.")

            ## Iterate through each node to add edges
            node1=0;
            for i in tqdm(np.arange(len(pos[0,:]))):
                # Iterate over all nodes

                # Assign bathymetryi 
                values1 = points[int(node1)][2];
                coords1 = G.nodes[node1]['pos'];
                
                # Get connection node ids
                connections = self.eaPoint.connectionNodeIDs[i,1:]

                for connection in connections:
                    # Iterate over connections

                    if (connection==self.eaPoint.connectionNodeIDs[:,0]).any():
                        # Connection found between value1 and value2. This would not happen
                        # if the connection was to a node over land.
                        idx2 = self.eaPoint.connectionNodeIDs[:,0][connection==self.eaPoint.connectionNodeIDs[:,0]][0]
                        node2 = np.argwhere(self.eaPoint.connectionNodeIDs[:,0]==idx2)[0][0]

                        # Assign bathymetryj
                        values2 = points[int(node2)][2];


                        if useGlobalDifference:
                            # Determine average property (e.g., bathymetry) between two nodes.
                            # propertyAve= (values1+values2)/2;
                            # Determine the minimum property (e.g., bathymetry) between two nodes.
                            # propertyMin= np.min(np.array([values1, values2]))
                            # Determine the absolute value of the inverse difference between two
                            # node properties. Set propertyInv to range of global data for values
                            # that are the same. Then normalize by the range 
                            SInv = 1/np.abs((values1-values2))
                            if np.isinf(SInv):
                                SInv = 1;

                            # Determine Exponential decaying property weight between two
                            stretch = .2
                            SExp = ( np.exp( (np.abs(dataRange) - np.abs(values1-values2))/(stretch*dataSTD) ) ) / np.exp( (np.abs(dataRange)/(stretch*dataSTD) ) )

                            # Calculate parabolic relationship for weight (AijExp and AijExpdata)
                            ## BreakPoint for exp-parabolic relationship change
                            breakPointWeight = stretch; # should be set to where curves are equal
                            breakPointWeight = 1
                            breakPointDiff = (dataRange - (stretch*dataSTD)*np.log(breakPointWeight*np.exp(dataRange/(stretch*dataSTD))) )

                            ## Recalse Inverse weight 
                            if not (breakPointWeight == 1):
                                SInv     = SInv    * (breakPointWeight*breakPointDiff)

                            ## Use inverse weight if difference in values are large.
                            if SExp<breakPointWeight:
                                S = SInv;
                            elif SExp>=breakPointWeight:
                                S = SExp;

                            # Note that setting the breakPointWeight to the following values
                            # has the following affect
                            #
                            # breakPointWeight = 0       --> Uses only Exponential relationship
                            # breakPointWeight > 1       --> Uses only Inverse relationship
                            # 0 < breakPointWeight < 1   --> Uses both Exponential & Inverse relationship. Used relationship depends on how similar properties are.
                        elif useEdgeDifference:
                            ## Define the upper and lower bound of the property difference (np.abs(values1-values2))
                            ## for an exponentially decaying of weight is calculated.
                            ## Value is calculated above.
                            # lowerbound = dataEdgeDiffSTD*factor;
                            # upperbound = dataEdgeDiffSTD/factor;
                            
                            ## Define the weight (S) at the upper bound of values1-values2 difference
                            ## Value is set above.
                            # S_at_upperbound = .1

                            ## Calculate the stretch factor for the exponential decay, such that
                            ## S(lowerbound) = 1 and S(upperbound) = S_at_upperbound.
                            ## Value is calculated above.
                            # stretchEdgeDifference = (lowerbound-upperbound)/np.log(S_at_upperbound)/dataSTD

                            ## Calculate the weight 
                            S  = ( np.exp( (np.abs(dataRange) - (0+np.abs(values1-values2)))/(stretchEdgeDifference*dataSTD) ) );

                            ## If value is within lowerbound distance then set connection strength to a value that
                            ## normalizes to 1.
                            if np.abs(values1-values2)<=lowerbound:
                                S = np.exp( ((np.abs(dataRange)- (lowerbound))/(stretchEdgeDifference*dataSTD) ) );

                            ## Normalize weights to weight value at lowerbound
                            S /= np.exp( ((np.abs(dataRange)- (lowerbound))/(stretchEdgeDifference*dataSTD) ) );

                            ## Set minimum S value to S_at_upperbound
                            if S < S_at_upperbound:
                                S = S_at_upperbound;

                        elif useEdgeGravity:
                            # Note that the gravity model represents node edges weights
                            # with (property1*property2)/(distanceV**2).

                            # Calculate the product of the property weights:
                            # The inverse distance squared is added with the
                            # later calculated nodeSpacingNormalizer.
                            S = values1*values2;

                        elif useLogistic:
                            # Difference in properties at nodes
                            diff = np.abs(values1-values2)
                            # Use the logistic function to calculated the edge weight component S.
                            S = logisticAttributes["L"]*(1+np.exp(-logisticAttributes["k"]*diff+logisticAttributes["shift"]))**(-1)

                        elif useNormPDFFittedSigmoid:
                            # Difference in properties at nodes
                            diff = np.abs(values1-values2)
                            # Use the pdf fitted sigmoid function
                            # to calculate the edge weight component S.
                            S = sigmoid(diff, *self.popt)

                        elif useQTGaussianSigmoid:
                            # Difference in properties at nodes
                            diff = np.abs(values1-values2);
                            # Transform from diff-space to gaussian-space
                            QTGdiff = qt.transform( np.reshape( np.array(diff), (1,1) ) );

                            # Apply stretch factor after QTGdiff is defined with a Guassian Transformer.
                            # This will give less weight to tail values of the distribution
                            QTGdiffStretch = 0.1; # Decimal percentage to stretch the QTGdiff value.
                            QTGdiff *= (1 + QTGdiffStretch);

                            # Use the logistic function to calculated the edge weight component S.
                            S = logisticAttributes["L"]*(1+np.exp(-logisticAttributes["k"]*QTGdiff+logisticAttributes["shift"]))**(-1)

                        elif useQTGaussianShiftedGaussianWeightDistribution:
                            # This method does the following to calculate weights
                            # 1. Filter outliers from difference data (using IQR method)
                            # 2. Convert difference data into gaussian (using QT method)
                            # 3. Calculate z-score of difference data between nodei and nodej
                            # 4. Given the z-score from step 3) calculate a CDF (0-1) value on a
                            # new distribution centered at 1 sigma (from the first distribution)
                            # and with a std of sigma/2 (from the first distribution).
                            # 4. Define weight as S=(1-CDF) 

                            # Difference in properties at nodes
                            diff = np.abs(values1-values2);
                            # Transform from diff-space to gaussian-space
                            QTGdiff = qt.transform( np.reshape( np.array(diff), (1,1) ) );
                            # Get probablity in stretched distribution
                            cdfCenter  = qtDissSTD*1
                            cdfStretch = qtDissSTD/3
                            CDF = stats.norm.cdf(QTGdiff, loc=cdfCenter, scale=cdfStretch)
                            # Divide by probablity in normal distribution. This
                            # scales probablility between 0-1.
                            # Note that:
                            #   S->1 for |value1 - value2|-> 0   and
                            #   S->0 for |value1 - value2|-> inf
                            S = (1-CDF);


                        # Note that this weight contains node spacing information
                        # (i.e., change in node density with latitude and increased \
                        # strength in with high latitude... )
                        coords2 = G.nodes[node2]['pos'];
                        distanceV = haversine_distance(coords1[0], coords1[1],
                                                       coords2[0], coords2[1],
                                                       1);
                        nodeSpacingNormalizer = distanceV**disPower;
                        

                        # Set edge
                        G.add_edge(node1, node2, bathyAve=S*nodeSpacingNormalizer);

                # Iterate node counter
                node1+=1;
                
            # Set some class parameters for testing purposes.
            self.G = G;

            # Look through all nodes and check for more than 4 connections
            if verbose:
                nodes = [self.G.degree[i] for i in range(len(self.G.degree))]
                edgeNode3 = np.argwhere(np.array(nodes)<4).T[0]
                edgeNode5 = np.argwhere(np.array(nodes)>4).T[0]
                print(edgeNode3, "nodes have only 3 edges shared with other nodes. This should occur for 8 nodes.")
                print(edgeNode5, "nodes have 5 edges shared with other nodes. This should not occur for any nodes.")
                del nodes, edgeNode5, edgeNode3


            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(method = detectionMethod["method"], minBasinCnt = minBasinCnt, resolution=detectionMethod["resolution"]);

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
            if write:
                nx.write_gml(G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
            

            ################
            ### Plotting ###
            ################
            if verbose:
                # Plot the network on a geographic map
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

                # Draw the nodes (points) on the map
                for node, data in G.nodes(data=True):
                    ax.plot(data['pos'][1], data['pos'][0], 'bo', markersize=1)  # longitude, latitude

                # Draw the edges (connections)
                for edge in G.edges(data=True):
                    node1, node2, weight = edge
                    lon1, lat1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
                    lon2, lat2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                    ax.plot([lon1, lon2], [lat1, lat2], 'k-', transform=ccrs.PlateCarree())

                # Add coastlines and gridlines
                if "Earth" in self.filename:
                    ax.coastlines()
                ax.gridlines()

                plt.title("Geographic Network of points")
                plt.show()


            ################
            ### Plotting ###
            ################
            
    def interp2regularGrid_python(self, dataIrregular=None, mask=True):
        """
        interp2regularGrid method is used to interpolate data to
        a regular grid given an input of irregular spaced data.

        Parameters
        -----------
        dataIrregular : NUMPY ARRAY
            3XN numpy array with columns of longitude, latitude, magnitude.
            The default is None. This will make the function define the 
            dataIrregular variable with basinIDs.
        mask : STRING
            The path to a netCDF4 file that can be used to mask the result
            of interpolation. The default is None.

        (Re)define
        -----------
        self.BasinIDA : NUMPY ARRAY
            A 2nxn array that hold basinID for each corresponding entry in
            self.lat and self.lon. 

        """

        '''
        # Save the irregular grid of data
        if dataIrregular == None:
            ## Get basin IDs from network object.
            tmpValuesID  = nx.get_node_attributes(self.G, "basinID");
            tmpValuesPos = nx.get_node_attributes(self.G, "pos");

            ## Define grid to hold irregularally spaced data
            dataIrregular = np.zeros((len(tmpValuesPos), 3))
            for i in tmpValuesID:
                dataIrregular[i,:] = np.array([tmpValuesPos[i][1], tmpValuesPos[i][0], tmpValuesID[i]['basinID']])
            np.savetxt("temp.txt", dataIrregular, delimiter='\n')

        ## Green spherical spline option
        os.system("gmt greenspline {0} -Rd -Sp -Z4 -I{1} -Ggrid.nc".format('temp.txt', self.EAinputs["resolution"]));

        ## Read the regular grid of data
        self.InterpData = Dataset('grid.nc', 'r');
        '''
        import copy as cp

        # Get basin IDs from network object.
        tmpValuesID  = nx.get_node_attributes(self.G, "basinID");
        tmpValuesPos = nx.get_node_attributes(self.G, "pos");

        # Define an array to hold longitude, latitude, and basinID
        dataIrregular = np.zeros((len(tmpValuesPos), 3))

        # Iterate over all nodes so each node's longitude, latitude,
        # and basinID can be added to the dataIrregular array.
        for i in tmpValuesID:
            dataIrregular[i,:] = np.array([tmpValuesPos[i][1], tmpValuesPos[i][0], tmpValuesID[i]['basinID']])
        
        # Define an array 2nxn to hold the basin IDs for the regular grid
        # on the surface of the a sphere (planet). 
        array = cp.deepcopy(self.lat)

        # Define a mapping function that maps node indecies on a irregular grid
        # to those on the regular grid. This will speed up calculations if this
        # function is called more than once.

        # Iterate over all latitude and longitudes of the input grid.
        for i in range(len(self.lat[:,0])):
            for j in range(len(self.lat[0,:])):
                # Find the distances from each regular grid point (i,j) to all
                # irregular grid points.
                x = haversine_distance(lat2= dataIrregular[:,1], lat1= self.lat[i,j],
                                        lon2= dataIrregular[:,0], lon1= self.lon[i,j],
                                        radius=1)

                # Assign the nearest basin ID to element (i,j) 
                array[i,j] = int(dataIrregular[np.argwhere(np.nanmin(x) == x)[0][0], 2])


        ## Apply the mask
        if mask:
            array[np.isnan(self.maskValue)] = np.nan

        
        self.BasinIDA = array;
    

    def interp2regularGrid(self,
                           dataIrregular=None,
                           mask=True,
                           propertyName="basinID"):
        """
        interp2regularGrid is a method used to interpolate
        equal area spaced nodes properties to a equal degree spaced
        grid.
        
        GMT version using nearest neighbor interpolation via nearneighbor.
        
        Parameters
        -----------
        dataIrregular : NUMPY ARRAY
            Equal area grid in a 3xn array with columns of lat, lon,
            and node property.
        mask : NUMPY ARRAY
            Mask to apply to interpolated grid.
        propertyName : STRING
            Name of property stored in graph.
        """

        # Set spatial resolution of interpolated field
        # Rounding to about 1.1 km resolution
        resolution = np.round( np.abs(np.diff(self.lat[:,0])[0]), 5)

        # 
        if dataIrregular is None:
            tmpValuesID  = nx.get_node_attributes(self.G, propertyName)
            tmpValuesPos = nx.get_node_attributes(self.G, "pos")
            dataIrregular = np.zeros((len(tmpValuesPos), 3))
            
            if propertyName == "basinID":
                # basin assignment to tmpValuesID is done
                # differently due to how it is stored
                # in nodes.
                for i in tmpValuesID:
                    dataIrregular[i, :] = np.array([
                        tmpValuesPos[i][1],  # lon
                        tmpValuesPos[i][0],  # lat
                        tmpValuesID[i][propertyName] # basinID
                    ])
            else:
                for i in tmpValuesID:
                    dataIrregular[i, :] = np.array([
                        tmpValuesPos[i][1],  # lon
                        tmpValuesPos[i][0],  # lat
                        tmpValuesID[i]       # property
                    ])
        else: 
            # Case where dataIrregular represents a list of
            # values corresponding to positions stored in
            # tmpValuesPos
            tmpValuesValues   = cp.deepcopy(dataIrregular)
            tmpValuesPos  = nx.get_node_attributes(self.G, "pos")
            dataIrregular = np.zeros((len(tmpValuesPos), 3))
            if propertyName == "basinID":
                # basin assignment to tmpValuesID is done
                # differently due to how it is stored
                # in nodes.
                for i in len(tmpValuesValues):
                    dataIrregular[i, :] = np.array([
                        tmpValuesPos[i][1],  # lon
                        tmpValuesPos[i][0],  # lat
                        tmpValuesID[i]       # Some input values on an irregualar grid
                    ])
            print("Add ... ")

                

        np.savetxt("temp_points.txt", dataIrregular, fmt="%.8f")

        R = "-R-180/180/-90/90"
        I = f"-I{resolution:0.1f}d"
        S = f"-S{1.5*resolution:0.1f}d"

        # Nearest neighbor interpolation
        os.system(f"gmt nearneighbor temp_points.txt {R} {I} {S} -rp -Nn -Gtemp_grid.nc")

        nc = Dataset("temp_grid.nc", "r")
        grid_var = list(nc.variables.keys())[-1]
        grid_data = np.flipud(nc.variables[grid_var][:]) # if .nc have rows decrease in latitude.
        #grid_data = nc.variables[grid_var][:] # if .nc have rows increase in latitude.
        nc.close()

        if mask:
            grid_data = np.where(np.isnan(self.maskValue), np.nan, grid_data)

        if propertyName == "basinID":
            self.BasinIDA = grid_data
        else:
            return grid_data

    def setEdgeParameter(self,
                         netCDF4Path,
                         readParm,
                         edgeParaOpt,
                         resample=False):
        """
        setEdgeParameter method is used to define the edge parameter
        that will be used with the community detection algorithm.

        Parameter
        ----------
        netCDF4Path : STRING
            A string path to a netCDF4 that contains the data to be
            used for and edge connections. 
        readParm : STRING
            The name of the parameter to use in the read netCDF4.
        edgeParaOpt : DICTIONARY
            A dictionary that might contain entries defining additional
            operations to take place after loading a netCDF4 to use for
            the edge parameter. Options currently include 'flipud' and
            'fliplr'. 
        resample : BOOLEAN
            An option to resample the edge parameter grid. The default
            is False.
        self.resolution : FLOAT
            Resolution, in degree, for the in Basins pbject analysis. 

        """
        # Get bathymetry resolution
        

        # Use GMT to resampled the input grid at the basin object's resolution
        # Note that this resampling should produce a cell-registered netCDF4
        if resample == True:
            os.system("gmt grdsample {0} -Rd -rp{2}d -G{1} -Vq".format(netCDF4Path,
                                                                    netCDF4Path.replace(".nc", "_resampled.nc"),
                                                                    np.diff(self.lon)[0][0]))
        else:
            os.system('cp {0} {1}'.format(netCDF4Path, netCDF4Path.replace(".nc", "_resampled.nc")));
        
        try: 
            # Read in the resampled netCDF4
            
            ncfile = Dataset(netCDF4Path.replace(".nc", "_resampled.nc"), mode='r')
            
            # Set the edge parameter
            self.edgeParm = ncfile[readParm][:].data;

            # Try other potential options 
            try:
                if edgeParaOpt['flipud']:
                    self.edgeParm = np.flipud(self.edgeParm);
            except:
                pass
            try:
                if edgeParaOpt['fliplr']:
                    self.edgeParm = np.fliplr(self.edgeParm);
            except:
                pass

            
            # Close netCDF4
            ncfile.close()
            
        except:
            print('readParm might not be a valid parameter in your input netCDF4.')

    def defineBasinsNonUnique(self, minBasinCnt = 3,
                     method = "Louvain",
                     reducedRes={"on":False,"factor":15},
                     edgeParaOpt={"Parameter":"Bathymetry", "readParm":None},
                     read=False,
                     write=False,
                     verbose=True):
        """
        defineBasinsNonUnique method will define basins with network analysis
        using either the Girvan-Newman or Louvain algorithm to define
        communities.

        Parameter
        ----------
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.
        method : STRING
            Determines the implemented community detection algorithm.
            The options are either Girvan-Newman or Louvain. The former
            is more robust with low scalability and the latter is practical
            but produces non-deterministic communities. The default is
            Louvain.
        reducedRes : DICTIONARY
            Option to reduce the resolution of the basin definition
            network calculation. Note that this should be turned
            off when doing analysis, and only kept on for testing
            purposes. The default is {"on":False,"factor":15}.
        edgeParaOpt : DICTIONARY
            Dictionary to hold options to describe node edge parameters.
            The input can either be define bathymetry or the path to 
            a netCDF4 file to use for connections. The default is
            {"Parameter":"Bathymetry", "readParm":None}.
            FIXME: Update based on implementation.
        read : BOOLEAN
            An option to read basin definitions for a given
            bathymetry model. The default is False. 
        write : BOOLEAN
            An option to write basin definitions for a given
            bathymetry model. The default is False. 
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.basinCnt : INT
            Number of basins in global bathymetry model.
        self.basinDis : NUMPY ARRAY
            binCnt x basinCnt array of bathymetry distributions.
        
            
        FIXME: NEED TO MAKE SURE reduceRes and resolution are interacting
        properly and not only for 1 degree resolution input bathymetry
        models.

        """
        ################################
        ### Read/Load edge Parameter ###
        ################################
        if edgeParaOpt['Parameter'] != "Bathymetry":
            self.setEdgeParameter(netCDF4Path=edgeParaOpt['Parameter'],
                                  readParm=edgeParaOpt['readParm'],
                                  edgeParaOpt=edgeParaOpt)


        ##########################
        ### Write/Load network ###
        ##########################
        if read:
            ####################
            ### Read network ###
            ####################
            self.G = nx.read_gml("{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), destringizer=float);
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara= self.edgeParm.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm[::self.reducedRes].T[::self.reducedRes].T.flatten();
            # Define resolution
            self.resolution = self.reducedRes*np.diff(self.lon)[0][0];
        
        else:
            ######################
            ### Create network ###
            ######################

            # Define counter and point dictionary
            cnt = 0.
            points = {};

            # Define resolution
            resolution = 1;
            
            # Only reduce resolution if option is set.
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf  = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
                areaWeightsf = self.areaWeights[::self.reducedRes].T[::self.reducedRes].T.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm[::self.reducedRes].T[::self.reducedRes].T.flatten();
            # Distance, in degrees, to a diagonal node.
            cornerDis = self.reducedRes/np.sin(np.pi/4);

            # Create dictionary and array of bathymetry points
            pos = np.zeros( (2, len(~np.isnan(bathymetryf))) );
            for i in tqdm( range(len(self.lonf)) ):
                bathymetryi = bathymetryf[i];
                areaWeighti = areaWeightsf[i];
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgeParai = edgePara[i];
                else:
                    # Use bathymetry as edge parameter
                    edgeParai = bathymetryi;
                if (~np.isnan(bathymetryi)) & (~np.isnan(edgeParai)):
                    points[int(cnt)] = (self.latf[i], self.lonf[i], bathymetryi, areaWeighti, edgeParai);    # (latitude, longitude, depth, areaWeight) w/ units (deg, deg, m, m2)
                    pos[:,int(cnt)] = np.array( [self.latf[i], self.lonf[i]] ); 
                    cnt+=1;

            # Create a graph
            G = nx.Graph()

            # Add nodes (points)
            for node, values in points.items():
                G.add_node(node, pos=values[0:2], depth=values[2], areaWeightm2=values[3], nodeAttribute1=values[4]);

            # Update to the above code block which Adds edges with weights based on geographic distance.
            # This code is significantly faster than the above code block

            ## Set bathymetry vector corresponding to node 
            nodebathymetryf = ~np.isnan(bathymetryf);
            nodeCntList = np.arange(0,len(pos[0,:]),1);

            ## Iterate through each node
            for node1, values1 in tqdm(points.items()):
                # Set coordinates of current iterated node
                coords1 = values1[0:2];

                # Find values surrounding coords1 (current iterated node)
                #coordsPotential = {}; cnt=0;
                indices = [];
                indicesAddPt = [];
                # Set above and below coordinate 
                if not (coords1[0] == np.max(self.latf)):
                    # Not North pole node
                    ## upper node
                    condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(1)
                if not (coords1[0] == np.min(self.latf)):
                    # Not South pole node
                    ## lower node
                    condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(2)

                # If on periodic boundary
                if coords1[1] == np.min(self.lonf):
                    # nodes on left (min) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(3)
                        indicesAddPt.append(3)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(4)
                        indicesAddPt.append(4)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], np.max(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(5)
                    indicesAddPt.append(5)

                elif coords1[1] == np.max(self.lonf):
                    # nodes on right (max) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(6)
                        indicesAddPt.append(6)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(7)
                        indicesAddPt.append(7)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], np.min(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(8)
                    indicesAddPt.append(8)

                else:
                    # Nodes not on boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(9)
                        indicesAddPt.append(9)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(10)
                        indicesAddPt.append(10)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(11)
                    indicesAddPt.append(11)
                    
                for idx in indices:
                    if not (idx.size == 0):
                        # Set node, coordinates, and bathymetry value of current iterated edge node.
                        node2 = idx[0];
                        coords2 = G.nodes[node2]['pos'];
                        values2 = G.nodes[node2]['nodeAttribute1'];
                        
                        
                        # Calculate geographic distance between points using geodesic distance.
                        bathyAve = (values1[4]+values2)/2;
                        

                        #hexsidelengnth = octPolylineLength(coords1, coords2, verbose=False);

                        # Note that this weight contains node spacing information
                        # (i.e., change in node density with latitude and increased \
                        # strength in with high latitude... )
                        distanceV = haversine_distance(coords1[0], coords1[1],
                                                    coords2[0], coords2[1],
                                                    1);
                        nodeSpacingNormalizer = 1/distanceV;

                        G.add_edge(node1, node2, bathyAve=bathyAve*nodeSpacingNormalizer);
                        #G.add_edge(node1, node2, bathyAve=bathyAve);

            # Set some class parameters for testing purposes.
            self.G = G;
            
            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(method = method, minBasinCnt = minBasinCnt);

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
            if write:
                nx.write_gml(G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
            

            ################
            ### Plotting ###
            ################
            if verbose:
                # Plot the network on a geographic map
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

                # Draw the nodes (points) on the map
                for node, data in G.nodes(data=True):
                    ax.plot(data['pos'][1], data['pos'][0], 'bo', markersize=1)  # longitude, latitude

                # Draw the edges (connections)
                for edge in G.edges(data=True):
                    node1, node2, weight = edge
                    lon1, lat1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
                    lon2, lat2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                    ax.plot([lon1, lon2], [lat1, lat2], 'k-', transform=ccrs.PlateCarree())

                # Add coastlines and gridlines
                if "Earth" in self.filename:
                    ax.coastlines()
                ax.gridlines()

                plt.title("Geographic Network of points")
                plt.show()


    def findCommunities(self,
                        detectionMethod,
                        method = "Louvain",
                        minBasinCnt=1,
                        minBasinLargerThanSmallMergers=False,
                        resolution=1):
        """
        findCommunities uses the Girvan-Newman or Louvain community
        detection algorithm to determine communities of nodes (basins).
        Then nodes of similar basins are given a basinID.

        
        Parameter
        ----------
        detectionMethod : DICTIONARY
            A dictionary of defined detection methods.
        method : STRING
            Determines the implemented community detection algorithm.
            The options are either Girvan-Newman or Louvain. The former
            is more robust with low scalability and the latter is practical
            but produces non-deterministic communities. The default is
            Louvain.
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.
        minBasinLargerThanSmallMergers : BOOLEAN
            An option that requires the minBasinCnt variable to equal
            to the number of merged basins that are larger than the
            small basins merger options defined in a mergerPackage.
        resolution : FLOAT
            The resolution value to be used with the Louvain community
            detection algorithm. Values greater than 1, makes the algorithm
            favor smaller communities (more communities). Values less than
            1, makes the algorithm favor larger communities (less communities).
            The default is 1.

        (Re)defines
        ------------
        self.communitiesFinal : LIST
            Python list of dictionaries, where each entry corresponds to community
            i, and the dictionary at entry i contains node indices within that
            community i.


        Return
        ----------
        None.        
        """

        # Imports
        import multiprocessing
        import ctypes
        import leidenalg
        import louvain
        
        import igraph as ig
        from sklearn.cluster import AgglomerativeClustering
        from cdlib import NodeClustering

        from collections import defaultdict
        import itertools



        # Assign defaults 

        ## Set ensembleSize parameter. Set to 1 if not defined
        try:
            ensembleSize = detectionMethod['ensembleSize'];
        except:
            ensembleSize = 1;
        try:
            constantSeeds = detectionMethod['constantSeeds'];
        except:
            constantSeeds = True;
        try:
            njobs = detectionMethod['njobs'];
        except:
            njobs = 1;
        try:
            detectionMethod['resolution'];
        except:
            detectionMethod['resolution'] = 1;

        if (method=="Leiden") | (method=="Leiden-Girvan-Newman"):
            # Optimization Strategy
            #OpStrat = leidenalg.CPMVertexPartition              # Constant potts model
            #OpStrat = louvain.ModularityVertexPartition;        # Modularity with no resolution parameter
            OpStrat = leidenalg.RBConfigurationVertexPartition  # Modularity with resolution parameter

        if (method=="Louvain") | (method=="Louvain-Girvan-Newman"):
            # Optimization Strategy
            #OpStrat = louvain.CPMVertexPartition                # Constant potts model
            #OpStrat = louvain.ModularityVertexPartition;        # Modularity with no resolution parameter
            OpStrat = louvain.RBConfigurationVertexPartition;   # Modularity with resolution parameter


        ########################
        ### Helper Functions ###
        ########################

        # Define function to find and return the node with 
        def mostCentralEdge(G):
            """
            mostCentralEdge function takes a graph and returns
            the node with the highest edge_betweenness_centrality.
            This function can be paired with nx.community.girvan_newman(...)
            to run the girvan-newman community detection algorithm
            for a edge weighted graph.

            Parameters
            -----------
            weight : STRING
                String name of the weighted edge to return from this
                function
            """
            centrality = nx.edge_betweenness_centrality(G, weight='bathyAve')
            return max(centrality, key=centrality.get)


        # def consensus_louvain(graph_nx,
        #                         resolution_parameter=1.0,
        #                         weight_attr="bathyAve",
        #                         runs=1,
        #                         distance_threshold=0.3):
        #     """
        #     consensus_louvain is a function that creates a consensus
        #     clustering from multiple Louvain runs with proper nod
        #     name handling and configurable threshold.

        #     graph_nx : NETWORKX GRAPH
        #         networkx constructed graph with nodes and edge
        #         connections with variable 'weight_attr' defined.
        #     resolution_parameter : FLOAT
        #         Leiden resolution parameter. Values larger than
        #         1 favor smaller (more) communities while a value
        #         smaller than 1 favors larger (less) communities.
        #     weight_attr : STRING
        #         Name of the graph edge weight to use for
        #         community calculation.
        #     runs : INT
        #         Number of Leiden used to create consensus.
        #     distance_threshold : FLOAT

        #     """
        #     # Stable node ordering
        #     nodes = sorted(graph_nx.nodes())
        #     n = len(nodes)
        #     node_to_idx = {node: i for i, node in enumerate(nodes)}
        #     idx_to_node = {i: node for node, i in node_to_idx.items()}

        #     # Build weighted edge list with consistent node labels
        #     edges = [(node_to_idx[u], node_to_idx[v], d.get(weight_attr, 1.0)) for u, v, d in graph_nx.edges(data=True)]
        #     g = ig.Graph()
        #     g.add_vertices(n)
        #     g.add_edges([(u, v) for u, v, w in edges])
        #     g.es["weight"] = [w for _, _, w in edges]
        #     g.vs["name"] = list(range(n))  # Stable index-named nodes

        #     # Initialize co-association matrix
        #     coassoc = np.zeros((n, n))


        #     for i in range(runs):
        #         part = louvain.find_partition(
        #             g,
        #             OpStrat,
        #             resolution_parameter=resolution_parameter,
        #             weights=g.es["weight"],
        #             seed=i
        #         )
        #         for community in part:
        #             for u in community:
        #                 for v in community:
        #                     coassoc[u, v] += 1

        #     # Normalize co-association matrix
        #     coassoc /= runs

        #     # Convert to dissimilarity for clustering
        #     distance = 1.0 - coassoc

        #     # Use Agglomerative Clustering with better threshold control
        #     model = AgglomerativeClustering(
        #         metric="precomputed",
        #         linkage="average",
        #         distance_threshold=distance_threshold,
        #         n_clusters=None
        #     )
        #     labels = model.fit_predict(distance)

        #     # Group nodes by cluster labels
        #     consensus_communities = [[] for _ in range(max(labels)+1)]
        #     for idx, label in enumerate(labels):
        #         consensus_communities[label].append(idx_to_node[idx])

        #     # Convert to sets
        #     consensus_communities = [set(c) for c in consensus_communities]

        #     return NodeClustering(
        #         communities=consensus_communities,
        #         graph=graph_nx,
        #         method_name="consensus_louvain_fixed",
        #         method_parameters={
        #             "resolution_parameter": resolution_parameter,
        #             "runs": runs,
        #             "distance_threshold": distance_threshold
        #         }
        #     )


        # def consensus_leiden(graph_nx,
        #                     resolution_parameter=1.0,
        #                     weight_attr="bathyAve",
        #                     runs=20,
        #                     distance_threshold=0.25,
        #                     OpStrat=OpStrat):
        #     """
        #     consensus_leiden is a function that creates a consensus
        #     clustering from multiple Leiden runs with proper nod
        #     name handling and configurable threshold.

        #     graph_nx : NETWORKX GRAPH
        #         networkx constructed graph with nodes and edge
        #         connections with variable 'weight_attr' defined.
        #     resolution_parameter : FLOAT
        #         Leiden resolution parameter. Values larger than
        #         1 favor smaller (more) communities while a value
        #         smaller than 1 favors larger (less) communities.
        #     weight_attr : STRING
        #         Name of the graph edge weight to use for
        #         community calculation.
        #     runs : INT
        #         Number of Leiden used to create consensus.
        #     distance_threshold : FLOAT

        #     """
        #     # Stable node ordering
        #     nodes = sorted(graph_nx.nodes())
        #     n = len(nodes)
        #     node_to_idx = {node: i for i, node in enumerate(nodes)}
        #     idx_to_node = {i: node for node, i in node_to_idx.items()}

        #     # Build weighted edge list with consistent node labels
        #     edges = [(node_to_idx[u], node_to_idx[v], d.get(weight_attr, 1.0)) for u, v, d in graph_nx.edges(data=True)]
        #     g = ig.Graph()
        #     g.add_vertices(n)
        #     g.add_edges([(u, v) for u, v, w in edges])
        #     g.es["weight"] = [w for _, _, w in edges]
        #     g.vs["name"] = list(range(n))  # Stable index-named nodes

        #     # Initialize co-association matrix
        #     coassoc = np.zeros((n, n))


        #     for i in range(runs):
        #         part = leidenalg.find_partition(
        #             g,
        #             OpStrat,
        #             resolution_parameter=resolution_parameter,
        #             weights=g.es["weight"],
        #             seed=i
        #         )
        #         for community in part:
        #             for u in community:
        #                 for v in community:
        #                     coassoc[u, v] += 1

        #     # Normalize co-association matrix
        #     coassoc /= runs

        #     # Convert to dissimilarity for clustering
        #     distance = 1.0 - coassoc

        #     # Use Agglomerative Clustering with better threshold control
        #     model = AgglomerativeClustering(
        #         metric="precomputed",
        #         linkage="average",
        #         distance_threshold=distance_threshold,
        #         n_clusters=None
        #     )
        #     labels = model.fit_predict(distance)

        #     # Group nodes by cluster labels
        #     consensus_communities = [[] for _ in range(max(labels)+1)]
        #     for idx, label in enumerate(labels):
        #         consensus_communities[label].append(idx_to_node[idx])

        #     # Convert to sets
        #     consensus_communities = [set(c) for c in consensus_communities]

        #     return NodeClustering(
        #         communities=consensus_communities,
        #         graph=graph_nx,
        #         method_name="consensus_leiden_fixed",
        #         method_parameters={
        #             "resolution_parameter": resolution_parameter,
        #             "runs": runs,
        #             "distance_threshold": distance_threshold
        #         }
        #     )
        def consensus_reduction_parallel_shared(graph_nx,
                                            resolution_parameter=1.0,
                                            weight_attr="bathyAve",
                                            runs=1,
                                            distance_threshold=0.3,
                                            n_jobs=1,
                                            partition_strategy=leidenalg.RBConfigurationVertexPartition,
                                            method="leiden",
                                            constantSeeds=True):
            """
            Parallel consensus clustering supporting Leiden or Louvain.

            Parameters
            ----------
            graph_nx : NETWORKX.GRAPH
                Input graph.
            resolution_parameter : FLOAT
                Louvain/Leiden resolution parameter.
            weight_attr : STRING
                Edge weight attribute name.
            runs : INT
                Number of runs to generate consensus.
            distance_threshold : FLOAT
                Distance threshold for final clustering.
            n_jobs : INT
                Number of parallel workers.
            partition_strategy : OBJECT
                Partition strategy (Leiden) or partition type (Louvain).
            method : STRING
                "leiden" or "louvain". The default is "leiden"
            """

            nodes = sorted(graph_nx.nodes())
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            idx_to_node = {i: node for node, i in node_to_idx.items()}

            edges = [(node_to_idx[u], node_to_idx[v], d.get(weight_attr, 1.0)) for u, v, d in graph_nx.edges(data=True)]
            g = ig.Graph()
            g.add_vertices(n)
            g.add_edges([(u, v) for u, v, w in edges])
            g.es["weight"] = [w for _, _, w in edges]
            g.vs["name"] = list(range(n))

            coassoc_base = multiprocessing.Array(ctypes.c_double, n * n, lock=True)

            if constantSeeds:
                seeds = list(range(runs))
            else:
                import random
                seeds = [];
                for _ in range(runs):
                    seeds.append(random.randint(0, 1e6))

            # For testing if seeds are constant or variable across runs.
            # print("\nseeds: {}\n".format(seeds))
            
            with multiprocessing.Pool(
                processes=n_jobs,
                initializer=_CReduction_init_worker,
                initargs=(g, coassoc_base, n, resolution_parameter, method, partition_strategy)
            ) as pool:
                pool.map(_CReduction_worker, seeds)

            coassoc = np.ctypeslib.as_array(coassoc_base.get_obj()).reshape((n, n))
            coassoc /= runs

            distance = 1.0 - coassoc
            model = AgglomerativeClustering(
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
                n_clusters=None
            )
            labels = model.fit_predict(distance)

            # Group nodes by cluster labels
            consensus_communities = [[] for _ in range(max(labels)+1)]
            for idx, label in enumerate(labels):
                consensus_communities[label].append(idx_to_node[idx])
            
            # Convert to sets
            consensus_communities = [set(c) for c in consensus_communities]

            return NodeClustering(
                communities=consensus_communities,
                graph=graph_nx,
                method_name=f"consensus_{method}_parallel_shared",
                method_parameters={
                    "resolution_parameter": resolution_parameter,
                    "runs": runs,
                    "distance_threshold": distance_threshold,
                    "n_jobs": n_jobs,
                    "partition_strategy": str(partition_strategy)
                }
            ), coassoc

        import numpy as np
        from collections import defaultdict

        def node_certainty_metrics(coassoc: np.ndarray, labels: np.ndarray, eps: float = 1e-9):
            """
            node_certainty_metrics method compute per-node certainty
            metrics from a co-association matrix and final labels.
            
            
            Returns :
            --------
            Dictionary
                'in_cohesion'
                'best_other'
                'margin'
                'silhouette'
                'assign_prob'
                'entropy'
                'core_score'
            """
            n = coassoc.shape[0]
            # Ensure diagonal = 1 (sometimes pooling can leave slight drift)
            coassoc = coassoc.copy()
            np.fill_diagonal(coassoc, 1.0)

            # Build communities (index lists)
            comm_to_idx = defaultdict(list);    # Dictionary of list
            for i, c in enumerate(labels):
                comm_to_idx[c].append(i)        # For community c append node index i
            comm_ids = list(comm_to_idx.keys()) # Get all community indices e.g., [0,1,2,...,99] for 100 communities.
            comm_lists = [np.array(comm_to_idx[c], dtype=int) for c in comm_ids]
            comm_of = {i: ci for ci, nodes in enumerate(comm_lists) for i in nodes}  # map node -> community index (0..k-1)

            k = len(comm_lists)
            # Precompute per-node per-community mean coassoc (including self, we will exclude later)
            # To avoid self-bias, we will subtract i when in its own community.
            mean_to_comm = np.zeros((n, k), dtype=float)

            for ci, members in enumerate(comm_lists):
                # For each node i, mean coassoc to members of community ci
                # coassoc[:, members] -> n x |members|; take mean across axis=1
                denom = len(members)
                if denom > 0:
                    mean_to_comm[:, ci] = coassoc[:, members].mean(axis=1)
                else:
                    mean_to_comm[:, ci] = 0.0

            # Now correct self-bias for a node's own community:
            # Replace mean with leave-one-out mean when computing own-community stats
            loo_mean_to_own = np.zeros(n, dtype=float)
            for i in range(n):
                ci = comm_of[i]
                members = comm_lists[ci]
                size = len(members)
                if size <= 1:
                    loo_mean_to_own[i] = 0.0
                else:
                    # (sum over members - self) / (size-1)
                    s = coassoc[i, members].sum() - 1.0
                    loo_mean_to_own[i] = s / (size - 1)

            # In-cluster cohesion (using leave-one-out)
            in_cohesion = loo_mean_to_own.copy()

            # Best-other overlap
            best_other = np.zeros(n, dtype=float)
            for i in range(n):
                ci = comm_of[i]
                # candidates are all communities except ci
                if k == 1:
                    best_other[i] = 0.0
                else:
                    # We used group means that included self if self is in that community; for other communities this is fine
                    others = [c for c in range(k) if c != ci]
                    # If some other community is a singleton with the node itself (can't happen), just safe-guard
                    best_other[i] = mean_to_comm[i, others].max() if others else 0.0

            margin = in_cohesion - best_other

            # Silhouette-like score using distances = 1 - coassoc
            silhouette = np.zeros(n, dtype=float)
            for i in range(n):
                ci = comm_of[i];        # Community index of node being evaluated
                own = comm_lists[ci];   # Numpy list of nodes in community ci
                size = len(own);        # Number of nodes in community ci
                if size <= 1:
                    # Case where node is the only node in the community
                    silhouette[i] = 0.0
                    continue
                
                # mean distance to own cluster (leave-one-out)
                # sum of coassociation between between node i and every other node in community ci: (coassoc[i, own].sum() - 1.0)
                # distance between node i and each node in community ci:            1.0 - (coassoc[i, own].sum() - 1)
                # average distance between node i and each node in community ci:    a
                a = (1.0 - (coassoc[i, own].sum() - 1.0) / (size - 1))  

                # mean distance to other clusters
                b = np.inf
                for cj, members in enumerate(comm_lists):
                    if cj == ci or len(members) == 0:
                        # if current evaluated community contains the current evaluated node
                        continue
                    # find determine closest (smallest distance) node outside of community ci. 
                    b = min(b, (1.0 - coassoc[i, members].mean()))
                if not np.isfinite(b):
                    silhouette[i] = 0.0
                else:
                    # Calculate silhouette value
                    denom = max(a, b)
                    silhouette[i] = (b - a) / denom if denom > eps else 0.0

            # Soft assignment probabilities p_i(C)
            # Use small epsilon to avoid zeros (helps entropy)
            soft = mean_to_comm + eps
            soft /= soft.sum(axis=1, keepdims=True)
            assign_prob = soft.max(axis=1)
            # Entropy in nats; convert to bits by / np.log(2) if desired
            entropy = -(soft * np.log(soft)).sum(axis=1)

            # Core score (higher means node is well-supported within its own community)
            core_score = np.zeros(n, dtype=float)
            for i in range(n):
                ci = comm_of[i]
                members = comm_lists[ci]
                if len(members) == 0:
                    core_score[i] = 0.0
                    continue
                numer = coassoc[i, members].sum()
                denom = np.sqrt(len(members)) * max(coassoc[i, members].max(), eps)
                core_score[i] = numer / denom

            return {
                "in_cohesion": in_cohesion,
                "best_other": best_other,
                "margin": margin,
                "silhouette": silhouette,
                "assign_prob": assign_prob,
                "entropy": entropy,
                "core_score": core_score,
            }
        
        ###################################################
        ### Community and Composite Community Detection ###
        ###################################################
        if method=="Girvan-Newman":
            ################################################
            ### Girvan-Newman community detection method ###
            ################################################
            # GIRVAN-NEWMAN COMMUNITY DETECTION
            self.communities = list(nx.community.girvan_newman(self.G, most_valuable_edge=mostCentralEdge));

            # Choose interation of the algorithm that has at least
            # minBasinCnt basins.
            interation = 0;
            while len(self.communities[interation]) < minBasinCnt:
                interation+=1;
            if interation > 0:
                interation-1;
            
            # Redefine the node community structure using Girvan Newman communities
            self.communitiesFinal = self.communities[interation];

        elif (method=="Leiden") | (method=="Leiden-Girvan-Newman"):
            # LEDIAN COMMUNITY DETECTION
            Rcommunities, coassocDiagonal = consensus_reduction_parallel_shared(self.G,
                                                                                resolution_parameter=detectionMethod['resolution'],
                                                                                distance_threshold=0.3,
                                                                                runs=ensembleSize,
                                                                                n_jobs=njobs,
                                                                                partition_strategy=OpStrat,
                                                                                method="leiden",
                                                                                constantSeeds=constantSeeds)

            # Set ensemble community uncertainty metrics
            # Convert communities to a label vector aligned with the sorted node list
            nodes_sorted = sorted(Rcommunities.graph.nodes())
            node_to_idx = {node:i for i, node in enumerate(nodes_sorted)}

            labels = np.full(len(nodes_sorted), -1, dtype=int)
            for cid, comm in enumerate(Rcommunities.communities):
                # cid  : community index
                # comm : list of nodes in community index cid 
                for node in comm:
                    labels[node_to_idx[node]] = cid

            metrics = node_certainty_metrics(coassocDiagonal, labels)

            # Example: attach per-node certainty back to node attributes
            for node, i in node_to_idx.items():
                self.G.nodes[node]["consensus_incohesion"] = float(metrics["in_cohesion"][i])
                self.G.nodes[node]["consensus_margin"]      = float(metrics["margin"][i])
                self.G.nodes[node]["consensus_silhouette"]  = float(metrics["silhouette"][i])
                self.G.nodes[node]["consensus_prob"]        = float(metrics["assign_prob"][i])
                self.G.nodes[node]["consensus_entropy"]     = float(metrics["entropy"][i])
                self.G.nodes[node]["consensus_core"]        = float(metrics["core_score"][i])

            # Set community structure to class attributes.
            Rcommunities                = Rcommunities.communities;
            self.Rcommunities           = cp.deepcopy(Rcommunities)
            self.RcommunitiesUnaltered  = cp.deepcopy(Rcommunities);
            self.communitiesFinal       = self.Rcommunities;
            self.coassocDiagonal        = coassocDiagonal;

        elif  (method=="Louvain") | (method=="Louvain-Girvan-Newman"):
            # LOUVAIN COMMUNITY DETECTION
            Rcommunities, coassocDiagonal = consensus_reduction_parallel_shared(self.G,
                                                                                resolution_parameter=detectionMethod['resolution'],
                                                                                distance_threshold=0.3,
                                                                                runs=ensembleSize,
                                                                                n_jobs=njobs,
                                                                                partition_strategy=OpStrat,
                                                                                method="louvain",
                                                                                constantSeeds=constantSeeds)
            
            # Set ensemble community uncertainty metrics
            # Convert communities to a label vector aligned with the sorted node list
            nodes_sorted = sorted(Rcommunities.graph.nodes())
            node_to_idx = {node:i for i, node in enumerate(nodes_sorted)}

            labels = np.full(len(nodes_sorted), -1, dtype=int)
            for cid, comm in enumerate(Rcommunities.communities):
                # cid  : community index
                # comm : list of nodes in community index cid 
                for node in comm:
                    labels[node_to_idx[node]] = cid

            metrics = node_certainty_metrics(coassocDiagonal, labels)

            # Example: attach per-node certainty back to node attributes
            for node, i in node_to_idx.items():
                self.G.nodes[node]["consensus_incohesion"] = float(metrics["in_cohesion"][i])
                self.G.nodes[node]["consensus_margin"]      = float(metrics["margin"][i])
                self.G.nodes[node]["consensus_silhouette"]  = float(metrics["silhouette"][i])
                self.G.nodes[node]["consensus_prob"]        = float(metrics["assign_prob"][i])
                self.G.nodes[node]["consensus_entropy"]     = float(metrics["entropy"][i])
                self.G.nodes[node]["consensus_core"]        = float(metrics["core_score"][i])

            # Set community structure to class attributes.
            Rcommunities                = Rcommunities.communities;
            self.Rcommunities           = cp.deepcopy(Rcommunities)
            self.RcommunitiesUnaltered  = cp.deepcopy(Rcommunities);
            self.communitiesFinal       = self.Rcommunities;
            self.coassocDiagonal        = coassocDiagonal;

        else:
            # Throw Error
            print(f"{method} is set incorrectly. It needs to be either Louvain/Leiden/Louvain-Girvan-Newman/Leiden-Girvan-Newman/Girvan-Newman.")

        if (method=="Leiden-Girvan-Newman") | (method=="Louvain-Girvan-Newman"):
            ## Mapping from node to community index from reduced (Leiden/Louvain) community detection
            node_to_comm = {}
            for idx, comm in enumerate(Rcommunities):
                for node in comm:
                    node_to_comm[node] = idx
            
            # Construct new graph with reduced (Leiden/Louvain) community consolidated nodes
            self.Gnew = nx.Graph()

            # Add *all* communities as nodes, even if disconnected
            self.Gnew.add_nodes_from(range(len(Rcommunities)))  # One node per community index

            # Propagate and sum node attributes from self.G to self.Gnew
            node_attrs_to_sum = ['areaWeightm2']
            comm_attr_sums = {attr: defaultdict(float) for attr in node_attrs_to_sum}

            # Sum attributes by community
            for node, data in self.G.nodes(data=True):
                comm = node_to_comm[node]
                for attr in node_attrs_to_sum:
                    val = data.get(attr, 0.0)
                    comm_attr_sums[attr][comm] += val

            # Assign summed attributes to Gnew community nodes
            for attr in node_attrs_to_sum:
                for comm, total in comm_attr_sums[attr].items():
                    self.Gnew.nodes[comm][attr] = total


            # Track summed weights between communities
            edge_weights = defaultdict(float)

            # Track unisolated reduced (Leiden/Louvain) communities (communities that connect to other communities).
            unisolatedCommunities = np.array([]);
            smallCommunities = np.array([]);
            # Iterate over all edges in the original graph
            for u, v, data in self.G.edges(data=True):
                cu = node_to_comm[u]
                cv = node_to_comm[v]
                weight = data.get('bathyAve', 1.0)

                if cu != cv:
                    # Undirected: sort community pair to avoid duplicates
                    edge = tuple(sorted((cu, cv)))
                    edge_weights[edge] += weight

            # Add weighted edges to Gnew
            for (cu, cv), edge_weight in edge_weights.items():
                self.Gnew.add_edge(cu, cv, bathyAve=edge_weight)

            # Apply Girvan–Newman algorithm to the simplified community graph
            comp = nx.community.girvan_newman(self.Gnew, most_valuable_edge=mostCentralEdge)
            

            if minBasinLargerThanSmallMergers:
                print("Girvan-Newman mergers are being conducted such there are at most {0} basins larger than {1} {2} surface area".format(detectionMethod['minBasinCnt'],np.max(detectionMethod['mergerPackage']['mergeSmallBasins']['threshold']), detectionMethod['mergerPackage']['mergeSmallBasins']['thresholdMethod']))
                # Iteratively run the Girvan-Newman algorithm until X communities greater than areaThreshold are detected.

                # Define attribute to merge
                node_attr = 'areaWeightm2'

                # Find total global area
                GlobalArea = sum(self.Gnew.nodes[n].get(node_attr, 0.0) for n in self.Gnew.nodes);
                
                # Change the threshold for 'small' basin based on what unit is deined.
                if detectionMethod['mergerPackage']['mergeSmallBasins']['thresholdMethod'] == "%":
                    # Defined small basin mergers with percent (%) total field area.
                    areaThreshold = GlobalArea*(np.max(detectionMethod['mergerPackage']['mergeSmallBasins']['threshold'])/100)
                else:
                    # Defined small basin mergers with SI unit (m) 
                    areaThreshold = np.max(detectionMethod['mergerPackage']['mergeSmallBasins']['threshold']);
            
                # Iterate through Girvan-Newman algorithm.
                for communities in comp:
                    # Convert tuple of sets to list of communities
                    community_list = list(communities)

                    # Compute areaWeightm2 sum for each community
                    areaSums = []
                    for comm in community_list:
                        totalArea = sum(self.Gnew.nodes[n].get(node_attr, 0.0) for n in comm)
                        areaSums.append(totalArea)

                    # Count communities exceeding threshold Y
                    large_comm_count = sum(area > areaThreshold for area in areaSums)

                    # Break when condition is met
                    if large_comm_count >= detectionMethod['minBasinCnt']:
                        # print("Final: large_comm_count, detectionMethod['minBasinCnt']", large_comm_count, detectionMethod['minBasinCnt'])
                        GNcommunities = community_list
                        break
            else:                
                # Remove merge communities until detectionMethod['mergerPackage']['minBasinCnt'] is reached
                limited = itertools.takewhile(lambda c: len(c) <= detectionMethod['minBasinCnt'], comp)
                for communities in limited:
                    GNcommunities = communities

            # Assign the Girvan-Newman community structure to a class attribute. 
            self.GNcommunities = GNcommunities

            # Map each Girvan–Newman community to its reduced (Leiden/Louvain) community
            reduced_to_gn = {}
            for idx, comm in enumerate(GNcommunities):
                for c in comm:
                    reduced_to_gn[c] = idx
            
            # Map each original node to a Girvan–Newman community via its reduced (Leiden/Louvain) community
            commNodes = [{} for _ in range(len(Rcommunities))]
            for commL in reduced_to_gn:
                commGN = reduced_to_gn[commL];
                
                try:
                    # Do not comment out. If this code can run then commNodes[commGN]
                    # has already been defined
                    len(commNodes[commGN]);
                    commNodes[commGN].update(Rcommunities[commL])
                except:
                    commNodes[commGN] = Rcommunities[commL]
                
            # Redefine the node community structure using reduced (Leiden/Louvain) & Girvan Newman composite communities
            self.communitiesFinal = commNodes;


        #####################################
        ### Set node attribute (basinIDs) ###
        #####################################
        
        ## Set the amount of unique basinIDs equal to the count
        ## of unique communities that the Girvan-Newman or Louvain
        ## algorithm found. This is not always equal to the minBasinCnt 
        basinIDTags = np.arange(len(self.communitiesFinal));

        basinIDs = {}; cnt = 0;
        for community in self.communitiesFinal:
            basinIDi = float(basinIDTags[cnt]);
            for nodeID in community:
                basinIDs[nodeID] = {"basinID": basinIDi};
            cnt+=1;
        nx.set_node_attributes(self.G, basinIDs, "basinID");

        # Iterpolate irregular grid to regular grid.
        # Method defines self.BasinIDA
        self.interp2regularGrid(mask=True)

    def createCommunityNodeColors(self, verbose=False):
        """
        createCommunityNodeColors method sets colors associated
        with different community nodes (e.g., basins in this case).


        Returns
        ----------
        node_colors : PYTHON LIST
            A list of hex code colors that correspond to a node's
            community.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.
        """
        # Get basin IDs from network object.
        tmpValues = nx.get_node_attributes(self.G, "basinID");

        # Record the count of basins
        basinIDi = 0
        for i in range(len(nx.get_node_attributes(self.G, "basinID"))):
            if int(tmpValues[i]["basinID"]) > basinIDi:
                basinIDi = int(tmpValues[i]["basinID"]);
        if verbose:
            print("There are {0:0.0f} basinIDs".format(basinIDi+1))

        # Define colors for basin identification.
        if basinIDi+1 <= 40:
            colors = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20b"].colors];
            colors2 = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20c"].colors]
            for i in range(len(colors2)):
                colors.append(colors2[i])
        else:
            # Resample a spectral colormap to the exact size of the basin count.
            colormap = mpl.colormaps["Spectral"];
            colormap = colormap.resampled(basinIDi+1);
            # Convert resampled spectral colormap to hexidecimal codes.
            colors = [mpl.colors.to_hex(i) for i in colormap(np.linspace(0, 1, basinIDi+1))[:, 0:3] ];

        node_colors = [];

        # Iterate through all bathymetry nodes.
        for i in range(len(nx.get_node_attributes(self.G, "basinID"))):
            basinIDi = int(tmpValues[i]["basinID"]);
            node_colors.append( colors[basinIDi] )

        return node_colors

    def visualizeCommunities(self,
                              cmapOpts={"cmap":"viridis",
                                        "cbar-title":"cbar-title",
                                        "cbar-range":[0,1]},
                              pltOpts={"valueType": "Bathymetry",
                                       "valueUnits": "m",
                                       "plotTitle":"",
                                       "plotZeroContour":False,
                                       "nodesize":5,
                                       "connectorlinewidth":1,
                                       "projection":"Mollweide"},
                              draw={"nodes":True,
                                    "connectors":True,
                                    "bathymetry":True,
                                    "coastlines":True,
                                    "gridlines":True},
                              saveSVG=False,
                              savePNG=False):
        """
        visualizeCommunities method creates a global map of nodes
        and connectings. The nodes communities are defined with
        different colors.


        Parameters
        ----------
        cmapOpts : DICTIONARY
            A set of options to format the color map and bar for the plot
        pltOpts : DICTIONARY
            A set of options to format the plot
        pltOpts : DICTIONARY
            A set of options to display different aspects of a basin.
        saveSVG : BOOLEAN
            An option to save an SVG output. The default is False.
        savePNG : BOOLEAN
            An option to save an PNG output. The default is False.
            
        Returns
        ----------
        None.        
        """
        # Make vector cooresponding to nodes that assigns each entry similar hex
        # color codes for similar community of nodes.
        node_colors = self.createCommunityNodeColors()
        
        # Plot the network on a geographic map

        ## Make figure
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(1, 1, height_ratios=[1]);  # 1 rows, 1 column, 

        ## Set projection type
        if pltOpts['projection'] == "Mollweide":
            projectionType = ccrs.Mollweide();
        elif pltOpts['projection'] == "Miller":
            projectionType = ccrs.Miller();
        elif pltOpts['projection'] == "Robinson":
            projectionType = ccrs.Robinson();
        elif pltOpts['projection'] == "Mercator":
            projectionType = ccrs.Mercator();

        ## Add axis
        ax = fig.add_subplot(gs[0], transform=ccrs.PlateCarree(), projection=projectionType);
        ax.axis('equal')

        ## Define local bathymetry variable (it will be mofified)
        if pltOpts['valueType'] == "Bathymetry":
            plotValue = self.bathymetry;
            lon = self.lon
            lat = self.lat
            pltOpts['valueType'] = {};
            pltOpts['valueType']['name'] = "Bathymetry"
        else:
            # If pltOpts['valueType'] is a dictionary the plot 'attribute' 
            # of the netcdf4 at path. 
            os.system("gmt grdsample {0} -Rd -rp{2}d -G{1} -Vq".format(pltOpts['valueType']['path'], "grid.nc", np.diff(self.lon)[0][0]))
            data = Dataset("grid.nc", 'r');
            plotValue = data[pltOpts['valueType']['attribute']][:].data;
            lon, lat = np.meshgrid(data['lon'][:], data['lat'][:])

            data.close()
            # Define cbar range
            cmapOpts['cbar-range'] = [np.nanmin(np.nanmin(plotValue)), np.nanmean(plotValue)+2*np.nanstd(plotValue)]


        ## Add bathymetry
        #print("plotValue",plotValue)
        #print("self.lon", self.lon.shape)
        #print("self.lat", self.lat.shape)
        #print("plotValue.shape", plotValue.shape)
        if draw['plotValue']:
            mesh = ax.pcolormesh(lon, lat, plotValue, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1])


        ## Add coastlines and gridlines
        ## Use 0 m contour line
        ## Set any np.nan values to 0.
        if draw['coastlines']:
            plotValue[np.isnan(plotValue)] = 0;
            zeroContour = ax.contour(lon, lat, plotValue, levels=[0], colors='black', transform=ccrs.PlateCarree())        
            

        ## Draw the edges (connections)
        if draw['connectors']:
            for edge in self.G.edges(data=True):
                node1, node2, weight = edge
                
                lon1, lat1 = self.G.nodes[node1]['pos'][1], self.G.nodes[node1]['pos'][0]
                lon2, lat2 = self.G.nodes[node2]['pos'][1], self.G.nodes[node2]['pos'][0]
                
                #minmaxlon = [np.min(self.lonf), np.max(self.lonf)]
                minmaxlon = [-180,180]
                
                value1 = ~( (minmaxlon[0]==lon1) and (minmaxlon[1]==lon2) );
                value2 = ~( (minmaxlon[1]==lon1) and (minmaxlon[0]==lon2) );
                if value1 & value2:
                    # Edge does not cross periodic boundary (this is done for visualization
                    # purposes only). With this condition, there are no values edges connected
                    # across the entire planetary surface (i.e., because they will not pass
                    # through the periodic boundary).
                    lines = ax.plot([lon1, lon2], [lat1, lat2], '-k', linewidth=pltOpts['connectorlinewidth'], transform=ccrs.PlateCarree())
                    for line in lines:
                        line.set_zorder(10); 

        ## Draw the nodes (points) on the map
        
        ### Format
        nodePosDict = self.G.nodes.data('pos');
        nodeBasinID = self.G.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
        

        ### Plot
        #### Define the color map and levels used for plotting basinIDs
        basincmap = mpl.colormaps["plasma"].resampled(len(np.unique(BasinID)));
        levels = np.arange(-.5, len(np.unique(BasinID)));

        #### Plot contour of scatter plot based on user inputs
        if draw['nodes']:
            if draw['connectors']:
                nodeplthandle1 = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, vmin=np.min(levels), vmax=np.max(levels), edgecolor='k', linewidths=pltOpts['connectorlinewidth'], s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            else:
                nodeplthandle1 = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, vmin=np.min(levels), vmax=np.max(levels), edgecolor='k', linewidths=pltOpts['connectorlinewidth']/4, s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            nodeplthandle1.set_zorder(11);
        elif draw['nodes-contour']:
            # If plotValue is plotted with contours then make alpha value of
            # the contours lower (makes more transparent)
            if draw['plotValue']:
                alpha = 0;
            else:
                alpha = 1;


            # Define arrays of latitude, longitude and basinIDs for plotting contours
            latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
            lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T

            BasinIDA = self.BasinIDA[::self.reducedRes].T[::self.reducedRes].T

            #BasinIDA = np.empty(np.shape(lonA));
            #BasinIDA[:] = np.nan;
            #for nodei in range(len(pos[:,1])):
            #    BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];
            
            cmap = mpl.colormaps["plasma"].resampled(len(np.unique(BasinID)))
            levels = np.arange(-.5, len(np.unique(BasinID)));
            nodeplthandle1 = ax.contourf(lonA, latA, BasinIDA, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, alpha=alpha)  # longitude, latitude
            nodeplthandle2 = ax.contour(lonA, latA, BasinIDA, levels=levels, colors='r', linewidths=.5, transform=ccrs.PlateCarree())
            nodeplthandle1.set_zorder(11);
            nodeplthandle2.set_zorder(12);
            # Plot basinID labels
            for BasinIDi in np.unique(BasinID):
                # If basin is on a periodic boundary when plot basinID at mean latitudes
                # and longitudes on the side of the periodic boundary which contains
                # the most amount of bathymetry points (not area)
                if (np.max(lonA[BasinIDA==BasinIDi]) == np.max(lonA)) & (np.min(lonA[BasinIDA==BasinIDi]) == np.min(lonA)):
                    # Basin crosses periodic boundary.
                    if len(lonA[(BasinIDA==BasinIDi) & (lonA<0)]) > len(lonA[(BasinIDA==BasinIDi) & (lonA>0)]):
                        # Left side of prime meridian has more points.
                        xloc = np.nanmean(lonA[(BasinIDA==BasinIDi) & (lonA<0)]);
                        yloc = np.nanmean(latA[(BasinIDA==BasinIDi) & (lonA<0)]);
                    else:
                        # Eight side of prime meridian has more, or equal, points.
                        xloc = np.nanmean(lonA[(BasinIDA==BasinIDi) & (lonA>0)]);
                        yloc = np.nanmean(latA[(BasinIDA==BasinIDi) & (lonA>0)]);
                else:
                    # Basin does not cross periodic boundary.
                    xloc = np.nanmean(lonA[(BasinIDA==BasinIDi)]);
                    yloc = np.nanmean(latA[(BasinIDA==BasinIDi)]);
                basinIDhandle1 = plt.text( xloc, yloc,  "{:0.0f}".format(BasinIDi), color='r', transform=ccrs.PlateCarree());
                basinIDhandle1.set_zorder(12);
                basinIDhandle2 = plt.plot( xloc, yloc, '.r', transform=ccrs.PlateCarree());
                basinIDhandle2[0].set_zorder(12);
                
                

            
        self.unBasinIS = np.unique(BasinID);

        ## Add a colorbar(s)
        if draw['plotValue']:
            basinIDcbarpad = .0;
        else:
            basinIDcbarpad = .05;
        
        # Basin IDs
        cbar1 = plt.colorbar(nodeplthandle1, ax=ax, orientation='horizontal', pad=0.0, aspect=40, shrink=0.7, cmap=basincmap);
        cbar1.set_label(label="Basin ID", size=12);
        self.cbar1 = cbar1
        if len(np.unique(BasinID))>10:
            cbar1.ax.tick_params(labelsize=8);  # Adjust the size of colorbar ticks
            
        else:
            cbar1.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks
        
        basinticklabels = [];
        if not (np.size(np.unique(BasinID))==1):
            #cbar1.set_ticks( np.arange(np.max(BasinID)/(2*(np.max(BasinID)+1)), np.max(BasinID), np.max(BasinID)/((np.max(BasinID)+1))) );  # Set custom tick positions
            cbar1.set_ticks( np.arange(0, np.max(BasinID)+1, 1) );  # Set custom tick positions
            for basini in np.arange(0, np.max(BasinID)+1, 1):
                basinticklabels.append( "{:0.0f}".format(basini) );
        else:
            cbar1.set_ticks( [0] );  # Set custom tick positions
            basinticklabels.append( "{:0.0f}".format(np.unique(BasinID)[0]) );
        
        cbar1.set_ticklabels( basinticklabels, rotation=90);  # Custom labels
        

        if draw['plotValue']:
            # Bathymetry
            cbar2 = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7);
            cbar2.set_label(label="{} [{}]".format(pltOpts['valueType']['name'], pltOpts['valueUnits']), size=12);
            cbar2.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks


        if draw['gridlines']:
            ax.gridlines()

        # Set x and y scale equal
        

        plt.title("Basin Networks ({})".format(self.body))

        # Save figure
        if savePNG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".png")), dpi=600)
        if saveSVG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".svg")))
        
        if (not savePNG) & (not saveSVG):
            plt.show()

    def mergeBasins(self, basinID1, basinID2, write=False):
        """
        mergeBasins method will merge basins with basin ID of basinID2
        to basins with ID basinID1. The resulting basin network is then
        rewitten to the original, overriding the original basinID network.
        Note that there is no need to reread the basin network. 

        Parameters
        ----------
        basinID1 : INT
            Basin ID to absorb basinID2.
        basinID2 : INT, or LIST OF INT
            Basin ID(s) to be absorbed by basinID1.
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute.
        """

        ##########################
        ### Merge basins Model ###
        ##########################
        # Get all node basin ids 
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );
        nodes = nx.get_node_attributes(self.G, "basinID");
        for nodeID in nodes:
            nodeBasinID[int(nodeID)] = nodes[float(nodeID)]['basinID'];
        
        # Change basin id(s) basinID2 to basinID1 (not in nodes yet)
        if np.size(basinID2) == 1:
            # A single basinID to convert to basinID1.
            nodeBasinID[nodeBasinID==basinID2] = basinID1;
        else:
            # Multiple (list) of basinIDs to convert to basinID1.
            for basinID2i in basinID2:
                nodeBasinID[nodeBasinID==basinID2i] = basinID1;

        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);
        
        # Set basin id to new merged basin
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");
        
        
        ###########################
        ### Write network Model ###
        ###########################
        if write:
            print("Network has been overwritten.")
            nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
        else: 
            print("Network has not been overwritten. New network only exist within this object instance.")

    def mergeSmallBasins(self, threshold, thresholdMethod, mergeMethod, write=False):
        """
        mergeSmallBasins method will merge basins below some threshold
        surface area with the closest basins of above that same threshold.
        The resulting basin network is then rewitten to the original,
        overriding the original basinID network. Note that there is no
        need to reread the basin network. 

        Parameters
        ----------
        threshold : FLOAT
            Basins smaller than this threshold, in [m2] or [%] of total
            seafloor area will be merged with the closest basin above
            this threshold in size.
        thresholdMethod : STRING
            A string that defines the input type for threshold. This should
            either be set to 'm2' or '%'.
        mergeMethod : STRING
            A string that defines the method to merge small basins with larger
            basins. This can either be set to 'centroid' or 'nearBasinEdge'.
            'centroid' will merge the small basin with nearest centroid of a
            large basin. 'nearBasinEdge' will compare the small basin centroid
            with the distance to all nodes. The small basin will then be assigned
            the same basin ID as the closest node part of a large basin. 
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute.
        """

        print("self.AOC", self.AOC)

        ##########################
        ### Merge basins Model ###
        ##########################
        # Define area scalar for thresholdMethod
        if thresholdMethod == '%':
            # Set methodScalar to AOC (total ocean surface) since we are
            # comparing with surface area in %.
            methodScalar = self.AOCMask/100;
        elif thresholdMethod == 'm2':
            # Set methodScalar to 1 since we are comparing with surface
            # area in meters.
            methodScalar = 1;

        # Define the number of define basins
        BasinIDmax = self.G
        distance = 1e20;

        # Get basin properties from nodes
        nodesID   = nx.get_node_attributes(self.G, "basinID");      # Dictionary of nodes and their basinID
        nodePos   = nx.get_node_attributes(self.G, "pos");          # Dictionary of nodes and their positions (lat,lon) [deg, deg]
        nodesAWm2 = nx.get_node_attributes(self.G,'areaWeightm2');  # Dictionary of nodes and their area [m2]


        # Set all node basin ids and position in an array
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );       # Array to hold node basinIDs
        nodePosA = np.zeros((len(self.G), 2), dtype=float );        # Array to hold basini node positions (lat,lon) [deg, deg]
        for nodeID in nodesID:
            # Set node basin ID to array entry
            nodeBasinID[int(nodeID)] = nodesID[float(nodeID)]['basinID'];
            # Set node position to array entry
            nodePosA[int(nodeID),:] = nodePos[float(nodeID)][:];

        # Set unique ids
        nodeBasinIDUnique = np.unique(nodeBasinID);

        # Get basin i areas & Calculate basin centroids (latitudes and
        # longitudes weighted by surface area of the node)
        basinArea = np.zeros(len(nodeBasinIDUnique));               # Array to hold basini area [m2]
        LatCentroid = np.zeros(len(nodeBasinIDUnique));             # Array to hold basini centroid latitude [deg]
        LonCentroid = np.zeros(len(nodeBasinIDUnique));             # Array to hold basini centroid longitude [deg]
        for i, basinID, nodePosi in zip(nodesAWm2, nodeBasinID, nodePosA):
            basinArea[int(basinID)] += nodesAWm2[i];                # Add node i to basinID basinArea area sum [m2].
            LatCentroid[int(basinID)] += nodePosi[0]*nodesAWm2[i]   # Add node i to basinID (basinArea area)*(latitude) sum [deg*m2].
            LonCentroid[int(basinID)] += np.deg2rad(nodePosi[1])*nodesAWm2[i]   # Add node i to basinID (basinArea area)*(longitude) sum [rad*m2]. Note that this radian conversion only works for data on [-180,180], not [0,360]


        ## Divide by the basinAreas to complete the calculation of area weighted centorids
        LatCentroid = LatCentroid/basinArea;                # Latitude basin centroid [deg]
        LonCentroid = np.rad2deg(LonCentroid/basinArea);    # Longitude basin centroid [deg]

        # Calculate the distance between all centroids
        ## Empty symmetric matrix representing distance between centroids.
        basinCentroidDist = np.zeros((len(LonCentroid),len(LatCentroid))); # [radians]
        ## Empty symmetric matrix representing distance between centroids.
        basinMergeAllowed = np.zeros((len(LonCentroid),len(LatCentroid)), dtype='bool'); # [boolean]
        ## Calculate distance between points
        ## Iterate over all points.
        for i in range(len(LatCentroid)):
            ## Iterate over all points again.
            for j in range(len(LatCentroid)):
                ## Only do calculation if point matches are not the same
                ## and are unique.
                if i<j:
                    # Calculate the distance between centroids
                    # Note that the spherical radius can be set to 1 since
                    # we are only concerned with finding the closest basin.
                    # The actual magnitude of the distance is irrelevant.
                    basinCentroidDist[i,j] = haversine_distance(LatCentroid[i],
                                                                LonCentroid[i],
                                                                LatCentroid[j],
                                                                LonCentroid[j],
                                                                radius=1)
                    # Set the symmetric entry
                    basinCentroidDist[j,i] = basinCentroidDist[i,j];
                
                    # Set whether connecting i to j would be connecting it to a
                    # basin larger than the threshold.
                    if basinArea[j]/methodScalar >= threshold:
                        # Receiving basin is larger than threshold
                        basinMergeAllowed[i,j] = True;
                    else:
                        # Receiving basin is smaller than threshold
                        basinMergeAllowed[i,j] = False;

                    if basinArea[i]/methodScalar >= threshold:
                        # Giving basin is larger than threshold
                        basinMergeAllowed[j,i] = True;
                    else:
                        # Giving basin is smaller than threshold
                        basinMergeAllowed[j,i] = False;

                elif i==j:
                    # Set diagonal entries to np.nan. This allows for simplified
                    # code later on.
                    basinCentroidDist[i,j] = np.nan;
                    # Set diagonal entries to False since a basin cannot be connect
                    # to itself.
                    basinMergeAllowed[j,i] = False;


        # Define an array with columns (oldBasinID, newBasinID)
        # that describes how basinID should be changed
        basinIDTrans = np.zeros( (len( basinArea[basinArea/methodScalar < threshold] ), 2), dtype=float );
        basinIDTrans[:]= np.nan
        basinMatchCnt = 0;

        # Iterate over all unique basin IDs
        for i in range(len(nodeBasinIDUnique)):
            # Basin id
            basinIDi = nodeBasinIDUnique[i];
            # If node i's basin is smaller than threshold.
            if (basinArea[i]/methodScalar < threshold):

                if mergeMethod == 'centroid':
                    # Select centroid distances for basin mergers
                    basiniCentroidDist = basinCentroidDist[i,:];

                    # Remove centroid distances that allow merger with
                    # basin below threshold size.
                    basiniCentroidDist[ ~basinMergeAllowed[i,:] ] = np.nan;
                    
                    # Find the basin ID to merge basinIDi into
                    basinIDnew = np.argwhere(basiniCentroidDist == np.nanmin(basiniCentroidDist))[0][0];

                elif mergeMethod == 'nearBasinEdge':
                    # Find distance from centroid to every other node
                    Distance = haversine_distance(LatCentroid[i],
                                                  LonCentroid[i],
                                                  nodePosA[:,0],
                                                  nodePosA[:,1],
                                                  radius=1);

                    # Find the smallest distance to basin that is greater than
                    # threshold size
                    # True if distance is smallest
                    #     logical2 = (Distance == np.nanmin(Distance));
                    # True if nodes are part of basins greater than the threshold
                    #     logical1 basinArea[[nodeBasinID]]/methodScalar >= threshold
                    logical1 = (basinArea[nodeBasinID.astype(int)]/methodScalar >= threshold);
                    logical2 = (Distance[logical1] == np.nanmin(Distance[logical1]));

                    basinIDnew = nodeBasinID[logical1][logical2][0]


                # Add to matches
                basinIDTrans[basinMatchCnt,:] = np.array([basinIDi, basinIDnew]);
                basinMatchCnt+=1;
        

        # Merge all the matched basinIDs
        # Change basin id(s) basinIDTrans[0] to basinIDTrans[1] (not in nodes yet)
        # Change basin id(s) basinID2 to basinID1 (not in nodes yet)
        if np.size(basinIDTrans) == 2:
            # basinIDTrans[0,0] is converted to basinIDTrans[0,1].
            nodeBasinID[nodeBasinID==basinIDTrans[0,0]] = basinIDTrans[0,1];
        else:
            # Multiple (list) of basinIDs to convert to basinIDTrans[0,i].
            for basinIDold, basinIDnew in basinIDTrans:
                nodeBasinID[nodeBasinID==basinIDold] = basinIDnew;
        
        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);
        
        # Set basin id to new merged basin
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");

        ###########################
        ### Write network Model ###
        ###########################
        if write:
            print("Network has been overwritten.")
            nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
        else: 
            print("Network has not been overwritten. New network only exist within this object instance.")

    def orderBasins(self, order, write=False):
        """
        orderBasins method will reorder the basin IDs in accordance with
        some input list.

        Parameters
        ----------
        order : LIST
            List of basin IDs, where the entry's index corresponds to new
            basinID. The value of the entry corresponds to the current basinID.
            For example, this might look like [2,0,1], meaning basin2->basin0,
            basin0->basin1, and basin1->basin2.
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute. 

        """
        ##########################
        ### Merge basins Model ###
        ##########################
        # Get all node basin ids 
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );
        nodes = nx.get_node_attributes(self.G, "basinID");
        for nodeID in nodes:
            nodeBasinID[int(nodeID)] = nodes[float(nodeID)]['basinID'];
        
        # Define the total number of basinIDs
        uniqueIDCnt = len(np.unique(nodeBasinID));

        # Change basinID ordering 
        for i in range(len(order)):
            originalID  = order[i];
            newID       = uniqueIDCnt+i;
            # Update basinID
            nodeBasinID[nodeBasinID==originalID] = newID;

        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);

        # Set basin id to new ordered basin IDs
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");

    def applyMergeBasinMethods(self, mergerID, mergerPackage, maxBasinCnt=1e5):
        '''
        applyMergeBasinMethods function takes a ExoCcycle basins object a
        mergerID and mergerPackage to merge basins defined by a predetermined
        merger strategy. Some of these have been define by the creators of
        ExoCcycle, but every other strategy must be defined as shown below


        Parameters
        -----------
        mergerID : INT
            An ID that describes the basins merges to take place as described
            in mergerPackage
        mergerPackage : DICTIONARY
            A dictionary that describes the general merger strategy and strategy
            for a given mergerID. It can be constructed as follows. The following
            package first merges basins that represent 0.1% then 0.5% of total
            basin surface area into the closest basin bigger than 0.1% and 0.5%.
            Next, if mergerID=0 then mergers0 will merger basin 0 with 0,8,9,10,
            basin 1 with ..., and basin 2 with ...
            
                mergerPackage = {'mergeSmallBasins': {'on':True,
                                                    'threshold':np.array([.1,.5]),
                                                    'thresholdMethod':'%',
                                                    'mergeMethod':'nearBasinEdge'},
                                'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                                'mergers0':  {'0':[0,8,9,10], '1':[...], '2':[...] },
                                'mergers5':  {'0':[...], '1':[...], '2':[...] },
                                'mergers...':{'0':[...], '1':[...], '2':[...] }
                                }
        maxBasinCnt : INT
            Maximum number of basins to allow in a bathymetry
            model. The default is 1e5.

        Redefine
        -------
        self : EXOCCYCLE OBJECT
            Object that describes the connection of nodes into basins by
            way of a community detection algorithm. Now with merged basins
            as described by mergerID and mergerPackage.                  
        
        '''
        
        # 1. Apply the small basin mergers
        try:
            if mergerPackage['mergeSmallBasins']['on']:
                # Iterate over all the small basin merger thresholds.
                for i in range(len(mergerPackage['mergeSmallBasins']['threshold'])):
                    self.mergeSmallBasins(
                        threshold      = mergerPackage['mergeSmallBasins']['threshold'][i],
                        thresholdMethod= mergerPackage['mergeSmallBasins']['thresholdMethod'],
                        mergeMethod    = mergerPackage['mergeSmallBasins']['mergeMethod'],
                        write=True);
                # Since graph is modified (not array), we must interpolate the grid
                # from irregular to regular grid after each merger.
                self.BasinIDA = self.interp2regularGrid(mask=True)
        except:
            pass;

        if maxBasinCnt==1e5:
            # Case: Merge basins by id
            # 2. Merge basins by ID
            try:
                # If the mergerID exists within the mergerPackage mergerIDs then proceed
                # with merging
                if (mergerPackage['mergerID']==mergerID).any():
                    for mainBasin in mergerPackage['mergers'+str(mergerID)]:
                        self.mergeBasins(mainBasin,
                                        mergerPackage['mergers'+str(mergerID)][mainBasin],
                                        write=True);
                        # Since graph is modified (not array), we must interpolate the grid
                        # from irregular to regular grid after each merger.
                        self.BasinIDA = self.interp2regularGrid(mask=True)

            except:
                pass;
            
            # 3. Rearrange basins (ordering is useful to keep consistently through temporal reconstructions)
            try:
                # Id the mergerID exists within the mergerPackage mergerIDs then proceed.
                if (mergerPackage['mergerID']==mergerID).any():
                    basinOrder = mergerPackage['arrange'+str(mergerID)];
                    self.orderBasins(basinOrder,
                                    write=True);
                    # Since graph is modified (not array), we must interpolate the grid
                    # from irregular to regular grid after each merger.
                    self.BasinIDA = self.interp2regularGrid(mask=True)
            except:
                pass;
        else:
            # Case: Merge basins by the sum node edge weights
            # Since this method does not modify the graph, we do not
            # need to the interpolate the irregular to regular grid
            # after each merger.

            # 
            self.calculateBasinParameters(verbose=False);
            
            self.mergeSmallestConnection(maxBasinCnt=maxBasinCnt,
                                         verbose=True);



        # 4. Report results from merging
        try:
            # If the verbose option has been chosen then plot the merged basins
            if mergerPackage['verbose']:
                blues_cm = mpl.colormaps['Blues'].resampled(100)
                self.visualizeCommunities( cmapOpts={"cmap":blues_cm,
                                                    "cbar-title":"cbar-title",
                                                    "cbar-range":[np.nanmin(np.nanmin(self.bathymetry)),
                                                                    np.nanmean(self.bathymetry)+2*np.nanstd(self.bathymetry)]},
                                            pltOpts={"valueType": "Bathymetry",
                                                    "valueUnits": "m",
                                                    "plotTitle":"{}".format(self.body),
                                                    "plotZeroContour":True,
                                                    "nodesize":1,
                                                    "connectorlinewidth":1,
                                                    "projection":"Miller"},
                                            draw={"nodes":False,
                                                "connectors":False,
                                                "bathymetry":False,
                                                "coastlines":True,
                                                "gridlines":False,
                                                "nodes-contour":True},
                                            saveSVG=False,
                                            savePNG=True)
        except:
            pass

    def mergeSmallestConnection(self, maxBasinCnt, verbose=True):
        """
        mergeSmallestConnection merges smallest basins and strongest
        connected basins with one another until maxBasinCnt number
        of basins exist.

        Parameters
        ----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.
        
        """
        # Define the starting number of basins
        #BasinCnt = len(np.unique(self.BasinIDA))-1;

        # Calculate connectivity and connective bathymetry distributions
        # & Calculate the strength between basins self.basinConnectionWeight
        self.calculateBasinConnectivityParameters(verbose=False)

        # Iterate until desired number of basins is obtained
        while self.basinCnt > maxBasinCnt:

            # Find the connection with highest basinConnectionWeight
            # Note that if np.argwhere(self.basinConnectionWeight, np.nanmax(self.basinConnectionWeight))
            # = [4,5]. This means that basin 4 should be mergered into basin 5.
            idx = np.argwhere(self.basinConnectionWeight == np.nanmax(self.basinConnectionWeight))[0]

            
            # Merge basin with 
            self.mergeBasins(idx[1], idx[0], write=False);
            
            
            # (Re)calculate basin and connectivity and connective bathymetry
            # distributions & Calculate the strength between basins
            # self.basinConnectionWeight
            self.calculateBasinParameters(verbose=False);
            self.calculateBasinConnectivityParameters(verbose=False)
            


        #self.calculateBasinConnectivityParameters(verbose=False)
        if verbose:
            print("self.basinConnectionWeight", self.basinConnectionWeight)

    def saveBasinNetwork(self):
        """
        saveBasinNetwork method will overwrite the original basinID network
        file.
        
        Returns
        --------
        None.
        """
        ###########################
        ### Write network Model ###
        ###########################
        nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)


    def calculateBasinParameters(self, binEdges=None, fieldNum="Field1", fldName=os.getcwd(), verbose=True):
        """
        calculateBasinParameters method will calculate basin bathymetry
        distributions, area and ocean volume fractions for basin.
        
        Parameters
        ----------
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        fieldNum : STRING, optional
            Name of the field to be used. The default is "Field1".
        fldName : STRING
            Directory to save figures to. The default is os.getcwd().
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Redefines
        ----------
        self.BasinIDA : NUMPY ARRAY
            An nx2n array of values corresponding to basin IDs. Location of no
            basins are given a fill value of np.nan. The size of this arry depends
            on input bathymetry models as well as user input self.reducedRes class
            attribute. 
        self.bathymetryAreaDistBasin : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryVolFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin volume, normalized to the volume of all ocean basins
            (excluding the high latitude ocean volume).
        self.bathymetryAreaFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin area, normalized to the total seafloor area (including
            the high latitude area).
        self.bathymetryAreaDist_wHighlatG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the inclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryAreaDistG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.binEdges : NUMPY LIST
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin.
        self.BasinParametersDefined : BOOLEAN
            Set to True to indicate that basin bathymetry parameters have been defined.
            
        Returns
        --------
        None.
        
        
        """
        # Define arrays for latitude, longitude, bathymetry (use the reduced resolution)
        latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
        lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T
        bathymetry = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T

        # Define array for basinIDs and corresponding node ids (use the reduced resolution)
        nodePosDict = self.G.nodes.data('pos');
        nodeBasinID = self.G.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
        
        # Define basinID and nodeid array
        self.interp2regularGrid(mask=True);
        #BasinIDA = np.empty(np.shape(lonA));
        #BasinIDA[:] = np.nan;
        #for nodei in range(len(pos[:,1])):
        #    BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];

        # Calculate basin distributions
        bathymetryAreaDistBasin, bathymetryVolFraction, bathymetryAreaFraction, bathymetryAreaFractionG, bathymetryAreaDist_wHighlatG, bathymetryAreaDistG, binEdges = Bathymetry.calculateBathymetryDistributionBasin(bathymetry, latA, lonA, self.BasinIDA, self.highlatlat, self.areaWeights, binEdges=binEdges, fldName=fldName, verbose=verbose)
        
        # Define basinID and nodeid array
        #self.BasinIDA = BasinIDA;

        # Set basin bathymetry parameters
        self.bathymetryAreaDistBasin = bathymetryAreaDistBasin
        self.bathymetryVolFraction   = bathymetryVolFraction
        self.bathymetryAreaFraction  = bathymetryAreaFraction
        self.bathymetryAreaFractionG = bathymetryAreaFractionG
        self.bathymetryAreaDist_wHighlatG = bathymetryAreaDist_wHighlatG
        self.bathymetryAreaDistG = bathymetryAreaDistG
        self.binEdges                = binEdges

        # Change class attribute to indicate that basin bathymetry
        # parameters have been defined.
        self.BasinParametersDefined = True;

    def calculateBasinConnectivityParameters(self, binEdges=None, disThres=444, fieldNum="Field1", fldName=os.getcwd(), verbose=True):
        """
        calculateBasinConnectivityParameters method is used to calculate
        basin connectivity parameters. The parameters are described in
        the returned parameters section.

        Parameters
        ----------
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        disThres : FLOAT, optional
            Value, in km, represents the distance threshold, away from
            a basin boundary, which will be consider part of that basin
            connection. The default value is 444.
        fieldNum : STRING, optional
            Name of the field to be used. The default is "Field1".
        fldName : STRING
            Directory to save figures to. The default is os.getcwd().
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Redefined
        --------
        self.basinCnt : INT
            Number of basins in model.
        self.basinConCnt : INT
            (self.basinCnt^2-self.basinCnt)/2. This value corresponds to the number
            of basin connections.
        self.validConCnt : INT
            Number of valid connections between basins (i.e., where self.bathymetryConDist)
            is not a vector of NaNs.
        self.connectingNodes : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of node
            IDs that correspond to a basin connection.
        self.connectiveBathy : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of
            bathymetry, in m, that correspond to a basin connection's nodes.
        self.connectiveAreaW : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of area
            weights that correspond to a basin connection's nodes.
        self.basinAreaConnection : NUMPY ARRAY
            A self.basinCnt x self.basinCnt matrix that enumerate the connections
            between basins. This is a useful book keeping attribute. 
        self.basinConnectionWeight : NUMPY ARRAY
            A self.basinCnt x self.basinCnt matrix holds weights describing how
            well connected two basins are. The connection strength is calculated
            based on connective bathymetry on a set of rules described within
            the body of code.
        self.bathymetryConDist : NUMPY ARRAY
            A self.basinCnt x self.basinCnt x (binEdges-1) matrix that holds area
            weighted bathymetry distributions of connective bathymetry between
            basins. Distributions sum to 100%. 
        self.basinConnectionDefined : BOOLEAN
            Set to positive to indicate basin connectivity parameters are calculated.
        """
        ############################################################
        ##### Find the connectivity between each set of basins #####
        ############################################################


        ############################################################
        ############# Find nodes boarding other basins #############
        ############################################################
        # Define the basin and connection count and graph
        self.basinCnt = len(np.unique(self.BasinIDA))-1
        self.basinConCnt = (self.basinCnt*self.basinCnt-self.basinCnt)//2
        BasinNodes = self.G;

        # Define node position array and basin ID vector
        nodePosDict = BasinNodes.nodes.data('pos');
        nodeBasinID = BasinNodes.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];

        # Create array of basins IDs
        latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
        lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T

        BasinIDA = self.BasinIDA[::self.reducedRes].T[::self.reducedRes].T

        #BasinIDA = np.empty(np.shape(lonA));
        #BasinIDA[:] = np.nan;
        #for nodei in range(len(pos[:,1])):
        #    BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];
        
        # Create an empty array to hold positions of basin nodes at basin's edges
        posEdges = np.zeros( (0, 2) );

        # Create an empty array to hold basin edge nodes IDs
        basinEdgeNodeBasinIDs1 = np.array([],dtype=float);
        basinEdgeNodeBasinIDs2 = np.array([],dtype=float);

        # Counter for number of nodes that are edges of basins
        cnt=0;

        # Populate basinEdgeNodeBasinIDs1 and basinEdgeNodeBasinIDs1 with
        # a basin edge node and its reciprocal.
        ## Iterate over all nodes
        for j in range(len(nodePosDict)):
            ## Iterate neighbors of node
            for i in BasinNodes.neighbors(j):
                ## If neighbors are not part of the same basin
                if nodeBasinID[i]['basinID'] != nodeBasinID[j]['basinID']:           
                    ## Add location of basin edge
                    posEdges = np.append(posEdges, [nodePosDict[j]], axis=0)
                    ## Add basin edge node to list                    
                    basinEdgeNodeBasinIDs1 = np.append(basinEdgeNodeBasinIDs1, 
                                                        nodeBasinID[j]['basinID']);
                    basinEdgeNodeBasinIDs2 = np.append(basinEdgeNodeBasinIDs2,
                                                        nodeBasinID[i]['basinID']);
                    ## Update counter
                    cnt+=1;
        
        ############################################################
        ############# Create basin connection ID array #############
        ############################################################
        # Define basinAreaConnection, a matrix which will hold an
        # enumeration of the connections between basins.
        self.basinAreaConnection = np.zeros((self.basinCnt,self.basinCnt), dtype=float);
        # Define self.basinAreaConnectionTracker, which keeps track of the 
        # symmetric basin connection (e.g., Atlantic-Indian for Indian-Atlantic)
        # has already been defined.
        basinAreaConnectionTracker = np.ones((self.basinCnt,self.basinCnt), dtype=bool);
        cnt=0

        # Populate basinAreaConnection with enumerations of basin
        # connections.
        ## Loop over basins
        for basini in range(self.basinCnt):
            ## Loop over basins
            for basinj in range(self.basinCnt):
                ## Loop over other basins
                if  (basini!=basinj) & basinAreaConnectionTracker[basinj,basini]:
                    ## Set basin connection ID
                    self.basinAreaConnection[basini,basinj] = cnt;
                    cnt+=1;
                    ## Indicate that basin connection ID has been set
                    basinAreaConnectionTracker[basini,basinj] = False;
                elif (basini!=basinj):
                    ## If symmetric calculation has already been done
                    ## then set the symmetric array value
                    self.basinAreaConnection[basini,basinj] = self.basinAreaConnection[basinj,basini]
                elif (basini==basinj):
                    self.basinAreaConnection[basini,basinj] = -1;
        
        ############################################################
        ####### Get the bathy/weights/nodeID for all nodes #########
        ############################################################
        allNodeIDs = np.array([], dtype=float);
        allNodeBathymetry = np.array([], dtype=float);
        allNodeAreaWeights = np.array([], dtype=float);
        for node in BasinNodes:
            allNodeIDs = np.append(allNodeIDs, node);
            allNodeBathymetry  = np.append(allNodeBathymetry, BasinNodes.nodes[node][fieldNum]);
            allNodeAreaWeights = np.append(allNodeAreaWeights, BasinNodes.nodes[node]['areaWeightm2']);
            
        
        ############################################################
        ### Set nodeIDs and bathymetry for each basin connection ###
        ############################################################

        # Define basin connective nodes, bathymetry, and area weights
        #  as list of list. This is done such that each basin connection
        # can have different numbers of nodes to describe them.
        self.connectiveBathy = np.empty(self.basinConCnt,dtype=object)
        self.connectiveAreaW = np.empty(self.basinConCnt,dtype=object)
        self.connectingNodes = np.empty(self.basinConCnt,dtype=object)
        for i in range(len(self.connectingNodes)):
            self.connectiveAreaW[i] = np.array([], dtype = float);
            self.connectiveBathy[i] = np.array([], dtype = float);
            self.connectingNodes[i] = np.array([], dtype = float);

        # Populate connectingNodes, for each basin connection, with
        # node IDs that are within disThres of the basin edges.
        ## Iterate over basin edge nodes
        for i in range(len(basinEdgeNodeBasinIDs1)):
            ## Calculate distance from all nodes to the edge of a
            ## basin edge node, i.
            distanceV = haversine_distance(pos[:,0], pos[:,1],
                                           posEdges[i,0], posEdges[i,1],
                                           self.radius*1e-3);
            ## If the distance is less then a threshold then set
            ## the node as a basin connective node
            logical = (distanceV<=disThres);
            
            ## Find the correct connection
            connectionID = int(basinEdgeNodeBasinIDs1[int(i)])
            connectionID = self.basinAreaConnection[int(basinEdgeNodeBasinIDs1[int(i)]),
                                                    int(basinEdgeNodeBasinIDs2[int(i)])]
            
            ## Add nodeIDs to appropriate basin connection list (using connectionID)
            ## And remove any repeat nodes
            if connectionID != -1:
                self.connectingNodes[int(connectionID)] = np.append(self.connectingNodes[int(connectionID)],
                                                                    allNodeIDs[logical] );
                self.connectingNodes[int(connectionID)] = np.unique(self.connectingNodes[int(connectionID)]);
    
        # Add bathymetry and area weights to appropriate basin
        # connection list (using connectionID).
        ## Loop over unique basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            self.connectiveBathy[int(connectionIDi)] = np.append(self.connectiveBathy[int(connectionIDi)],
                                                                 allNodeBathymetry[self.connectingNodes[int(connectionIDi)].astype('int')] );
            self.connectiveAreaW[int(connectionIDi)] = np.append(self.connectiveAreaW[int(connectionIDi)],
                                                                 allNodeAreaWeights[self.connectingNodes[int(connectionIDi)].astype('int')] );
        
        ############################################################
        # Calculate bathymetry distributions for basin connections #
        ############################################################

        # If binEdges are not defined in method input then define with
        # LOSCAR's, a carbon cycle model, default distribution.
        if binEdges is None:
            binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    
        # Setup array to hold connective bathymetry distributions.
        #self.bathymetryConDist = {};
        self.bathymetryConDist = np.zeros( (self.basinCnt, self.basinCnt, len(binEdges)-1) ,dtype=float)
    
        # Calculate all basin connection bathymetry distributions.
        ## Iterate over basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            # Calculate a basin connection bathymetry distributions.
            bathymetryConDisti, binEdges = np.histogram((1e-3)*self.connectiveBathy[int(connectionIDi)],
                                                             weights = self.connectiveAreaW[int(connectionIDi)],
                                                             bins=binEdges);
            # Add the distribution information to dictionary.
            for symmeticIndex in np.argwhere(self.basinAreaConnection == int(connectionIDi)):
                self.bathymetryConDist[ symmeticIndex[0], symmeticIndex[1] ] = 100*(bathymetryConDisti/np.sum(bathymetryConDisti));
        
        ############################################################
        ############### Calculate connection weights ###############
        ############################################################
        ## Define basinConnectionWeight
        self.basinConnectionWeight = np.zeros( (self.basinCnt, self.basinCnt), dtype=float)

        ## Iterate over basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            
            # Define the connection area (size).
            areaofConnection =  np.sum(self.connectiveAreaW[connectionIDi]);

            for symmeticIndex in np.argwhere(self.basinAreaConnection == int(connectionIDi)):
                ## Iterate over symmetric IDs
                # Define basins' area (size)
                areaofBasin1 =      self.bathymetryAreaFractionG["Basin{}".format(symmeticIndex[0])];
                areaofBasin2 =      self.bathymetryAreaFractionG["Basin{}".format(symmeticIndex[1])];
                
                # Set weight
                self.basinConnectionWeight[ symmeticIndex[0], symmeticIndex[1] ] = (1/areaofBasin1)*np.nansum( areaofConnection*self.bathymetryConDist[ symmeticIndex[0], symmeticIndex[1] ]/100 );

        # Change class variable to indicate that basin connectivity
        # parameters have been defined.
        self.basinConnectionDefined = True;
        
        # Calculate the number of valid connections between basins
        self.validConCnt = 0;
        for i in range(self.basinCnt):
            for j in range(self.basinCnt):
                if (i>j) & (~np.isnan(self.bathymetryConDist[i,j,:])).any():
                    self.validConCnt += 1;

        ############################################################
        ####################### Plot results #######################
        ############################################################
        if verbose:
            # Report the basin connectivity distributions
            # print("Bin edges used:\n", binEdges);
            # print("Bathymetry area distribution including high latitude bathymetry:\n");
            # for connectionIDi in range(self.basinConCnt):
            #     idx = np.argwhere(connectionIDi==self.basinAreaConnection)[0];
            #     print(self.bathymetryConDist[idx[0], idx[1]]);
            
            # Plot the basin IDs, connectivity nodes, and their
            # bathymetry distributions
            self.plotBasinConnections(pos, binEdges, fieldNum=fieldNum, fldName=fldName, savePNG=True, saveSVG=True);

    def plotBasinConnections(self, pos, binEdges=None, fieldNum = "Field1",
                             fldName = os.getcwd(), savePNG=False, saveSVG=False, fidName = "BasinConnections.png"):
        """
        plotBasinConnections is used to plot results calculating from
        running the method calculateBasinConnectivityParameters.

        Parameters
        -----------
        pos : NUMPY ARRAY
            An nx2 array with columns of latitude and longitude, in degrees. This
            array should represent the locations of basin nodes.
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        fieldNum : STRING, optional
            Name of the field to be used. The default is "Field1".
        fldName : STRING
            Directory to save figures to. The default is os.getcwd().
        savePNG : BOOLEAN
            An option to save an PNG output. The default is False.
        saveSVG : BOOLEAN
            An option to save an SVG output. The default is False.

        Returns
        --------
        None.
        """
        # If binEdges are not defined in method input then define with
        # LOSCAR's, a carbon cycle model, default distribution.
        if binEdges is None:
            binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);


        # Set up the Mollweide projection
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1]);  # 2 rows, 1 column, with both row heights equal.

        ax1 = fig.add_subplot(gs[0], projection=ccrs.Mollweide());

        # Create colormap (basins IDs)
        ## Set colormap
        cmap1 = plt.get_cmap("Pastel1")
        ## Extract basinCnt colors from the colormap
        colors_rgb1 = [cmap1(i) for i in range(self.basinCnt)]
        ## Convert RGB to hex
        colors1 = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb1]
        ## Create a custom colormap from the list of colors
        custom_cmap1 = LinearSegmentedColormap.from_list("custom_pastel", colors1, N=256)

        # Create colormap (Connection IDs)
        cmap2 = plt.get_cmap("Dark2")
        ## Extract basinCnt colors from the colormap
        colors_rgb2 = [cmap2(i) for i in range(self.validConCnt)]
        ## Convert RGB to hex
        colors2 = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb2]
        ## Create a custom colormap from the list of colors
        custom_cmap2 = LinearSegmentedColormap.from_list("custom_pastel", colors2, N=256)

        # Plot basin contourf and coastlines

        ## Add the plot using pcolormesh
        mesh = ax1.pcolormesh(self.lon, self.lat, self.BasinIDA, cmap=custom_cmap1, transform=ccrs.PlateCarree(), zorder=0)

        ## Add coastlines
        ### Set any np.nan values to 0.
        bathymetry = self.bathymetry
        bathymetry[np.isnan(bathymetry)] = 0;
        ### Plot coastlines.
        zeroContour = ax1.contour(self.lon, self.lat, bathymetry,levels=[0], colors='black', transform=ccrs.PlateCarree())

        # Plot basin connection contour.

        # ## Define global array of connective bathymetry
        # BC = np.empty((np.shape(self.lat)));
        # BC[:] = np.nan;
        # for connectingNodei in range(len(self.connectingNodes)):
        #     for lat, lon in pos[self.connectingNodes[connectingNodei].astype('int')]:
        #         BC[(self.lat==lat)&(self.lon==lon)] = connectingNodei; # FIXME: Only works for equal spacing (in degrees)


        # ## Plot 
        # plt.contourf(self.lon, self.lat, BC,
        #              cmap=custom_cmap2,
        #              transform=ccrs.PlateCarree(), zorder=1);
        def resolutionScaledPoints(ax, resolution=1):
            """
            Convert a resolution in degrees to an approximate marker size in pt2
            that appears uniform on a global PlateCarree map.
            """
            fig = ax.get_figure()
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width_inch, height_inch = bbox.width, bbox.height

            # Use the global degree range
            deg_width = 360  # Longitude range
            deg_height = 180  # Latitude range

            deg_per_inch_x = deg_width / width_inch
            deg_per_inch_y = deg_height / height_inch

            # Approximate with the average
            deg_per_inch = np.mean([deg_per_inch_x, deg_per_inch_y])

            diameter_inch = resolution / deg_per_inch
            radius_pt = (diameter_inch * 72) / 2.0  # 1 inch = 72 points
            area_pt = np.pi * radius_pt**2
            return area_pt

        scatter_area = resolutionScaledPoints(ax1, resolution=self.Fields[fieldNum]['resolution'])

        validCon = 0;
        for connectingNodei in range(len(self.connectingNodes)):
            connection = pos[self.connectingNodes[connectingNodei].astype('int')]
            if np.size(connection) != 0:
                plt.scatter(connection[:,1], connection[:,0],
                            s=scatter_area,
                            color=colors2[validCon],
                            transform=ccrs.PlateCarree(), zorder=1);
                validCon+=1
        #self.pos= pos;



        # Plot gridlines
        ax1.gridlines()

        # Plot bathymetry distributions of basin connections.

        ## Set new axis to plot on
        ax2 = fig.add_subplot(gs[1]);

        ## Define factors for plotting
        factor1 = .1
        factor2 = .15
        if self.basinConCnt%2:
            factor3 = 0;
        else:
            factor3 = 1;

        ## Iteratively plot basin bathymetry distributions
        print(binEdges)
        validConi = 0;
        for i in range(self.basinConCnt):
            # Calculate index of distribution
            idx = np.argwhere(i==self.basinAreaConnection)[0];
            # Check if distribution is valid (i.e., if basins are connected)
            if (~np.isnan(self.bathymetryConDist[idx[0],idx[1]])).any():
                # Distribution is valid; now plot
                plt.bar(x=binEdges[1:]+(validConi)*((factor2)*np.diff(binEdges)) - (validCon-factor3)*(factor2*np.diff(binEdges))/2,                        
                        height=self.bathymetryConDist[idx[0],idx[1]],
                        width=factor1*np.diff(binEdges),
                        label= "Connection {:0.0f}".format(validConi),
                        color=colors2[validConi])
                validConi += 1

        ## Format ticks
        plt.xticks(binEdges[1:]);
        plt.yticks(np.arange(0,35,5));

        ## Format Labels
        plt.legend();
        plt.title("Basin Connective Bathymetry Distributions")
        plt.xlabel("Bathymetry Bins [km]");
        plt.ylabel("Seafloor Area [%]");

        ## Format figure format
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)


        # Save figure
        if savePNG:
            plt.savefig("{}/{}".format(fldName,fidName), dpi=600)
        if saveSVG:
            plt.savefig("{}/{}".format(fldName,fidName.replace(".png", ".svg")))
            
    def saveCcycleParameter(self, fieldNum="Field1", verbose=True):
        """
        saveCcycleParameter method create a new netCDF4 containing 
        the original bathymetry model, areaweights, and other global
        bathymetry parameters. Additionally, the new netCDF4 will
        contain a basinID array as well as bathymetry distributions,
        area and ocean water volume fractions for those basins.

        This method produces a netCDF4 that contains two groups
        (CycleParms and Arrays). These groups might look like the
        following for a global ocean system that has 3 basins:

        group /basinConnections:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            basinConnectionBathymetry(BasinID, BasinID, binEdges)


        group /CycleParms:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            float32 binEdges(binEdges),
            float64 Global-whighlat(binEdges),
            float64 Global(binEdges),
            float64 basin-0(binEdges),
            float64 basin-1(binEdges),
            float64 basin-2(binEdges),
            float32 BasinID(BasinID),
            float64 fdvol(BasinID),
            float64 fanoc(BasinID),
            float64 highlatlat(),
            float64 highlatA(),
            float64 AOC()
            float64 VOC(),
            groups: 

        group /Arrays:
            dimensions(sizes): lat(180), lon(360)
            variables(dimensions):
            float32 lat(lat),
            float32 lon(lon),
            float64 bathymetry(lat, lon),
            float64 basinIDArray(lat, lon),
            float64 areaWeights(lat)
            groups: 

        Parameters
        -----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        --------
        None.
        """
        # Set netCDF4 filename
        BathyPath = "{0}/{1}".format(self.dataDir, self.filename.replace(".nc", "_wBasins.nc"));
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4')

        # Report what values will be stored in saved netCDF4
        storedValuesStrV = ["---", "---"]
        if self.BasinParametersDefined:
            storedValuesStrV[0] = "";
        else:
            storedValuesStrV[0] = "not ";
        if self.basinConnectionDefined:
            storedValuesStrV[1] = "";
        else:
            storedValuesStrV[1] = "not ";
        if verbose:
            print("Basin bathymetry parameters are {}being stored in netCDF4 group CycleParms".format(storedValuesStrV[0]));
            print("Basin connectivity bathymetry parameters are {}being stored in netCDF4 group basinConnections".format(storedValuesStrV[1]));

        # Define groups
        ArraysGroup = ncfile.createGroup("Arrays")
        if self.BasinParametersDefined:
            CycleParmsGroup = ncfile.createGroup("CycleParms")
        if self.basinConnectionDefined:
            BasinConnectionsGroup = ncfile.createGroup("basinConnections")

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ArraysGroup.createDimension('lat', len(self.bathymetry[:,0]));     # latitude axis
        lon_dim = ArraysGroup.createDimension('lon', len(self.bathymetry[0,:]));     # longitude axis
        if self.BasinParametersDefined:
            binEdges_dim = CycleParmsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = CycleParmsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        if self.basinConnectionDefined:
            binEdges_dim = BasinConnectionsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = BasinConnectionsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        
        # Define lat/lon with the same names as dimensions to make variables.
        lat = ArraysGroup.createVariable('lat', np.float32, ('lat',));
        lat.units = 'degrees_north'; lat.long_name = 'latitude';
        lon = ArraysGroup.createVariable('lon', np.float32, ('lon',));
        lon.units = 'degrees_east'; lon.long_name = 'longitude';

        # Define a 2D variable to hold the elevation data
        bathy = ArraysGroup.createVariable('bathymetry',np.float64,('lat','lon'))
        bathy.units = 'meters'
        bathy.standard_name = 'bathymetry'

        # Define a 2D variable to hold the basinID information
        basinIDArray = ArraysGroup.createVariable('basinIDArray',np.float64,('lat','lon'))
        basinIDArray.units = 'ID'
        basinIDArray.standard_name = 'BasinID'

        # Define vector as function with longitude dependence
        areaWeights = ArraysGroup.createVariable('areaWeights',np.float64,('lat',))
        areaWeights.units = 'meters sq'
        areaWeights.standard_name = 'areaWeights'

        # Define variables for bathymetry distributions (vectors)
        if self.BasinParametersDefined:
            ## Global
            binEdges = CycleParmsGroup.createVariable('binEdges', np.float32, ('binEdges',));
            binEdges.units = 'km'; binEdges.long_name = 'km depth';

            distribution_whighlat = CycleParmsGroup.createVariable('Global-whighlat', np.float64, ('binEdges',))
            distribution_whighlat.units = 'kernal distribution'
            distribution_whighlat.standard_name = 'Global-whighlat'

            distribution = CycleParmsGroup.createVariable('Global', np.float64, ('binEdges',))
            distribution.units = 'kernal distribution'
            distribution.standard_name = 'Global'

            ## Basins Scale Variables
            ### Basin distribution
            distributionBasins = {};
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)] = CycleParmsGroup.createVariable('basin-{:0.0f}'.format(basinIDi), np.float64, ('binEdges',));
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].units = 'kernal distribution';
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].standard_name = 'basin-{:0.0f}'.format(basinIDi);

            ### BasinID
            BasinID = CycleParmsGroup.createVariable('BasinID', np.float32, ('BasinID',));
            BasinID.units = 'ID'; binEdges.long_name = 'BasinID';

            ### Basin Volume fractions
            fdvol = CycleParmsGroup.createVariable('fdvol', np.float64, ('BasinID',))
            fdvol.units = '%'
            fdvol.standard_name = 'fdvol'
            fdvol.long_name = "fdvol the sum of which is equal to 100% (of VOC now within the high latitude area)"

            ### Basin Area fractions (sum to 100% - highlatA/AOC)
            fanoc = CycleParmsGroup.createVariable('fanoc', np.float64, ('BasinID',))
            fanoc.units = '%'
            fanoc.standard_name = 'fanoc'
            fanoc.long_name = "fanoc the sum of which is equal to 100% - highlatA/AOC"

            # Define single values parameters (e.g., VOC, AOC, high latitude cutoff)
            highlatlat = CycleParmsGroup.createVariable('highlatlat', None)
            highlatlat.units = 'degrees'
            highlatlat.standard_name = 'highlatlat'

            highlatA = CycleParmsGroup.createVariable('highlatA', None)
            highlatA.units = 'meters sq'
            highlatA.standard_name = 'highlatA'

            VOC = CycleParmsGroup.createVariable('VOC', None)
            VOC.units = 'meters cubed'
            VOC.standard_name = 'VOC'

            AOC = CycleParmsGroup.createVariable('AOC', None)
            AOC.units = 'meters sq'
            AOC.standard_name = 'AOC'

        if self.basinConnectionDefined:
            basinConnectionBathymetry = BasinConnectionsGroup.createVariable('basinConnectionBathymetry', np.float32, ('BasinID', 'BasinID', 'binEdges'));
            basinConnectionBathymetry.units = 'km'; basinConnectionBathymetry.long_name = 'km depth';
        
        # Format title
        ncfile.title='{} Bathymetry created from topography resampled at {:0.0f} degrees. NetCDF4 includes carbon cycle bathymetry parameters'.format(self.body, self.Fields[fieldNum]['resolution'])
        
        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        bathy[:] = self.bathymetry;
        basinIDArray[:] = self.BasinIDA;
        areaWeights[:] = self.areaWeights[:,0];


        if self.BasinParametersDefined:
            # Add bathymetry distribution information
            distribution_whighlat[:] = self.bathymetryAreaDist_wHighlatG;
            distribution[:] = self.bathymetryAreaDistG;
            binEdges[:] = self.binEdges[1:];

            # Add basin distribution information
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)][:] = self.bathymetryAreaDistBasin['Basin{:0.0f}'.format(basinIDi)]

            # Add basin area and volume fractions
            fdvolValues = np.zeros(len(self.bathymetryAreaDistBasin));
            fanocValues = np.zeros(len(self.bathymetryAreaDistBasin));
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                fdvolValues[basinIDi] = self.bathymetryVolFraction['Basin{:0.0f}'.format(basinIDi)];
                fanocValues[basinIDi] = self.bathymetryAreaFraction['Basin{:0.0f}'.format(basinIDi)];
            fdvol[:] = fdvolValues;
            fanoc[:] = fanocValues;

            # Add attributes
            highlatlat[:] = self.highlatlat;
            highlatA[:] = self.highlatA;
            VOC[:] = self.VOC;
            AOC[:] = self.AOC;
        
        if self.basinConnectionDefined:
            basinConnectionBathymetry[:] = self.bathymetryConDist;
            

        # Close the netcdf
        ncfile.close();

        # Report contents of the created netCDF4
        if verbose:
            # Open netCDF4
            ncfile = Dataset(BathyPath, mode='r', format='NETCDF4')

            # Report netCDF4 contents
            print("Group\tVariable\t\t\tDimensions\t\t\t\tShape")
            print("--------------------------------------------------------------------------------------")
            for groupi in ncfile.groups:
                print(groupi);
                for variable in ncfile[groupi].variables:
                    if len(variable) != 20: 
                        variablePrint = variable.ljust(25)
                    print("\t"+variablePrint.ljust(25)+
                        "\t"+str(ncfile[groupi][variable].dimensions).ljust(35)+
                        "\t"+str(ncfile[groupi][variable].shape).ljust(35))
            
            # Close netCDF4
            ncfile.close();



#######################################################################
###################### Basin definition functions #####################
#######################################################################


class Basins():
    """
    Basins is a class meant to construct basins and bathymetry properties
    given a bathymetry model netCDF4.
    """

    def __init__(self, dataDir, filename, body):
        """
        Initialization of Basins class.

        Parameter
        ----------
        dataDir : STRING
            A directory which you store local data within. Note that this
            function will download directories [data_dir]/topographies
        filename : STRING
            Output file name 
        body : STRING
            Name of the input terrestial body. This is only used for
            visualization purposes, not naming convections when writing
            model files.
        
        Define
        ----------
        self.bathymetry : NUMPY ARRAY
        self.areaWeights : NUMPY ARRAY
        self.lat : NUMPY ARRAY
        self.lon : NUMPY ARRAY
        self.radius : FLOAT

        """

        # Read netCDF4 bathymetry file
        self.nc = Dataset("{}/{}".format(dataDir, filename));
        self.dataDir = dataDir;
        self.filename = filename;
        self.body = body;

        # Set latitude/longitude/elev
        self.lon, self.lat  = np.meshgrid( np.array(self.nc['lon'][:]), np.array(self.nc['lat'][:]) );
        self.bathymetry = np.array(self.nc['bathymetry'][:]);
        self.areaWeights = np.reshape(np.repeat(np.array(self.nc["areaWeights"][:]), np.shape(self.nc["bathymetry"][:])[1] ), np.shape(self.nc["bathymetry"]) );
        self.radius = np.sqrt( np.sum(np.sum(self.areaWeights))/(4*np.pi) );
        self.highlatlat = self.nc['highlatlat'][:].data;
        self.highlatA = self.nc['highlatA'][:].data;
        self.VOC = self.nc['VOC'][:].data;
        self.AOC = self.nc['AOC'][:].data;    


        # Define class attributes to be redefined throughout analysis
        ## Have basin connection been defined.
        self.basinConnectionDefined = False;
        ## Have basin bathymetry parameters been defined.
        self.BasinParametersDefined = False;
        
        # Close file  
        self.nc.close();

    def defineBasins(self, minBasinCnt = 3,
                     method = "Louvain",
                     reducedRes={"on":False,"factor":15},
                     read=False,
                     write=False,
                     verbose=True):
        """
        defineBasins method will define basins with network analysis
        using either the Girvan-Newman or Louvain algorithm to define
        communities.

        Parameter
        ----------
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.
        method : STRING
            Determines the implemented community detection algorithm.
            The options are either Girvan-Newman or Louvain. The former
            is more robust with low scalability and the latter is practical
            but produces non-deterministic communities. The default is
            Louvain.
        reducedRes : DICTIONARY
            Option to reduce the resolution of the basin definition
            network calculation. Note that this should be turned
            off when doing analysis, and only kept on for testing
            purposes. The default is {"on":False,"factor":15}.
        read : BOOLEAN
            An option to read basin definitions for a given
            bathymetry model. The default is False. 
        write : BOOLEAN
            An option to write basin definitions for a given
            bathymetry model. The default is False. 
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.basinCnt : INT
            Number of basins in global bathymetry model.
        self.basinDis : NUMPY ARRAY
            binCnt x basinCnt array of bathymetry distributions.
        
            
        FIXME: NEED TO MAKE SURE reduceRes and resolution are interacting
        properly and not only for 1 degree resolution input bathymetry
        models.

        """

        ##########################
        ### Write/Load network ###
        ##########################
        if read:
            ####################
            ### Read network ###
            ####################
            self.G = nx.read_gml("{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), destringizer=float);
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
            
            # Define resolution
            self.resolution = self.reducedRes*np.diff(self.lon)[0][0];
        
        else:
            ######################
            ### Create network ###
            ######################

            # Define counter and point dictionary
            cnt = 0.
            points = {};

            # Define resolution
            resolution = 1;
            
            # Only reduce resolution if option is set.
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf  = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
                areaWeightsf = self.areaWeights[::self.reducedRes].T[::self.reducedRes].T.flatten();

            # Distance, in degrees, to a diagonal node.
            cornerDis = self.reducedRes/np.sin(np.pi/4);

            # Create dictionary and array of bathymetry points
            pos = np.zeros( (2, len(~np.isnan(bathymetryf))) );
            for i in tqdm( range(len(self.lonf)) ):
                bathymetryi = bathymetryf[i];
                areaWeighti = areaWeightsf[i];
                if (~np.isnan(bathymetryi)):
                    points[int(cnt)] = (self.latf[i], self.lonf[i], bathymetryi, areaWeighti);    # (latitude, longitude, depth, areaWeight) w/ units (deg, deg, m, m2)
                    pos[:,int(cnt)] = np.array( [self.latf[i], self.lonf[i]] ); 
                    cnt+=1;

            # Create a graph
            G = nx.Graph()

            # Add nodes (points)
            for node, values in points.items():
                G.add_node(node, pos=values[0:2], depth=values[2], areaWeightm2=values[3]);

            # Update to the above code block which Adds edges with weights based on geographic distance.
            # This code is significantly faster than the above code block

            ## Set bathymetry vector corresponding to node 
            nodebathymetryf = ~np.isnan(bathymetryf);
            nodeCntList = np.arange(0,len(pos[0,:]),1);

            ## Iterate through each node
            for node1, values1 in tqdm(points.items()):
                # Set coordinates of current iterated node
                coords1 = values1[0:2];

                # Find values surrounding coords1 (current iterated node)
                #coordsPotential = {}; cnt=0;
                indices = [];
                indicesAddPt = [];
                # Set above and below coordinate 
                if not (coords1[0] == np.max(self.latf)):
                    # Not North pole node
                    ## upper node
                    condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(1)
                if not (coords1[0] == np.min(self.latf)):
                    # Not South pole node
                    ## lower node
                    condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(2)

                # If on periodic boundary
                if coords1[1] == np.min(self.lonf):
                    # nodes on left (min) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(3)
                        indicesAddPt.append(3)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(4)
                        indicesAddPt.append(4)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], np.max(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(5)
                    indicesAddPt.append(5)

                elif coords1[1] == np.max(self.lonf):
                    # nodes on right (max) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(6)
                        indicesAddPt.append(6)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(7)
                        indicesAddPt.append(7)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], np.min(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(8)
                    indicesAddPt.append(8)

                else:
                    # Nodes not on boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(9)
                        indicesAddPt.append(9)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(10)
                        indicesAddPt.append(10)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(11)
                    indicesAddPt.append(11)
                    
                for idx in indices:
                    if not (idx.size == 0):
                        # Set node, coordinates, and bathymetry value of current iterated edge node.
                        node2 = idx[0];
                        coords2 = G.nodes[node2]['pos'];
                        values2 = G.nodes[node2]['depth'];
                        
                        # Calculate geographic distance between points using geodesic distance.
                        bathyAve= (values1[2]+values2)/2;
                        hexsidelengnth = octPolylineLength(coords1, coords2, verbose=False);
                                                
                        G.add_edge(node1, node2, bathyAve=bathyAve*hexsidelengnth);
            

            # Set some class parameters for testing purposes.
            self.G = G;

            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(method = method, minBasinCnt = minBasinCnt);

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
            if write:
                nx.write_gml(G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
            

            ################
            ### Plotting ###
            ################
            if verbose:
                # Plot the network on a geographic map
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

                # Draw the nodes (points) on the map
                for node, data in G.nodes(data=True):
                    ax.plot(data['pos'][1], data['pos'][0], 'bo', markersize=1)  # longitude, latitude

                # Draw the edges (connections)
                for edge in G.edges(data=True):
                    node1, node2, weight = edge
                    lon1, lat1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
                    lon2, lat2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                    ax.plot([lon1, lon2], [lat1, lat2], 'k-', transform=ccrs.PlateCarree())

                # Add coastlines and gridlines
                if "Earth" in self.filename:
                    ax.coastlines()
                ax.gridlines()

                plt.title("Geographic Network of points")
                plt.show()

    def setEdgeParameter(self,
                         netCDF4Path,
                         readParm,
                         edgeParaOpt,
                         resample=False):
        """
        setEdgeParameter method is used to define the edge parameter
        that will be used with the community detection algorithm.

        Parameter
        ----------
        netCDF4Path : STRING
            A string path to a netCDF4 that contains the data to be
            used for and edge connections. 
        readParm : STRING
            The name of the parameter to use in the read netCDF4.
        edgeParaOpt : DICTIONARY
            A dictionary that might contain entries defining additional
            operations to take place after loading a netCDF4 to use for
            the edge parameter. Options currently include 'flipud' and
            'fliplr'. 
        resample : BOOLEAN
            An option to resample the edge parameter grid. The default
            is False.
        self.resolution : FLOAT
            Resolution, in degree, for the in Basins pbject analysis. 

        """
        # Get bathymetry resolution
        

        # Use GMT to resampled the input grid at the basin object's resolution
        # Note that this resampling should produce a cell-registered netCDF4
        if resample == True:
            os.system("gmt grdsample {0} -Rd -rp{2}d -G{1} -Vq".format(netCDF4Path,
                                                                    netCDF4Path.replace(".nc", "_resampled.nc"),
                                                                    np.diff(self.lon)[0][0]))
        else:
            os.system('cp {0} {1}'.format(netCDF4Path, netCDF4Path.replace(".nc", "_resampled.nc")));
        
        try: 
            # Read in the resampled netCDF4
            
            ncfile = Dataset(netCDF4Path.replace(".nc", "_resampled.nc"), mode='r')
            
            # Set the edge parameter
            self.edgeParm = ncfile[readParm][:].data;

            # Try other potential options 
            try:
                if edgeParaOpt['flipud']:
                    self.edgeParm = np.flipud(self.edgeParm);
            except:
                pass
            try:
                if edgeParaOpt['fliplr']:
                    self.edgeParm = np.fliplr(self.edgeParm);
            except:
                pass

            
            # Close netCDF4
            ncfile.close()
            
        except:
            print('readParm might not be a valid parameter in your input netCDF4.')


    def defineBasinsNonUnique(self, minBasinCnt = 3,
                     method = "Louvain",
                     reducedRes={"on":False,"factor":15},
                     edgeParaOpt={"Parameter":"Bathymetry", "readParm":None},
                     read=False,
                     write=False,
                     verbose=True):
        """
        defineBasins method will define basins with network analysis
        using either the Girvan-Newman or Louvain algorithm to define
        communities.

        Parameter
        ----------
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.
        method : STRING
            Determines the implemented community detection algorithm.
            The options are either Girvan-Newman or Louvain. The former
            is more robust with low scalability and the latter is practical
            but produces non-deterministic communities. The default is
            Louvain.
        reducedRes : DICTIONARY
            Option to reduce the resolution of the basin definition
            network calculation. Note that this should be turned
            off when doing analysis, and only kept on for testing
            purposes. The default is {"on":False,"factor":15}.
        edgeParaOpt : DICTIONARY
            Dictionary to hold options to describe node edge parameters.
            The input can either be define bathymetry or the path to 
            a netCDF4 file to use for connections. The default is
            {"Parameter":"Bathymetry", "readParm":None}.
            FIXME: Update based on implementation.
        read : BOOLEAN
            An option to read basin definitions for a given
            bathymetry model. The default is False. 
        write : BOOLEAN
            An option to write basin definitions for a given
            bathymetry model. The default is False. 
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.basinCnt : INT
            Number of basins in global bathymetry model.
        self.basinDis : NUMPY ARRAY
            binCnt x basinCnt array of bathymetry distributions.
        
            
        FIXME: NEED TO MAKE SURE reduceRes and resolution are interacting
        properly and not only for 1 degree resolution input bathymetry
        models.

        """
        ################################
        ### Read/Load edge Parameter ###
        ################################
        if edgeParaOpt['Parameter'] != "Bathymetry":
            self.setEdgeParameter(netCDF4Path=edgeParaOpt['Parameter'],
                                  readParm=edgeParaOpt['readParm'],
                                  edgeParaOpt=edgeParaOpt)


        ##########################
        ### Write/Load network ###
        ##########################
        if read:
            ####################
            ### Read network ###
            ####################
            self.G = nx.read_gml("{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), destringizer=float);
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara= self.edgeParm.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm[::self.reducedRes].T[::self.reducedRes].T.flatten();
            # Define resolution
            self.resolution = self.reducedRes*np.diff(self.lon)[0][0];
        
        else:
            ######################
            ### Create network ###
            ######################

            # Define counter and point dictionary
            cnt = 0.
            points = {};

            # Define resolution
            resolution = 1;
            
            # Only reduce resolution if option is set.
            if not reducedRes['on']:
                self.reducedRes = np.diff(self.lon)[0][0];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                areaWeightsf = self.areaWeights.flatten();
                bathymetryf = self.bathymetry.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm.flatten();
            else:
                self.reducedRes = reducedRes['factor'];
                self.latf = self.lat[::self.reducedRes].T[::self.reducedRes].T.flatten();
                self.lonf = self.lon[::self.reducedRes].T[::self.reducedRes].T.flatten();
                bathymetryf  = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();
                areaWeightsf = self.areaWeights[::self.reducedRes].T[::self.reducedRes].T.flatten();
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgePara = self.edgeParm[::self.reducedRes].T[::self.reducedRes].T.flatten();
            # Distance, in degrees, to a diagonal node.
            cornerDis = self.reducedRes/np.sin(np.pi/4);

            # Create dictionary and array of bathymetry points
            pos = np.zeros( (2, len(~np.isnan(bathymetryf))) );
            for i in tqdm( range(len(self.lonf)) ):
                bathymetryi = bathymetryf[i];
                areaWeighti = areaWeightsf[i];
                if edgeParaOpt['Parameter']!="Bathymetry":
                    edgeParai = edgePara[i];
                else:
                    # Use bathymetry as edge parameter
                    edgeParai = bathymetryi;
                if (~np.isnan(bathymetryi)) & (~np.isnan(edgeParai)):
                    points[int(cnt)] = (self.latf[i], self.lonf[i], bathymetryi, areaWeighti, edgeParai);    # (latitude, longitude, depth, areaWeight) w/ units (deg, deg, m, m2)
                    pos[:,int(cnt)] = np.array( [self.latf[i], self.lonf[i]] ); 
                    cnt+=1;

            # Create a graph
            G = nx.Graph()

            # Add nodes (points)
            for node, values in points.items():
                G.add_node(node, pos=values[0:2], depth=values[2], areaWeightm2=values[3], nodeAttribute1=values[4]);

            # Update to the above code block which Adds edges with weights based on geographic distance.
            # This code is significantly faster than the above code block

            ## Set bathymetry vector corresponding to node 
            nodebathymetryf = ~np.isnan(bathymetryf);
            nodeCntList = np.arange(0,len(pos[0,:]),1);

            ## Iterate through each node
            for node1, values1 in tqdm(points.items()):
                # Set coordinates of current iterated node
                coords1 = values1[0:2];

                # Find values surrounding coords1 (current iterated node)
                #coordsPotential = {}; cnt=0;
                indices = [];
                indicesAddPt = [];
                # Set above and below coordinate 
                if not (coords1[0] == np.max(self.latf)):
                    # Not North pole node
                    ## upper node
                    condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(1)
                if not (coords1[0] == np.min(self.latf)):
                    # Not South pole node
                    ## lower node
                    condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(2)

                # If on periodic boundary
                if coords1[1] == np.min(self.lonf):
                    # nodes on left (min) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(3)
                        indicesAddPt.append(3)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(4)
                        indicesAddPt.append(4)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], np.max(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(5)
                    indicesAddPt.append(5)

                elif coords1[1] == np.max(self.lonf):
                    # nodes on right (max) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(6)
                        indicesAddPt.append(6)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(7)
                        indicesAddPt.append(7)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], np.min(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(8)
                    indicesAddPt.append(8)

                else:
                    # Nodes not on boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(9)
                        indicesAddPt.append(9)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]+self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-self.reducedRes, coords1[1]-self.reducedRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(10)
                        indicesAddPt.append(10)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-self.reducedRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(11)
                    indicesAddPt.append(11)
                    
                for idx in indices:
                    if not (idx.size == 0):
                        # Set node, coordinates, and bathymetry value of current iterated edge node.
                        node2 = idx[0];
                        coords2 = G.nodes[node2]['pos'];
                        values2 = G.nodes[node2]['nodeAttribute1'];
                        
                        
                        # Calculate geographic distance between points using geodesic distance.
                        bathyAve = (values1[4]+values2)/2;
                        

                        #hexsidelengnth = octPolylineLength(coords1, coords2, verbose=False);

                        # Note that this weight contains node spacing information
                        # (i.e., change in node density with latitude and increased \
                        # strength in with high latitude... )
                        distanceV = haversine_distance(coords1[0], coords1[1],
                                                       coords2[0], coords2[1],
                                                       1);
                        nodeSpacingNormalizer = 1/distanceV;
                        print(coords1[0], coords1[1],
                              coords2[0], coords2[1])
                        

                        G.add_edge(node1, node2, bathyAve=bathyAve*nodeSpacingNormalizer);
                        #G.add_edge(node1, node2, bathyAve=bathyAve);
            



            # Set some class parameters for testing purposes.
            self.G = G;

            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(method = method, minBasinCnt = minBasinCnt);

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
            if write:
                nx.write_gml(G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
            

            ################
            ### Plotting ###
            ################
            if verbose:
                # Plot the network on a geographic map
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

                # Draw the nodes (points) on the map
                for node, data in G.nodes(data=True):
                    ax.plot(data['pos'][1], data['pos'][0], 'bo', markersize=1)  # longitude, latitude

                # Draw the edges (connections)
                for edge in G.edges(data=True):
                    node1, node2, weight = edge
                    lon1, lat1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
                    lon2, lat2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                    ax.plot([lon1, lon2], [lat1, lat2], 'k-', transform=ccrs.PlateCarree())

                # Add coastlines and gridlines
                if "Earth" in self.filename:
                    ax.coastlines()
                ax.gridlines()

                plt.title("Geographic Network of points")
                plt.show()

    def findCommunities(self, method = "Louvain", minBasinCnt=1):
        """
        findCommunities uses the Girvan-Newman or Louvain community
        detection algorithm to determine communities of nodes (basins).
        Then nodes of similar basins are given a basinID.

        
        Parameter
        ----------
        method : STRING
            Determines the implemented community detection algorithm.
            The options are either Girvan-Newman or Louvain. The former
            is more robust with low scalability and the latter is practical
            but produces non-deterministic communities. The default is
            Louvain.
        minBasinCnt : INT
            The minimum amount of basins the user chooses to define
            for the given bathymetry model input.

        Return
        ----------
        None.        
        """

        if method=="Girvan-Newman":
            # Run Girvan-Newman algorithm
            self.communities = list(nx.community.girvan_newman(self.G));

            # Choose interation of the algorithm that has at least
            # minBasinCnt basins.
            interation = 0;
            while len(self.communities[interation]) < minBasinCnt:
                interation+=1;
            if interation > 0:
                interation-1;
            self.communitiesFinal = self.communities[interation];

        else:
            self.communitiesFinal = nx.community.louvain_communities(self.G, weight='bathyAve', resolution=1, threshold=1e-12, seed=1)
        # Set node attribute (basinIDs)

        ## Set the amount of unique basinIDs equal to the count
        ## of unique communities that the Girvan-Newman or Louvain
        ## algorithm found. This is not always equal to the minBasinCnt 
        basinIDTags = np.arange(len(self.communitiesFinal));

        basinIDs = {}; cnt = 0;
        for community in self.communitiesFinal:
            basinIDi = float(basinIDTags[cnt]);
            for nodeID in community:
                basinIDs[nodeID] = {"basinID": basinIDi};
            cnt+=1;
        nx.set_node_attributes(self.G, basinIDs, "basinID");

    def createCommunityNodeColors(self, verbose=False):
        """
        createCommunityNodeColors method sets colors associated
        with different community nodes (e.g., basins in this case).


        Returns
        ----------
        node_colors : PYTHON LIST
            A list of hex code colors that correspond to a node's
            community.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.
        """
        # Get basin IDs from network object.
        tmpValues = nx.get_node_attributes(self.G, "basinID");

        # Record the count of basins
        basinIDi = 0
        for i in range(len(nx.get_node_attributes(self.G, "basinID"))):
            if int(tmpValues[i]["basinID"]) > basinIDi:
                basinIDi = int(tmpValues[i]["basinID"]);
        if verbose:
            print("There are {0:0.0f} basinIDs".format(basinIDi+1))

        # Define colors for basin identification.
        if basinIDi+1 <= 40:
            colors = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20b"].colors];
            colors2 = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20c"].colors]
            for i in range(len(colors2)):
                colors.append(colors2[i])
        else:
            # Resample a spectral colormap to the exact size of the basin count.
            colormap = mpl.colormaps["Spectral"];
            colormap = colormap.resampled(basinIDi+1);
            # Convert resampled spectral colormap to hexidecimal codes.
            colors = [mpl.colors.to_hex(i) for i in colormap(np.linspace(0, 1, basinIDi+1))[:, 0:3] ];

        node_colors = [];

        # Iterate through all bathymetry nodes.
        for i in range(len(nx.get_node_attributes(self.G, "basinID"))):
            basinIDi = int(tmpValues[i]["basinID"]);
            node_colors.append( colors[basinIDi] )

        return node_colors

    def visualizeCommunities(self,
                              cmapOpts={"cmap":"viridis",
                                        "cbar-title":"cbar-title",
                                        "cbar-range":[0,1]},
                              pltOpts={"valueType": "Bathymetry",
                                       "valueUnits": "m",
                                       "plotTitle":"",
                                       "plotZeroContour":False,
                                       "nodesize":5,
                                       "connectorlinewidth":1,
                                       "projection":"Mollweide"},
                              draw={"nodes":True,
                                    "connectors":True,
                                    "bathymetry":True,
                                    "coastlines":True,
                                    "gridlines":True},
                              saveSVG=False,
                              savePNG=False):
        """
        visualizeCommunities method creates a global map of nodes
        and connectings. The nodes communities are defined with
        different colors.


        Parameters
        ----------
        cmapOpts : DICTIONARY
            A set of options to format the color map and bar for the plot
        pltOpts : DICTIONARY
            A set of options to format the plot
        pltOpts : DICTIONARY
            A set of options to display different aspects of a basin.
        saveSVG : BOOLEAN
            An option to save an SVG output. The default is False.
        savePNG : BOOLEAN
            An option to save an PNG output. The default is False.
            
        Returns
        ----------
        None.        
        """
        # Make vector cooresponding to nodes that assigns each entry similar hex
        # color codes for similar community of nodes.
        node_colors = self.createCommunityNodeColors()
        
        # Plot the network on a geographic map

        ## Make figure
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(1, 1, height_ratios=[1]);  # 1 rows, 1 column, 

        ## Set projection type
        if pltOpts['projection'] == "Mollweide":
            projectionType = ccrs.Mollweide();
        elif pltOpts['projection'] == "Miller":
            projectionType = ccrs.Miller();
        elif pltOpts['projection'] == "Robinson":
            projectionType = ccrs.Robinson();
        elif pltOpts['projection'] == "Mercator":
            projectionType = ccrs.Mercator();

        ## Add axis
        ax = fig.add_subplot(gs[0], transform=ccrs.PlateCarree(), projection=projectionType);
        ax.axis('equal')

        ## Define local bathymetry variable (it will be mofified)
        bathymetry = self.bathymetry;


        ## Add bathymetry
        if draw['bathymetry']:
            mesh = ax.pcolormesh(self.lon, self.lat, bathymetry, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1])


        ## Add coastlines and gridlines
        ## Use 0 m contour line
        ## Set any np.nan values to 0.
        if draw['coastlines']:
            bathymetry[np.isnan(bathymetry)] = 0;
            zeroContour = ax.contour(self.lon, self.lat, bathymetry, levels=[0], colors='black', transform=ccrs.PlateCarree())        
            

        ## Draw the edges (connections)
        if draw['connectors']:
            for edge in self.G.edges(data=True):
                node1, node2, weight = edge
                
                lon1, lat1 = self.G.nodes[node1]['pos'][1], self.G.nodes[node1]['pos'][0]
                lon2, lat2 = self.G.nodes[node2]['pos'][1], self.G.nodes[node2]['pos'][0]
                
                minmaxlon = [np.min(self.lonf), np.max(self.lonf)]
                
                value1 = ~( (minmaxlon[0]==lon1) and (minmaxlon[1]==lon2) );
                value2 = ~( (minmaxlon[1]==lon1) and (minmaxlon[0]==lon2) );
                if value1 & value2:
                    # Edge does not cross periodic boundary (this is done for visualization
                    # purposes only). With this condition, there are no values edges connected
                    # across the entire planetary surface (i.e., because they will not pass
                    # through the periodic boundary).
                    lines = ax.plot([lon1, lon2], [lat1, lat2], '-k', linewidth=pltOpts['connectorlinewidth'], transform=ccrs.PlateCarree())
                    for line in lines:
                        line.set_zorder(10); 

        ## Draw the nodes (points) on the map
        
        ### Format
        nodePosDict = self.G.nodes.data('pos');
        nodeBasinID = self.G.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
        

        ### Plot
        #### Define the color map and levels used for plotting basinIDs
        basincmap = mpl.colormaps["plasma"].resampled(len(np.unique(BasinID)));
        levels = np.arange(-.5, len(np.unique(BasinID)));

        #### Plot contour of scatter plot based on user inputs
        if draw['nodes']:
            if draw['connectors']:
                nodeplthandle1 = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, vmin=np.min(levels), vmax=np.max(levels), edgecolor='k', linewidths=pltOpts['connectorlinewidth'], s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            else:
                nodeplthandle1 = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, vmin=np.min(levels), vmax=np.max(levels), edgecolor='k', linewidths=pltOpts['connectorlinewidth']/4, s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            nodeplthandle1.set_zorder(11);
        elif draw['nodes-contour']:
            # If bathymetry is plotted with contours then make alpha value of
            # the contours lower (makes more transparent)
            if draw['bathymetry']:
                alpha = 0;
            else:
                alpha = 1;


            # Define arrays of latitude, longitude and basinIDs for plotting contours
            latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
            lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T
            BasinIDA = np.empty(np.shape(lonA));
            BasinIDA[:] = np.nan;
            for nodei in range(len(pos[:,1])):
                BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];
            
            cmap = mpl.colormaps["plasma"].resampled(len(np.unique(BasinID)))
            levels = np.arange(-.5, len(np.unique(BasinID)));
            nodeplthandle1 = ax.contourf(lonA, latA, BasinIDA, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, alpha=alpha)  # longitude, latitude
            nodeplthandle2 = ax.contour(lonA, latA, BasinIDA, levels=levels, colors='r', linewidths=.5, transform=ccrs.PlateCarree())
            nodeplthandle1.set_zorder(11);
            nodeplthandle2.set_zorder(12);
            # Plot basinID labels
            for BasinIDi in np.unique(BasinID):
                # If basin is on a periodic boundary when plot basinID at mean latitudes
                # and longitudes on the side of the periodic boundary which contains
                # the most amount of bathymetry points (not area)
                if (np.max(lonA[BasinIDA==BasinIDi]) == np.max(lonA)) & (np.min(lonA[BasinIDA==BasinIDi]) == np.min(lonA)):
                    # Basin crosses periodic boundary.
                    if len(lonA[(BasinIDA==BasinIDi) & (lonA<0)]) > len(lonA[(BasinIDA==BasinIDi) & (lonA>0)]):
                        # Left side of prime meridian has more points.
                        xloc = np.nanmean(lonA[(BasinIDA==BasinIDi) & (lonA<0)]);
                        yloc = np.nanmean(latA[(BasinIDA==BasinIDi) & (lonA<0)]);
                    else:
                        # Eight side of prime meridian has more, or equal, points.
                        xloc = np.nanmean(lonA[(BasinIDA==BasinIDi) & (lonA>0)]);
                        yloc = np.nanmean(latA[(BasinIDA==BasinIDi) & (lonA>0)]);
                else:
                    # Basin does not cross periodic boundary.
                    xloc = np.nanmean(lonA[(BasinIDA==BasinIDi)]);
                    yloc = np.nanmean(latA[(BasinIDA==BasinIDi)]);
                basinIDhandle1 = plt.text( xloc, yloc,  "{:0.0f}".format(BasinIDi), color='r', transform=ccrs.PlateCarree());
                basinIDhandle1.set_zorder(12);
                basinIDhandle2 = plt.plot( xloc, yloc, '.r', transform=ccrs.PlateCarree());
                basinIDhandle2[0].set_zorder(12);
                
                

            
        self.unBasinIS = np.unique(BasinID);

        ## Add a colorbar(s)
        if draw['bathymetry']:
            basinIDcbarpad = .0;
        else:
            basinIDcbarpad = .05;
        
        # Basin IDs
        cbar1 = plt.colorbar(nodeplthandle1, ax=ax, orientation='horizontal', pad=0.0, aspect=40, shrink=0.7, cmap=basincmap);
        cbar1.set_label(label="Basin ID", size=12);
        self.cbar1 = cbar1
        if len(np.unique(BasinID))>10:
            cbar1.ax.tick_params(labelsize=8);  # Adjust the size of colorbar ticks
            
        else:
            cbar1.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks
        
        basinticklabels = [];
        if not (np.size(np.unique(BasinID))==1):
            #cbar1.set_ticks( np.arange(np.max(BasinID)/(2*(np.max(BasinID)+1)), np.max(BasinID), np.max(BasinID)/((np.max(BasinID)+1))) );  # Set custom tick positions
            cbar1.set_ticks( np.arange(0, np.max(BasinID)+1, 1) );  # Set custom tick positions
            for basini in np.arange(0, np.max(BasinID)+1, 1):
                basinticklabels.append( "{:0.0f}".format(basini) );
        else:
            cbar1.set_ticks( [0] );  # Set custom tick positions
            basinticklabels.append( "{:0.0f}".format(np.unique(BasinID)[0]) );
        
        cbar1.set_ticklabels( basinticklabels, rotation=90);  # Custom labels
        

        if draw['bathymetry']:
            # Bathymetry
            cbar2 = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7);
            cbar2.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
            cbar2.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks


        if draw['gridlines']:
            ax.gridlines()

        # Set x and y scale equal
        

        plt.title("Basin Networks ({})".format(self.body))

        # Save figure
        if savePNG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".png")), dpi=600)
        if saveSVG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".svg")))

    def mergeBasins(self, basinID1, basinID2, write=False):
        """
        mergeBasins method will merge basins with basin ID of basinID2
        to basins with ID basinID1. The resulting basin network is then
        rewitten to the original, overriding the original basinID network.
        Note that there is no need to reread the basin network. 

        Parameters
        ----------
        basinID1 : INT
            Basin ID to absorb basinID2.
        basinID2 : INT, or LIST OF INT
            Basin ID(s) to be absorbed by basinID1.
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute.
        """

        ##########################
        ### Merge basins Model ###
        ##########################
        # Get all node basin ids 
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );
        nodes = nx.get_node_attributes(self.G, "basinID");
        for nodeID in nodes:
            nodeBasinID[int(nodeID)] = nodes[float(nodeID)]['basinID'];
        
        # Change basin id(s) basinID2 to basinID1 (not in nodes yet)
        if np.size(basinID2) == 1:
            # A single basinID to convert to basinID1.
            nodeBasinID[nodeBasinID==basinID2] = basinID1;
        else:
            # Multiple (list) of basinIDs to convert to basinID1.
            for basinID2i in basinID2:
                nodeBasinID[nodeBasinID==basinID2i] = basinID1;

        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);
        
        # Set basin id to new merged basin
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");
        
        
        ###########################
        ### Write network Model ###
        ###########################
        if write:
            print("Network has been overwritten.")
            nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
        else: 
            print("Network has not been overwritten. New network only exist within this object instance.")

    def mergeSmallBasins(self, threshold, thresholdMethod, mergeMethod, write=False):
        """
        mergeSmallBasins method will merge basins below some threshold
        surface area with the closest basins above that same threshold.
        The resulting basin network is then rewitten to the original,
        overriding the original basinID network. Note that there is no
        need to reread the basin network. 

        Parameters
        ----------
        threshold : FLOAT
            Basins smaller than this threshold, in [m2] or [%] of total
            seafloor area will be merged with the closest basin above
            this threshold in size.
        thresholdMethod : STRING
            A string that defines the input type for threshold. This should
            either be set to 'm2' or '%'.
        mergeMethod : STRING
            A string that defines the method to merge small basins with larger
            basins. This can either be set to 'centroid' or 'nearBasinEdge'.
            'centroid' will merge the small basin with nearest centroid of a
            large basin. 'nearBasinEdge' will compare the small basin centroid
            with the distance to all nodes. The small basin will then be assigned
            the same basin ID as the closest node part of a large basin. 
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute.
        """


        ##########################
        ### Merge basins Model ###
        ##########################
        # Define area scalar for thresholdMethod
        if thresholdMethod == '%':
            # Set methodScalar to AOC (total ocean surface) since we are
            # comparing with surface area in %.
            methodScalar = self.AOC/100;
        elif thresholdMethod == 'm2':
            # Set methodScalar to 1 since we are comparing with surface
            # area in meters.
            methodScalar = 1;

        # Define the number of defined basins
        BasinIDmax = self.G
        distance = 1e20;

        # Get basin properties from nodes
        nodesID = nx.get_node_attributes(self.G, "basinID");        # Dictionary of nodes and their basinID
        nodePos = nx.get_node_attributes(self.G, "pos");            # Dictionary of nodes and their positions (lat,lon) [deg, deg]
        nodesAWm2 = nx.get_node_attributes(self.G,'areaWeightm2');  # Dictionary of nodes and their area [m2]


        # Set all node basin ids and position in an array
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );       # Array to hold node basinIDs
        nodePosA = np.zeros((len(self.G), 2), dtype=float );        # Array to hold basini node positions (lat,lon) [deg, deg]
        for nodeID in nodesID:
            # Set node basin ID to array entry
            nodeBasinID[int(nodeID)] = nodesID[float(nodeID)]['basinID'];
            # Set node position to array entry
            nodePosA[int(nodeID),:] = nodePos[float(nodeID)][:];

        # Set unique ids
        nodeBasinIDUnique = np.unique(nodeBasinID);

        # Get basin i areas & Calculate basin centroids (latitudes and
        # longitudes weighted by surface area of the node)
        basinArea = np.zeros(len(nodeBasinIDUnique));               # Array to hold basini area [m2]
        LatCentroid = np.zeros(len(nodeBasinIDUnique));             # Array to hold basini centroid latitude [deg]
        LonCentroid = np.zeros(len(nodeBasinIDUnique));             # Array to hold basini centroid longitude [deg]
        for i, basinID, nodePosi in zip(nodesAWm2, nodeBasinID, nodePosA):
            basinArea[int(basinID)]   += nodesAWm2[i];              # Add node i to basinID basinArea area sum [m2].
            LatCentroid[int(basinID)] += nodePosi[0]*nodesAWm2[i]   # Add node i to basinID (basinArea area)*(latitude) sum [deg*m2].
            LonCentroid[int(basinID)] += np.deg2rad(nodePosi[1])*nodesAWm2[i]   # Add node i to basinID (basinArea area)*(longitude) sum [rad*m2]. Note that this radian conversion only works for data on [-180,180], not [0,360]

        ## Divide by the basinAreas to complete the calculation of area weighted centorids
        LatCentroid = LatCentroid/basinArea;                # Latitude basin centroid [deg]
        LonCentroid = np.rad2deg(LonCentroid/basinArea);    # Longitude basin centroid [deg]

        # Calculate the distance between all centroids
        ## Empty symmetric matrix representing distance between centroids.
        basinCentroidDist = np.zeros((len(LonCentroid),len(LatCentroid))); # [radians]
        ## Empty symmetric matrix representing distance between centroids.
        basinMergeAllowed = np.zeros((len(LonCentroid),len(LatCentroid)), dtype='bool'); # [boolean]
        ## Calculate distance between points
        ## Iterate over all points.
        for i in range(len(LatCentroid)):
            ## Iterate over all points again.
            for j in range(len(LatCentroid)):
                ## Only do calculation if point matches are not the same
                ## and are unique.
                if i<j:
                    # Calculate the distance between centroids
                    # Note that the spherical radius can be set to 1 since
                    # we are only concerned with finding the closest basin.
                    # The actual magnitude of the distance is irrelevant.
                    basinCentroidDist[i,j] = haversine_distance(LatCentroid[i],
                                                                LonCentroid[i],
                                                                LatCentroid[j],
                                                                LonCentroid[j],
                                                                radius=1)
                    # Set the symmetric entry
                    basinCentroidDist[j,i] = basinCentroidDist[i,j];
                
                    # Set whether connecting i to j would be connecting it to a
                    # basin larger than the threshold.
                    if basinArea[j]/methodScalar >= threshold:
                        # Receiving basin is larger than threshold
                        basinMergeAllowed[i,j] = True;
                    else:
                        # Receiving basin is smaller than threshold
                        basinMergeAllowed[i,j] = False;

                    if basinArea[i]/methodScalar >= threshold:
                        # Giving basin is larger than threshold
                        basinMergeAllowed[j,i] = True;
                    else:
                        # Giving basin is smaller than threshold
                        basinMergeAllowed[j,i] = False;

                elif i==j:
                    # Set diagonal entries to np.nan. This allows for simplified
                    # code later on.
                    basinCentroidDist[i,j] = np.nan;
                    # Set diagonal entries to False since a basin cannot be connect
                    # to itself.
                    basinMergeAllowed[j,i] = False;


        # Define an array with columns (oldBasinID, newBasinID)
        # that describes how basinID should be changed
        basinIDTrans = np.zeros( (len( basinArea[basinArea/methodScalar < threshold] ), 2), dtype=float );
        basinIDTrans[:]= np.nan
        basinMatchCnt = 0;

        # Iterate over all unique basin IDs
        for i in range(len(nodeBasinIDUnique)):
            # Basin id
            basinIDi = nodeBasinIDUnique[i];
            # If node i's basin is smaller than threshold.
            if (basinArea[i]/methodScalar < threshold):

                if mergeMethod == 'centroid':
                    # Select centroid distances for basin mergers
                    basiniCentroidDist = basinCentroidDist[i,:];

                    # Remove centroid distances that allow merger with
                    # basin below threshold size.
                    basiniCentroidDist[ ~basinMergeAllowed[i,:] ] = np.nan;
                    
                    # Find the basin ID to merge basinIDi into
                    basinIDnew = np.argwhere(basiniCentroidDist == np.nanmin(basiniCentroidDist))[0][0];

                elif mergeMethod == 'nearBasinEdge':
                    print(f"nearbasin merge: {i}, {basinIDi} is getting merged.")
                    # Find distance from centroid to every other node
                    Distance = haversine_distance(LatCentroid[i],
                                                  LonCentroid[i],
                                                  nodePosA[:,0],
                                                  nodePosA[:,1],
                                                  radius=1);

                    # Find the smallest distance to basin that is greater than
                    # threshold size
                    # True if distance is smallest
                    #     logical2 = (Distance == np.nanmin(Distance));
                    # True if nodes are part of basins greater than the threshold
                    #     logical1 basinArea[[nodeBasinID]]/methodScalar >= threshold
                    logical1 = (basinArea[nodeBasinID.astype(int)]/methodScalar >= threshold);
                    logical2 = (Distance[logical1] == np.nanmin(Distance[logical1]));

                    basinIDnew = nodeBasinID[logical1][logical2][0]


                # Add to matches
                basinIDTrans[basinMatchCnt,:] = np.array([basinIDi, basinIDnew]);
                basinMatchCnt+=1;
        

        # Merge all the matched basinIDs
        # Change basin id(s) basinIDTrans[0] to basinIDTrans[1] (not in nodes yet)
        # Change basin id(s) basinID2 to basinID1 (not in nodes yet)
        if np.size(basinIDTrans) == 2:
            # basinIDTrans[0,0] is converted to basinIDTrans[0,1].
            nodeBasinID[nodeBasinID==basinIDTrans[0,0]] = basinIDTrans[0,1];
        else:
            # Multiple (list) of basinIDs to convert to basinIDTrans[0,i].
            for basinIDold, basinIDnew in basinIDTrans:
                nodeBasinID[nodeBasinID==basinIDold] = basinIDnew;
        
        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);
        
        # Set basin id to new merged basin
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");

        ###########################
        ### Write network Model ###
        ###########################
        if write:
            print("Network has been overwritten.")
            nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)
        else: 
            print("Network has not been overwritten. New network only exist within this object instance.")

    def orderBasins(self, order, write=False):
        """
        orderBasins method will reorder the basin IDs in accordance with
        some input list.

        Parameters
        ----------
        order : LIST
            List of basin IDs, where the entry's index corresponds to new
            basinID. The value of the entry corresponds to the current basinID.
            For example, this might look like [2,0,1], meaning basin2->basin0,
            basin0->basin1, and basin1->basin2.
        write : BOOLEAN
            An option to write over the original network. The default is False.
        
        Re(defines)
        ------------
        self.G's basinID node attribute. 

        """
        ##########################
        ### Merge basins Model ###
        ##########################
        # Get all node basin ids 
        nodeBasinID = np.zeros( (len(self.G)), dtype=float );
        nodes = nx.get_node_attributes(self.G, "basinID");
        for nodeID in nodes:
            nodeBasinID[int(nodeID)] = nodes[float(nodeID)]['basinID'];
        
        # Define the total number of basinIDs
        uniqueIDCnt = len(np.unique(nodeBasinID));

        # Change basinID ordering 
        for i in range(len(order)):
            originalID  = order[i];
            newID       = uniqueIDCnt+i;
            # Update basinID
            nodeBasinID[nodeBasinID==originalID] = newID;

        # Reset basin ID indexing
        uniqueIDs = np.unique(nodeBasinID);
        for i in range(len(uniqueIDs)):
            nodeBasinID[nodeBasinID==uniqueIDs[i]] = float(i);

        # Set basin id to new ordered basin IDs
        basinIDs = {};
        for nodeID in self.G:
            basinIDs[int(nodeID)] = {"basinID": nodeBasinID[int(nodeID)]};
            
        nx.set_node_attributes(self.G, basinIDs, "basinID");

    def applyMergeBasinMethods(self, mergerID, mergerPackage, maxBasinCnt=1e5):
        '''
        applyMergeBasinMethods function takes a ExoCcycle basins object a
        mergerID and mergerPackage to merge basins defined by a predetermined
        merger strategy. Some of these have been define by the creators of
        ExoCcycle, but every other strategy must be defined as shown below


        Parameters
        -----------
        mergerID : INT
            An ID that describes the basins merges to take place as described
            in mergerPackage
        mergerPackage : DICTIONARY
            A dictionary that describes the general merger strategy and strategy
            for a given mergerID. It can be constructed as follows. The following
            package first merges basins that represent 0.1% then 0.5% of total
            basin surface area into the closest basin bigger than 0.1% and 0.5%.
            Next, if mergerID=0 then mergers0 will merger basin 0 with 0,8,9,10,
            basin 1 with ..., and basin 2 with ...
            
                mergerPackage = {'mergeSmallBasins': {'on':True,
                                                    'threshold':np.array([.1,.5]),
                                                    'thresholdMethod':'%',
                                                    'mergeMethod':'nearBasinEdge'},
                                'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                                'mergers0':  {'0':[0,8,9,10], '1':[...], '2':[...] },
                                'mergers5':  {'0':[...], '1':[...], '2':[...] },
                                'mergers...':{'0':[...], '1':[...], '2':[...] }
                                }
        maxBasinCnt : INT
            Maximum number of basins to allow in a bathymetry
            model. The default is 1e5.

        Redefine
        -------
        self : EXOCCYCLE OBJECT
            Object that describes the connection of nodes into basins by
            way of a community detection algorithm. Now with merged basins
            as described by mergerID and mergerPackage.                  
        
        '''
        
        # 1. Apply the small basin mergers
        try:
            if mergerPackage['mergeSmallBasins']['on']:
                # Iterate over all the small basin merger thresholds.
                for i in range(len(mergerPackage['mergeSmallBasins']['threshold'])):
                    self.mergeSmallBasins(
                        threshold      = mergerPackage['mergeSmallBasins']['threshold'][i],
                        thresholdMethod= mergerPackage['mergeSmallBasins']['thresholdMethod'],
                        mergeMethod    = mergerPackage['mergeSmallBasins']['mergeMethod']);
        except:
            pass;

        if maxBasinCnt==1e5:
            # Case: Merge basins by id
            # 2. Merge basins by ID
            try:
                # If the mergerID exists within the mergerPackage mergerIDs then proceed
                # with merging
                if (mergerPackage['mergerID']==mergerID).any():
                    for mainBasin in mergerPackage['mergers'+str(mergerID)]:
                        self.mergeBasins(mainBasin,
                                        mergerPackage['mergers'+str(mergerID)][mainBasin],
                                        write=False);
            except:
                pass;
            
            # 3. Rearrange basins (ordering is useful to keep consistently through temporal reconstructions)
            try:
                # Id the mergerID exists within the mergerPackage mergerIDs then proceed.
                if (mergerPackage['mergerID']==mergerID).any():
                    basinOrder = mergerPackage['arrange'+str(mergerID)];
                    self.orderBasins(basinOrder,
                                    write=False);
            except:
                pass;
        else:
            # Case: Merge basins by the sum node edge weights 
            self.calculateBasinParameters(verbose=False);
            
            self.mergeSmallestConnection(maxBasinCnt=maxBasinCnt,
                                         verbose=True);





        # 4. Report results from merging
        try:
            # If the verbose option has been chosen then plot the merged basins
            if mergerPackage['verbose']:
                blues_cm = mpl.colormaps['Blues'].resampled(100)
                self.visualizeCommunities( cmapOpts={"cmap":blues_cm,
                                                    "cbar-title":"cbar-title",
                                                    "cbar-range":[np.nanmin(np.nanmin(self.bathymetry)),
                                                                    np.nanmean(self.bathymetry)+2*np.nanstd(self.bathymetry)]},
                                            pltOpts={"valueType": "Bathymetry",
                                                    "valueUnits": "m",
                                                    "plotTitle":"{}".format(self.body),
                                                    "plotZeroContour":True,
                                                    "nodesize":1,
                                                    "connectorlinewidth":1,
                                                    "projection":"Miller"},
                                            draw={"nodes":False,
                                                "connectors":False,
                                                "bathymetry":False,
                                                "coastlines":True,
                                                "gridlines":False,
                                                "nodes-contour":True},
                                            saveSVG=False,
                                            savePNG=True)
        except:
            pass

    def mergeSmallestConnection(self, maxBasinCnt, verbose=True):
        """
        mergeSmallestConnection merges smallest basins and strongest
        connected basins with one another until maxBasinCnt number
        of basins exist.

        Parameters
        ----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.
        
        """
        # Define the starting number of basins
        #BasinCnt = len(np.unique(self.BasinIDA))-1;

        # Calculate connectivity and connective bathymetry distributions
        # & Calculate the strength between basins self.basinConnectionWeight
        self.calculateBasinConnectivityParameters(verbose=False)

        # Iterate until desired number of basins is obtained
        while self.basinCnt > maxBasinCnt:

            # Find the connection with highest basinConnectionWeight
            # Note that if np.argwhere(self.basinConnectionWeight, np.nanmax(self.basinConnectionWeight))
            # = [4,5]. This means that basin 4 should be mergered into basin 5.
            idx = np.argwhere(self.basinConnectionWeight == np.nanmax(self.basinConnectionWeight))[0]

            
            # Merge basin with 
            self.mergeBasins(idx[1], idx[0], write=False);
            
            
            # (Re)calculate basin and connectivity and connective bathymetry
            # distributions & Calculate the strength between basins
            # self.basinConnectionWeight
            self.calculateBasinParameters(verbose=False);
            self.calculateBasinConnectivityParameters(verbose=False)
            


        #self.calculateBasinConnectivityParameters(verbose=False)
        if verbose:
            print("self.basinConnectionWeight", self.basinConnectionWeight)

    def saveBasinNetwork(self):
        """
        saveBasinNetwork method will overwrite the original basinID network
        file.
        
        Returns
        --------
        None.
        """
        ###########################
        ### Write network Model ###
        ###########################
        nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)


    def calculateBasinIDA(self):

        # Define arrays for latitude, longitude, bathymetry (use the reduced resolution)
        latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
        lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T

        # Define array for basinIDs and corresponding node ids (use the reduced resolution)
        nodePosDict = self.G.nodes.data('pos');
        nodeBasinID = self.G.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
        
        BasinIDA = np.empty(np.shape(lonA));
        BasinIDA[:] = np.nan;
        for nodei in range(len(pos[:,1])):
            BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];

        self.BasinIDA = BasinIDA;

    def calculateBasinParameters(self, binEdges=None, verbose=True):
        """
        calculateBasinParameters method will calculate basin bathymetry
        distributions, area and ocean volume fractions for basin.
        
        Parameters
        ----------
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Redefines
        ----------
        self.BasinIDA : NUMPY ARRAY
            An nx2n array of values corresponding to basin IDs. Location of no
            basins are given a fill value of np.nan. The size of this arry depends
            on input bathymetry models as well as user input self.reducedRes class
            attribute. 
        self.bathymetryAreaDistBasin : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryVolFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin volume, normalized to the volume of all ocean basins
            (excluding the high latitude ocean volume).
        self.bathymetryAreaFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin area, normalized to the total seafloor area (including
            the high latitude area).
        self.bathymetryAreaDist_wHighlatG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the inclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryAreaDistG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.binEdges : NUMPY LIST
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin.
        self.BasinParametersDefined : BOOLEAN
            Set to True to indicate that basin bathymetry parameters have been defined.
            
        Returns
        --------
        None.
        
        
        """
        # Define arrays for latitude, longitude, bathymetry (use the reduced resolution)
        latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
        lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T
        bathymetry = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T

        # Define array for basinIDs and corresponding node ids (use the reduced resolution)
        nodePosDict = self.G.nodes.data('pos');
        nodeBasinID = self.G.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
        
        BasinIDA = np.empty(np.shape(lonA));
        BasinIDA[:] = np.nan;
        for nodei in range(len(pos[:,1])):
            BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];

        # Calculate basin distributions
        bathymetryAreaDistBasin, bathymetryVolFraction, bathymetryAreaFraction, bathymetryAreaFractionG, bathymetryAreaDist_wHighlatG, bathymetryAreaDistG, binEdges = Bathymetry.calculateBathymetryDistributionBasin(bathymetry, latA, lonA, BasinIDA, self.highlatlat, self.areaWeights, binEdges=binEdges, verbose=verbose)
        
        # Define basinID and nodeid array
        self.BasinIDA = BasinIDA;

        # Set basin bathymetry parameters
        self.bathymetryAreaDistBasin = bathymetryAreaDistBasin
        self.bathymetryVolFraction   = bathymetryVolFraction
        self.bathymetryAreaFraction  = bathymetryAreaFraction
        self.bathymetryAreaFractionG = bathymetryAreaFractionG
        self.bathymetryAreaDist_wHighlatG = bathymetryAreaDist_wHighlatG
        self.bathymetryAreaDistG = bathymetryAreaDistG
        self.binEdges                = binEdges

        # Change class attribute to indicate that basin bathymetry
        # parameters have been defined.
        self.BasinParametersDefined = True;

    def calculateBasinConnectivityParameters(self, binEdges=None, disThres=444, verbose=True):
        """
        calculateBasinConnectivityParameters method is used to calculate
        basin connectivity parameters. The parameters are described in
        the returned parameters section.

        Parameters
        ----------
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        disThres : FLOAT, optional
            Value, in km, represents the distance threshold, away from
            a basin boundary, which will be consider part of that basin
            connection. The default value is 444.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Redefined
        --------
        self.basinCnt : INT
            Number of basins in model.
        self.basinConCnt : INT
            (self.basinCnt^2-self.basinCnt)/2. This value corresponds to the number
            of basin connections.
        self.validConCnt : INT
            Number of valid connections between basins (i.e., where self.bathymetryConDist)
            is not a vector of NaNs.
        self.connectingNodes : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of node
            IDs that correspond to a basin connection.
        self.connectiveBathy : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of
            bathymetry, in m, that correspond to a basin connection's nodes.
        self.connectiveAreaW : NUMPY ARRAY
            self.basinConCnt length list of lists. Each list holds the set of area
            weights that correspond to a basin connection's nodes.
        self.basinAreaConnection : NUMPY ARRAY
            A self.basinCnt x self.basinCnt matrix that enumerate the connections
            between basins. This is a useful book keeping attribute. 
        self.basinConnectionWeight : NUMPY ARRAY
            A self.basinCnt x self.basinCnt matrix holds weights describing how
            well connected two basins are. The connection strength is calculated
            based on connective bathymetry on a set of rules described within
            the body of code.
        self.bathymetryConDist : NUMPY ARRAY
            A self.basinCnt x self.basinCnt x (binEdges-1) matrix that holds area
            weighted bathymetry distributions of connective bathymetry between
            basins. Distributions sum to 100%. 
        self.basinConnectionDefined : BOOLEAN
            Set to positive to indicate basin connectivity parameters are calculated.
        """
        ############################################################
        ##### Find the connectivity between each set of basins #####
        ############################################################


        ############################################################
        ############# Find nodes boarding other basins #############
        ############################################################
        # Define the basin and connection count and graph
        self.basinCnt = len(np.unique(self.BasinIDA))-1
        self.basinConCnt = (self.basinCnt*self.basinCnt-self.basinCnt)//2
        BasinNodes = self.G;

        # Define node position array and basin ID vector
        nodePosDict = BasinNodes.nodes.data('pos');
        nodeBasinID = BasinNodes.nodes.data('basinID');
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];

        # Create array of basins IDs
        latA = self.lat[::self.reducedRes].T[::self.reducedRes].T
        lonA = self.lon[::self.reducedRes].T[::self.reducedRes].T
        BasinIDA = np.empty(np.shape(lonA));
        BasinIDA[:] = np.nan;
        for nodei in range(len(pos[:,1])):
            BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];
        
        # Create an empty array to hold positions of basin nodes at basin's edges
        posEdges = np.zeros( (0, 2) );

        # Create an empty array to hold basin edge nodes IDs
        basinEdgeNodeBasinIDs1 = np.array([],dtype=float);
        basinEdgeNodeBasinIDs2 = np.array([],dtype=float);

        # Counter for number of nodes that are edges of basins
        cnt=0;

        # Populate basinEdgeNodeBasinIDs1 and basinEdgeNodeBasinIDs1 with
        # a basin edge node and its reciprocal.
        ## Iterate over all nodes
        for j in range(len(nodePosDict)):
            ## Iterate neighbors of node
            for i in BasinNodes.neighbors(j):
                ## If neighbors are not part of the same basin
                if nodeBasinID[i]['basinID'] != nodeBasinID[j]['basinID']:           
                    ## Add location of basin edge
                    posEdges = np.append(posEdges, [nodePosDict[j]], axis=0)
                    ## Add basin edge node to list                    
                    basinEdgeNodeBasinIDs1 = np.append(basinEdgeNodeBasinIDs1, 
                                                        nodeBasinID[j]['basinID']);
                    basinEdgeNodeBasinIDs2 = np.append(basinEdgeNodeBasinIDs2,
                                                        nodeBasinID[i]['basinID']);
                    ## Update counter
                    cnt+=1;
        
        ############################################################
        ############# Create basin connection ID array #############
        ############################################################
        # Define basinAreaConnection, a matrix which will hold an
        # enumeration of the connections between basins.
        self.basinAreaConnection = np.zeros((self.basinCnt,self.basinCnt), dtype=float);
        # Define self.basinAreaConnectionTracker, which keeps track of the 
        # symmetric basin connection (e.g., Atlantic-Indian for Indian-Atlantic)
        # has already been defined.
        basinAreaConnectionTracker = np.ones((self.basinCnt,self.basinCnt), dtype=bool);
        cnt=0

        # Populate basinAreaConnection with enumerations of basin
        # connections.
        ## Loop over basins
        for basini in range(self.basinCnt):
            ## Loop over basins
            for basinj in range(self.basinCnt):
                ## Loop over other basins
                if  (basini!=basinj) & basinAreaConnectionTracker[basinj,basini]:
                    ## Set basin connection ID
                    self.basinAreaConnection[basini,basinj] = cnt;
                    cnt+=1;
                    ## Indicate that basin connection ID has been set
                    basinAreaConnectionTracker[basini,basinj] = False;
                elif (basini!=basinj):
                    ## If symmetric calculation has already been done
                    ## then set the symmetric array value
                    self.basinAreaConnection[basini,basinj] = self.basinAreaConnection[basinj,basini]
                elif (basini==basinj):
                    self.basinAreaConnection[basini,basinj] = -1;
        
        ############################################################
        ####### Get the bathy/weights/nodeID for all nodes #########
        ############################################################
        allNodeIDs = np.array([], dtype=float);
        allNodeBathymetry = np.array([], dtype=float);
        allNodeAreaWeights = np.array([], dtype=float);
        for node in BasinNodes:
            allNodeIDs = np.append(allNodeIDs, node);
            allNodeBathymetry  = np.append(allNodeBathymetry, BasinNodes.nodes[node]['depth']);
            allNodeAreaWeights = np.append(allNodeAreaWeights, BasinNodes.nodes[node]['areaWeightm2']);
            
        
        ############################################################
        ### Set nodeIDs and bathymetry for each basin connection ###
        ############################################################

        # Define basin connective nodes, bathymetry, and area weights
        #  as list of list. This is done such that each basin connection
        # can have different numbers of nodes to describe them.
        self.connectiveBathy = np.empty(self.basinConCnt,dtype=object)
        self.connectiveAreaW = np.empty(self.basinConCnt,dtype=object)
        self.connectingNodes = np.empty(self.basinConCnt,dtype=object)
        for i in range(len(self.connectingNodes)):
            self.connectiveAreaW[i] = np.array([], dtype = float);
            self.connectiveBathy[i] = np.array([], dtype = float);
            self.connectingNodes[i] = np.array([], dtype = float);

        # Populate connectingNodes, for each basin connection, with
        # node IDs that are within disThres of the basin edges.
        ## Iterate over basin edge nodes
        for i in range(len(basinEdgeNodeBasinIDs1)):
            ## Calculate distance from all nodes to the edge of a
            ## basin edge node, i.
            distanceV = haversine_distance(pos[:,0], pos[:,1],
                                           posEdges[i,0], posEdges[i,1],
                                           self.radius*1e-3);
            ## If the distance is less then a threshold then set
            ## the node as a basin connective node
            logical = (distanceV<=disThres);
            
            ## Find the correct connection
            connectionID = int(basinEdgeNodeBasinIDs1[int(i)])
            connectionID = self.basinAreaConnection[int(basinEdgeNodeBasinIDs1[int(i)]),
                                                    int(basinEdgeNodeBasinIDs2[int(i)])]
            
            ## Add nodeIDs to appropriate basin connection list (using connectionID)
            ## And remove any repeat nodes
            if connectionID != -1:
                self.connectingNodes[int(connectionID)] = np.append(self.connectingNodes[int(connectionID)],
                                                                    allNodeIDs[logical] );
                self.connectingNodes[int(connectionID)] = np.unique(self.connectingNodes[int(connectionID)]);
    
        # Add bathymetry and area weights to appropriate basin
        # connection list (using connectionID).
        ## Loop over unique basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            self.connectiveBathy[int(connectionIDi)] = np.append(self.connectiveBathy[int(connectionIDi)],
                                                                 allNodeBathymetry[self.connectingNodes[int(connectionIDi)].astype('int')] );
            self.connectiveAreaW[int(connectionIDi)] = np.append(self.connectiveAreaW[int(connectionIDi)],
                                                                 allNodeAreaWeights[self.connectingNodes[int(connectionIDi)].astype('int')] );
        
        ############################################################
        # Calculate bathymetry distributions for basin connections #
        ############################################################

        # If binEdges are not defined in method input then define with
        # LOSCAR's, a carbon cycle model, default distribution.
        if binEdges is None:
            binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    
        # Setup array to hold connective bathymetry distributions.
        #self.bathymetryConDist = {};
        self.bathymetryConDist = np.zeros( (self.basinCnt, self.basinCnt, len(binEdges)-1) ,dtype=float)
    
        # Calculate all basin connection bathymetry distributions.
        ## Iterate over basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            # Calculate a basin connection bathymetry distributions.
            bathymetryConDisti, binEdges = np.histogram((1e-3)*self.connectiveBathy[int(connectionIDi)],
                                                             weights = self.connectiveAreaW[int(connectionIDi)],
                                                             bins=binEdges);
            # Add the distribution information to dictionary.
            for symmeticIndex in np.argwhere(self.basinAreaConnection == int(connectionIDi)):
                self.bathymetryConDist[ symmeticIndex[0], symmeticIndex[1] ] = 100*(bathymetryConDisti/np.sum(bathymetryConDisti));
        
        ############################################################
        ############### Calculate connection weights ###############
        ############################################################
        ## Define basinConnectionWeight
        self.basinConnectionWeight = np.zeros( (self.basinCnt, self.basinCnt), dtype=float)

        ## Iterate over basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            
            # Define the connection area (size).
            areaofConnection =  np.sum(self.connectiveAreaW[connectionIDi]);

            for symmeticIndex in np.argwhere(self.basinAreaConnection == int(connectionIDi)):
                ## Iterate over symmetric IDs
                # Define basins' area (size)
                areaofBasin1 =      self.bathymetryAreaFractionG["Basin{}".format(symmeticIndex[0])];
                areaofBasin2 =      self.bathymetryAreaFractionG["Basin{}".format(symmeticIndex[1])];
                
                # Set weight
                self.basinConnectionWeight[ symmeticIndex[0], symmeticIndex[1] ] = (1/areaofBasin1)*np.nansum( areaofConnection*self.bathymetryConDist[ symmeticIndex[0], symmeticIndex[1] ]/100 );


        # Change class variable to indicate that basin connectivity
        # parameters have been defined.
        self.basinConnectionDefined = True;
        
        # Calculate the number of valid connections between basins
        self.validConCnt = 0;
        for i in range(self.basinCnt):
            for j in range(self.basinCnt):
                if (i>j) & (~np.isnan(self.bathymetryConDist[i,j,:])).any():
                    self.validConCnt += 1;

        ############################################################
        ####################### Plot results #######################
        ############################################################
        if verbose:
            # Report the basin connectivity distributions
            # print("Bin edges used:\n", binEdges);
            # print("Bathymetry area distribution including high latitude bathymetry:\n");
            # for connectionIDi in range(self.basinConCnt):
            #     idx = np.argwhere(connectionIDi==self.basinAreaConnection)[0];
            #     print(self.bathymetryConDist[idx[0], idx[1]]);
            
            # Plot the basin IDs, connectivity nodes, and their
            # bathymetry distributions
            self.plotBasinConnections(pos, binEdges);

    def plotBasinConnections(self, pos, binEdges=None,
                             savePNG=False, saveSVG=False, outputDir = os.getcwd(), fidName = "plotBasinConnections.png"):
        """
        plotBasinConnections is used to plot results calculating from
        running the method calculateBasinConnectivityParameters.
        
        Parameters
        -----------
        pos : NUMPY ARRAY
            An nx2 array with columns of latitude and longitude, in degrees. This
            array should represent the locations of basin nodes.
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        savePNG : BOOLEAN
            An option to save an PNG output. The default is False.
        saveSVG : BOOLEAN
            An option to save an SVG output. The default is False.

        Returns
        --------
        None.
        """

        # If binEdges are not defined in method input then define with
        # LOSCAR's, a carbon cycle model, default distribution.
        if binEdges is None:
            binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    

        # Set up the Mollweide projection
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1]);  # 2 rows, 1 column, with both row heights equal.

        ax1 = fig.add_subplot(gs[0], projection=ccrs.Mollweide());

        # Create colormap (basins IDs)
        ## Set colormap
        cmap1 = plt.get_cmap("Pastel1")
        ## Extract basinCnt colors from the colormap
        colors_rgb1 = [cmap1(i) for i in range(self.basinCnt)]
        ## Convert RGB to hex
        colors1 = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb1]
        ## Create a custom colormap from the list of colors
        custom_cmap1 = LinearSegmentedColormap.from_list("custom_pastel", colors1, N=256)

        # Create colormap (Connection IDs)
        cmap2 = plt.get_cmap("Dark2")
        ## Extract basinCnt colors from the colormap
        colors_rgb2 = [cmap2(i) for i in range(self.validConCnt)]
        ## Convert RGB to hex
        colors2 = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb2]
        ## Create a custom colormap from the list of colors
        custom_cmap2 = LinearSegmentedColormap.from_list("custom_pastel", colors2, N=256)


        # Plot basin contourf and coastlines

        ## Add the plot using pcolormesh
        mesh = ax1.pcolormesh(self.lon, self.lat, self.BasinIDA, cmap=custom_cmap1, transform=ccrs.PlateCarree())

        ## Add coastlines
        ### Set any np.nan values to 0.
        bathymetry = self.bathymetry
        bathymetry[np.isnan(bathymetry)] = 0;
        ### Plot coastlines.
        zeroContour = ax1.contour(self.lon, self.lat, bathymetry,levels=[0], colors='black', transform=ccrs.PlateCarree())

        # Plot basin connection contour.

        ## Define global array of connective bathymetry
        BC = np.empty((np.shape(self.lat)));
        BC[:] = np.nan;
        for connectingNodei in range(len(self.connectingNodes)):
            for lat, lon in pos[self.connectingNodes[connectingNodei].astype('int')]:
                BC[(self.lat==lat)&(self.lon==lon)] = connectingNodei;

        ## Plot 
        plt.contourf(self.lon, self.lat, BC,
                     cmap=custom_cmap2,
                     transform=ccrs.PlateCarree());
        

        # Plot gridlines
        ax1.gridlines()
        
        # Plot bathymetry distributions of basin connections.

        ## Set new axis to plot on
        ax2 = fig.add_subplot(gs[1]);

        ## Define factors for plotting
        factor1 = .1
        factor2 = .25
        if self.basinCnt%2:
            factor3 = 0.5;
        else:
            factor3 = 0;

        ## Iteratively plot basin bathymetry distributions
        validConi = 0;
        for i in range(self.basinConCnt):
            # Calculate index of distribution
            idx = np.argwhere(i==self.basinAreaConnection)[0];
            # Check if distribution is valid (i.e., if basins are connected)
            if (~np.isnan(self.bathymetryConDist[idx[0],idx[1]])).any():
                # Distribution is valid; now plot
                plt.bar(x=binEdges[1:]-(factor2/2)*(self.validConCnt/2 - i -factor3)*np.diff(binEdges),
                        height=self.bathymetryConDist[idx[0],idx[1]],
                        width=factor1*np.diff(binEdges),
                        label= "Connection {:0.0f}".format(validConi),
                        color=colors2[validConi])
                validConi += 1

        ## Format ticks
        plt.xticks(binEdges[1:]);
        plt.yticks(np.arange(0,35,5));

        ## Format Labels
        plt.legend();
        plt.title("Basin Connective Bathymetry Distributions")
        plt.xlabel("Bathymetry Bins [km]");
        plt.ylabel("Seafloor Area [%]");

        ## Format figure format
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        

        # Save figure
        if savePNG:
            plt.savefig("{}/{}".format(outputDir,fidName), dpi=600)
        if saveSVG:
            plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))
            
    def saveCcycleParameter(self, verbose=True):
        """
        saveCcycleParameter method create a new netCDF4 containing 
        the original bathymetry model, areaweights, and other global
        bathymetry parameters. Additionally, the new netCDF4 will
        contain a basinID array as well as bathymetry distributions,
        area and ocean water volume fractions for those basins.

        This method produces a netCDF4 that contains two groups
        (CycleParms and Arrays). These groups might look like the
        following for a global ocean system that has 3 basins:

        group /basinConnections:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            basinConnectionBathymetry(BasinID, BasinID, binEdges)


        group /CycleParms:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            float32 binEdges(binEdges),
            float64 Global-whighlat(binEdges),
            float64 Global(binEdges),
            float64 basin-0(binEdges),
            float64 basin-1(binEdges),
            float64 basin-2(binEdges),
            float32 BasinID(BasinID),
            float64 fdvol(BasinID),
            float64 fanoc(BasinID),
            float64 highlatlat(),
            float64 highlatA(),
            float64 AOC()
            float64 VOC(),
            groups: 

        group /Arrays:
            dimensions(sizes): lat(180), lon(360)
            variables(dimensions):
            float32 lat(lat),
            float32 lon(lon),
            float64 bathymetry(lat, lon),
            float64 basinIDArray(lat, lon),
            float64 areaWeights(lat)
            groups: 

        Parameters
        -----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        --------
        None.
        """
        # Set netCDF4 filename
        BathyPath = "{0}/{1}".format(self.dataDir, self.filename.replace(".nc", "_wBasins.nc"));
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4')

        # Report what values will be stored in saved netCDF4
        storedValuesStrV = ["---", "---"]
        if self.BasinParametersDefined:
            storedValuesStrV[0] = "";
        else:
            storedValuesStrV[0] = "not ";
        if self.basinConnectionDefined:
            storedValuesStrV[1] = "";
        else:
            storedValuesStrV[1] = "not ";
        if verbose:
            print("Basin bathymetry parameters are {}being stored in netCDF4 group CycleParms".format(storedValuesStrV[0]));
            print("Basin connectivity bathymetry parameters are {}being stored in netCDF4 group basinConnections".format(storedValuesStrV[1]));

        # Define groups
        ArraysGroup = ncfile.createGroup("Arrays")
        if self.BasinParametersDefined:
            CycleParmsGroup = ncfile.createGroup("CycleParms")
        if self.basinConnectionDefined:
            BasinConnectionsGroup = ncfile.createGroup("basinConnections")

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ArraysGroup.createDimension('lat', len(self.bathymetry[:,0]));     # latitude axis
        lon_dim = ArraysGroup.createDimension('lon', len(self.bathymetry[0,:]));     # longitude axis
        if self.BasinParametersDefined:
            binEdges_dim = CycleParmsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = CycleParmsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        if self.basinConnectionDefined:
            binEdges_dim = BasinConnectionsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = BasinConnectionsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        
        # Define lat/lon with the same names as dimensions to make variables.
        lat = ArraysGroup.createVariable('lat', np.float32, ('lat',));
        lat.units = 'degrees_north'; lat.long_name = 'latitude';
        lon = ArraysGroup.createVariable('lon', np.float32, ('lon',));
        lon.units = 'degrees_east'; lon.long_name = 'longitude';

        # Define a 2D variable to hold the elevation data
        bathy = ArraysGroup.createVariable('bathymetry',np.float64,('lat','lon'))
        bathy.units = 'meters'
        bathy.standard_name = 'bathymetry'

        # Define a 2D variable to hold the basinID information
        basinIDArray = ArraysGroup.createVariable('basinIDArray',np.float64,('lat','lon'))
        basinIDArray.units = 'ID'
        basinIDArray.standard_name = 'BasinID'

        # Define vector as function with longitude dependence
        areaWeights = ArraysGroup.createVariable('areaWeights',np.float64,('lat',))
        areaWeights.units = 'meters sq'
        areaWeights.standard_name = 'areaWeights'

        # Define variables for bathymetry distributions (vectors)
        if self.BasinParametersDefined:
            ## Global
            binEdges = CycleParmsGroup.createVariable('binEdges', np.float32, ('binEdges',));
            binEdges.units = 'km'; binEdges.long_name = 'km depth';

            distribution_whighlat = CycleParmsGroup.createVariable('Global-whighlat', np.float64, ('binEdges',))
            distribution_whighlat.units = 'kernal distribution'
            distribution_whighlat.standard_name = 'Global-whighlat'

            distribution = CycleParmsGroup.createVariable('Global', np.float64, ('binEdges',))
            distribution.units = 'kernal distribution'
            distribution.standard_name = 'Global'

            ## Basins Scale Variables
            ### Basin distribution
            distributionBasins = {};
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)] = CycleParmsGroup.createVariable('basin-{:0.0f}'.format(basinIDi), np.float64, ('binEdges',));
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].units = 'kernal distribution';
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].standard_name = 'basin-{:0.0f}'.format(basinIDi);

            ### BasinID
            BasinID = CycleParmsGroup.createVariable('BasinID', np.float32, ('BasinID',));
            BasinID.units = 'ID'; binEdges.long_name = 'BasinID';

            ### Basin Volume fractions
            fdvol = CycleParmsGroup.createVariable('fdvol', np.float64, ('BasinID',))
            fdvol.units = '%'
            fdvol.standard_name = 'fdvol'
            fdvol.long_name = "fdvol the sum of which is equal to 100% (of VOC now within the high latitude area)"

            ### Basin Area fractions (sum to 100% - highlatA/AOC)
            fanoc = CycleParmsGroup.createVariable('fanoc', np.float64, ('BasinID',))
            fanoc.units = '%'
            fanoc.standard_name = 'fanoc'
            fanoc.long_name = "fanoc the sum of which is equal to 100% - highlatA/AOC"

            # Define single values parameters (e.g., VOC, AOC, high latitude cutoff)
            highlatlat = CycleParmsGroup.createVariable('highlatlat', None)
            highlatlat.units = 'degrees'
            highlatlat.standard_name = 'highlatlat'

            highlatA = CycleParmsGroup.createVariable('highlatA', None)
            highlatA.units = 'meters sq'
            highlatA.standard_name = 'highlatA'

            VOC = CycleParmsGroup.createVariable('VOC', None)
            VOC.units = 'meters cubed'
            VOC.standard_name = 'VOC'

            AOC = CycleParmsGroup.createVariable('AOC', None)
            AOC.units = 'meters sq'
            AOC.standard_name = 'AOC'

        if self.basinConnectionDefined:
            basinConnectionBathymetry = BasinConnectionsGroup.createVariable('basinConnectionBathymetry', np.float32, ('BasinID', 'BasinID', 'binEdges'));
            basinConnectionBathymetry.units = 'km'; basinConnectionBathymetry.long_name = 'km depth';
        
        # Format title
        ncfile.title='{} Bathymetry created from topography resampled at {:0.0f} degrees. NetCDF4 includes carbon cycle bathymetry parameters'.format(self.body, self.resolution)
        
        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        bathy[:] = self.bathymetry;
        basinIDArray[:] = self.BasinIDA;
        areaWeights[:] = self.areaWeights[:,0];


        if self.BasinParametersDefined:
            # Add bathymetry distribution information
            distribution_whighlat[:] = self.bathymetryAreaDist_wHighlatG;
            distribution[:] = self.bathymetryAreaDistG;
            binEdges[:] = self.binEdges[1:];

            # Add basin distribution information
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)][:] = self.bathymetryAreaDistBasin['Basin{:0.0f}'.format(basinIDi)]

            # Add basin area and volume fractions
            fdvolValues = np.zeros(len(self.bathymetryAreaDistBasin));
            fanocValues = np.zeros(len(self.bathymetryAreaDistBasin));
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                fdvolValues[basinIDi] = self.bathymetryVolFraction['Basin{:0.0f}'.format(basinIDi)];
                fanocValues[basinIDi] = self.bathymetryAreaFraction['Basin{:0.0f}'.format(basinIDi)];
            fdvol[:] = fdvolValues;
            fanoc[:] = fanocValues;

            # Add attributes
            highlatlat[:] = self.highlatlat;
            highlatA[:] = self.highlatA;
            VOC[:] = self.VOC;
            AOC[:] = self.AOC;
        
        if self.basinConnectionDefined:
            basinConnectionBathymetry[:] = self.bathymetryConDist;
            

        # Close the netcdf
        ncfile.close();

        # Report contents of the created netCDF4
        if verbose:
            # Open netCDF4
            ncfile = Dataset(BathyPath, mode='r', format='NETCDF4')

            # Report netCDF4 contents
            print("Group\tVariable\t\t\tDimensions\t\t\t\tShape")
            print("--------------------------------------------------------------------------------------")
            for groupi in ncfile.groups:
                print(groupi);
                for variable in ncfile[groupi].variables:
                    if len(variable) != 20: 
                        variablePrint = variable.ljust(25)
                    print("\t"+variablePrint.ljust(25)+
                        "\t"+str(ncfile[groupi][variable].dimensions).ljust(35)+
                        "\t"+str(ncfile[groupi][variable].shape).ljust(35))
            
            # Close netCDF4
            ncfile.close();




class BasinsSynth():
    """
    BasinsSynth is a class meant to construct basins and bathymetry properties
    given a synthetic bathymetry model created with classes within the
    Bathymetry module.
    """

    def __init__(self, dataDir, filename, radius):
        """
        Initialization of BasinsSynth class.

        Parameter
        ----------
        dataDir : STRING
            A directory which you will store local data within. Note that
            this function will write to directories [data_dir]/bathymetries.
        filename : STRING
            Output file name 
        radius : FLOAT
            The radius of the synthetic planet bathymetry, in m.
        
        Define
        ----------
        self.bathymetry : NUMPY ARRAY
        self.areaWeights : NUMPY ARRAY
        self.lat : NUMPY ARRAY
        self.lon : NUMPY ARRAY
        self.radius : FLOAT

        """

        # Read netCDF4 bathymetry file
        self.dataDir = dataDir;
        self.filename = filename;

        # Set Planet radius
        self.radius = radius;

        # Define class attributes to be redefined throughout analysis
        ## Have basin connection been defined.
        self.basinConnectionDefined = False;
        ## Have basin bathymetry parameters been defined.
        self.BasinParametersDefined = False;
            
    def setBathymetryBins(self, bathymetryBins):
        '''
        setBasinCnt is a method used to define the bathymetry
        distibution bin edges.
        
        '''

    def defineBasinParameters(self,
                              BasinCnt = 3,
                              Distribution = None,
                              binEdges = None,
                              AOC = None,
                              VOC = None,
                              fanoc=np.array([.30, .30, .30, .10]),
                              fdvol=np.array([.333,.333,.334]),
                              verbose=True):
        """
        defineBasins method will define basins given a BasinCnt and
        input bathymetry distribution.

        Parameter
        ----------
        BasinCnt : INT
            The amount of basins the user chooses to define for the
            given synthetic bathymetry model. The default is 3.
        Distribution : NUMPY VECTOR
            An n length vector corresponding to a global bathymetry
            distribution, in %. The sum of the vector should be 100%.
        binEdges : NUMPY VECTOR
            AN n+1 numpy list of bin edges, in km, to where bathymetry
            distributions were calculated over. Note that anything deeper
            than the last bin edge should be defined within the last bin.
        AOC : FLOAT
            The total surface area of the seafloor of a bathymetry model [m2].
        VOC : FLOAT
            The total ocean volume of a bathymetry model [m3].
        fanoc : NUMPY VECTOR
            BasinCnt+1 length vector corresponding to the amount of
            surface area covered by each basin [in decimal percent]
            and a high latitude ocean box represented in Earth system
            models (see the Earth system model named LOSCAR for
            interpretation of this box). Note that the sum of the vector
            should equal 1. The default value is np.array([.30,.30,.30, 10]).
        fdvol : NUMPY VECTOR
            BasinCnt length vector corresponding to the amount of
            basin volume [in decimal percent] with respect to total ocaen
            volume. Note that this value does not represent any ocean basin
            volume within the high latitude box region. The default value
            is np.array([.333,.333,.334,]).
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Define
        ----------
        self.bathymetryAreaDistBasin : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryVolFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin volume, normalized to the volume of all ocean basins
            (excluding the high latitude ocean volume).
        self.bathymetryAreaFraction : DICTIONARY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            the precent basin area, normalized to the total seafloor area (including
            the high latitude area).
        self.bathymetryAreaDist_wHighlatG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the inclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryAreaDistG : NUMPY ARRAY
            A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
            a histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.binEdges : NUMPY LIST
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin.
        self.highlatlat : FLOAT
            The latitude at which seafloor area at and above this latitude amounts to 
            200*self.highlatA/self.AOC % of the total seafloor area. This value is set
            to None since it does not have a strict definition in synthetic bathymetry
            models.
        self.VOC : FLOAT
            The total ocean volume of a bathymetry model [m3].
        self.AOC : FLOAT
            The total surface area of the seafloor of a bathymetry model [m2].
        self.highlatA : FLOAT
            The area contained within the high latitude region. This is normally
            a set value in a Earth (planetary) system model. In LOSCAR, this value
            is set to 10%.

        """

        # Define the seafloor area and volume distribution between basins
        # and the high latitude box. [Decimal percent]
        self.bathymetryVolFraction = {};
        self.bathymetryAreaFraction = {};
        for basinIDi in range(BasinCnt):
            self.bathymetryVolFraction['Basin{:0.0f}'.format(basinIDi)]     = fdvol[basinIDi];
            self.bathymetryAreaFraction['Basin{:0.0f}'.format(basinIDi)]    = fanoc[basinIDi];

        # Add bathymetry distribution information
        self.binEdges                      = binEdges; # [0,...,6.5];

        # Set synthetic bathymetry distributions
        ## Global scale
        self.bathymetryAreaDist_wHighlatG  = Distribution;
        self.bathymetryAreaDistG           = Distribution;

        # Basin scale
        self.bathymetryAreaDistBasin = {};
        for basinIDi in range(BasinCnt):
            self.bathymetryAreaDistBasin['Basin{:0.0f}'.format(basinIDi)] = Distribution;

        # Add attributes
        self.highlatlat = None; # Is not defined for synthetic bathymetry models
        self.VOC = VOC
        self.AOC = AOC
        self.highlatA = fanoc[-1]*self.AOC;

        # Change boolean to indicate that basin bathymetry parameters have been defined.
        self.BasinParametersDefined = True;

        

    def defineBasinConnectivityParameters(self,
                                          BasinCnt = 3,
                                          Distribution = None,
                                          verbose=True):
        """
        defineBasinConnectivityParameters is a method used to define
        basin connectivity bathymetry for synthetic bathymetry models.
        Note that this method sets connective bathymetry equal to a 
        common global bathymetry distribution.

        Parameter
        ----------
        BasinCnt : INT
            The amount of basins the user chooses to define for the
            given synthetic bathymetry model. The default is 3.
        Distribution : NUMPY VECTOR
            An n length vector corresponding to a global bathymetry
            distribution, in %. The sum of the vector should be 100%.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Redefined
        --------
        self.bathymetryConDist : NUMPY ARRAY
            A self.basinCnt x self.basinCnt x (binEdges-1) matrix that holds area
            weighted bathymetry distributions of connectivity bathymetry between
            basins. Distributions sum to 100%. 
        """

        # Setup array to hold connective bathymetry distributions.
        self.bathymetryConDist = np.zeros( (self.basinCnt, self.basinCnt, len(binEdges)-1) ,dtype=float)

        # Populate the array with bathymetry distributions

        dsf

        self.basinConnectionDefined = True;
        pass

    def saveCcycleParameter(self, verbose=True):
        """
        saveCcycleParameter method create a new netCDF4 containing 
        the synthetic bathymetry parameters.

        This method produces a netCDF4 that contains two groups
        (CycleParms and Arrays). These groups might look like the
        following for a global ocean system that has 3 basins:

        group /basinConnections:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            basinConnectionBathymetry(BasinID, BasinID, binEdges)


        group /CycleParms:
            dimensions(sizes): binEdges(13), BasinID(3)
            variables(dimensions):
            float32 binEdges(binEdges),
            float64 Global-whighlat(binEdges),
            float64 Global(binEdges),
            float64 basin-0(binEdges),
            float64 basin-1(binEdges),
            float64 basin-2(binEdges),
            float32 BasinID(BasinID),
            float64 fdvol(BasinID),
            float64 fanoc(BasinID),
            float64 highlatlat(),
            float64 highlatA(),
            float64 AOC()
            float64 VOC(),
            groups: 

        group /Arrays:
            dimensions(sizes): lat(180), lon(360)
            variables(dimensions):
            float32 lat(lat),
            float32 lon(lon),
            float64 bathymetry(lat, lon),
            float64 basinIDArray(lat, lon),
            float64 areaWeights(lat)
            groups: 

        Parameters
        -----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        --------
        None.
        """
        # Set netCDF4 filename
        BathyPath = "{0}/{1}".format(self.dataDir, self.filename);
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4')

        # Report what values will be stored in saved netCDF4
        storedValuesStrV = ["---", "---"]
        if self.BasinParametersDefined:
            storedValuesStrV[0] = "";
        else:
            storedValuesStrV[0] = "not ";
        if self.basinConnectionDefined:
            storedValuesStrV[1] = "";
        else:
            storedValuesStrV[1] = "not ";
        if verbose:
            print("Basin bathymetry parameters are {}being stored in netCDF4 group CycleParms".format(storedValuesStrV[0]));
            print("Basin connectivity bathymetry parameters are {}being stored in netCDF4 group basinConnections".format(storedValuesStrV[1]));

        # Define groups
        if self.BasinParametersDefined:
            CycleParmsGroup = ncfile.createGroup("CycleParms")
        if self.basinConnectionDefined:
            BasinConnectionsGroup = ncfile.createGroup("basinConnections")

        # Define dimension (latitude, longitude, and bathymetry distributions)
        if self.BasinParametersDefined:
            binEdges_dim = CycleParmsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = CycleParmsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        if self.basinConnectionDefined:
            binEdges_dim = BasinConnectionsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
            basinID_dim = BasinConnectionsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID

        # Define variables for bathymetry distributions (vectors)
        if self.BasinParametersDefined:
            ## Global
            binEdges = CycleParmsGroup.createVariable('binEdges', np.float32, ('binEdges',));
            binEdges.units = 'km'; binEdges.long_name = 'km depth';

            distribution_whighlat = CycleParmsGroup.createVariable('Global-whighlat', np.float64, ('binEdges',))
            distribution_whighlat.units = 'kernal distribution'
            distribution_whighlat.standard_name = 'Global-whighlat'

            distribution = CycleParmsGroup.createVariable('Global', np.float64, ('binEdges',))
            distribution.units = 'kernal distribution'
            distribution.standard_name = 'Global'

            ## Basins Scale Variables
            ### Basin distribution
            distributionBasins = {};
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)] = CycleParmsGroup.createVariable('basin-{:0.0f}'.format(basinIDi), np.float64, ('binEdges',));
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].units = 'kernal distribution';
                distributionBasins['Basin{:0.0f}'.format(basinIDi)].standard_name = 'basin-{:0.0f}'.format(basinIDi);

            ### BasinID
            BasinID = CycleParmsGroup.createVariable('BasinID', np.float32, ('BasinID',));
            BasinID.units = 'ID'; binEdges.long_name = 'BasinID';

            ### Basin Volume fractions
            fdvol = CycleParmsGroup.createVariable('fdvol', np.float64, ('BasinID',))
            fdvol.units = '%'
            fdvol.standard_name = 'fdvol'
            fdvol.long_name = "fdvol the sum of which is equal to 100% (of VOC now within the high latitude area)"

            ### Basin Area fractions (sum to 100% - highlatA/AOC)
            fanoc = CycleParmsGroup.createVariable('fanoc', np.float64, ('BasinID',))
            fanoc.units = '%'
            fanoc.standard_name = 'fanoc'
            fanoc.long_name = "fanoc the sum of which is equal to 100% - highlatA/AOC"

            # Define single values parameters (e.g., VOC, AOC, high latitude cutoff)
            highlatlat = CycleParmsGroup.createVariable('highlatlat', None)
            highlatlat.units = 'degrees'
            highlatlat.standard_name = 'highlatlat'

            highlatA = CycleParmsGroup.createVariable('highlatA', None)
            highlatA.units = 'meters sq'
            highlatA.standard_name = 'highlatA'

            VOC = CycleParmsGroup.createVariable('VOC', None)
            VOC.units = 'meters cubed'
            VOC.standard_name = 'VOC'

            AOC = CycleParmsGroup.createVariable('AOC', None)
            AOC.units = 'meters sq'
            AOC.standard_name = 'AOC'

        if self.basinConnectionDefined:
            basinConnectionBathymetry = BasinConnectionsGroup.createVariable('basinConnectionBathymetry', np.float32, ('BasinID', 'BasinID', 'binEdges'));
            basinConnectionBathymetry.units = 'km'; basinConnectionBathymetry.long_name = 'km depth';
        
        # Format title
        ncfile.title='Bathymetry created from synthetic models. NetCDF4 includes carbon cycle bathymetry parameters'.format()
        
        # Populate the variables
        if self.BasinParametersDefined:
            # Add bathymetry distribution information
            distribution_whighlat[:] = self.bathymetryAreaDist_wHighlatG;
            distribution[:] = self.bathymetryAreaDistG;
            binEdges[:] = self.binEdges[1:];

            # Add basin distribution information
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                distributionBasins['Basin{:0.0f}'.format(basinIDi)][:] = self.bathymetryAreaDistBasin['Basin{:0.0f}'.format(basinIDi)]

            # Add basin area and volume fractions
            fdvolValues = np.zeros(len(self.bathymetryAreaDistBasin));
            fanocValues = np.zeros(len(self.bathymetryAreaDistBasin));
            for basinIDi in range(len(self.bathymetryAreaDistBasin)):
                fdvolValues[basinIDi] = self.bathymetryVolFraction['Basin{:0.0f}'.format(basinIDi)];
                fanocValues[basinIDi] = self.bathymetryAreaFraction['Basin{:0.0f}'.format(basinIDi)];
            fdvol[:] = fdvolValues;
            fanoc[:] = fanocValues;

            # Add attributes
            highlatlat[:] = self.highlatlat;
            highlatA[:] = self.highlatA;
            VOC[:] = self.VOC;
            AOC[:] = self.AOC;
        
        if self.basinConnectionDefined:
            basinConnectionBathymetry[:] = self.bathymetryConDist;
            

        # Close the netcdf
        ncfile.close();

        # Report contents of the created netCDF4
        if verbose:
            # Open netCDF4
            ncfile = Dataset(BathyPath, mode='r', format='NETCDF4')

            # Report netCDF4 contents
            print("Group\tVariable\t\t\tDimensions\t\t\t\tShape")
            print("--------------------------------------------------------------------------------------")
            for groupi in ncfile.groups:
                print(groupi);
                for variable in ncfile[groupi].variables:
                    if len(variable) != 20: 
                        variablePrint = variable.ljust(25)
                    print("\t"+variablePrint.ljust(25)+
                        "\t"+str(ncfile[groupi][variable].dimensions).ljust(35)+
                        "\t"+str(ncfile[groupi][variable].shape).ljust(35))
            
            # Close netCDF4
            ncfile.close();



def haversine_distance(lat1, lon1, lat2, lon2, radius):
    """
    haversine_distance function calculate the great-circle distance
    between two points on a sphere using the Haversine formula.
    
    Parameters
    ----------
    lat1 : FLOAT
        Coordinate of first point latitude, in degree.
    lon1 : FLOAT
        Coordinate of first point longitude, in degree.
    lat2 : FLOAT or NUMPY VECTOR
        Coordinate of second point latitude, in degree.
    lon2 : FLOAT or NUMPY VECTOR
        Coordinate of second point longitude, in degree.
    radius : FLOAT
        Radius of the sphere.
    
    Returns
    --------
    distance : FLAOT
        Distance between the two points in the same unitas the radius.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    
    # Numerical issues can cause points that are on top of one another to
    # be assigned negative a values. This ensures a is always equal to or
    # greater than 0.
    # if np.size(a) != 1:
    #     a[a<0] = 0;
    # elif a<0:
    #     a = 0;

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance
    distance = radius * c
    return distance

def octPolylineLength(coords1, coords2, resolution=1, verbose=True):
    """
    octPolylineLength function will define the length of a ~hexagonal
    prism side between two coordinate points. Note that on the surface
    of a sphere, a point must be surrounded by a hexagonal "like" volume,
    such that low and adjacent higher latitude hexagonal like shapes
    will have the same surface side length.

    This function finds two points (point1 and point2) that
    represent a polyline interface between between input coordinates
    (coords1, coords2). The legnth between these points, on a surface
    of a sphere, is calculated and returned
    
    Parameter
    ----------
    coords1 : object
        Tuple of latitude and longitude, in degrees. 
    coords2 : INT
        Tuple of latitude and longitude, in degrees.
    resolution : FLOAT
        The resolution of the input coordinates, in degrees. The default
        is 1.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Returns
    --------
    length : FLOAT
        The length of the hexagonal like shape side that is shared between
        input coordinates (coords1 and coords1), as defined on a unit sphere. 
    
    """

    # Define some value(s) of the hexagonal prism
    ## (resolution/2) * (1+2/np.sqrt(2))**(-1)
    w = (resolution/2)*(0.4142135623730951);    # Half edge length of hexagonal prism [deg].
    #a = 2*w;                                    # Edge length of hexagonal prism [deg].
    #b = a*0.7071067811865475;                   # a * sin(np.ppi/4)

    # Find the points to calculate a polyline between.

    ## Take the absolute value of both latitude coordinates
    ## then find the hexagonal like prism surface from the 
    ## lowest latitude point.
    ## This is used for if points are not along the same longitude.
    coord = ( np.min([np.abs(coords1[0]), np.abs(coords2[0])]), np.min([np.abs(coords1[1]), np.abs(coords2[1])]) );
    if coords1[0]==coords2[0]:
        # If points along same latitude line

        # Note that longitude differences are centered
        # around the 0 longitude points. This is okay to
        # do since point locations of longitudes do not
        # change polyline lengths.
        point1 = (coord[0]+w, 0);       # top right corner
        point2 = (coord[0]-w, 0);       # bottom right corner
        if verbose:
            print("same lat points", point1, point2)

    elif coords1[1]==coords2[1]:
        # If points along same longitudinal line 
        point1 = (coord[0]+resolution/2, -w);    # top left corner
        point2 = (coord[0]+resolution/2, w);     # right left corner
        if verbose:
            print("same lon points", point1, point2)
    else:
        # If points along different lat/lon (diagonal edge)
        point1 = (coord[0]+resolution/2,    w);
        point2 = (coord[0]+w,               resolution/2);
        if verbose:
            print("diagonal points", point1, point2)

    # Calculate distance on surface of sphere.
    length = haversine_distance(point1[0], point1[1], point2[0], point2[1], radius=1);
    return length



#######################################################################
######################### Basin merger packages #######################
#######################################################################
def mergerPackages(package = '', verbose=True):
    '''
    mergerPackages returns predefined basin merger methods for some
    seafloor models/reconstructions. Users can create their own as 
    well!

    Parameter
    ----------
    package : STRING
        The name of a basin merger package.


    Return
    -------
    mergerPackage : DICTIONARY
        A dictionary that describes the general merger strategy and strategy
        for a given mergerID. It can be constructed as follows. The following
        package first merges basins that represent 0.1% then 0.5% of total
        basin surface area into the closest basin bigger than 0.1% and 0.5%.
        Next, if mergerID=0 then mergers0 will merger basin 0 with 0,8,9,10,
        basin 1 with ..., and basin 2 with ...
        
            mergerPackage = {'mergeSmallBasins': {'on':True,
                                                'threshold':np.array([.1,.5]),
                                                'thresholdMethod':'%',
                                                'mergeMethod':'nearBasinEdge'},
                            'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                            'mergers0':  {'0':[0,8,9,10], '1':[...], '2':[...] },
                            'mergers5':  {'0':[...], '1':[...], '2':[...] },
                            'mergers...':{'0':[...], '1':[...], '2':[...] }
                            }
    
    '''
    # Define a merger package that does not merge and basins
    mergerPackage = {'mergeSmallBasins': {'on':False,
                                                'threshold':np.array([.1]),
                                                'thresholdMethod':'%',
                                                'mergeMethod':'nearBasinEdge'},
                            'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                            'mergers0':  {'0':[0], '1':[1], '2':[2] },
                            }

    # Define merger package
    if package == 'EarthRecon3Basins_CM2009':
        # Package merges to have basins larger than 0.5% total surface of oceans.
        # Then basins are merged into
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1,.5]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'mergerID': np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]),
                        'mergers0':   {'0':[0,1],            '1':[1,2,4,5],     '2':[2,3,4,5,6]},
                        'mergers5':   {'0':[0,1,2,6],        '1':[1,4,5,6],     '2':[2,3]},
                        'mergers10':  {'0':[0,6,7,9,10],     '1':[1,4],         '2':[2,3,4,5]},
                        'mergers15':  {'0':[0,1],            '1':[1,2,4,6],     '2':[2,3,4,5,6]},
                        'mergers20':  {'0':[0,1,6],          '1':[1,3,4,5],     '2':[2,3,4,5]},
                        'mergers25':  {'0':[0,3,4,9],        '1':[1,2,4],       '2':[2,3,4,5]},
                        'mergers30':  {'0':[0,4,5,6,7],      '1':[1,2,4,5],     '2':[2]},
                        'mergers35':  {'0':[0,2,5,7,8],      '1':[1,4,5],       '2':[2,3]},
                        'mergers40':  {'0':[0,3],            '1':[1,3,4,5,6],   '2':[2,3,4,5]},
                        'mergers45':  {'0':[0,3,9],          '1':[1,3,4,6,7],   '2':[2,3,4]},
                        'mergers50':  {'0':[0,4,6,7,9],      '1':[1,2,4],       '2':[2,3,4]},
                        'mergers55':  {'0':[0,2,5],          '1':[1,3,4,5,6,8], '2':[2,3,4]},
                        'mergers60':  {'0':[0,2,4,6,8],      '1':[1,3,4],       '2':[2,3,4]},
                        'mergers65':  {'0':[0,2,4,5,6,9,10], '1':[1,3,5],       '2':[2,3]},
                        'mergers70':  {'0':[0,2],            '1':[1,6,8],       '2':[2,3,4,5,6]},
                        'mergers75':  {'0':[0,8,10],         '1':[1,2,3],       '2':[2,3,4,5,6]},
                        'mergers80':  {'0':[0,3],            '1':[1,2,5,6,7],   '2':[2,3,4,5]},
                        'arrange0':  [2,0,1],
                        'arrange5':  [1,2,0],
                        'arrange15': [2,0,1],
                        'arrange20': [2,0,1],
                        'arrange25': [2,1,0],
                        'arrange30': [1,2,0],
                        'arrange35': [1,2,0],
                        'arrange40': [2,0,1],
                        'arrange45': [2,0,1],
                        'arrange50': [2,1,0],
                        'arrange55': [2,0,1],
                        'arrange60': [2,1,0],
                        'arrange65': [1,2,0],
                        'arrange70': [1,0,2],
                        'arrange80': [2,0,1],
                        'verbose':True};
        pass

    if package == 'Lite':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1, .5]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'Lite1':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1, .5, 1]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'Lite2':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.05]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'Lite3':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'Lite4':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.03]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'LiteShelf0,3degree':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([0.03, .15]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'Test':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.15]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass
    elif package == 'None':
        # Package only merges to have basins larger than 0.5% total surface of oceans
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([0]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'verbose':True};
        pass

    elif package == 'EarthRecon3BasinsRK2021_H_2,10e-12':
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1,.5]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                        'mergers0':  {'0':[0,2,6], '1':[1,2,3,5,8], '2':[2,3,4,5]},
                        'arrange0': [2,0,1],
                        'verbose':True};
        pass

    elif package == 'EarthRecon3BasinsRK2021_H_4,80e-12':
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1,.5]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                        'mergers0':  {'0':[0,2,6], '1':[1,2,3,5,8], '2':[2,3,4,5]},
                        'arrange0': [2,0,1],
                        'verbose':True};
        pass
        
    elif package == 'EarthRecon3BasinsRK2021_H_8,00e-12':
        mergerPackage = {'mergeSmallBasins': {'on':True,
                                              'threshold':np.array([.1,.5]),
                                              'thresholdMethod':'%',
                                              'mergeMethod':'nearBasinEdge'},
                        'mergerID': np.array([0, 5, 10, 15, 20, 25]),
                        'mergers0':  {'0':[0,8,9,10], '1':[1,2,6], '2':[2,3,4,5] },
                        'verbose':True};

    return mergerPackage

#######################################################################
################ Process Global Ocean Physics Reanalysis ##############
#######################################################################
class GLORYS12V1_QT():
    '''
    The GLORYS12V1 class is used to download GLORYS12V1
    data and format it to be used with the ExoCcycle model.

    '''

    def __init__(self, options = {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1", "year":[1994], "data": "bottomT", "depthAve":[0,100]}):
        """
        Initialization of GLORYS12V1
        
        
        Parameters
        -----------
        options : DICTIONARY
            download : BOOLEAN
            dataDir : STRING
            year : LIST
            data : STRING
                Data to be averaged
                    'bottomT' : sea_water_potential_temperature_at_sea_floor
                    'thetao' : sea_water_potential_temperature [depth dimension]
                    'so' : sea_water_salinity [depth dimension]
            depthAve : LIST
                2 element list, describing the range over which an output
                value should be averaged. Note that averaging is only used
                if the input variable has a depth dimension.
            The default is {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1",
            "year":[1994], "data": "thetao", "depthAve":[0,100]}.
        """

        # Assign options to object
        self.options = options;

        # Assign general name of netCDF file
        self.options["netCDFGeneral"] = "mercatorglorys12v1_gl12_mean_YEARMONTH.nc";
    
        # Define initial attributes
        self.areaWeightsA = None;



    def download(self):
        """
        download method is used to download the GLORYS12V1
        data (netCDFs) and store them in a data directory
        accessed by the ExoCcycle library.
        """
        FIXME


    def averageModels(self):
        """
        averageModels is used to make an averaged netCDF model
        given all netCDFs stored in the data directory.  
        """

        # Create list of netCDFs to average
        self.ListOfNetCDFs = [];
        for i in range(len(self.options['year'])):
            for month in range(12):
                readFile = self.options["netCDFGeneral"].replace("YEAR", str(self.options['year'][i])).replace("MONTH", "{}".format(month+1).zfill(2))
                self.ListOfNetCDFs.append(readFile)
        
        # Iterate through all netCDF4 files, copying only the used variables
        self.ListOfSimpNetCDFs = [];
        for i in range(len(self.ListOfNetCDFs)):
            self.simplifyNetCDF(
                inputPath=self.options["dataDir"]+"/"+self.ListOfNetCDFs[i],
                outputPath=self.options["dataDir"]+"/file{}.nc".format(i),
                variableList=['longitude', 'latitude', self.options['data']]
            )
            self.ListOfSimpNetCDFs.append(self.options["dataDir"]+"/file{}.nc".format(i))

        # Create a gmt command and use to gmt to average all netCDF4s.
        ## Create list of files to add
        SimpNetCDFs = " ".join(self.ListOfSimpNetCDFs)
        ## Create list of adds
        adds=[];
        for i in range(len(self.ListOfSimpNetCDFs)-1):
            adds.append("ADD")
        adds = " ".join(adds);
        ## Define the command
        outputFileName = "{0}_average_{1}_{2}m_QTAveraged.nc".format(self.options["dataDir"]+"/"+self.options['data'], self.options['depthAve'][0], self.options['depthAve'][1])
        GMTcommand = "gmt grdmath {0} {1} {2} DIV = {3}".format(SimpNetCDFs, adds, len(self.ListOfSimpNetCDFs), outputFileName)
        ## Use the command
        os.system(GMTcommand)
        ## Apply a mask to the averaged grid (FIXME: No longer need)
        #os.system("gmt grdmath {0} {1} OR = {1}".format(self.options["dataDir"]+"/"+self.ListOfSimpNetCDFs[0], outputFileName)
        
    def makeAreaWeightGrid(self, step, latRange, lonRange):
        areaWeightsA, longitudes, latitudes, totalArea, totalAreaCalculated = areaWeights(resolution = step,
                                                                                          radius = 1,
                                                                                          LonStEd = [lonRange[0]-step/2,lonRange[1]+step/2],
                                                                                          LatStEd = [latRange[0]-step/2,latRange[1]+step/2])
        self.areaWeights = areaWeightsA
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.totalArea = totalArea
        self.totalAreaCalculated = totalAreaCalculated

    def simplifyNetCDF(self, inputPath="path/file.nc", outputPath="~/file2.nc",
    variableList=['longitude', 'latitude', 'variable']):
        """
        simplifyNetCDF method reads a NetCDF4 file and writes a new NetCDF4 file
        with only lat, lon, and 'variable' variables.

        Parameters
        -----------
        input_path : STRING
            Path to the input NetCDF4 file.
        output_path : STRING
            Path to save the new NetCDF4 file.
        variableList : LIST OF STRINGS
            3 length list of strings that correspond to an
            x (longitude), y (latitude), and z (e.g., bathymetry)
            variable. The default is ['latitude', 'logitude',
            'variable'].
        """
        from sklearn.preprocessing import QuantileTransformer
        
        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(outputPath)

        # Open the original NetCDF file (file1.nc) in read mode
        with Dataset(inputPath, 'r') as src:
            # Define area weights if not already defined
            if self.areaWeightsA is None:
                # Defines: self.areaWeights, self.longitudes, self.latitudes, self.totalArea, self.totalAreaCalculated
                self.makeAreaWeightGrid(step      = src['latitude'].step,
                                        latRange  = [src['latitude'].valid_min, src['latitude'].valid_max],
                                        lonRange  = [src['longitude'].valid_min, src['longitude'].valid_max])

            # Create a new NetCDF file (file2.nc) in write mode
            with Dataset(outputPath, 'w', format="NETCDF4_CLASSIC") as dst:
                # Copy global attributes
                if "title" in src.ncattrs():
                    dst.title = src.title  # Preserve title attribute

                # Copy lat & lon dimensions
                for dim_name in [variableList[0], variableList[1]]:
                    if dim_name in src.dimensions:
                        dst.createDimension(dim_name, len(src.dimensions[dim_name]))

                # Copy lat & lon variables
                # for var_name in [variableList[0], variableList[1]]:
                #     if var_name in src.variables:
                #         var = src.variables[var_name]
                #         dst_var = dst.createVariable(var_name, np.float32, var.dimensions)
                #         dst_var[:] = var[:]  # Copy data
                #         # Copy attributes
                #         for attr in var.ncattrs():
                #             try:
                #                 dst_var.setncatts({attr: var.getncattr(attr)})
                #             except:
                #                 pass
                if variableList[0] in src.variables:
                    lat = src.variables[variableList[0]]
                    lat_dst_var = dst.createVariable(variableList[0], np.float32, lat.dimensions)
                    lat_dst_var[:] = lat[:]  # Copy data
                    # Copy attributes
                    # for attr in lat.ncattrs():
                    #     try:
                    #         lat_dst_var.setncatts({attr: lat.getncattr(attr)})
                    #     except:
                    #         pass
                if variableList[1] in src.variables:
                    lon = src.variables[variableList[1]]
                    lon_dst_var = dst.createVariable(variableList[1], np.float32, lon.dimensions)
                    lon_dst_var[:] = lon[:]  # Copy data
                    # Copy attributes
                    # for attr in lon.ncattrs():
                    #     try:
                    #         lon_dst_var.setncatts({attr: lon.getncattr(attr)})
                    #     except:
                    #         pass


                # Copy variable and rename it to 'z'
                if variableList[2] in src.variables:
                    z_var = src.variables[variableList[2]]
                    dst_z = dst.createVariable("z", np.float32, (z_var.dimensions[-2],z_var.dimensions[-1]))
                    if len(z_var.dimensions) == 3:
                        # Copy data
                        dst_z[:] = z_var[:]
                    else:
                        # Define depth variable and logical defining
                        # depth range to average over
                        depth = src.variables['depth'][:];
                        layerThickness = np.diff( np.append(0, depth) )
                        depthLogical = (depth>self.options['depthAve'][0])&(depth<self.options['depthAve'][1])

                        # Define array to hold sum of layered values
                        # (Assumes top/first layer contains all possible values as non-nan)
                        LayerIdx = np.argwhere(depthLogical.data).T
                        if np.size(LayerIdx) > 1:
                            LayerIdx = LayerIdx[0]
                            topLayerIdx = np.argwhere(depthLogical.data)[0][0];
                        else:
                            topLayerIdx = np.argwhere(depthLogical.data)[0];
                        sum = z_var[:][0][topLayerIdx].data*0
                        intervals = z_var[:][0][topLayerIdx].data*0
                        layerThickness = layerThickness[depthLogical]
                        dataLayers = z_var[:][0][depthLogical]

                        # Loop over depth intervals
                        for i in range(len(layerThickness)):
                            # Assign working layer
                            dataLayer = dataLayers[i]
                            # print(dataLayer)
                            # print(dataLayer.data)
                            # print(dataLayer.mask)
                            # print( np.reshape(dataLayer.data[~dataLayer.mask], (np.size(dataLayer.data[~dataLayer.mask]), 1)) )
                            # print("i, layerThickness", i, layerThickness)

                            # Check if field is empty. This might be the case for the deepest layer
                            if np.size(dataLayer.data[~dataLayer.mask]) == 0:
                                continue

                            # Apply an area weighted quantile transformation to layer
                            qt = QuantileTransformer(n_quantiles=1000,
                                                     random_state=0,
                                                     output_distribution='normal')

                            qt.fit_transform( np.reshape(dataLayer.data[~dataLayer.mask], (np.size(dataLayer.data[~dataLayer.mask]), 1)) )    
                            dataLayerTransformed = qt.transform(np.reshape( dataLayer.data, (np.size(dataLayer.data),1) ) )
                            dataLayerTransformed = np.reshape( dataLayerTransformed, np.shape(dataLayer.data) )

                            # Add quantile transformed data to running sum
                            sum[~dataLayer.mask]       += dataLayerTransformed[ ~dataLayer.mask ]*layerThickness[i];
                            intervals[~dataLayer.mask] += layerThickness[i]

                        # Copy top layer and replace .data with averaged values 
                        # average = cp.deepcopy( z_var[:][0][topLayerIdx] )
                        # average.data[:] = (sum/intervals)
                        # dst_z[:] = average.data
                        dst_z[:] = sum/intervals

                    # Copy attributes
                    # for attr in z_var.ncattrs():
                    #     if attr != "_FillValue":
                    #         try:
                    #             #print("attr",attr)
                    #             dst_z.setncatts({attr: z_var.getncattr(attr)})
                    #         except:
                    #             pass


    def readnetCDF(self, year, month):
        # Define the file name for year and month
        
        readFile = self.options["dataDir"]+"/"+self.options["netCDFGeneral"].replace("YEAR", str(year)).replace("MONTH", "{}".format(month).zfill(2))

        # Read the netCDF file
        return Dataset(readFile, "r");



class GLORYS12V1():
    '''
    The GLORYS12V1 class is used to download GLORYS12V1
    data and format it to be used with the ExoCcycle model.

    '''

    def __init__(self, options = {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1", "year":[1994], "data": "bottomT", "depthAve":[0,100]}):
        """
        Initialization of GLORYS12V1
        
        
        Parameters
        -----------
        options : DICTIONARY
            download : BOOLEAN
            dataDir : STRING
            year : LIST
            data : STRING
                Data to be averaged
                    'bottomT' : sea_water_potential_temperature_at_sea_floor
                    'thetao' : sea_water_potential_temperature [depth dimension]
                    'so' : sea_water_salinity [depth dimension]
            depthAve : LIST
                2 element list, describing the range over which an output
                value should be averaged. Note that averaging is only used
                if the input variable has a depth dimension.
            The default is {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1",
            "year":[1994], "data": "thetao", "depthAve":[0,100]}.
        """

        # Assign options to object
        self.options = options;

        # Assign general name of netCDF file
        self.options["netCDFGeneral"] = "mercatorglorys12v1_gl12_mean_YEARMONTH.nc";



    def download(self):
        """
        download method is used to download the GLORYS12V1
        data (netCDFs) and store them in a data directory
        accessed by the ExoCcycle library.
        """
        FIXME


    def averageModels(self):
        """
        averageModels is used to make an averaged netCDF model
        given all netCDFs stored in the data directory.  
        """

        # Create list of netCDFs to average
        self.ListOfNetCDFs = [];
        for i in range(len(self.options['year'])):
            for month in range(12):
                readFile = self.options["netCDFGeneral"].replace("YEAR", str(self.options['year'][i])).replace("MONTH", "{}".format(month+1).zfill(2))
                self.ListOfNetCDFs.append(readFile)
        
        # Iterate through all netCDF4 files, copying only the used variables
        self.ListOfSimpNetCDFs = [];
        for i in range(len(self.ListOfNetCDFs)):
            self.simplifyNetCDF(
                inputPath=self.options["dataDir"]+"/"+self.ListOfNetCDFs[i],
                outputPath=self.options["dataDir"]+"/file{}.nc".format(i),
                variableList=['longitude', 'latitude', self.options['data']]
            )
            self.ListOfSimpNetCDFs.append(self.options["dataDir"]+"/file{}.nc".format(i))

        # Create a gmt command and use to gmt to average all netCDF4s.
        ## Create list of files to add
        SimpNetCDFs = " ".join(self.ListOfSimpNetCDFs)
        ## Create list of adds
        adds=[];
        for i in range(len(self.ListOfSimpNetCDFs)-1):
            adds.append("ADD")
        adds = " ".join(adds);
        ## Define the command
        outputFileName = "{0}_average_{1}_{2}m.nc".format(self.options["dataDir"]+"/"+self.options['data'], self.options['depthAve'][0], self.options['depthAve'][1])
        GMTcommand = "gmt grdmath {0} {1} {2} DIV = {3}".format(SimpNetCDFs, adds, len(self.ListOfSimpNetCDFs), outputFileName)
        ## Use the command
        os.system(GMTcommand)
        ## Apply a mask to the averaged grid (FIXME: No longer need)
        #os.system("gmt grdmath {0} {1} OR = {1}".format(self.options["dataDir"]+"/"+self.ListOfSimpNetCDFs[0], outputFileName)
        

    def simplifyNetCDF(self, inputPath="path/file.nc", outputPath="~/file2.nc",
    variableList=['longitude', 'latitude', 'variable']):
        """
        simplifyNetCDF method reads a NetCDF4 file and writes a new NetCDF4 file
        with only lat, lon, and 'variable' variables.

        Parameters
        -----------
        input_path : STRING
            Path to the input NetCDF4 file.
        output_path : STRING
            Path to save the new NetCDF4 file.
        variableList : LIST OF STRINGS
            3 length list of strings that correspond to an
            x (longitude), y (latitude), and z (e.g., bathymetry)
            variable. The default is ['latitude', 'logitude',
            'variable'].
        """
        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(outputPath)

        # Open the original NetCDF file (file1.nc) in read mode
        with Dataset(inputPath, 'r') as src:
            # Create a new NetCDF file (file2.nc) in write mode
            with Dataset(outputPath, 'w', format="NETCDF4_CLASSIC") as dst:
                # Copy global attributes
                if "title" in src.ncattrs():
                    dst.title = src.title  # Preserve title attribute

                # Copy lat & lon dimensions
                for dim_name in [variableList[0], variableList[1]]:
                    if dim_name in src.dimensions:
                        dst.createDimension(dim_name, len(src.dimensions[dim_name]))

                # Copy lat & lon variables
                # for var_name in [variableList[0], variableList[1]]:
                #     if var_name in src.variables:
                #         var = src.variables[var_name]
                #         dst_var = dst.createVariable(var_name, np.float32, var.dimensions)
                #         dst_var[:] = var[:]  # Copy data
                #         # Copy attributes
                #         for attr in var.ncattrs():
                #             try:
                #                 dst_var.setncatts({attr: var.getncattr(attr)})
                #             except:
                #                 pass
                if variableList[0] in src.variables:
                    lat = src.variables[variableList[0]]
                    lat_dst_var = dst.createVariable(variableList[0], np.float32, lat.dimensions)
                    lat_dst_var[:] = lat[:]  # Copy data
                    # Copy attributes
                    # for attr in lat.ncattrs():
                    #     try:
                    #         lat_dst_var.setncatts({attr: lat.getncattr(attr)})
                    #     except:
                    #         pass
                if variableList[1] in src.variables:
                    lon = src.variables[variableList[1]]
                    lon_dst_var = dst.createVariable(variableList[1], np.float32, lon.dimensions)
                    lon_dst_var[:] = lon[:]  # Copy data
                    # Copy attributes
                    # for attr in lon.ncattrs():
                    #     try:
                    #         lon_dst_var.setncatts({attr: lon.getncattr(attr)})
                    #     except:
                    #         pass


                # Copy variable and rename it to 'z'
                if variableList[2] in src.variables:
                    z_var = src.variables[variableList[2]]
                    dst_z = dst.createVariable("z", np.float32, (z_var.dimensions[-2],z_var.dimensions[-1]))
                    if len(z_var.dimensions) == 3:
                        # Copy data
                        dst_z[:] = z_var[:]
                    else:
                        # Define depth variable and logical defining
                        # depth range to average over
                        depth = src.variables['depth'][:];
                        layerThickness = np.diff( np.append(0, depth) )
                        depthLogical = (depth>self.options['depthAve'][0])&(depth<self.options['depthAve'][1])

                        # Define array to hold sum of layered values
                        # (Assumes top/first layer contains all possible values as non-nan)
                        LayerIdx = np.argwhere(depthLogical.data).T
                        if np.size(LayerIdx) > 1:
                            LayerIdx = LayerIdx[0]
                            topLayerIdx = np.argwhere(depthLogical.data)[0][0];
                        else:
                            topLayerIdx = np.argwhere(depthLogical.data)[0];
                        sum = z_var[:][0][topLayerIdx].data*0
                        intervals = z_var[:][0][topLayerIdx].data*0
                        layerThickness = layerThickness[depthLogical]
                        dataLayers = z_var[:][0][depthLogical]

                        # Loop over depth intervals
                        for i in range(len(layerThickness)):
                            sum[~dataLayers[i].mask]       += dataLayers[i].data[ ~dataLayers[i].mask ]*layerThickness[i];
                            intervals[~dataLayers[i].mask] += layerThickness[i]

                        # Copy top layer and replace .data with averaged values 
                        # average = cp.deepcopy( z_var[:][0][topLayerIdx] )
                        # average.data[:] = (sum/intervals)
                        # dst_z[:] = average.data
                        dst_z[:] = sum/intervals

                    # Copy attributes
                    # for attr in z_var.ncattrs():
                    #     if attr != "_FillValue":
                    #         try:
                    #             #print("attr",attr)
                    #             dst_z.setncatts({attr: z_var.getncattr(attr)})
                    #         except:
                    #             pass


    def readnetCDF(self, year, month):
        # Define the file name for year and month
        
        readFile = self.options["dataDir"]+"/"+self.options["netCDFGeneral"].replace("YEAR", str(year)).replace("MONTH", "{}".format(month).zfill(2))

        # Read the netCDF file
        return Dataset(readFile, "r");


class GLORYS12V1_old():
    '''
    The GLORYS12V1 class is used to download GLORYS12V1
    data and format it to be used with the ExoCcycle model.

    '''

    def __init__(self, options = {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1", "year":[1994], "data": "bottomT", "depthAve":[0,100]}):
        """
        Initialization of GLORYS12V1
        
        
        Parameters
        -----------
        options : DICTIONARY
            download : BOOLEAN
            dataDir : STRING
            year : LIST
            data : STRING
                Data to be averaged
                    'bottomT' : sea_water_potential_temperature_at_sea_floor
                    'thetao' : sea_water_potential_temperature [depth dimension]
                    'so' : sea_water_salinity [depth dimension]
            depthAve : LIST
                2 element list, describing the range over which an output
                value should be averaged. Note that averaging is only used
                if the input variable has a depth dimension.
            The default is {"download": False, "dataDir":os.getcwd()+"/GLORYS12V1",
            "year":[1994], "data": "thetao", "depthAve":[0,100]}.
        """

        # Assign options to object
        self.options = options;

        # Assign general name of netCDF file
        self.options["netCDFGeneral"] = "mercatorglorys12v1_gl12_mean_YEARMONTH.nc";



    def download(self):
        """
        download method is used to download the GLORYS12V1
        data (netCDFs) and store them in a data directory
        accessed by the ExoCcycle library.
        """
        FIXME


    def averageModels(self):
        """
        averageModels is used to make an averaged netCDF model
        given all netCDFs stored in the data directory.  
        """

        # Create list of netCDFs to average
        self.ListOfNetCDFs = [];
        for i in range(len(self.options['year'])):
            for month in range(12):
                readFile = self.options["netCDFGeneral"].replace("YEAR", str(self.options['year'][i])).replace("MONTH", "{}".format(month+1).zfill(2))
                self.ListOfNetCDFs.append(readFile)
        
        # Iterate through all netCDF4 files, copying only the used variables
        self.ListOfSimpNetCDFs = [];
        for i in range(len(self.ListOfNetCDFs)):
            self.simplifyNetCDF(
                inputPath=self.options["dataDir"]+"/"+self.ListOfNetCDFs[i],
                outputPath=self.options["dataDir"]+"/file{}.nc".format(i),
                variableList=['longitude', 'latitude', self.options['data']]
            )
            self.ListOfSimpNetCDFs.append(self.options["dataDir"]+"/file{}.nc".format(i))

        # Create a gmt command and use to gmt to average all netCDF4s.
        ## Create list of files to add
        SimpNetCDFs = " ".join(self.ListOfSimpNetCDFs)
        ## Create list of adds
        adds=[];
        for i in range(len(self.ListOfSimpNetCDFs)-1):
            adds.append("ADD")
        adds = " ".join(adds);
        ## Define the command
        outputFileName = "{0}_average_{1}_{2}m.nc".format(self.options["dataDir"]+"/"+self.options['data'], self.options['depthAve'][0], self.options['depthAve'][1])
        GMTcommand = "gmt grdmath {0} {1} {2} DIV = {3}".format(SimpNetCDFs, adds, len(self.ListOfSimpNetCDFs), outputFileName)
        ## Use the command
        os.system(GMTcommand)
        ## Apply a mask to the averaged grid (FIXME: No longer need)
        #os.system("gmt grdmath {0} {1} OR = {1}".format(self.options["dataDir"]+"/"+self.ListOfSimpNetCDFs[0], outputFileName)
        

    def simplifyNetCDF(self, inputPath="path/file.nc", outputPath="~/file2.nc",
    variableList=['longitude', 'latitude', 'variable']):
        """
        simplifyNetCDF method reads a NetCDF4 file and writes a new NetCDF4 file
        with only lat, lon, and 'variable' variables.

        Parameters
        -----------
        input_path : STRING
            Path to the input NetCDF4 file.
        output_path : STRING
            Path to save the new NetCDF4 file.
        variableList : LIST OF STRINGS
            3 length list of strings that correspond to an
            x (longitude), y (latitude), and z (e.g., bathymetry)
            variable. The default is ['latitude', 'logitude',
            'variable'].
        """
        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(outputPath)

        # Open the original NetCDF file (file1.nc) in read mode
        with Dataset(inputPath, 'r') as src:
            # Create a new NetCDF file (file2.nc) in write mode
            with Dataset(outputPath, 'w', format="NETCDF4_CLASSIC") as dst:
                # Copy global attributes
                if "title" in src.ncattrs():
                    dst.title = src.title  # Preserve title attribute

                # Copy lat & lon dimensions
                for dim_name in [variableList[0], variableList[1]]:
                    if dim_name in src.dimensions:
                        dst.createDimension(dim_name, len(src.dimensions[dim_name]))

                # Copy lat & lon variables
                # for var_name in [variableList[0], variableList[1]]:
                #     if var_name in src.variables:
                #         var = src.variables[var_name]
                #         dst_var = dst.createVariable(var_name, np.float32, var.dimensions)
                #         dst_var[:] = var[:]  # Copy data
                #         # Copy attributes
                #         for attr in var.ncattrs():
                #             try:
                #                 dst_var.setncatts({attr: var.getncattr(attr)})
                #             except:
                #                 pass
                if variableList[0] in src.variables:
                    lat = src.variables[variableList[0]]
                    lat_dst_var = dst.createVariable(variableList[0], np.float32, lat.dimensions)
                    lat_dst_var[:] = lat[:]  # Copy data
                    # Copy attributes
                    # for attr in lat.ncattrs():
                    #     try:
                    #         lat_dst_var.setncatts({attr: lat.getncattr(attr)})
                    #     except:
                    #         pass
                if variableList[1] in src.variables:
                    lon = src.variables[variableList[1]]
                    lon_dst_var = dst.createVariable(variableList[1], np.float32, lon.dimensions)
                    lon_dst_var[:] = lon[:]  # Copy data
                    # Copy attributes
                    # for attr in lon.ncattrs():
                    #     try:
                    #         lon_dst_var.setncatts({attr: lon.getncattr(attr)})
                    #     except:
                    #         pass


                # Copy variable and rename it to 'z'
                if variableList[2] in src.variables:
                    z_var = src.variables[variableList[2]]
                    dst_z = dst.createVariable("z", np.float32, (z_var.dimensions[-2],z_var.dimensions[-1]))
                    if len(z_var.dimensions) == 3:
                        # Copy data
                        dst_z[:] = z_var[:]
                        print("test1")
                    else:
                        print("test2")
                        # Define depth variable and logical defining
                        # depth range to average over
                        depth = src.variables['depth'][:];
                        layerThickness = np.diff( np.append(0, depth) )
                        depthLogical = (depth>self.options['depthAve'][0])&(depth<self.options['depthAve'][1])
                        
                        # Copy data
                        x = np.nanmean(z_var[:][0][depthLogical]*layerThickness[depthLogical, np.newaxis, np.newaxis], axis=0) / np.sum(layerThickness[depthLogical])
                        self.x1 = z_var[:][0][depthLogical]
                        self.x2 = layerThickness[depthLogical, np.newaxis, np.newaxis]
                        self.x3 = np.sum(layerThickness[depthLogical])

                        dst_z[:] = x.data

                    # Copy attributes
                    # for attr in z_var.ncattrs():
                    #     if attr != "_FillValue":
                    #         try:
                    #             #print("attr",attr)
                    #             dst_z.setncatts({attr: z_var.getncattr(attr)})
                    #         except:
                    #             pass


    def readnetCDF(self, year, month):
        # Define the file name for year and month
        
        readFile = self.options["dataDir"]+"/"+self.options["netCDFGeneral"].replace("YEAR", str(year)).replace("MONTH", "{}".format(month).zfill(2))

        # Read the netCDF file
        return Dataset(readFile, "r");
    


#######################################################################
################# Extra Functions, might not be used ##################
#######################################################################
def polygonAreaOnSphere(vertices, radius=6371e3):
    """
    Calculate the area of a polygon on a sphere using spherical excess formula.
    Note that great circle arcs are drawn between input vertices.
    
    Parameters:
        vertices : LIST OF TUPLES
            List of (latitude, longitude) points representing the polygon's vertices.
            Latitudes and longitudes should be in degrees.
        radius : FLOAT
            Radius of the sphere.
        
    Returns:
        float: Area of the polygon on the sphere in square kilometers.
    """
    
    def to_radians(deg):
        return np.radians(deg)

    # Convert latitude and longitude to radians
    vertices_rad = [(to_radians(lat), to_radians(lon)) for lat, lon in vertices]

    total_angle = 0.0
    n = len(vertices_rad)
    print(vertices_rad)
    
    # Loop through the vertices using the spherical excess formula
    for i in range(n):
        print(i)
        lat1, lon1 = vertices_rad[i]
        lat2, lon2 = vertices_rad[(i + 1) % n]
        print(i, lat1, lon1, lat2, lon2)
        
        # Compute the angle between two vertices
        delta_lon = lon2 - lon1
        total_angle += np.arctan2(np.tan(lat2 / 2 + np.pi / 4) * np.sin(delta_lon), np.tan(lat1 / 2 + np.pi / 4));

    # Spherical excess formula: Area = (sum of angles - (n-2)*pi) * radius^2
    spherical_excess = abs(total_angle - (n - 2) * np.pi)

    # Return the area on the sphere
    area = spherical_excess * (radius ** 2)

    return area







def octPolylineLengthplt():
    """
    the octPolylineLengthplt function is used to test the implementation of
    octPolylineLength. This function shows the calulation of octagonal like
    shape sides for increasing latitudes, for hexagons centered at from 0.5
    to 89.5 degrees latitude and with 1 degree resolution.
    
    """

    resolution = 1
    rlatv = np.arange(resolution/2, 90, resolution)
    lowerv = []
    upperv = []
    leftrightv = []
    diav = []


    #print(rlatv)
    for rlat in rlatv:
        leftright =  octPolylineLength((rlat,rlon), (rlat,rlon+resolution), resolution=resolution, verbose=False)

        upper = octPolylineLength((rlat,rlon), (rlat+resolution,rlon), resolution=resolution, verbose=False)
        lower = octPolylineLength((rlat,rlon), (rlat-resolution,rlon), resolution=resolution, verbose=False)

        dia = octPolylineLength((rlat,rlon), (rlat+resolution,rlon+resolution), resolution=resolution, verbose=False)
        
        lowerv.append(lower)
        upperv.append(upper)
        leftrightv.append(leftright)
        diav.append(dia)
    #print( octPolylineLength((rlat,rlon), (rlat+resolution,rlon+resolution)) )

    #print("lower \t", lowerv)
    #print("upper \t", upperv)
    #print("leftright \t", leftrightv)
    #print("dia \t", diav)


    import matplotlib.pyplot as plt

    r = 6371;
    plt.plot(rlatv, r*np.array(lowerv), label="lower");
    plt.plot(rlatv, r*np.array(upperv), label="upper");
    plt.plot(rlatv, r*np.array(diav), label="dia");
    plt.plot(rlatv, r*np.array(leftrightv), label="leftright");

    plt.xlabel("latitude [deg]");
    plt.ylabel("length [km]");
    plt.legend();
