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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx # type: ignore
from netCDF4 import Dataset 
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# For progress bars
from tqdm.auto import tqdm # used for progress bar

from ExoCcycle import Bathymetry # type: ignore


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


def areaWeights(resolution = 1, radius = 6371e3, verbose=True):
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
    Y = np.arange(90-resolution/2,-90-resolution/2, -resolution);
    X = np.arange(-180+resolution/2, 180+resolution/2, resolution);

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

def plotGlobal(lat, lon, values,
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
    fig = plt.figure(figsize=(10, 5))

    ## Set up the Mollweide projection
    ax = plt.axes(projection=ccrs.Mollweide())

    ## Add the plot using pcolormesh
    mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                         vmin=cmapOpts['cbar-range'][0], vmax=cmapOpts['cbar-range'][1])
    if pltOpts["plotZeroContour"]:
        # Set any np.nan values to 0.
        values[np.isnan(values)] = 0;
        zeroContour = ax.contour(lon, lat, values, levels=[0], colors='black', transform=ccrs.PlateCarree())

    ## Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
    cbar.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
    cbar.ax.tick_params(labelsize=10)  # Adjust the size of colorbar ticks

    ## Add gridlines
    ax.gridlines()

    ## Set a title
    plt.title(pltOpts['plotTitle'])

    # Save figure
    if savePNG:
        plt.savefig("{}/{}".format(outputDir,fidName), dpi=600)
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
                        hexsidelengnth = hexPolylineLength(coords1, coords2, verbose=False);
                        
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
            self.communitiesFinal = nx.community.louvain_communities(self.G, weight='bathyAve', resolution=.1, threshold=1e-12, seed=1)

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

        # Define the number of define basins
        BasinIDmax = self.G
        distance = 1e20;

        # Get basin properties from nodes
        nodesID = nx.get_node_attributes(self.G, "basinID");          # Dictionary of nodes and their basinID
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
        colors_rgb2 = [cmap2(i) for i in range(self.basinCnt)]
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
        for i in range(self.basinConCnt):
            idx = np.argwhere(i==self.basinAreaConnection)[0];
            plt.bar(x=binEdges[1:]-(factor2/2)*(self.basinConCnt/2 - i -factor3)*np.diff(binEdges),
                    height=self.bathymetryConDist[idx[0],idx[1]],
                    width=factor1*np.diff(binEdges),
                    label= "Connection {:0.0f}".format(i),
                    color=colors2[i])
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
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance
    distance = radius * c
    return distance

def hexPolylineLength(coords1, coords2, resolution=1, verbose=True):
    """
    hexPolylineLength function will define the length of a ~hexagonal
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
                                              'threshold':np.array([.1,.5]),
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







def hexPolylineLengthplt():
    """
    the hexPolylineLengthplt function is used to test the implementation of
    hexPolylineLength. This function shows the calulation of hexagonal like
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
        leftright =  EC.utils.hexPolylineLength((rlat,rlon), (rlat,rlon+resolution), resolution=resolution, verbose=False)

        upper = EC.utils.hexPolylineLength((rlat,rlon), (rlat+resolution,rlon), resolution=resolution, verbose=False)
        lower = EC.utils.hexPolylineLength((rlat,rlon), (rlat-resolution,rlon), resolution=resolution, verbose=False)

        dia = EC.utils.hexPolylineLength((rlat,rlon), (rlat+resolution,rlon+resolution), resolution=resolution, verbose=False)
        
        lowerv.append(lower)
        upperv.append(upper)
        leftrightv.append(leftright)
        diav.append(dia)
    #print( EC.utils.hexPolylineLength((rlat,rlon), (rlat+resolution,rlon+resolution)) )

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