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
import networkx as nx
from netCDF4 import Dataset 
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# For GUI (FIXME: choosing basin definitions, might not be needed anymore)
import tkinter as tk
from tkinter import messagebox

# For progress bars
from tqdm import tqdm


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
        import networkx as nx
        from geopy.distance import geodesic


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
                bathymetryf = self.bathymetry[::self.reducedRes].T[::self.reducedRes].T.flatten();

            # Distance, in degrees, to a diagonal node.
            cornerDis = self.reducedRes/np.sin(np.pi/4);

            # Create dictionary and array of bathymetry points
            pos = np.zeros( (2, len(~np.isnan(bathymetryf))) );
            for i in tqdm( range(len(self.lonf)) ):
                bathymetryi = bathymetryf[i];
                if (~np.isnan(bathymetryi)):
                    points[int(cnt)] = (self.latf[i], self.lonf[i], bathymetryi);    # (latitude, longitude, depth) w/ units (deg, deg, m)
                    pos[:,int(cnt)] = np.array( [self.latf[i], self.lonf[i]] ); 
                    cnt+=1;

            # Create a graph
            G = nx.Graph()

            # Add nodes (points)
            for node, values in points.items():
                G.add_node(node, pos=values[0:2], depth=values[2])

            """
            # Add edges with weights based on geographic distance            
            for node1, values1 in tqdm(points.items()):
                coords1 = values1[0:2];
                for node2, values2 in points.items():
                    coords2 = values2[0:2];
                    if node1 != node2: # If the nodes are the same
                        
                        if (np.abs(points[node1][0]-points[node2][0])<cornerDis) & (np.abs(points[node1][1]-points[node2][1])<cornerDis):
                            # (Within cornerDis in latitude) & (Within cornerDis in longitude)
                            # If the nodes are adjacent seafloor node

                            # Calculate geographic distance between points using geodesic distance
                            bathyAve= (values1[2]+values2[2])/2;
                            hexsidelengnth = hexPolylineLength(coords1, coords2, verbose=False);
                            G.add_edge(node1, node2, bathyAve=bathyAve*hexsidelengnth)
                        elif (points[node1][1]==np.min(self.lonf)) & (np.abs(points[node1][0]-points[node2][0])<cornerDis) & (np.abs(points[node1][1]+points[node2][1])<cornerDis):
                            # (On left most boundary) & (Within cornerDis in latitude) & (Within cornerDis in longitude)
                            # If nodes are adjacent seafloor nodes and at a periodic boundary 
                            bathyAve= (values1[2]+values2[2])/2;
                            hexsidelengnth = hexPolylineLength(coords1, (points[node2][0], -1*points[node2][1]), verbose=False);
                            G.add_edge(node1, node2, bathyAve=bathyAve*hexsidelengnth)
            """    

            # Update to the above code block which Adds edges with weights based on geographic distance.
            # This code is significantly faster than the above code block

            ## Set bathymetry vector corresponding to node 
            nodebathymetryf = ~np.isnan(bathymetryf);
            nodeCntList = np.arange(0,len(pos[0,:]),1)

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
                        
                        #G.add_edge(node1, node2, bathyAve=hexsidelengnth);
                        G.add_edge(node1, node2, bathyAve=bathyAve*hexsidelengnth);
            #print("Bathymetry is not being used as a weight for node edges")



            # Set some class parameters for testing purposes.
            self.G = G;

            # Find communities of nodes using the Girvan-Newman algorithm
            self.findCommunities(method = method, minBasinCnt = minBasinCnt);

            # Find nodes with high inbetweenness
            # FIXME:
            

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
        findCommunities uses the Girvan-Newman algorithm to determine
        communities of nodes (basins). Then nodes of similar basins
        are given a basinID.

        
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
        ## of unique communities that the Girvan-Newman algorithm
        ## found. This is not always equal to the minBasinCnt 
        basinIDTags = np.arange(len(self.communitiesFinal));

        basinIDs = {}; cnt = 0;
        for community in self.communitiesFinal:
            basinIDi = float(basinIDTags[cnt]);
            for nodeID in community:
                basinIDs[nodeID] = {"basinID": basinIDi};
            cnt+=1;
        nx.set_node_attributes(self.G, basinIDs, "basinID");



    def createCommunityNodeColors(self):
        """
        createCommunityNodeColors method sets colors associated
        with different community nodes (e.g., basins in this case).


        Returns
        ----------
        node_colors : PYTHON LIST
            A list of hex code colors that correspond to a node's
            community.
        """
        # Define colors for basin identification.
        colors = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20b"].colors];
        colors2 = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20c"].colors]
        for i in range(len(colors2)):
            colors.append(colors2[i])

        node_colors = [];

        # Get basin IDs from network object.
        tmpValues = nx.get_node_attributes(self.G, "basinID");

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
        rewitten, overriding the original basinID network. Note that
        there is no need to reread the basin network. 

        Parameters
        ----------
        basinID1 : INT
            Basin ID to absorb basinID2.
        basinID2 : INT, or LIST OF INT
            Basin ID(s) to be absorbed by basinID1.
        write : BOOLEAN
            An option to write over the original network.
        
        Returns
        --------
        None.
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

        Defines
        ----------
        BasinIDA : NUMPY ARRAY
            FIXME: ADD
        bathymetryAreaDistBasin : DICTIONARY
            FIXME: ADD
        bathymetryAreaFraction : DICTIONARY
            FIXME: ADD
        bathymetryVolFraction : DICTIONARY
            FIXME: ADD
        binEdges : NUMPY ARRAY
            FIXME: ADD
        
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
        #nodeIDDictionary = self.G.nodes;
        pos = np.zeros( (len(nodePosDict), 2) );
        BasinID = np.zeros( (len(nodeBasinID), 1) );
        #nodeID = np.zeros( (len(nodeBasinID), 1) );
        for i in range(len(nodePosDict)):
            pos[i,:] = nodePosDict[i];
            BasinID[i] = nodeBasinID[i]['basinID'];
            #nodeID[i] = nodeIDDictionary[i];
        
        BasinIDA = np.empty(np.shape(lonA));
        BasinIDA[:] = np.nan;
        nodeIDA = np.empty(np.shape(lonA));
        #nodeIDA[:] = np.nan;
        for nodei in range(len(pos[:,1])):
            BasinIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = BasinID[nodei];
            #nodeIDA[(lonA==pos[nodei,1])&(latA==pos[nodei,0])] = nodeID[nodei];

        # Calculate basin distributions
        bathymetryAreaDistBasin, bathymetryVolFraction, bathymetryAreaFraction, bathymetryAreaDist_wHighlatG, bathymetryAreaDistG, binEdges = Bathymetry.calculateBathymetryDistributionBasin(bathymetry, latA, BasinIDA, self.highlatlat, self.areaWeights, binEdges=binEdges, verbose=verbose)
        
        # Define basinID and nodeid array
        self.BasinIDA = BasinIDA;
        #self.nodeIDA = nodeIDA;

        # Set basin bathymetry parameters
        self.bathymetryAreaDistBasin = bathymetryAreaDistBasin
        self.bathymetryAreaFraction  = bathymetryAreaFraction
        self.bathymetryVolFraction   = bathymetryVolFraction
        self.binEdges                = binEdges
        self.bathymetryAreaDist_wHighlatG = bathymetryAreaDist_wHighlatG
        self.bathymetryAreaDistG = bathymetryAreaDistG

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

        Returns
        --------
            basinAreaConnection : NUMPY ARRAY
                An area weighted array that describes the surface area
                connection between all basins (e.g., a 3x3 would describe
                the surface connection between 3 basins).

            ARRAY2 : NUMPY ARRAY
                A 3x3xlen(binEdges) size array with each entry in the 3x3
                array representing bathymetry distributions.
        
        """
        print("working progress")
        ############################################################
        ##### Find the connectivity between each set of basins #####
        ############################################################


        ############################################################
        ############# Find nodes boarding other basins #############
        ############################################################
        # Define the basin count and graph
        basinCnt = len(np.unique(self.BasinIDA))-1
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
        basinAreaConnection = np.zeros((basinCnt,basinCnt), dtype=float);
        # Define basinAreaConnectionTracker, which keeps track if the 
        # symmetric basin connection (e.g., Atlantic-Indian for Indian-Atlantic)
        # has already been defined.
        basinAreaConnectionTracker = np.ones((basinCnt,basinCnt), dtype=bool);
        cnt=0

        # Populate basinAreaConnection with enumerations of basin
        # connections.
        ## Loop over basins
        for basini in range(basinCnt):
            ## Loop over basins
            for basinj in range(basinCnt):
                ## Loop over other basins
                if  (basini!=basinj) & basinAreaConnectionTracker[basinj,basini]:
                    ## Set basin connection ID
                    basinAreaConnection[basini,basinj] = cnt;
                    cnt+=1;
                    ## Indicate that basin connection ID has been set
                    basinAreaConnectionTracker[basini,basinj] = False;
                elif (basini!=basinj):
                    ## If symmetric calculation has already been done
                    ## then set the symmetric array value
                    basinAreaConnection[basini,basinj] = basinAreaConnection[basinj,basini]
                elif (basini==basinj):
                    basinAreaConnection[basini,basinj] = -1;
        
        ############################################################
        ####### Get the bathymetry and nodeID for all nodes ########
        ############################################################
        allNodeIDs = np.array([], dtype=float);
        allNodeBathymetry = np.array([], dtype=float);
        for node in BasinNodes:
            allNodeIDs = np.append(allNodeIDs, node);
            allNodeBathymetry = np.append(allNodeBathymetry, BasinNodes.nodes[node]['depth']);
        
        ############################################################
        ### Set nodeIDs and bathymetry for each basin connection ###
        ############################################################

        # Define basin connective nodes and bathymetry as list of list.
        # This is done such that each basin connection can have different
        # numbers of nodes to describe them.
        self.connectiveBathy = np.empty((basinCnt*basinCnt-basinCnt)//2,dtype=object)
        self.connectingNodes = np.empty((basinCnt*basinCnt-basinCnt)//2,dtype=object)
        for i in range(len(self.connectingNodes)):
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
            connectionID = basinAreaConnection[int(basinEdgeNodeBasinIDs1[int(i)]),
                                               int(basinEdgeNodeBasinIDs2[int(i)])]
            
            ## Add nodeIDs to appropriate basin connection list (using connectionID)
            ## And remove any repeat nodes
            if connectionID != -1:
                self.connectingNodes[int(connectionID)] = np.append(self.connectingNodes[int(connectionID)],
                                                                    allNodeIDs[logical] );
                self.connectingNodes[int(connectionID)] = np.unique(self.connectingNodes[int(connectionID)]);
    
        # Add bathymetry to appropriate basin connection list
        # (using connectionID).
        ## Loop over unique basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            self.connectiveBathy[int(connectionIDi)] = np.append(self.connectiveBathy[int(connectionIDi)],
                                                                 allNodeBathymetry[self.connectingNodes[int(connectionIDi)].astype('int')] );
        
        ############################################################
        # Calculate bathymetry distributions for basin connections #
        ############################################################

        # If binEdges are not defined in method input then define with
        # LOSCAR's, a carbon cycle model, default distribution.
        if binEdges is None:
            binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    
        # Setup dictionary to hold basin distributions.
        self.bathymetryConDist = {};
    
        # Calculate all basin connection bathymetry distributions.
        ## Iterate over basin connections
        for connectionIDi in range(len(self.connectingNodes)):
            print("Need to add area weights to properly represent bathymetry distribution at basin connection.")
            # Calculate a basin connection bathymetry distributions.
            self.bathymetryConDisti, binEdges = np.histogram((1e-3)*self.connectiveBathy[connectionIDi],
                                                             bins=binEdges);
            # Add the distribution information to dictionary.
            self.bathymetryConDist['Connection{:0.0f}'.format(connectionIDi)] = 100*(self.bathymetryConDisti/np.sum(self.bathymetryConDisti));
    
        ############################################################
        ####################### Plot results #######################
        ############################################################
        if verbose:
            # Report the basin connectivity distributions
            print("Bin edges used:\n", binEdges);
            print("Bathymetry area distribution including high latitude bathymetry:\n");
            for connectionIDi in range(len(self.connectingNodes)):
                print(self.bathymetryConDist['Connection{:0.0f}'.format(connectionIDi)]);
            
            # Plot the basin IDs, connectivity nodes, and their
            # bathymetry distributions
            self.plotBasinConnections(pos, basinCnt, binEdges);


    def plotBasinConnections(self, pos,  basinCnt, binEdges=None, verbose=True):
        """
        plotBasinConnections is used to plot results calculating from
        running the method calculateBasinConnectivityParameters.
        
        Parameters
        -----------
        pos : NUMPY ARRAY
            An nx2 array with columns of latitude and longitude, in degrees. This
            array should represent the locations of basin nodes.
        basinCnt : INT
            Number of basins in model.
        binEdges : NUMPY LIST, optional
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin. The default is None, but this is modified to 
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
            the code.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

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

        # Plot basin contourf and coastlines

        ## Add the plot using pcolormesh
        mesh = ax1.pcolormesh(self.lon, self.lat, self.BasinIDA, cmap="Pastel1", transform=ccrs.PlateCarree())

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

        ## Set colormap
        cmap = plt.get_cmap("Dark2")
        ## Extract basinCnt colors from the colormap
        colors_rgb = [cmap(i) for i in range(basinCnt)]
        ## Convert RGB to hex
        colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb]
        ## Create a custom colormap from the list of colors
        custom_cmap = LinearSegmentedColormap.from_list("custom_pastel", colors, N=256)

        ## Plot 
        plt.contourf(self.lon, self.lat, BC,
                     cmap=custom_cmap,
                     transform=ccrs.PlateCarree());
        

        # Plot gridlines
        ax1.gridlines()
        

        # Plot bathymetry distributions of basin connections.

        ## Set new axis to plot on
        ax2 = fig.add_subplot(gs[1]);


        ## Define factors for plotting
        factor1 = .1
        factor2 = .25
        if len(self.bathymetryConDist)%2:
            factor3 = 0.5;
        else:
            factor3 = 0;
        cnt = len(self.bathymetryConDist);

        ## Iteratively plot basin bathymetry distributions
        for i in range(len(self.bathymetryConDist)):
            plt.bar(x=binEdges[1:]-(factor2/2)*(cnt/2 - i -factor3)*np.diff(binEdges),
                    height=self.bathymetryConDist['Connection{:0.0f}'.format(i)],
                    width=factor1*np.diff(binEdges),
                    label= "Connection {:0.0f}".format(i),
                    color=colors[i])
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
        #if savePNG:
        #    plt.savefig("{}/{}".format(outputDir,fidName), dpi=600)
        #if saveSVG:
        #    plt.savefig("{}/{}".format(outputDir,fidName.replace(".png", ".svg")))






                
    



    def saveCcycleParameter(self):
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
            FIXME: Working progress to include bathymetry distributions
            that connect basins... these could be formated as the following

            1) An area weighted array that describes the surface area connection
            between all basins (e.g., a 3x3 would describe the surface connection
            between 3 basins)

            2) An array of bathymetry distributions (3x3xlen(binEdges))


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
            float64 VOC(),
            float64 AOC()
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

        Returns
        --------
        None.
        """
        # Set netCDF4 filename
        BathyPath = "{0}/{1}".format(self.dataDir, self.filename.replace(".nc", "_wBasins.nc"));
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4')
        CycleParmsGroup = ncfile.createGroup("CycleParms")
        ArraysGroup = ncfile.createGroup("Arrays")

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ArraysGroup.createDimension('lat', len(self.bathymetry[:,0]));     # latitude axis
        lon_dim = ArraysGroup.createDimension('lon', len(self.bathymetry[0,:]));     # longitude axis
        binEdges_dim = CycleParmsGroup.createDimension('binEdges', len(self.binEdges[1:]));              # distribution
        basinID_dim = CycleParmsGroup.createDimension('BasinID', len(self.bathymetryAreaDistBasin));     # BasinID
        
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
        
        # Format title
        ncfile.title='{} Bathymetry created from topography resampled at {:0.0f} degrees. NetCDF4 includes carbon cycle bathymetry parameters'.format(self.body, self.resolution)
        
        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        bathy[:] = self.bathymetry;
        basinIDArray[:] = self.BasinIDA;
        areaWeights[:] = self.areaWeights[:,0];

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

        # Close the netcdf
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
    lat2 : FLOAT
        Coordinate of second point latitude, in degree.
    lon2 : FLOAT
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