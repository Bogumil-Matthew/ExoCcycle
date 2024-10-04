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
        provied values. The default is 6371e3.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Returns
    -------
    areaWeights : NUMPY ARRAY
        An array of global degree to area weights. The size is dependent on
        input resolution. The sum of the array equals 4 pi radius^2 for 
        sufficiently high resolution.
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
        using the Girvan-Newman algorithm to define communities.

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


        # basins are determine based on a connectivity
        # algorithm. Note that this might be appropriate
        # for a diffusion only ocean system, but currents
        # are not taken into account (this would require
        # knowing much more information about the body's
        # properties).
        
        # Calculate ocean basin volume distribution
        basinVolDis = self.areaWeights*self.bathymetry;
        # Run a k-mean analysis (note that this must have periodic x boundaries)
        # See https://www.mdpi.com/2073-8994/14/6/1237 and https://github.com/kpodlaski/periodic-kmeans
        # For methods that modify the pyClustering library
        # https://github.com/kpodlaski/periodic-kmeans

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
        
            # Define resolution
            resolution = 1;
            
            # Only reduce resolution if option is set. Note that this must
            # be consistent with written network
            if not reducedRes['on']:
                reduceRes = reducedRes['factor'];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                reduceRes = reducedRes['factor'];
                self.latf = self.lat[::reduceRes].T[::reduceRes].T.flatten();
                self.lonf = self.lon[::reduceRes].T[::reduceRes].T.flatten();
                bathymetryf = self.bathymetry[::reduceRes].T[::reduceRes].T.flatten();
        
        elif write:
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
                reduceRes = reducedRes['factor'];
                self.latf = self.lat.flatten();
                self.lonf = self.lon.flatten();
                bathymetryf = self.bathymetry.flatten();
            else:
                reduceRes = reducedRes['factor'];
                self.latf = self.lat[::reduceRes].T[::reduceRes].T.flatten();
                self.lonf = self.lon[::reduceRes].T[::reduceRes].T.flatten();
                bathymetryf = self.bathymetry[::reduceRes].T[::reduceRes].T.flatten();

            # Distance, in degrees, to a diagonal node.
            cornerDis = reduceRes/np.sin(np.pi/4);

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
                    condition = (pos.T==np.array( [coords1[0]+reduceRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(1)
                if not (coords1[0] == np.min(self.latf)):
                    # Not South pole node
                    ## lower node
                    condition = (pos.T==np.array( [coords1[0]-reduceRes, coords1[1]] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] )
                    indicesAddPt.append(2)

                # If on periodic boundary
                if coords1[1] == np.min(self.lonf):
                    # nodes on left (min) boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, coords1[1]+reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(3)
                        indicesAddPt.append(3)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, coords1[1]+reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, np.max(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] )
                        indicesAddPt.append(4)
                        indicesAddPt.append(4)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+reduceRes] ));
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
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, coords1[1]-reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(6)
                        indicesAddPt.append(6)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, np.min(self.lonf)] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, coords1[1]-reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(7)
                        indicesAddPt.append(7)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], np.min(self.lonf)] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-reduceRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    indicesAddPt.append(8)
                    indicesAddPt.append(8)

                else:
                    # Nodes not on boundary
                    if not (coords1[0] == np.max(self.latf)):
                        # Not North pole node
                        ## upper right node
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, coords1[1]+reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## upper left node
                        condition = (pos.T==np.array( [coords1[0]+reduceRes, coords1[1]-reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(9)
                        indicesAddPt.append(9)
                    if not (coords1[0] == np.min(self.latf)):
                        # Not South pole node
                        ## lower right node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, coords1[1]+reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        ## lower left node
                        condition = (pos.T==np.array( [coords1[0]-reduceRes, coords1[1]-reduceRes] ));
                        condition = (condition[:,0]==True) & (condition[:,1]==True);
                        indices.append( nodeCntList[condition] );
                        indicesAddPt.append(10)
                        indicesAddPt.append(10)
                    ## right node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]+reduceRes] ));
                    condition = (condition[:,0]==True) & (condition[:,1]==True);
                    indices.append( nodeCntList[condition] );
                    ## left node
                    condition = (pos.T==np.array( [coords1[0], coords1[1]-reduceRes] ));
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

            # Find nodes with high inbetweenness
            # FIXME:
            

            ###########################
            ### Write network Model ###
            ###########################
            # Write network
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
        colors = [mpl.colors.to_hex(i) for i in mpl.colormaps["tab20"].colors][1:];
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
                                       "connectorlinewidth":1},
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

        ## Add axis
        ax = fig.add_subplot(gs[0], transform=ccrs.PlateCarree(), projection=ccrs.Mollweide());

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
        basincmap =mpl.colors.ListedColormap(np.unique(node_colors));
        if draw['nodes']:
            if draw['connectors']:
                nodeplthandle = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, edgecolor='k', linewidths=pltOpts['connectorlinewidth'], s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            else:
                nodeplthandle = ax.scatter(pos[:,1], pos[:,0], marker='o', c=BasinID, edgecolor=node_colors, linewidths=pltOpts['connectorlinewidth']/4, s=pltOpts['nodesize'], transform=ccrs.PlateCarree(), cmap=basincmap)  # longitude, latitude
            nodeplthandle.set_zorder(11);

        ## Add a colorbar(s)
        if draw['bathymetry']:
            basinIDcbarpad = .0;
        else:
            basinIDcbarpad = .05;
        
        # Basin IDs
        cbar1 = plt.colorbar(nodeplthandle, ax=ax, orientation='horizontal', pad=0.0, aspect=40, shrink=0.7, cmap=basincmap);
        cbar1.set_label(label="Basin ID", size=12);
        cbar1.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks
        basinticklabels = [];
        if not (np.size(np.unique(BasinID))==1):
            cbar1.set_ticks( np.arange(np.max(BasinID)/(2*(np.max(BasinID)+1)), np.max(BasinID), np.max(BasinID)/((np.max(BasinID)+1))) );  # Set custom tick positions
            for basini in np.arange(0, np.max(BasinID)+1, 1):
                basinticklabels.append( "{:0.0f}".format(basini) );
        else:
            cbar1.set_ticks( [0] );  # Set custom tick positions
            basinticklabels.append( "{:0.0f}".format(np.unique(BasinID)[0]) );
        
        cbar1.set_ticklabels( basinticklabels );  # Custom labels

        if draw['bathymetry']:
            # Bathymetry
            cbar2 = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7);
            cbar2.set_label(label="{} [{}]".format(pltOpts['valueType'], pltOpts['valueUnits']), size=12);
            cbar2.ax.tick_params(labelsize=10);  # Adjust the size of colorbar ticks


        if draw['gridlines']:
            ax.gridlines()

        plt.title("Basin Networks ({})".format(self.body))

        # Save figure
        if savePNG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".png")), dpi=600)
        if saveSVG:
            plt.savefig("{}/{}".format(self.dataDir,self.filename.replace(".nc",".svg")))

    def mergeBasins(self, basinID1, basinID2):
        """
        mergeBasins method will merge basins with basin ID of basinID2
        to basins with ID basinID1. The resulting basin network is then
        rewitten, overriding the original basinID network. Note that
        there is no need to reread the basin network. 

        Parameters
        ----------
        basinID1 : INT
            Basin ID to absorb basinID2.
        basinID2 : INT
            Basin ID to be absorbed by basinID1.
        
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
        
        # Change basin ids basinID2 to basinID1 (not in nodes yet)
        nodeBasinID[nodeBasinID==basinID2] = basinID1;

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
        nx.write_gml(self.G, "{}/{}".format(self.dataDir, self.filename.replace(".nc","_basinNetwork.gml")), stringizer=str)



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
        FIXME: ADD
    coords2 : INT
        FIXME: ADD
    bathyAve : BOOLEAN
        FIXME: ADD
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