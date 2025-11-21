#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:20:00 2024

@author: Matthew Bogumil
"""

#######################################################################
############################### Imports ###############################
#######################################################################
from ExoCcycle import utils # type: ignore
from ExoCcycle import plotHelper # type: ignore
import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # used for progress bar
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import copy as cp

#######################################################################
#################### ExoCcycle Create Bathymetries ####################
#######################################################################   
class BathyMeasured():
    """
    Measured topography/bathymetry helper for Venus, Earth, Mars, and Moon.

    This class downloads publicly available planetary topography datasets,
    reprojects / converts them to NetCDF when needed, resamples to a chosen
    angular resolution, and derives a bathymetry model by flooding the
    topography to satisfy either an ocean-area or water-volume constraint.
    It also exports bathymetry grids and related carbon-cycle distributions
    in NetCDF format for downstream use in ExoCcycle.

    Notes
    -----
    - Downloaded sources (indicative):
        * Venus (Magellan): USGS Planetary Data (GeoTIFF → NetCDF) - https://astrogeology.usgs.gov/search/map/venus_magellan_global_c3_mdir_colorized_topographic_mosaic_6600m
        * Venus (PDS IMG): PDS Geosciences Node (IMG → NetCDF) - https://pds-geosciences.wustl.edu/mgn/mgn-v-rss-5-gravity-l2-v1/mg_5201/images/topogrd.img
        * Earth: ETOPO1 (NetCDF/GRD) - https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/cell_registered/netcdf/
        * Mars: MOLA (NetCDF) - https://github.com/andrebelem/PlanetaryMaps/raw/v1.0/mola32.nc
        * Moon: Planetary Geodesy Data Archive (PGDA) - https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_pixel_202405.grd
    - Several steps invoke external CLI tools (GMT, GDAL, wget). Ensure they
      are installed and available on ``PATH``.
    - Longitudes are normalized to ``[-180, 180]`` and latitudes to
      ``[-90, 90]``; cell registration is preserved during resampling.

    Attributes (set after initialization or workflow steps)
    -------------------------------------------------------
    model : str
        One of ``{"Venus2","Venus","Earth","Mars","Moon"}``.
    fidName : str
        Canonical filename of the raw source grid for the selected body.
    variables : dict
        Mapping of variable names present in the source NetCDF/GRD/IMG
        (e.g., ``{"lat":"lat", "lon":"lon", "elev":"z"}``).
    initiallykm : bool
        Whether the input elevation is given in kilometers (converted to meters).
    radiuskm : float
        Planetary radius (km) used for area computations.
    highlatP : float
        Decimal fraction of ocean area designated as “high-latitude box”
        for LOSCAR-style parameterization.
    resolution : float
        Target angular resolution (degrees) set by :meth:`readTopo`.
    data_dir : str
        Root directory used to store inputs/outputs.
    lon, lat : ndarray
        2-D cell-registered longitudes/latitudes (deg) for resampled grids.
    elev : ndarray
        2-D topography (meters; positive above sea level).
    bathymetry : ndarray
        2-D seafloor depth (meters; positive), land masked as NaN.
    areaWeights : ndarray
        2-D area weight per cell (m²) for the selected radius/resolution.
    AOC, VOC : float
        Ocean surface area (m²) and ocean volume (m³) of derived bathymetry.
    highlatA : float
        High-latitude ocean area (m²) implied by ``highlatP``.
    highlatlat : float
        Latitude cutoff (deg) delimiting the high-latitude region.
    bathymetryAreaDist, bathymetryAreaDist_wHighlat : ndarray
        Kernelized/semi-binned bathymetry distributions (global w/o and with
        high-latitude contribution).
    binEdges : ndarray
        Bin edges (km) used for bathymetry distributions.
    nc, bathync : netCDF4.Dataset
        Open handles to resampled topography and saved bathymetry NetCDFs
        (when used).
    """    
    '''
    Measured topography (~bathymetry) Venus/Mars/Moon/present-day Earth.

    getTopo is a method to download different topography models.
        getTopo(self, body = {'Mars':'True', 'Venus':'False', 'Moon':'False', 'Earth':'False'}):


    
    readTopo is a method to read in a downloaded topography models. They are interpolated and
    saved to the same input directory. Topography model was already saved with a chosen resolution
    then this model is read instead.
        readTopo(self, data_dir, new_resolution = 1, verbose=True):
    
    setSeaLevel is a method to 
        setSeaLevel(self, basinVolume = {"on":True, 'uncompactedVol':None}, oceanArea = {"on":True, "area":0.7}, isostaticCompensation = {"on":False}):
        1. Might be useful for flexural Isostasy:
            https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JB014571
            https://pages.uoregon.edu/rdorsey/BasinAnalysis/AngevineEtal1990/Chapt%205%20Flexure.pdf
        2. Might be useful for the Isostasy
            https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073334

    saveBathymetry is a method used to save bathymetry created after running setSeaLevel
        saveBathymetry(self, filePath):

    loadBathymetry is a method used to load bathymetry created after running setSeaLevel() and saveBathymetry()
        loadBathymetry(self, filePath):

    '''
    
    def __init__(self, body=None):
        """
        Initialize a planetary dataset preset.

        Parameters
        ----------
        body : {"Venus2","Venus","Earth","Mars","Moon"} or None
            Planetary body/preset to configure. Sets source filenames,
            variable names, unit flags, planet radius, and default
            high-latitude fraction. If ``None`` or unrecognized, prints a
            helpful message and leaves the object unconfigured.

        Side Effects
        ------------
        Sets attributes: ``model``, ``fidName``, ``variables``, ``initiallykm``,
        ``radiuskm``, ``highlatP`` based on selection.

        Notes
        -----
        - ``"Venus2"`` refers to the Magellan C3-MDIR colorized topographic
          mosaic (USGS; GeoTIFF).
        - ``"Venus"`` refers to the PDS ``topogrd.img`` product.
        """

        if body == None:
            print('No body was chosen. Define input body as "Venus" | "Earth" | "Mars" | "Moon"');
        else:
            if body.lower() == "venus2":
                self.model = "Venus2";
                self.fidName = "Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc";
                self.variables = {"lat":"lat", "lon":"lon", "elev":["Band1","Band2","Band3"]};
                self.initiallykm = False;
                self.radiuskm = 6051.8;
                self.highlatP = .10; # this is the hb[10] value from LOSCAR.
            elif body.lower() == "venus":
                self.model = "Venus";
                self.fidName = "topogrd.nc";
                self.variables = {"lat":"lat", "lon":"lon", "elev":"elev"};
                self.initiallykm = True;
                self.radiuskm = 6051.8;
                self.highlatP = .10; # this is the hb[10] value from LOSCAR.
            elif body.lower() == "earth":
                self.model = "Earth";
                self.fidName = "ETOPO1_Ice_c_gdal.nc";
                self.variables = {"lat":"lat", "lon":"lon", "elev":"z"};
                self.initiallykm = False;
                self.radiuskm = 6371.0;
                self.highlatP = .10; # this is the hb[10] value from LOSCAR.
            elif body.lower() == "mars":
                self.model = "Mars";
                self.fidName = "mola32.nc";
                self.variables = {"lat":"latitude", "lon":"longitude", "elev":"alt"};
                self.initiallykm = False;
                self.radiuskm = 3389.5;
                self.highlatP = .10; # this is the hb[10] value from LOSCAR.
            elif body.lower() == "moon":
                self.model = "Moon";
                self.fidName = "LDEM64_PA_pixel_202405.nc";
                self.variables = {"lat":"lat", "lon":"lon", "elev":"z"};
                self.initiallykm = True;
                self.radiuskm = 1737.4;
                self.highlatP = .10; # this is the hb[10] value from LOSCAR.
            else:
                print('Chosen a implemented body. Define input body as "Venus" | "Earth" | "Mars" | "Moon"');

    def getTopo(self, data_dir, verbose=True):
        """
        Download (if needed) and convert the selected body's topography.

        The method creates a structured directory tree under
        ``{data_dir}/topographies/{Body}``, pulls remote datasets using
        ``wget``, converts GRD/IMG/GeoTIFF to NetCDF (via GDAL/GMT), and
        optionally produces a quicklook PostScript using GMT.

        Parameters
        ----------
        data_dir : str
            Root directory for data storage. Subfolders will be created per body.
        verbose : bool, optional
            If True, emits GMT preview commands to generate simple plots.

        Returns
        -------
        None

        Creates
        -------
        {data_dir}/topographies/{Body}/<source>.nc
            NetCDF representation of the raw/global topography.
        {data_dir}/topographies/{Body}/{Body}.ps
            (Optional) Quicklook GMT plot when ``verbose=True``.

        Dependencies
        ------------
        Requires external tools: ``wget``, ``gdalwarp``, ``gdal_translate``,
        and/or ``gmt`` depending on the source.
        """
        # Make directory for storing topography model
        utils.create_file_structure([data_dir+"/topographies",
                                     data_dir+"/topographies/Earth",
                                     data_dir+"/topographies/Venus",
                                     data_dir+"/topographies/Mars",
                                     data_dir+"/topographies/Moon"],
                                     root = True,
                                     verbose=verbose)

        # Download model
        if self.model == "Venus2":
            if not os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif".format(data_dir, self.model)):
                os.system("wget -O {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif https://planetarymaps.usgs.gov/mosaic/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif".format(data_dir, self.model));
            if not os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif".format(data_dir, self.model)):
                os.system("export PROJ_IGNORE_CELESTIAL_BODY=YES &&\
                        gdalwarp -t_srs EPSG:4326 {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif".format(data_dir, self.model));
            if not os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc".format(data_dir, self.model)):
                os.system("gdal_translate -of NetCDF {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc".format(data_dir, self.model));
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc -JN0/5i -Crelief -P -K -Vq> {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Venus":
            if not os.path.exists("{0}/topographies/{1}/topogrd.dat".format(data_dir, self.model)):
                os.system("wget -O {0}/topographies/{1}/topogrd.dat https://pds-geosciences.wustl.edu/mgn/mgn-v-rss-5-gravity-l2-v1/mg_5201/topo/topogrd.dat".format(data_dir, self.model));
            if not os.path.exists("{0}/topographies/{1}/topogrd.nc".format(data_dir, self.model)):
                # Write netCDF file
                ## Read .dat

                elevmodel = np.loadtxt("{0}/topographies/{1}/topogrd.dat".format(data_dir, self.model))
                elevmodel = elevmodel.flatten()
                elevmodel = np.flipud(elevmodel.reshape(180,360))
                elevmodel = np.roll(elevmodel, 60)
                resolution = 1;
                lonmodel, latmodel = np.meshgrid(np.arange(-180+resolution/2, 180, resolution),
                                                 np.arange(-90+resolution/2,  90,  resolution))

                #fid = open("{0}/topographies/{1}/topogrd.img".format(data_dir, self.model), 'rb');
                #elevmodel = np.fromfile(fid, dtype=np.uint8);
                #resolution = 1
                #lonmodel, latmodel = np.meshgrid(np.arange(-180, 180, 1), np.arange(-90+resolution/2, 90-resolution/2, resolution) )
                #elevmodel = elevmodel.reshape(180,360);
                ## Make netCDF file
                ncfile = Dataset("{0}/topographies/{1}/topogrd.nc".format(data_dir, self.model), mode='w', format='NETCDF4_CLASSIC') 

                ## Define dimension
                lat_dim = ncfile.createDimension('lat', len(elevmodel[:,0]));        # latitude axis
                lon_dim = ncfile.createDimension('lon', len(elevmodel[0,:]));       # longitude axis
                
                ### Define lat/lon with the same names as dimensions to make variables.
                lat = ncfile.createVariable('lat', np.float32, ('lat',));
                lat.units = 'degrees_north'; lat.long_name = 'latitude';
                lon = ncfile.createVariable('lon', np.float32, ('lon',));
                lon.units = 'degrees_east'; lon.long_name = 'longitude';

                ## Define a 2D variable to hold the elevation data
                elev = ncfile.createVariable('elev',np.float64,('lat','lon'))
                elev.units = 'meters'
                elev.standard_name = 'elevation'
                
                ## Format
                ncfile.title='{} Topography resampled at {:0.0f}deg'.format(self.model, 1)

                ## Populate the variables
                lat[:]  = latmodel[:,0];
                lon[:]  = lonmodel[0,:];
                elev[:] = elevmodel;
                
                ## Close the netcdf
                ncfile.close(); 
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/topogrd.nc -JN0/5i -Crelief -P -K -Vq > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Earth":
            if not os.path.exists("{0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd.gz".format(data_dir, self.model)):
                os.system("wget -O {0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd.gz https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/cell_registered/netcdf/ETOPO1_Ice_c_gdal.grd.gz".format(data_dir, self.model))
            if not os.path.exists("{0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd".format(data_dir, self.model)):
                os.system("yes N | gzip -k -d {0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd.gz".format(data_dir, self.model));
            if not os.path.exists("{0}/topographies/{1}/ETOPO1_Ice_c_gdal.nc".format(data_dir, self.model)):
                os.system("gmt grdconvert {0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd {0}/topographies/{1}/ETOPO1_Ice_c_gdal.nc -fg -Vq".format(data_dir, self.model))
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/ETOPO1_Ice_c_gdal.nc -JN0/5i -Crelief -P -K -Vq > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Mars":
            if not os.path.exists("{0}/topographies/{1}/mola32.nc".format(data_dir, self.model)):
                os.system("wget -O {0}/topographies/{1}/mola32.nc https://github.com/andrebelem/PlanetaryMaps/raw/v1.0/mola32.nc".format(data_dir, self.model))
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/mola32.nc -JN0/5i -Crelief -P -K -Vq > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Moon":
            if not os.path.exists("{0}/topographies/{1}/LDEM64_PA_pixel_202405.grd".format(data_dir, self.model)):
                os.system("wget -O {0}/topographies/{1}/LDEM64_PA_pixel_202405.grd https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_pixel_202405.grd".format(data_dir, self.model));
            if not os.path.exists("{0}/topographies/{1}/LDEM64_PA_pixel_202405.nc".format(data_dir, self.model)):
                os.system("gmt grdconvert {0}/topographies/{1}/LDEM64_PA_pixel_202405.grd {0}/topographies/{1}/LDEM64_PA_pixel_202405.nc -fg -Vq".format(data_dir, self.model));
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/LDEM64_PA_pixel_202405.nc -JN0/5i -Crelief -P -K -Vq > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));

    def readTopo(self, data_dir, new_resolution=1, verbose=True):
        """
        Load and resample topography to a uniform lat/lon grid.

        If a resampled NetCDF at the requested resolution already exists, it
        is read directly; otherwise the raw grid is opened, optionally
        resampled via GMT for very high input resolutions (>3600 columns),
        normalized to conventional lat/lon ranges, and then interpolated
        to a cell-registered grid of spacing ``new_resolution`` degrees
        using nearest-neighbor interpolation.

        Parameters
        ----------
        data_dir : str
            Root directory containing the per-body ``topographies`` folder.
        new_resolution : float, optional
            Output angular resolution in degrees (e.g., 1, 0.5). Default 1.
        verbose : bool, optional
            If True, prints array shapes/ranges and plots a global map.

        Defines
        -------
        self.resolution : float
            Set to ``new_resolution``.
        self.data_dir : str
        self.lon, self.lat : ndarray
            2-D cell-registered longitudes/latitudes (deg).
        self.elev : ndarray
            2-D elevation (meters).
        Also writes the resampled topography NetCDF for reuse.

        Writes
        ------
        {data_dir}/topographies/{Body}/{Body}_resampled_{res}deg.nc

        Returns
        -------
        None

        Notes
        -----
        - Input units in km (e.g., some Venus/Moon products) are converted to meters.
        - Longitudes >180 are wrapped to ``[-180,180]`` and arrays are rolled so that
          longitudes are monotonic.
        - The output grid is cell-registered, spanning the full sphere.
        """
        # Set the object's resolution for topography/bathymetry calculations
        self.resolution = new_resolution;

        # Set the location of data storage
        self.data_dir = data_dir;

        # Define the resampled topography output file name.
        if new_resolution == int(new_resolution):
            resampledTopoPath = "{0}/topographies/{1}/{1}_resampled_{2:0.0f}deg.nc".format(data_dir,  self.model, new_resolution);
        elif new_resolution*10 == int(new_resolution*10):
            resampledTopoPath = "{0}/topographies/{1}/{1}_resampled_{2:0.1f}deg.nc".format(data_dir,  self.model, new_resolution);
        else:
            new_resolution = round(new_resolution,1);
            self.new_resolution = new_resolution;
            print("Resolution was set to: {} degrees".format(self.new_resolution))
            resampledTopoPath = "{0}/topographies/{1}/{1}_resampled_{2:0.1f}deg.nc".format(data_dir,  self.model, new_resolution);

        # If the resampled topography model already exists then read model
        # instead of continuing.
        if os.path.exists(resampledTopoPath):
            # Read resampled topography file
            self.nc = Dataset(resampledTopoPath);

            # Set latitude/longitude/elev
            self.lon, self.lat  = np.meshgrid( np.array(self.nc['lon'][:]), np.array(self.nc['lat'][:]) );
            self.elev = np.array(self.nc['elev'][:]);

            if verbose:
                print("Shape of input topography arrays")
                print(np.shape(self.lon));
                print(np.shape(self.lat));
                print(np.shape(self.elev));
                print((self.lon));
                print((self.lat));
                print((self.elev));
            
            # Plot global topography model
            plotHelper.plotGlobal(self.lat, self.lon, self.elev,
                             cmapOpts={"cmap":"viridis",
                                       "cbar-title":"cbar-title",
                                       "cbar-range":[np.min(np.min(self.elev)),np.max(np.max(self.elev))]},
                             pltOpts={"valueType": "Topography",
                                      "valueUnits": "m",
                                      "plotTitle":"{} resampled at {:0.0f} degree resolution".format(self.model, new_resolution),
                                      "plotZeroContour":False})
            # return
            return
        
        else:
            # Load netCdf file
            TopoPath = "{}/topographies/{}/{}".format(data_dir, self.model, self.fidName);
            self.nc = Dataset(TopoPath);

            # If the grid is very large (higher than 6 arc minute resoluion) then resample
            # using gmt then reload netCDF
            if len( np.array(self.nc[self.variables['elev']][:])[0,:] ) > 3600:
                # Close dataset
                self.nc.close();
                # Resample and reread dataset 
                os.system("gmt grdsample {0} -Rd -I{2}d -rp -G{1} -Vq".format(TopoPath, TopoPath.replace(".nc", "_resampled.nc"), new_resolution))
                TopoPath = TopoPath.replace(".nc", "_resampled.nc");
                self.nc = Dataset(TopoPath);
                # Need to redefine the lat/lon/elevation naming scheme
                self.variables = {"lat":"lat", "lon":"lon", "elev":"z"};

            # Define latitude/longitude/elevation
            self.lat_netcdf  = np.array(self.nc[self.variables['lat']][:]);
            self.lon_netcdf  = np.array(self.nc[self.variables['lon']][:]);
            if self.initiallykm:
                self.elev_netcdf = np.array(self.nc[self.variables['elev']][:])*1e3;
            else:
                self.elev_netcdf = np.array(self.nc[self.variables['elev']][:]);
            
            # Close file
            self.nc.close()

            # Check dimenions of lat/lon make mesh grids if not already
            if len(np.shape(self.lat_netcdf)) == 1:
                # lat and lon are vectors, not arrays

                # Make lat and lon into arrays
                self.lon_netcdf, self.lat_netcdf = np.meshgrid(self.lon_netcdf, self.lat_netcdf);
            
            # Make latitude and longitude defined on [-90,90] and [0,360], respectively.
            if np.max(np.max(self.lat_netcdf)) > 90:
                print('changing latitude')
                self.lat_netcdf[self.lat_netcdf<90] = 90 - self.lat_netcdf[self.lat_netcdf<90];
                self.lat_netcdf[self.lat_netcdf>90] = self.lat_netcdf[self.lat_netcdf>90] - 90;

            if np.max(np.max(self.lon_netcdf)) > 180:
                print('changing longitude')
                self.lon_netcdf[self.lon_netcdf>180] = self.lon_netcdf[self.lon_netcdf>180] - 360;

                # Roll all arrays such that they start and end at -180, and 180 longitude, respectively.
                columns = np.argwhere(self.lon_netcdf[0,:]==np.max(self.lon_netcdf[0,:]) )[0][0];
                self.lon_netcdf  = np.roll(self.lon_netcdf, columns+1);
                #self.lat_netcdf  = np.roll(self.lat_netcdf, columns+1);
                self.elev_netcdf = np.roll(self.elev_netcdf, columns+1);
            
            # Create grid to interpolate to
            X = np.arange(-180+new_resolution/2, 180+new_resolution/2, new_resolution)
            Y = np.arange(90-new_resolution/2, -90-new_resolution/2, -new_resolution)
            if Y[0] != -Y[-1]:
                Y = np.array([new_resolution/2, -new_resolution/2])
                while (Y[0]+new_resolution)<90:
                    Y = np.append( Y[0]+new_resolution, Y)
                while -90 < (Y[-1]-new_resolution):
                    Y = np.append( Y, Y[-1]-new_resolution)
            if X[0] != -X[-1]:
                X = np.array([-new_resolution/2, new_resolution/2])
                while -180<(X[0]-new_resolution):
                    X = np.append( X[0]-new_resolution, X)
                while (X[-1]+new_resolution)<180:
                    X = np.append( X, X[-1]+new_resolution)

            self.lon, self.lat = np.meshgrid(X, Y)
            #self.lon, self.lat = np.meshgrid(np.arange(-180+new_resolution/2, 180+new_resolution/2, new_resolution), np.arange(90-new_resolution/2, -90-new_resolution/2, -new_resolution) )

            # Interpolate new latitude/longitude/elevation
            if verbose:
                print("Shape of input topography arrays")
                print(np.shape(self.lon_netcdf));
                print(np.shape(self.lat_netcdf));
                print(np.shape(self.elev_netcdf));
                print((self.lon_netcdf));
                print((self.lat_netcdf));
                print((self.elev_netcdf));
            
            interp = NearestNDInterpolator(list(zip(self.lon_netcdf.flatten(), self.lat_netcdf.flatten())), self.elev_netcdf.flatten());
            self.elev = interp(self.lon, self.lat);


            # Ensure latitude and longitude range from negative to positive values
            if not self.lat[:,0][0] < self.lat[:,0][-1]:
                self.lat  = np.flipud(self.lat);
                self.elev = np.flipud(self.elev);
            if not self.lon[0,:][0] < self.lon[0,:][-1]:
                self.lon  = np.fliplr(self.lon);
                self.elev = np.fliplr(self.elev);
    
            # Write resampled topography to be read for later use.
            
            # Make ncfile
            ncfile = Dataset(resampledTopoPath, mode='w', format='NETCDF4_CLASSIC') 

            # Define dimension
            lat_dim = ncfile.createDimension('lat', len(self.elev[:,0]));        # latitude axis
            lon_dim = ncfile.createDimension('lon', len(self.elev[0,:]));       # longitude axis
            #elev_dim = ncfile.createDimension('elev', self.elev);
            
            # Define lat/lon with the same names as dimensions to make variables.
            lat = ncfile.createVariable('lat', np.float32, ('lat',));
            lat.units = 'degrees_north'; lat.long_name = 'latitude';
            lon = ncfile.createVariable('lon', np.float32, ('lon',));
            lon.units = 'degrees_east'; lon.long_name = 'longitude';

            # Define a 2D variable to hold the elevation data
            elev = ncfile.createVariable('elev',np.float64,('lat','lon'))
            elev.units = 'meters'
            elev.standard_name = 'elevation'
            
            # Format
            ncfile.title='{} Topography resampled at {:0.0f} degrees.'.format(self.model, new_resolution)

            # Populate the variables
            lat[:]  = self.lat[:,0];
            lon[:]  = self.lon[0,:];
            elev[:] = self.elev;
            
            # Close the netcdf
            ncfile.close(); 
             
    def setSeaLevel(self,
                    basinVolume={"on": True, "uncompactedVol": None},
                    oceanArea={"on": True, "area": 0.7},
                    isostaticCompensation={"on": False},
                    verbose=True):
        """
        Derive bathymetry from the resampled topography by flooding.

        One of two constraints is used::
        1) **Area constraint** (``oceanArea['on']``): Flood until oceans cover
           the specified decimal fraction of the globe.
        2) **Volume constraint** (``basinVolume['on']``): Flood until the total
           ocean volume equals ``uncompactedVol`` (m³).

        Optionally, a placeholder is provided for isostatic compensation of
        water loading (not implemented).

        Parameters
        ----------
        basinVolume : dict
            ``{"on": bool, "uncompactedVol": float or None}``. Enable and set
            the target ocean volume (m³).
        oceanArea : dict
            ``{"on": bool, "area": float}``. Enable and set target ocean area
            as a decimal fraction (e.g., 0.7 for 70%).
        isostaticCompensation : dict
            ``{"on": bool}``. If True, calls a placeholder routine
            (currently non-functional).
        verbose : bool, optional
            If True, prints progress when using the volume method.

        Defines
        -------
        self.bathymetry : ndarray
            Seafloor depth (meters, positive); land cells are NaN.
        self.areaWeights : ndarray
            Per-cell area weights (m²) from a spherical model at planet radius.
        self.AOC : float
            Ocean surface area (m²).
        self.VOC : float
            Ocean volume (m³).
        self.highlatlat, self.highlatA : float
            High-latitude cutoff latitude (deg) and high-latitude ocean area (m²),
            via ``calculateHighLatA``.
        self.bathymetryAreaDist, self.bathymetryAreaDist_wHighlat : ndarray
            Global bathymetry distributions (excluding/including high-latitude),
            via ``calculateBathymetryDistributionGlobal``.
        self.binEdges : ndarray
            Bin edges (km) used for distributions.

        Returns
        -------
        None

        Notes
        -----
        - Flooding is applied as an incremental sea-level rise (1 m steps for
          area constraint, 10 m steps for volume constraint) until the target
          is met.
        - Requires helper functions available as ``utils.areaWeights``,
          ``calculateHighLatA`` and ``calculateBathymetryDistributionGlobal``.
        - The isostatic compensation routine is a stub.
        """

        # Define methods for creating bathymetry models.
        def oceanAreaMethod(topography, oceanArea, areaWeights, isostaticCompensation, verbose=True):
            """
            oceanAreaMethod method is use to calculate and define bathymetry as the
            topography flooded until the oceans cover some decimal percentage of 
            the planet.


            Parameters
            ----------
            topography : NUMPY ARRAY
                nx2n array representing cell registered topography, in m.
            oceanArea : DICTIONARY
                Option to define bathymetry by flooding topography until
                oceanArea['area'], decimal percent, of global area is covered
                with oceans.
            areaWeights : NUMPY ARRAY
                An array of global degree to area weights. The size is dependent on
                input resolution. The sum of the array equals 4 pi radius^2 for 
                sufficiently high resolution, in m2.
            isostaticCompensation : DICTIONARY
                An option to apply isostatic compensation to for ocean loading
                on the topography. Option assumes a uniform physical properties
                of the lithosphere.
            verbose : BOOLEAN, optional
                Reports more information about process. The default is True.

            Returns
            -------
            bathymetry : NUMPY ARRAY
                nx2n array representing seafloor depth, in m, with positive values.
                Bathymetry is define by the method of this function. Any areas above
                sea-level are assigned a value of np.nan.
            """
            # Iterate flooding (meter-by-meter) until the desired amount of ocean
            # planetary surface area is flooded.
            calculatedOceanAreaDPercent = 0;
            deepestSeafloor = 0;
            while calculatedOceanAreaDPercent < oceanArea['area']:
                # Add 1 meter of flooding
                deepestSeafloor += 1;
                bathymetry = topography-deepestSeafloor;
                # Do isostatic compensation calculation for ocean loading
                if isostaticCompensation['on']:
                    bathymetry = isostaticCompensationMethod(bathymetry);
                # Calculate ocean area, in m2.
                AOC = np.sum(np.sum( areaWeights[bathymetry<0] ))
                # Calculate ocean area, in decimal percent.
                calculatedOceanAreaDPercent = AOC / np.sum(np.sum( areaWeights ));
        
            # Set topography in bathymetry variable to np.nan.
            bathymetry[bathymetry>=0] = np.nan;

            return np.abs(bathymetry)


        def waterVolumeMethod(topography, basinVolume, areaWeights, isostaticCompensation, verbose=True):
            """
            waterVolumeMethod method is use to calculate and define bathymetry as the
            topography flooded until the oceans contain some input value of ocean
            water, in m3.


            Parameters
            ----------
            topography : NUMPY ARRAY
                nx2n array representing cell registered topography, in m.
            basinVolume : DICTIONARY
                Option to define bathymetry by flooding topography with
                basinVolume['uncompactedVol'] amount of ocean water, in m3.
            areaWeights : NUMPY ARRAY
                An array of global degree to area weights. The size is dependent on
                input resolution. The sum of the array equals 4 pi radius^2 for 
                sufficiently high resolution, in m2.
            isostaticCompensation : DICTIONARY
                An option to apply isostatic compensation to for ocean loading
                on the topography. Option assumes a uniform physical properties
                of the lithosphere.
            verbose : BOOLEAN, optional
                Reports more information about process. The default is True.

            Returns
            -------
            bathymetry : NUMPY ARRAY
                nx2n array representing seafloor depth, in m, with positive values.
                Bathymetry is define by the method of this function. Any areas above
                sea-level are assigned a value of np.nan.
            """
            # Iterate flooding (meter-by-meter) until the desired amount of ocean
            # volume produced.
            VOC = 0;
            deepestSeafloor = 0;
            while VOC < basinVolume['uncompactedVol']:
                # Add 10 meter of flooding
                deepestSeafloor += 10;
                bathymetry = topography-deepestSeafloor;
                # Do isostatic compensation calculation for ocean loading
                if isostaticCompensation['on']:
                    bathymetry = isostaticCompensationMethod(bathymetry);
                # Calculate basin volume, in m3.
                VOC = np.abs( np.sum(np.sum( bathymetry[bathymetry<0]*areaWeights[bathymetry<0] )) );
                if verbose:
                    print("Deepest seafloor is {0:2.0f} m with total basin volume of {1:2.2e} m3".format(deepestSeafloor, VOC))

            # Set topography in bathymetry variable to np.nan.
            bathymetry[bathymetry>=0] = np.nan; 

            return np.abs(bathymetry)            

        def isostaticCompensationMethod(topography,  loadingProp, lithosphereProp):
            """
            isostaticCompensationMethod is a method used to recalculate bathymetry
            due to the loading of lithosphere.

            topography : NUMPY ARRAY
                nx2n array representing cell registered topography, in m. Note
                that there should be some bathymetry (represented with negative
                values in the topography array).
            loadingProp : Dictionary
                A dictionary of loading material properties. This should include the
                following properties
                    'uniformDensity' : loading material density, in kg/m3
            lithosphereProp : DICTIONARY
                A dictionary representing loaded lithosphere properties. 

            
            Returns
            -------
            bathymetry : NUMPY ARRAY
                nx2n array representing seafloor depth, in m, with negative values,
                and topography with positive values.







            1. Might be useful for flexural Isostasy:
                https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JB014571
                https://pages.uoregon.edu/rdorsey/BasinAnalysis/AngevineEtal1990/Chapt%205%20Flexure.pdf
            2. Might be useful for the Isostasy
                https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073334


            """
            print("working progress")

        # Show options chosen for bathymetry model creation
        methodChoice = None;
        methodChoice
        if basinVolume['on']:
            methodChoice = "basin volume constraint";
        if oceanArea['on']:
            if methodChoice is None:
                methodChoice = "basin area constraint";
            else:
                print("{} was also chosen. Using {}.".format(methodChoice));

        
        # Define degree to area weights
        areaWeights, longitudes, latitudes, totalArea, totalAreaCalculated = utils.areaWeights(resolution = self.resolution, radius = self.radiuskm*1e3, verbose=False);
        self.areaWeights = areaWeights;

        # Define initial bathymetry model with minimum elevation set to 0 m (sea-level)
        # Note that topography represents values above sea-level with positive values.
        topography = self.elev - np.min(np.min(self.elev));

        # Feed bathymetry initial model into selected bathymetry calculation function. 
        if methodChoice == "basin volume constraint":
            bathymetry = waterVolumeMethod(topography, basinVolume, areaWeights, isostaticCompensation, verbose = False)
        elif methodChoice == "basin area constraint":
            bathymetry = oceanAreaMethod(topography, oceanArea, areaWeights, isostaticCompensation, verbose = False)

        # Calculate and define properties of bathymetry model
        self.bathymetry = bathymetry;
        self.AOC = np.nansum(np.nansum( areaWeights[~np.isnan(self.bathymetry)] ))
        self.VOC = np.sum(np.sum( (bathymetry*areaWeights)[~np.isnan(self.bathymetry)] ))
        
        ## Sets self.highlatA and self.highlatlat
        self.highlatlat, self.highlatA = calculateHighLatA(self.bathymetry, self.lat, areaWeights, self.highlatP, verbose=False);

        ## Define global distribution of sea seafloor depths
        ## Both self.bathymetryAreaDist and self.bathymetryAreaDist_wHighlat
        ## are define here.
        self.bathymetryAreaDist, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistributionGlobal(self.bathymetry, self.lat, self.highlatlat, areaWeights, binEdges = None, verbose=True);

    def saveBathymetry(self, verbose=True):
        """
        Save the derived bathymetry and summary parameters to NetCDF.

        The output file contains gridded bathymetry plus global area weights,
        bathymetry distributions (with/without high-latitude region),
        bin edges, and scalar integrals (AOC, VOC, etc.).

        Parameters
        ----------
        verbose : bool, optional
            If True, creates the output directory tree if needed
            and prints paths/actions.

        Writes
        ------
        {data_dir}/bathymetries/{Body}/{Body}_resampled_{res}deg.nc
            Variables:
              - ``bathymetry(lat,lon)`` : m
              - ``lat(lat)`` : deg_north
              - ``lon(lon)`` : deg_east
              - ``areaWeights(lat)`` : m² (per-latitude row representative)
              - ``binEdges(binEdges)`` : km
              - ``bathymetry-distribution-G(binEdges)`` : kernel distribution
              - ``bathymetry-distribution-whighlat-G(binEdges)`` : kernel distribution
              - ``highlatlat`` : deg
              - ``highlatA`` : m²
              - ``VOC`` : m³
              - ``AOC`` : m²

        Returns
        -------
        None

        Notes
        -----
        - The latitude/longitude axes are written as vectors paired with
          cell-registered 2-D data.
        - Title metadata encodes the body and resolution.
        """
        # Make directory for storing bathymetry model(s)
        utils.create_file_structure([self.data_dir+"/bathymetries",
                                     self.data_dir+"/bathymetries/Earth",
                                     self.data_dir+"/bathymetries/Venus",
                                     self.data_dir+"/bathymetries/Mars",
                                     self.data_dir+"/bathymetries/Moon"],
                                     root = True,
                                     verbose=verbose)     
        
        # Set netCDF4 filename
        if self.resolution == int(self.resolution):
            BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.1f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        elif self.resolution*10 == int(self.resolution*10):
            BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.1f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        #BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.0f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4_CLASSIC') 

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ncfile.createDimension('lat', len(self.bathymetry[:,0]));     # latitude axis
        lon_dim = ncfile.createDimension('lon', len(self.bathymetry[0,:]));     # longitude axis
        binEdges_dim = ncfile.createDimension('binEdges', len(self.binEdges[1:]));      # distribution
        
        # Define lat/lon with the same names as dimensions to make variables.
        lat = ncfile.createVariable('lat', np.float32, ('lat',));
        lat.units = 'degrees_north'; lat.long_name = 'latitude';
        ncfile.variables['lat'].axis = 'Y';
        ncfile.variables['lat'].actual_range = [-90,90];

        lon = ncfile.createVariable('lon', np.float32, ('lon',));
        lon.units = 'degrees_east'; lon.long_name = 'longitude';
        ncfile.variables['lon'].axis = 'X'
        ncfile.variables['lon'].actual_range = [-180,180];

        # Define a 2D variable to hold the elevation data
        bathy = ncfile.createVariable('bathymetry',np.float64,('lat','lon'))
        bathy.units = 'meters'
        bathy.standard_name = 'bathymetry'

        # Define vector as function with longitude dependence
        areaWeights = ncfile.createVariable('areaWeights',np.float64,('lat',))
        areaWeights.units = 'meters sq'
        areaWeights.standard_name = 'areaWeights'

        # Define variables for bathymetry distributions (vectors)
        binEdges = ncfile.createVariable('binEdges', np.float32, ('binEdges',));
        binEdges.units = 'km'; binEdges.long_name = 'km depth';

        distribution_whighlat = ncfile.createVariable('bathymetry-distribution-whighlat-G', np.float64, ('binEdges',))
        distribution_whighlat.units = 'kernel distribution'
        distribution_whighlat.standard_name = 'bathymetry-distribution-whighlat-G'

        distribution = ncfile.createVariable('bathymetry-distribution-G', np.float64, ('binEdges',))
        distribution.units = 'kernel distribution'
        distribution.standard_name = 'bathymetry-distribution-G'

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
        
        # Format title
        ncfile.title='{} Bathymetry created from topography resampled at {:0.1f} degrees.'.format(self.model, self.resolution)

        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        bathy[:] = self.bathymetry;
        areaWeights[:] = self.areaWeights[:,0];

        # Add bathymetry distribution information
        distribution_whighlat[:] = self.bathymetryAreaDist_wHighlat;
        distribution[:] = self.bathymetryAreaDist;
        binEdges[:] = self.binEdges[1:];

        # Add attributes
        highlatlat[:] = self.highlatlat;
        highlatA[:] = self.highlatA;
        VOC[:] = self.VOC;
        AOC[:] = self.AOC;

        # Close the netcdf
        ncfile.close();

    def readBathymetry(self, verbose=True):
        """
        Open a previously saved bathymetry NetCDF for this body/resolution.

        Parameters
        ----------
        verbose : bool, optional
            Unused here (reserved for symmetry with other methods).

        Returns
        -------
        None

        Side Effects
        ------------
        Opens and stores a handle to the dataset at
        ``self.bathync = Dataset(path, 'r')``. The path is resolved using
        ``self.data_dir``, ``self.model``, and ``self.resolution``.
        """
        if self.resolution == int(self.resolution):
            BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.0f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        elif self.resolution*10 == int(self.resolution*10):
            BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.1f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        
        # Make new .nc file
        self.bathync = Dataset(BathyPath, mode='r', format='NETCDF4_CLASSIC') 


class BathyRecon():
    """
    Reconstructs global Earth bathymetry through time using a workflow
    modeled after Bogumil et al. (2024, PNAS, doi:10.1073/pnas.2400232121).

    Pipeline (conceptual)
    ---------------------
    1. Convert reconstructed oceanic lithosphere age (paleo-isochrons)
       to first-order bathymetry via a chosen age–depth relation.
    2. Add isostatically compensated deep-marine sediment thickness
       using an empirical age/latitude relationship.
    3. Fill continental shelves, flooded continents, and deep marine
       not represented by plate reconstructions with paleoDEMs.
    4. Apply a eustatic sea-level adjustment (Haq87 long-term curve).
    5. Deform the global bathymetry distribution to match present-day
       uncertainty and reproduce target ocean-container volumes (VOC).

    Inputs
    ------
    This class expects directory trees with NetCDF grids for:
      • paleoDEMs:  files named like  <prefix>_<Age>Ma.nc
      • oceanLith:  files named like  <prefix>-<Age>.nc
      • etopo:      present-day topography/bathymetry NetCDF (single file)

    Attributes (high level)
    -----------------------
    directories : dict
        Paths for 'paleoDEMs', 'oceanLith', and 'etopo'.
    ESL : pandas.DataFrame
        Long-term Haq87 eustatic sea-level curve with columns ['Ma','m'].
    radiuskm : float
        Earth radius (km), default 6371.0.
    thermalSubMethod : dict
        Thermal subsidence method settings (see `setThermalMethod`).
    VOCValuesPD, VOCValues, VOCAgeValues : ndarray
        Present-day and time-dependent target ocean volumes and their ages.
    paleoDEMsfids, paleoDEMsAges : ndarray
        PaleodeM filenames and their parsed ages (Ma).
    oceanLithfids, oceanLithReconAges : ndarray
        Lithosphere-age filenames and their parsed ages (Ma).
    etopofid : str
        Filename for present-day ETOPO/Topo NetCDF.

    Notes
    -----
    • Many steps call external tools (`gmt`, `wget`) and assume they are on PATH.
    • Numerous intermediate arrays (e.g., `self.topography`, `self.bathymetry`,
      `self.areaWeights`, etc.) are defined during `run` or helper methods.
    """

    def __init__(self, directories):
        """
        Initialize the reconstruction with source directories and sea-level curve.

        Parameters
        ----------
        directories : dict
            Must contain:
              - 'paleoDEMs' : path to paleo digital elevation models (NetCDF).
              - 'oceanLith' : path to ocean lithosphere age grids (NetCDF).
              - 'etopo'     : path to directory with a single present-day NetCDF.

            File naming conventions are assumed:
              • paleoDEMs: <prefix>_<Age>Ma.nc (Age as float/int)
              • oceanLith: <prefix>-<Age>.nc   (Age as float/int)

        Defines
        -------
        self.ESL : pandas.DataFrame
            Haq87 long-term sea-level curve with columns ['Ma','m'].
        self.directories : dict
            Stored copy of input directories.
        self.paleoDEMsfids, self.paleoDEMsAges : ndarray
            Filenames and parsed ages (Ma) for paleoDEMs.
        self.oceanLithfids, self.oceanLithReconAges : ndarray
            Filenames and parsed ages (Ma) for lithosphere age grids.
        self.etopofid : str
            Resolved present-day ETOPO/Topo filename (first if multiple found).
        self.radiuskm : float
            Earth radius (km).
        self.thermalSubMethod : dict
            Default thermal subsidence method {'type': 'CM2009'}.
        self.VOCValuesPD, self.VOCValues, self.VOCAgeValues : ndarray
            Default VOC targets (constant by default).

        Notes
        -----
        Expects the included Haq87 curve at `IncludedData/Haq87_SealevelCurve_Longterm.dat`.
        """
        # Set the directories to read 
        filenameESL     = '/IncludedData/Haq87_SealevelCurve_Longterm.dat';
        pathScript      = os.path.dirname(os.path.realpath(__file__));

        # Read ESL curve 
        self.ESL = pd.read_csv(pathScript+filenameESL, sep=' ', names=['Ma','m'], index_col=False)

        # Define the directories dictionary attribute and check that appropriate paths
        # have been defined.
        self.directories = directories;

        # Check that all directories exist
        for path in self.directories:
            try: os.path.isdir(path)
            except: print("{} does not exist".format(path))

        # Set paleoDEM directory/file information
        ## Set filenames
        self.paleoDEMsfids = np.array([i for i in os.listdir(self.directories['paleoDEMs']) if not ('.cache' in i) and ('.nc' in i)]);
        ## Set reconstruction ages
        self.paleoDEMsAges = np.array([float(i.split('Ma.nc')[0].split('_')[-1]) for i in self.paleoDEMsfids]);

        # Set paleoDEM directory/file information
        ## Set filenames
        self.oceanLithfids = np.array([i for i in os.listdir(self.directories['oceanLith']) if not ('.cache' in i) and ('.nc' in i)]);
        ## Set reconstruction ages
        self.oceanLithReconAges = np.array([float(i.split('.nc')[0].split('-')[-1]) for i in self.oceanLithfids]);

        # Set etopo directory/file information
        ## Set filenames
        self.etopofid = np.array([i for i in os.listdir(self.directories['etopo']) if not ('.cache' in i) and ('.nc' in i)]);
        if len(self.etopofid) == 1:
            self.etopofid = self.etopofid[0];
        else:
            print("Multiple netCDF4 files were read from the given etopo directory: {0}.\n{1} will be read and used as the present-day topography throughout this analysis".format(self.etopofid, self.etopofid[0]))
            self.etopofid = self.etopofid[0];
        
        # Set the radius of planet
        self.radiuskm = 6371.0;

        # Set the default thermal subsidence method
        self.thermalSubMethod = {'type':"CM2009"};
    
        # Ocean basin volume is defined based on analysis from Bogumil et al. (2024) https://doi.org/10.1073/pnas.2400232121.
        constVOC = True;
        self.VOCValuesPD      = np.array([1.3350350e+18]); # From downsampled topo6 as defined in https://doi.org/10.1073/pnas.2400232121
        if constVOC:
            print("Ocean basin volume is constant through reconstruction period.")
            self.VOCValues      = cp.deepcopy(self.VOCValuesPD);
            self.VOCAgeValues   = np.arange(0,85,5);
        else:
            # As defined as in Bogumil et al. (2024) https://doi.org/10.1073/pnas.2400232121
            self.VOCValues = 1.3350350e+18; # From downsampled topo6
            self.VOCAgeValues   = np.arange(0,85,5);
            # percent of total surface water stored in glaciers which would be ocean
            # water at studied time period.
            self.VOCValues = np.empty(size=self.VOCAgeValues);
            for i in range(self.VOCAgeValues):
                if self.VOCAgeValues[i] >= 40:
                    per_ice_melt = 2.1;         # [%] - No glaciers
                elif (self.VOCAgeValues[i] >= 5) & (self.VOCAgeValues[i]<40):
                    per_ice_melt = 2.1/2;       # [%] - Half glacier volume
                else:
                    per_ice_melt = 0;           # [%] - Present day glaciers
                self.VOCValues[i] = (100/(100-per_ice_melt))*cp.deepcopy(self.VOCValuesPD);

    def setThermalMethod(self, thermalSubMethod={'type':"CM2009"}, verbose=True):
        """
        Choose the age–depth relationship used for thermal subsidence.

        Parameters
        ----------
        thermalSubMethod : dict, optional
            For 'CM2009':
                {'type': 'CM2009'}
            For 'RK2021':
                {'type': 'RK2021', 'H': <float W/m/K>, 'MORDepthkm': <float>}
            Suggested H values: 2.1e-12 (PD), 3.2e-12 (1.7 Ga), 4.8e-12 (2.8 Ga), 6.4e-12 (3.5 Ga), 8e-12 (3.95 Ga).
        verbose : bool, optional
            If True, prints guidance when H differs from tabulated values.

        Redefines
        ---------
        self.thermalSubMethod : dict
            Method dictionary used by `addThermalSub`.

        Options
        • 'CM2009' (default): Crosby & McKenzie (2009) piecewise age–depth relation.
        • 'RK2021'         : Rosas & Korenaga (2021) parameterized model supporting
                              variable internal heating H and MOR depth.
        """
        # Set method to use for thermal subsidence of ocean lithosphere
        self.thermalSubMethod['type'] = thermalSubMethod['type'];

        if self.thermalSubMethod['type'] == "RK2021":
            # Set interval heating value if RK2021 meethod is used.
            self.thermalSubMethod['H'] = thermalSubMethod['H'];

            # Set mid-ocean-ridge depth
            self.thermalSubMethod['MORDepthkm'] = thermalSubMethod['MORDepthkm'];

            # Report
            if verbose:
                HAvailable = np.array([2.1e-12, 3.2e-12, 4.8e-12, 6.4e-12, 8e-12]);
                ## Find the closest H value from the RK2021 paper
                ## Find the index of that internal heating value
                i = np.argwhere(np.min(np.abs(HAvailable-self.thermalSubMethod['H'])) == np.abs(HAvailable-self.thermalSubMethod['H']))[0][0]

                if 10 < 100*np.abs(HAvailable[i]-self.thermalSubMethod['H'])/self.thermalSubMethod['H']:
                    print("\nUser defined internal heating {:1.1e} [W/m/K] is {:0.0f}% (large/small) than the closest internal heating value used in RK2021 ({:1.1e} [W/m/K]). H = {:1.1e} [W/m/K] will be used.".format(thermalSubMethod['H'],
                                                                                                                                                                                                                       -100*(HAvailable[i]-self.thermalSubMethod['H'])/self.thermalSubMethod['H'],
                                                                                                                                                                                                                       HAvailable[i],
                                                                                                                                                                                                                       HAvailable[i]))
                    print("\nInternal heating values of H [W/m/K] = 2.1e-12 (present-day) 3.2e-12 (1.7 Ga), 4.8e-12 (2.8 Ga), 6.4e-12 (3.5 Ga), and 8e-12 (3.95 Ga) available for analysis.")

    def run(self, startMa=80, endMa=0, deltaMyr=5, resolution=1,
            maxBasinCnt=1e5, findBasins=True, verbose=True):
        """
        Execute the reconstruction loop and save outputs (per time slice).

        Steps (per time slice)
        ----------------------
        1. Read/prepare lithosphere age grid and initialize topography.
        2. Add thermal subsidence via `addThermalSub`.
        3. Add sediment thickness and isostatic compensation.
        4. Merge paleoDEMs for continental/flooded regions.
        5. Apply eustatic sea-level adjustment (Haq87; optionally scaled).
        6. Compute area weights, bathymetry, and global diagnostics (AOC/VOC).
        7. For 'CM2009': optionally apply volume-of-ocean (VOC) correction
           based on present-day mismatch (`addVOCCorrection`).
        8. Identify/merge basins and compute basin parameters/connectivity.
        9. Save bathymetry and basin outputs for ExoCcycle.

        Parameters
        ----------
        startMa : int, optional
            Oldest reconstruction age (Ma). Default 80.
        endMa : int, optional
            Youngest reconstruction age (Ma). Default 0.
        deltaMyr : int, optional
            Temporal step (Myr). Default 5.
        resolution : float, optional
            Spatial grid resolution (degrees). Default 1.
        maxBasinCnt : int, optional
            Maximum number of allowed basins for merging heuristics.
        findBasins : bool, optional
            If True, perform basin identification/merging workflow.
        verbose : bool, optional
            If True, print progress and generate quick looks.

        Side Effects
        ------------
        • Writes per-age NetCDF files via `saveBathymetry`.
        • Creates/updates `self.basins` with computed basin properties.
        • May create plots/diagnostics when `verbose=True`.

        Notes
        -----
        Requires helper utilities from `utils` and external GMT.
        """
        # Define all periods to bathymetry for.
        reconAgeVec = list(np.arange(endMa, startMa+deltaMyr, deltaMyr));

        # Loop over all periods.
        for reconAge in (pbar := tqdm(reconAgeVec)):
            # 1. Add ocean lithosphere age-depth relationship 
            # 1a. Read ocean lithosphere age grid
            self.getOceanLithosphereAgeGrid(age=reconAge, resolution=1, fuzzyAge=False);
            # 1b. Define topography, latitude, and longitude arrays using ocean lithosphere inputs
            self.lon, self.lat = np.meshgrid(self.oceanLithAge['lon'], self.oceanLithAge['lat']);
            self.topography = np.empty(shape=np.shape(self.lon));
            self.topography[:] = np.nan;

            # 1c. Use age grid to calculate seafloor depth from age-depth relationship
            # Note that bathymetry is represented with positive values.
            self.topography = self.addThermalSub(self.topography, self.oceanLithAge['z'][:].data, self.lat, method=self.thermalSubMethod, verbose=False);
            
            # 2. Isostatically compensation (sed thickness, litho age) that also include
            # sediment thickness.
            # Note that bathymetry is represented with positive values.
            self.topography = self.getIsostaticCorrection(self.topography, self.oceanLithAge['z'][:].data, self.lat, self.lon, verbose=False)
            
            # 3. Add paleoDEM
            # 3a. Read paleoDEM (defined as self.paleoDEM)
            self.getDEM(age=reconAge, resolution=1, fuzzyAge=False);

            # 3b. Add paleoDEM
            # Note that bathymetry will now be represented with negative values.
            self.topography = -self.topography;
            self.topography[np.isnan(self.oceanLithAge['z'][:].data)] = self.paleoDEM['z'][:].data[np.isnan(self.oceanLithAge['z'][:].data)]

            # 4. Add eustatic sea-level curve to represent flooding.
            # Note that this change in sea-level is only applied over ocean
            # seafloor modeled with seafloor ages (i.e., seafloor represented
            # by paleoDEMs is not changed). The sealevel is also applied at
            # 65% of its value which is consistent with continental flooding
            # (see https://www.earthbyte.org/webdav/ftp/Data_Collections/Scotese_Wright_2018_PaleoDEM/Scotese_Wright2018_PALEOMAP_PaleoDEMs.pdf) 
            if self.thermalSubMethod['type'] == "CM2009":
                self.topography, ESLi = self.getESL(self.topography, self.oceanLithAge['z'][:].data, reconAge, factor=.65, verbose=False);

            # 5. Close the ocean lithospheric age grids netCDF4
            self.oceanLithAge.close()

            # 6. Create global area weights array
            if reconAge == reconAgeVec[0]:
                areaWeights, longitudes, latitudes, totalArea, totalAreaCalculated = utils.areaWeights(resolution = 1, radius = self.radiuskm*1e3, verbose=False);
                self.areaWeights = areaWeights;
            
            # 7. Define bathymetry
            self.bathymetry = cp.deepcopy(self.topography);
            self.bathymetry[self.bathymetry>0] = np.nan;
            self.bathymetry = (-1)*self.bathymetry

            # 8. Calculate and define properties of bathymetry model
            self.AOC = np.nansum(np.nansum( areaWeights[~np.isnan(self.bathymetry)] ))
            self.VOC = np.sum(np.sum( (self.bathymetry*areaWeights)[~np.isnan(self.bathymetry)] ))
            
            ## Sets self.highlatA and self.highlatlat
            self.highlatP = .10; # This is the hb[10] value from LOSCAR, % seafloor area in high latitude box.
            self.highlatlat, self.highlatA = calculateHighLatA(self.bathymetry, self.lat, areaWeights, self.highlatP, verbose=False);

            ## Define global distribution of sea seafloor depths
            ## Both self.bathymetryAreaDist and self.bathymetryAreaDist_wHighlat
            ## are define here.
            self.bathymetryAreaDist, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistributionGlobal(self.bathymetry, self.lat, self.highlatlat, areaWeights, binEdges = None, verbose=True);


            if self.thermalSubMethod['type'] == "RK2021":
                # 9. Save bathymetry model w/o the ocean volume corrections
                self.data_dir = os.getcwd();
                self.resolution = resolution;
                self.model = "EarthRecon3BasinsRK2021_"+("H_{:2.2e}".format( self.thermalSubMethod['H'] )).replace(".",',');

                ## Create directory for storing bathymetry models
                utils.create_file_structure(list_of_directories=['/bathymetries/'+self.model])

                self.saveBathymetry(reconAge, verbose=True);

                # 10. Find basins (Note that this is a partially manual process)
                ## Define basins class for finding basins
                basins = utils.Basins(dataDir=os.getcwd()+"/bathymetries/{}".format(self.model),
                                    filename="{}_{}deg_{}Ma.nc".format(self.model, resolution, reconAge),
                                    body=self.model);

                # Define basins based on user input boundaries
                # If the file exist then read file, otherwise write
                if os.path.isfile("{}/{}".format(basins.dataDir, basins.filename.replace(".nc","_basinNetwork.gml"))):
                    basins.defineBasins(minBasinCnt = 3,
                                        method = "Louvain",
                                        reducedRes={"on":True,"factor":1},
                                        read=True,
                                        write=False,
                                        verbose=False)
                else:
                    basins.defineBasins(minBasinCnt = 3,
                                        method = "Louvain",
                                        reducedRes={"on":True,"factor":1},
                                        read=False,
                                        write=True,
                                        verbose=False)
                
                basins.applyMergeBasinMethods(reconAge,
                                                utils.mergerPackages(self.model),
                                                maxBasinCnt=maxBasinCnt);

                
                # Assign basins as a BathyRecon class attribute.
                self.basins = basins;
                        

            elif self.thermalSubMethod['type'] == "CM2009":
                # 9. Save bathymetry model w/o the ocean volume corrections
                # FIXME: Could to define this at a higher level.
                self.data_dir = os.getcwd();
                self.resolution = resolution;
                self.model = "EarthRecon3Basins"
                #self.model = "EarthRecon3_4Basins"

                ## Create directory for storing bathymetry models
                utils.create_file_structure(list_of_directories=['/bathymetries/'+self.model])

                self.saveBathymetry(reconAge, verbose=True);


                # 10. Find basins (Note that this is a partially manual process)
                ## Define basins class for finding basins
                basins = utils.Basins(dataDir=os.getcwd()+"/bathymetries/{}".format(self.model),
                                    filename="{}_{}deg_{}Ma.nc".format(self.model, resolution, reconAge),
                                    body=self.model);

                # Define basins based on user input boundaries
                # If the file exist then read file, otherwise write
                if os.path.isfile("{}/{}".format(basins.dataDir, basins.filename.replace(".nc","_basinNetwork.gml"))):
                    basins.defineBasins(minBasinCnt = 3,
                                        method = "Louvain",
                                        reducedRes={"on":True,"factor":1},
                                        read=True,
                                        write=False,
                                        verbose=False)
                else:
                    basins.defineBasins(minBasinCnt = 3,
                                        method = "Louvain",
                                        reducedRes={"on":True,"factor":1},
                                        read=False,
                                        write=True,
                                        verbose=False)
                
                try:
                    basins.applyMergeBasinMethods(reconAge,
                                                  utils.mergerPackages("{}_{}".format(self.model, self.thermalSubMethod['type'])),
                                                  maxBasinCnt=maxBasinCnt);
                except:
                    pass
                
                # Assign basins as a BathyRecon class attribute.
                self.basins = basins;
                if findBasins==True:
                    continue;

                # 11. Apply ocean volume correction based on is representation of present-day
                # bathymetry.
                # 11a. Set the ocean volume expected at the reconstruction period.
                self.VOCTarget = self.getVOCi(reconAge);

                # 11b. Apply the ocean volume correction based on the misfit of present-day
                # distributions and the expected paleo ocean volume.            
                if reconAge == 0:
                    self.bathymetry, self.sxbin_p = self.addVOCCorrection(reconAge, self.bathymetry, self.VOCTarget, resolution=1, verbose=True)
                else:
                    try:
                        # If self.etopoKernelDis was defined (i.e., the present-day analysis was previously done) then
                        # the volume correction will be applied to the topography model.
                        

                        # Apply the ocean volume correction
                        self.bathymetry, self.sxbin_p = self.addVOCCorrection(reconAge, self.bathymetry, self.VOCTarget, resolution=1, verbose=True)
                    except:
                        # Present-day analysis was never done, so self.etopoKernelDis is not defined and will not be applied.
                        pass
                
                # 11c. Assign the VOC corrected bathymetry to the basins object
                self.basins.bathymetry = self.bathymetry;
            

            # 12. Save bathymetry model (w/ the ocean volume corrections) in standardized ExoCcycle outputs.

            # 12a. Calculate basin bathymetry parameters
            # Note: the basin bathymetry distributions will be calculated
            # with volume corrected bathymetry since self.basins.bathymetry
            # was updated with the ocean basin volume corrected bathymetry.
            self.basins.calculateBasinParameters(verbose=True)

            # 12b. Calculate basin connectivity parameters
            # Note: the ocean basin volume correction is applied but basin
            # connectivity parameters are calculated with the original
            # (non-VOC corrected) bathymetry. Therefore, there will be some
            # inconsistencies between basin connectivity parameters and
            # bathymetry distributions.
            self.basins.calculateBasinConnectivityParameters(verbose=True)

            # 12c. Expand original bathymetry netCDF4 file by writting a new basin bathymetry
            # netCDF4 file that also contains basin bathymetry parameters.
            self.basins.saveCcycleParameter(verbose=True);

            # Report
            if verbose:
                blues_cm = mpl.colormaps['Blues'].resampled(100)
                self.highlatlat = 90
                plotHelper.plotGlobalwHist(self.lat, self.lon, self.bathymetry,
                                     self.binEdges, self.bathymetryAreaDist_wHighlat, self.bathymetryAreaDist, self.highlatlat,
                                     outputDir = os.getcwd(),
                                     fidName = "plotGlobal-Test.png",
                                     cmapOpts={"cmap":blues_cm,
                                               "cbar-title":"cbar-title",
                                                "cbar-range":[np.nanmin(np.nanmin(self.bathymetry)),
                                                              np.nanmean(self.bathymetry)+2*np.nanstd(self.bathymetry)]},
                                     pltOpts={"valueType": "Bathymetry",
                                              "valueUnits": "m",
                                              "plotTitle":"",
                                              "plotZeroContour":False},
                                     saveSVG=False,
                                     savePNG=False)
            
    def getVOCi(self, age):
        """
        Return the target global ocean-container volume (VOC) at a given age.

        Parameters
        ----------
        age : int
            Time (Ma) for which to return VOC.

        Returns
        -------
        float
            Target ocean basin volume at `age` (m³). If a constant VOC is used,
            returns a scalar independent of age.
        """
        if len(self.VOCValues) == 1:
            return self.VOCValues[0];
        else:
            return self.VOCValues[self.VOCAgeValues == age];

    def saveBathymetry(self, reconAge, verbose=True):
        """
        Save the (optionally VOC-corrected) bathymetry and diagnostics to NetCDF.

        File layout
        -----------
        Dimensions:
            lat (deg)   : [-90, 90]
            lon (deg)   : [-180, 180]
            binEdges    : upper bound edges (km) for depth bins

        Variables:
            bathymetry(lat,lon)                           : m (positive depth; land=NaN)
            lat(lat), lon(lon)                            : deg
            areaWeights(lat)                              : m² (per-lat row representative)
            binEdges(binEdges)                            : km
            bathymetry-distribution-G(binEdges)           : % area (kernel)
            bathymetry-distribution-whighlat-G(binEdges)  : % area (kernel; incl. high-lat)
            highlatlat                                    : deg
            highlatA                                      : m²
            AOC                                           : m²
            VOC                                           : m³

        Parameters
        ----------
        reconAge : int
            Reconstruction age (Ma) to encode in filename.
        verbose : bool, optional
            If True, ensures directories exist and prints path.

        Returns
        -------
        None
        """
        # Make directory for storing bathymetry model(s)
        utils.create_file_structure([self.data_dir+"/bathymetries",
                                     self.data_dir+"/bathymetries/EarthRecon",
                                     self.data_dir+"/bathymetries/EarthRecon3Basins",
                                     self.data_dir+"/bathymetries/EarthRecon3_4Basins"],
                                     root = True,
                                     verbose=verbose)     
        
        # Set netCDF4 filename
        BathyPath = "{0}/bathymetries/{1}/{1}_{2:0.0f}deg_{3:0.0f}Ma.nc".format(self.data_dir,  self.model, self.resolution, reconAge);
        
        # Make new .nc file
        ncfile = Dataset(BathyPath, mode='w', format='NETCDF4_CLASSIC') 

        # Define dimension (latitude, longitude, and bathymetry distributions)
        lat_dim = ncfile.createDimension('lat', len(self.bathymetry[:,0]));     # latitude axis
        lon_dim = ncfile.createDimension('lon', len(self.bathymetry[0,:]));     # longitude axis
        binEdges_dim = ncfile.createDimension('binEdges', len(self.binEdges[1:]));      # distribution
        
        # Define lat/lon with the same names as dimensions to make variables.
        lat = ncfile.createVariable('lat', np.float32, ('lat',));
        lat.units = 'degrees_north'; lat.long_name = 'latitude';
        lon = ncfile.createVariable('lon', np.float32, ('lon',));
        lon.units = 'degrees_east'; lon.long_name = 'longitude';

        # Define a 2D variable to hold the elevation data
        bathy = ncfile.createVariable('bathymetry',np.float64,('lat','lon'))
        bathy.units = 'meters'
        bathy.standard_name = 'bathymetry'

        # Define vector as function with longitude dependence
        areaWeights = ncfile.createVariable('areaWeights',np.float64,('lat',))
        areaWeights.units = 'meters sq'
        areaWeights.standard_name = 'areaWeights'

        # Define variables for bathymetry distributions (vectors)
        binEdges = ncfile.createVariable('binEdges', np.float32, ('binEdges',));
        binEdges.units = 'km'; binEdges.long_name = 'km depth';

        distribution_whighlat = ncfile.createVariable('bathymetry-distribution-whighlat-G', np.float64, ('binEdges',))
        distribution_whighlat.units = 'kernel distribution'
        distribution_whighlat.standard_name = 'bathymetry-distribution-whighlat-G'

        distribution = ncfile.createVariable('bathymetry-distribution-G', np.float64, ('binEdges',))
        distribution.units = 'kernel distribution'
        distribution.standard_name = 'bathymetry-distribution-G'

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
        
        # Format title
        ncfile.title='{} Bathymetry created from topography resampled at {:0.0f} degrees.'.format(self.model, self.resolution)

        # Populate the variables
        lat[:]  = self.lat[:,0];
        lon[:]  = self.lon[0,:];
        bathy[:] = self.bathymetry;
        areaWeights[:] = self.areaWeights[:,0];

        # Add bathymetry distribution information
        distribution_whighlat[:] = self.bathymetryAreaDist_wHighlat;
        distribution[:] = self.bathymetryAreaDist;
        binEdges[:] = self.binEdges[1:];

        # Add attributes
        highlatlat[:] = self.highlatlat;
        highlatA[:] = self.highlatA;
        VOC[:] = self.VOC;
        AOC[:] = self.AOC;

        # Close the netcdf
        ncfile.close();

    def getDEM(self, age, resolution, fuzzyAge=False):
        """
        Load a paleoDEM for the requested time and resample to the target grid.

        Parameters
        ----------
        age : int
            Target reconstruction age (Ma).
        resolution : float
            Output grid spacing (degrees).
        fuzzyAge : bool, optional
            If True, allows nearest match within small tolerance (±0.5 Myr).

        Defines
        -------
        self.paleoDEM : netCDF4.Dataset
            Temporary NetCDF with resampled paleoDEM (`z` variable).

        Notes
        -----
        Uses `gmt grdsample` to resample and convert to cell-registered grid.
        """
        # Find the paleoDEM file most closely related to the
        # reconstruction period.
        paleoDEMfidi = self.directories['paleoDEMs']+"/"+self.paleoDEMsfids[ np.min(np.abs(self.paleoDEMsAges - age))==np.abs(self.paleoDEMsAges - age) ][0]

        # Use gmt to copy the paleoDEM, resampling into the user
        # defined resolution. Note that this code also converts
        # the paleoDEM from grid line to cell registered.
        os.system("gmt grdsample {0} -G{1} -I{2} -Rd -T -Vq".format(paleoDEMfidi,
                                                                os.getcwd()+'/tempPaleoDEMi.nc',
                                                                resolution))
        
        # Read paleoDEM
        self.paleoDEM = Dataset(os.getcwd()+'/tempPaleoDEMi.nc');

        # Delete paleoDEM
        os.system("rm {}".format(os.getcwd()+'/tempPaleoDEMi.nc'))

    def getOceanLithosphereAgeGrid(self, age, resolution, fuzzyAge=False):
        """
        Load an ocean lithosphere age grid nearest to the requested time.

        Parameters
        ----------
        age : int
            Target reconstruction age (Ma).
        resolution : float
            Output grid spacing (degrees).
        fuzzyAge : bool, optional
            If True, allows nearest match within small tolerance.

        Defines
        -------
        self.oceanLithAge : netCDF4.Dataset
            Temporary NetCDF with resampled lithosphere ages (`z` variable),
            plus `lat`/`lon` axes.

        Notes
        -----
        Uses `gmt grdsample` to resample and convert to cell-registered grid.
        """
        # Find the paleoDEM file most closely related to the
        # reconstruction period.
        oceanLithAgefidi = self.directories['oceanLith']+"/"+self.oceanLithfids[ np.min(np.abs(self.oceanLithReconAges - age))==np.abs(self.oceanLithReconAges - age) ][0]

        # Use gmt to copy the paleoDEM, resampling into the user
        # defined resolution. Note that this code also converts
        # the paleoDEM from grid line to cell registered.
        os.system("gmt grdsample {0} -G{1} -I{2} -Rd -T -Vq".format(oceanLithAgefidi,
                                                                os.getcwd()+'/tempOecanLithAgei.nc',
                                                                resolution))
        
        # Read paleoDEM
        self.oceanLithAge = Dataset(os.getcwd()+'/tempOecanLithAgei.nc');

        # Delete paleoDEM
        os.system("rm {}".format(os.getcwd()+'/tempOecanLithAgei.nc'))

    def addThermalSub(self, topography, seafloorAge, latitude,
                    method={'type': 'CM2009'}, verbose=True):
        """
        Apply thermal subsidence to an oceanic lithosphere surface using empirical
        or parameterized age–depth relationships.

        The function computes first-order bathymetric depth as a function of
        seafloor age, representing the thermal contraction of the lithosphere
        during conductive and convective cooling phases. The resulting depth field
        is superimposed onto the existing topography array.

        Parameters
        ----------
        topography : numpy.ndarray
            2-D array of surface elevation or base topography (m). Values at
            oceanic locations will be overwritten by predicted oceanic depth.
        seafloorAge : numpy.ndarray
            2-D grid of seafloor ages (Myr). NaN where non-oceanic.
        latitude : numpy.ndarray
            2-D grid of latitudes (degrees), corresponding to `seafloorAge`.
        method : dict, optional
            Dictionary defining the thermal subsidence scheme. Defaults to
            ``{'type': 'CM2009'}``. Accepted values are listed in *Notes*.
        verbose : bool, optional
            If True, prints debug information or diagnostic comparisons.

        Returns
        -------
        numpy.ndarray
            Updated 2-D array of bathymetric depth (m) representing the thermal
            subsidence of the oceanic lithosphere.

        Notes
        -----
        **Available Methods**
            - ``'CM2009'`` — Crosby & McKenzie (2009):  
            Empirical piecewise relation between seafloor age and depth.  
            - ``'RK2021'`` — Rosas & Korenaga (2021):  
            Physically parameterized model with variable internal heating (H)
            and mid-ocean ridge depth.

        **Parameters used for RK2021 calculations**

            a1      = 0.001      (Slope, conductive phase [1/Myr^0.5]),  
            a2      = 0.0228     (Slope, convective phase [-]),  
            b2      = 0.0452     (Intercept, conductive phase [-]),  
            tmax    = 500        (Maximum seafloor age [Myr]),  
            ti      = 2.5        (Axis intercept (Fig. 2a RK2021) [Myr]),  
            rhoW    = 1000       (Water density [kg/m^3]),  
            rhoM    = 3300       (Surface mantle density [kg/m^3]),  
            rho0    = 4000       (Reference mantle density [kg/m^3]),  
            d       = 2.9e6      (Mantle thickness [m]),  
            kappa   = 4          (Thermal conductivity [W/m/K]),  
            alpha   = 1e-6       (Thermal expansivity [1/K]),  
            deltaT  = 1350       (Mantle potential temperature [K]),  
            nu0     = 1.3e20     (Reference viscosity [Pa·s]  

        **Varying Parameters (from RK2021, Fig. 2a)**

            H* — Non-dimensionalized internal heating parameter (varies)  
            t* — Onset time of asthenospheric convection (varies)

        These values may be digitized or interpolated from Rosas & Korenaga (2021)
        to reconstruct the age–depth relation for specific internal heating regimes.
        """

        if method['type'] == "RK2021":
            # Rosas and Korenaga (2021) age-depth relationship for different internal
            # heating scenarios.
            parameters = {}
            parameters['set1'] = {};

            # Conversion
            Myr2s = (1e6)*365*24*60*60; # [s/Myr]

            # All parameters are taken from Rosas and Korenaga (2021)
            parameters['set1']['a1'] = 0.001;           # Slope for linear relation between H* in the conductive phase [1/s^(1/2)]
            parameters['set1']['a2'] = 0.0228;          # Slope for linear relation between H* in the convection phase [-]
            parameters['set1']['b2'] = 0.0452;          # Intercept for linear relation between H* in the conductive phase [-]
            parameters['set1']['tmax'] = 500*Myr2s;     # Maximum age of seafloor [s]
            parameters['set1']['ti'] = (2.5**2)*Myr2s;  # Intersection with horizontal axis (ext fig 2a) [s]
            parameters['set1']['rhoW'] = 1000;          # Surface mantle density [kg/m3]
            parameters['set1']['rhoM'] = 3300;          # Density of water [kg/m3]

            parameters['set1']['rho0'] = 4000;          # Reference density [kg/m3]
            parameters['set1']['d'] = 2900e3;           # Thickness of manle [m]
            parameters['set1']['K'] = 4;                # Thermal conductivity [W/m/K]
            parameters['set1']['kappa'] = 1e-6;         # Thermal diffusivity [m2/s]
            parameters['set1']['alpha'] = 3e-5;         # Thermal expansivity [1/K]
            parameters['set1']['deltaT'] = 1350;        # Mantle potential temperature [K]

            parameters['set1']['nu0'] = 1.3e20;         # Asthenospheric viscosity [Pa s]
            parameters['set1']['H'] = np.array([2.1, 3.2, 4.8, 6.4, 8])*1e-12; # Internal heating [W/m/K]
            parameters['set1']['Hstar'] = (parameters['set1']['rho0']*parameters['set1']['d']**2/(parameters['set1']['K']*parameters['set1']['deltaT'])) * parameters['set1']['H'];

            # Digitized from figure 2a (Rosas and Korenaga 2021)
            parameters['set1']['tstar'] = (np.array([16.966824644549764,
                                                    16.303317535545023,
                                                    15.734597156398102,
                                                    15.165876777251183,
                                                    14.786729857819903])**2)*Myr2s; # Onset time pf sublithospheric of convection[s]


            # Functions for seafloor subsidence from Rosas and Korenaga (2021)
            def whs(alpha, deltaT, rhoM, rhoW, kappa, age):
                '''
                Function for seafloor subsidence (i.e., cooling of a semi-infinite 2D halfspace from RK2021)
                
                Parameters
                -----------
                alpha : FLOAT
                    Thermal expansivity, in [1/K].
                deltaT : FLOAT
                    Mantle potential temperature, in [K].
                rhoM : FLOAT
                    Surface mantle density density, in kg/m3.
                rhoW : FLOAT
                    Water density, in kg/m3.
                kappa :
                    Thermal diffusivity, in [m2/s].
                age : NUMPY ARRAY
                    Seafloor age, in [s].

                Return
                -------
                Depth of seafloor with respect to a mid-ocean-ridge depth, in m. 

                '''
                return 2*(alpha*deltaT*( rhoM / (rhoM-rhoW) )) * ( (kappa * age) / np.pi )**(1/2);

            def ws(whs, a1, a2, b2, ti, tmax, tstar, Hstar, age):
                '''
                Function for correction of seafloor subsidence (i.e., cooling of a
                semi-infinite 2D halfspace from RK2021) when mantle internal heating
                changes with respect to present-day.
                
                Parameters
                -----------
                whs : NUMPY ARRAY
                    Depth of seafloor with respect to a mid-ocean-ridge depth and
                    as calculated from whs (RK2021), in [m].
                a1 : FLOAT
                    Slope for linear relation between H* in the conductive phase,
                    in [1/Myr^(1/2)].
                a2 : FLOAT
                    Slope for linear relation between H* in the convection phase,
                    in [-].
                b2 : FLOAT
                    Intercept for linear relation between H* in the conductive
                    phase, in [-].
                ti : FLOAT
                    Intersection with horizontal axis (ext fig 2a), in [Myr].
                tmax : FLOAT
                    Maximum age of seafloor, in [Myr].
                tstar : FLOAT
                    Onset of asthenospheric convection, in [Myr].
                Hstar : FLOAT
                    Non-dimensionalized internal heating, in [-].
                age : NUMPY ARRAY
                    Seafloor age, in [Myr].

                Return
                -------
                Depth of seafloor with respect to a mid-ocean-ridge depth, in m. 

                '''
                # Conductive dominated
                whs[age<tstar] = whs[age<tstar]*(1 - a1*Hstar*(np.sqrt(age[age<tstar])-np.sqrt(ti)) )
                # Convection dominated
                whs[age>=tstar]  = whs[age>=tstar]*(1 - a1*Hstar*(np.sqrt(tstar)-np.sqrt(ti)) +\
                                                (Hstar*(a1*(np.sqrt(tstar)-np.sqrt(ti))-a2)-b2)*\
                                                (np.sqrt(age[age>=tstar])-np.sqrt(tstar))/\
                                                (np.sqrt(tmax)-np.sqrt(tstar)))
                return whs
            
            # Determine the onset of asthenospheric convection from
            # the input internal heating.
            ## Find the closest H value from the RK2021 paper
            ## Find the index of that internal heating value
            i = np.argwhere(np.min(np.abs(parameters['set1']['H']-method['H'])) == np.abs(parameters['set1']['H']-method['H']))[0][0]

            # Calculate cooling from a semi-infinite 2D half-space
            whsi = whs(parameters['set1']['alpha'],
                       parameters['set1']['deltaT'],
                       parameters['set1']['rhoM'],
                       parameters['set1']['rhoW'],
                       parameters['set1']['kappa'],
                       seafloorAge*Myr2s)
            
            # Apply the correction terms to whsi for internal conditions
            # and add the depth of the MOR.
            wsi  = ws(cp.deepcopy(whsi),
                    parameters['set1']['a1'],
                    parameters['set1']['a2'],
                    parameters['set1']['b2'],
                    parameters['set1']['ti']/Myr2s,
                    parameters['set1']['tmax']/Myr2s,
                    parameters['set1']['tstar'][i]/Myr2s,
                    parameters['set1']['Hstar'][i],
                    seafloorAge) + (1e3)*method['MORDepthkm'];
        
            # Set seafloor area with no seafloor ages to nan.
            wsi[seafloorAge<0]=np.nan;
            wsi[np.isnan(seafloorAge)]=np.nan

            # Modify the input topography with thermal subsidence calculations.
            topography = wsi;

        else:
            # Crosby and Mckenzie (2009) age-depth relationship piecewise logicals
            age_eq_less_than_75=(seafloorAge<=75)&(seafloorAge>-1)
            age_eq_less_than_160=(seafloorAge<=160)&(seafloorAge>75)
            age_greater_than_160=seafloorAge>160

            # Convert age to depth using Crosby and Mckenzie (2009) age-depth
            # relationship for oceanic crust
            depth=np.zeros(seafloorAge.shape)
            depth[age_eq_less_than_75]=2652+324*np.sqrt(seafloorAge[age_eq_less_than_75])
            depth[age_eq_less_than_160]=5028+5.26*seafloorAge[age_eq_less_than_160]-250*np.sin((seafloorAge[age_eq_less_than_160]-75)/30)
            depth[age_greater_than_160]=5750
            depth[seafloorAge<0]=np.nan
            depth[np.isnan(seafloorAge)]=np.nan

            # Modify the input topography with thermal subsidence calculations.
            topography = depth;
        
        # Report
        if verbose:
            print('--------------seafloorAge')
            print(seafloorAge)
            print(~np.isnan(seafloorAge))
            print(seafloorAge.shape)
            print('--------------depth')
            print(depth)
            print(~np.isnan(depth))
            print(depth.shape)
            print('--------------topography')
            print(topography)
            print(~np.isnan(topography))
            print(topography.shape)
        
        ## Return depth
        return topography

    def getESL(self, topography, seafloorAge, age, factor=1, verbose=True):
        """
        Apply eustatic sea-level offset (Haq87) to oceanic regions only.

        Parameters
        ----------
        topography : numpy.ndarray
            2-D topography with oceanic depths (m; positive).
        seafloorAge : numpy.ndarray
            2-D seafloor age grid (Myr); NaN over continents/paleoDEM areas.
        age : float
            Time (Ma) for which to sample the Haq87 sea-level curve.
        factor : float, optional
            Fraction of the Haq87 value to apply (0–1). Default 1.0.
        verbose : bool, optional
            If True, prints the applied offset.

        Returns
        -------
        (numpy.ndarray, float)
            Tuple of (modified topography, applied ESL in meters).
        """
        # Resolution of input (read) eustatic sealevel curve, in myr.
        resolution = .09;
        
        ## Apply Haq-87 sea-level curve to ocean regions only (Haq_87_SL_temp=0 if opt_Haq87_SL==False)
        # Inceased sea level is added to depth since depth is positive
        Haq_87_SL_temp = self.ESL.loc[np.abs(self.ESL['Ma']-age)<resolution]['m'].values[0]

        # Modify the topography with sealevel
        # Note that this change in sea-level is only applied over ocean
        # seafloor modeled with seafloor ages (i.e., seafloor represented
        # by paleoDEMs is not changed).
        topography[~np.isnan(seafloorAge)] -= Haq_87_SL_temp*factor;

        # Report
        if verbose:
            print("The eustatic sea-level at {0:.0f} Ma with respect to present-day is {1:0.1f} m.".format(age, Haq_87_SL_temp))

        return topography, Haq_87_SL_temp  

    def getSed(self, seafloorAge, latitude):
        """
        Estimate deep-marine sediment thickness from age/latitude (Straume et al., 2019).

        Parameters
        ----------
        seafloorAge : numpy.ndarray
            2-D seafloor age grid (Myr); NaN where not oceanic.
        latitude : numpy.ndarray
            2-D latitude grid (deg).

        Returns
        -------
        numpy.ndarray
            Sediment thickness (m) computed from a second-order polynomial in
            latitude modulated by sqrt(age); NaN where seafloorAge is invalid.
        """
        import copy as cp
        
        ## Calculate sediment thickness w/ Straume et al. (2019) age-sediment thickness relationship
        sedThick = cp.deepcopy(seafloorAge);
        sedThick[seafloorAge<0]=np.nan;
        sedThick = np.sqrt(seafloorAge)*(52-2.46*np.abs(latitude)+0.045*np.square(np.abs(latitude)));
        
        return sedThick

    def getIsostaticCorrection(self, topography, seafloorAge, latitude, longitude, verbose=True):
        """
        Add sediment thickness and apply isostatic compensation to oceanic depths.

        The load of seawater + sediments is compensated following Hoggard et al. (2017)
        and Athy (1930) porosity decay, yielding a corrected bathymetric depth.


        Parameters
        ----------
        topography : numpy.ndarray
            2-D bathymetry/topography (m) before sediment/isostatic terms.
        seafloorAge : numpy.ndarray
            2-D seafloor age (Myr); NaN over continents.
        latitude, longitude : numpy.ndarray
            2-D coordinate grids (deg).
        verbose : bool, optional
            If True, plots a quicklook of bathymetry change.

        Returns
        -------
        numpy.ndarray
            Bathymetry/topography with sediment loading + isostatic correction applied (m).
        """
        # Calculate sediment thickness
        sedThickness = self.getSed(seafloorAge, latitude)

        ## Apply Hoggard et al. (2017) isostatic correction using calculated sediment thicknesses
        # Define densities (Hoggard et al., 2017; Athy, 1930)
        rho_w   = 1.03e3;      	# [Mg/m^3] Density of water (Hoggard et al., 2017)
        rho_a   = 3.20e3;       	# [Mg/m^3] Density of Asthenosphere (Hoggard et al., 2017)
        phi0    = 0.61;         	# [-]  (Athy, 1930)
        lamda   = 3.9e3;        	# [m]  (Athy, 1930)
        rho_sg  = 2.65e3;       	# [Mg/m^3] Density of solid grains (Hoggard et al., 2017)
        def isostaticCorrection(oceanThickness, sedThickness):
            rho_s=rho_sg + ((phi0*lamda)/oceanThickness)*(rho_w-rho_sg)*(1-np.exp(-1*oceanThickness/lamda));
            compensatedSedPackages=((rho_a-rho_s)/(rho_a-rho_w))*sedThickness;
            print("Check that the output has the correct sign for positive value bathymetry")
            return oceanThickness-compensatedSedPackages
        
        # Add sediment thickness and correction term to topography
        topographyOut = isostaticCorrection(topography, sedThickness);
        

        # Plot the change in bathymetry resulting from the sediment/isostatic correction
        if verbose:
            # Report
            import matplotlib.pyplot as plt
            fig = plt.figure()
            # Plot lithosphere ages overlain by coastlines from the paleoDEMS.
            plt.contourf(longitude, latitude, topographyOut-topography)
            plt.colorbar(label="Change in bathymetry [m]")
            plt.xlim([-180,180])
            plt.ylim([-90,90])
            plt.axis('equal')


    
        return topographyOut

    def addVOCCorrection(self, age, bathymetry, VOCTarget, resolution, verbose=True):
        """
        Adjust bathymetry distributions to match a target global ocean volume (VOC).

        Concept
        -------
        This function ensures that modeled ocean basin volumes remain consistent
        with a prescribed target VOC across geologic time.

        - At present-day (`age == 0`), the model computes the depth-binned
        distribution mismatch between the simulated bathymetry and modern
        ETOPO data. From this difference, a correction vector (`sxbin_p`)
        is derived and cached.

        - For nonzero ages, the bathymetry is redistributed within basins
        according to `sxbin_p` and each basin’s fractional volume until
        the desired global VOC is achieved.

        Parameters
        ----------
        age : int
            Geological time (Ma).  
            If 0, compute and store the correction spectrum for reference.
        bathymetry : numpy.ndarray
            2-D bathymetry array (m; positive downward).  
            Land or emergent regions should be represented as NaN.
        VOCTarget : float
            Target global ocean volume (m³) to be matched after redistribution.
        resolution : float
            Working grid spacing (degrees) used for ETOPO resampling and
            histogram comparisons.
        verbose : bool, optional
            If True, generates diagnostic messages and comparison plots.  
            Default is True.

        Returns
        -------
        tuple of numpy.ndarray
            Updated bathymetry array and the correction spectrum (`sxbin_p`)
            expressed as percent change per depth bin.

        Notes
        -----
        - Uses ``self.basins`` to preserve basin boundaries during redistribution.
        - The correction spectrum (`sxbin_p`) is stored internally when
        `age == 0` and reused for other ages to maintain continuity.
        """

        # Set the global ocean basin volume for 
        #VOCTarget = self.getVOCi(age);


        def plotFnc(bathymetryAreaDist, bathymetryAreaDistEtopo, binEdges, areaTotal, areaTotalEtopo, volTotal, volTotalEtopo, bathymetry, etopo):
            '''
            plotFnc is used to plot the difference between the model bathymetry
            and actual present-day bathymetry. Positive values mean the model
            overrepresents bathymetry at a given depth.

            '''
            ## Set up the Mollweide projection
            fig = plt.figure(figsize=(8, 8))
            gs = GridSpec(2, 1, height_ratios=[3, 1]);  # 2 rows, 1 column, with the first row 3 times taller

            ax1 = fig.add_subplot(gs[0], projection=ccrs.Mollweide());

            ## Add the plot using pcolormesh
            bathymetryplt = cp.deepcopy(bathymetry);
            etopoplt = cp.deepcopy(etopo);
            bathymetryplt[np.isnan(bathymetryplt)] = 0;
            etopoplt[np.isnan(etopoplt)] = 0;
            bathymetryplt[bathymetryplt!=0] = bathymetryplt[bathymetryplt!=0]/bathymetryplt[bathymetryplt!=0];
            etopoplt[etopoplt!=0] = etopoplt[etopoplt!=0]/etopoplt[etopoplt!=0];
            difference = bathymetryplt-etopoplt;
            # 0     Representing bathymetry.
            # -1    measured bathymetry was not modeled.
            # 1     bathymetry was modeled where no measured bathymetry exists.
            mesh = ax1.scatter(self.lon[difference!=0], self.lat[difference!=0], c=difference[difference!=0], s=1, transform=ccrs.PlateCarree(), cmap='RdGy');

            ## Add a colorbar
            cbar = plt.colorbar(mesh, ax=ax1, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)
            cbar.ax.tick_params(labelsize=10, )  # Adjust the size of colorbar ticks
            cbar.ax.set_xticks([-1,1])
            cbar.ax.set_xticklabels(['Bathymetry\nMissing from Model', 'Bathymetry\nMissing from Etopo']) 

            ## Add gridlines
            ax1.gridlines()

            ## Set a title
            plt.title("Earth (Modeled/Measured) Bathymetry")

            ## Make histogram plot
            ax2 = fig.add_subplot(gs[1]);
            
            factor1 = .2
            factor2 = .25
            plt.bar(x=binEdges[1:]-(factor2/2)*np.diff(binEdges),
                    height=bathymetryAreaDistEtopo,
                    width=factor1*np.diff(binEdges),
                    label= "Mismatch")
            plt.bar(x=binEdges[1:]+(factor2/2)*np.diff(binEdges),
                    height=bathymetryAreaDist,
                    width=factor1*np.diff(binEdges),
                    label= "Mismatch")
            
            plt.plot([-1, 8], [0, 0], 'k', linewidth=.5)

            # Plot annoation of total area and volume difference
            deltaAOC = 100*(areaTotal-areaTotalEtopo)/areaTotalEtopo;
            deltaVOC = 100*(volTotal-volTotalEtopo)/volTotalEtopo;


            bbox = dict(boxstyle="round", fc="0.9", color='blue')
            plt.annotate(
                f"$\Delta$AOC = {deltaAOC:.1f}%\n$\Delta$VOC = {deltaVOC:.1f}%",
                fontsize=12,
                xy=(.1, 20),
                bbox=bbox)

            # ticks
            plt.xticks(binEdges[1:]);
            plt.yticks(np.arange(0,35,5));

            # Labels
            plt.legend();
            plt.title("(Modeled - Measured) Bathymetry Distribution")
            plt.xlabel("Bathymetry Bins [km]");
            plt.ylabel("Seafloor Area [%]\nOverrepresented in Model");

            # figure format
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.xlim([0,7])



        def plotFnc2(lat, lon, values,
                     areaTotal, areaTotalEtopo, volTotal, volTotalEtopo,
                            binEdges, bathymetryAreaDistVOCcorrection, bathymetryAreaDistEtopo,
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
            plotFnc2 function is used to plot global ranging datasets that
            are represented with evenly spaced latitude and longitude values.
            This function is modified from plotGlobalwHist and is used to 
            plot and compare VOC corrected bathymetry vs etopo bathymetry.

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
                    height=bathymetryAreaDistVOCcorrection,
                    width=factor1*np.diff(binEdges),
                    label= "VOC Corrected bathymetry")
            plt.bar(x=binEdges[1:]+(factor2/2)*np.diff(binEdges),
                    height=bathymetryAreaDistEtopo,
                    width=factor1*np.diff(binEdges),
                    label= "Etopo Bathymetry")
            

            # Plot annoation of total area and volume difference
            deltaAOC = 100*(areaTotal-areaTotalEtopo)/areaTotalEtopo;
            deltaVOC = 100*(volTotal-volTotalEtopo)/volTotalEtopo;


            bbox = dict(boxstyle="round", fc="0.9", color='blue')
            plt.annotate(
                f"$\Delta$AOC = {deltaAOC:.1f}%\n$\Delta$VOC = {deltaVOC:.1f}%",
                fontsize=12,
                xy=(.1, 20),
                bbox=bbox)
            
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
                    

        # If the bathymetry is representing present day then calculate
        # the correction distribution to apply to all later paleo
        # bathymetry reconstructions.
        if age == 0:
            # Create copy of the measured present-day topography that is
            # at the same resolution as the reconstruction models.
            os.system("gmt grdsample {0} -G{1} -I{2} -Rd -Vq".format(self.directories['etopo']+"/"+self.etopofid,
                                                                    os.getcwd()+'/tempetopo.nc',
                                                                    resolution))
            # Read etopo1 and define etopo bathymetry
            etopo = Dataset(os.getcwd()+'/tempetopo.nc')
            self.etopobathymetry = cp.deepcopy(etopo['z'][:].data);
            self.etopobathymetry[self.etopobathymetry>0] = np.nan;
            self.etopobathymetry = (-1)*self.etopobathymetry


            # Delete paleoDEM
            os.system("rm {}".format(os.getcwd()+'/tempetopo.nc'))

            # Create bathymetry histogram (for model at 0 Ma)
            self.bathymetryAreaDist, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistributionGlobal(bathymetry, self.lat, 90, self.areaWeights, binEdges = None, verbose=False);

            # Create bathymetry histogram (for etopo)
            self.bathymetryAreaDistEtopo, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistributionGlobal(self.etopobathymetry, self.lat, 90, self.areaWeights, binEdges = None, verbose=False);

            # Mismatch between modeled and measured bathymetry (Positive magitudes represent % of seafloor overrepresented 
            # NORMALIZED TO GLOBAL SEAFLOOR AREA) 
            self.bathymetryAreaDistMismatchGNorm = (self.bathymetryAreaDist - self.bathymetryAreaDistEtopo);

            # Mismatch between modeled and measured bathymetry (Positive magitudes represent % of seafloor overrepresented
            # NORMALIZED TO MISMATCHED SEAFLOOR AREA)
            #self.bathymetryAreaDistMismatchMMNorm = 100*self.bathymetryAreaDistMismatchGNorm/np.sum(np.abs(self.bathymetryAreaDistMismatchGNorm));

            if verbose:
                plotFnc(self.bathymetryAreaDistMismatchGNorm,
                        self.bathymetryAreaDistMismatchGNorm,
                        self.binEdges,
                        np.nansum(self.areaWeights[~np.isnan(bathymetry)]),
                        np.nansum(self.areaWeights[~np.isnan(self.etopobathymetry)]),
                        np.nansum(bathymetry*self.areaWeights),
                        np.nansum(self.etopobathymetry*self.areaWeights),
                        bathymetry,
                        self.etopobathymetry)
                
            # Calculate basin parameters. This produces the following useful values
            ## self.basins.BasinIDA : Global array with values corresponding to basin ID
            ## self.basins.bathymetryAreaDistBasin : ["Basin0", "Basin1",...].  Note that
            ##      this distribution is calculated with the exclusion of high latitude
            ##      distribution of seafloor depths. This is what is normally inputted into
            ##      the LOSCAR carbon cycle model.
            ## self.basins.bathymetryVolFraction : ["Basin0", "Basin1",...]. Each entry contains
            ##      the precent basin volume, normalized to the volume of all ocean basins
            ##      (excluding the high latitude ocean volume).
            ## self.basins.bathymetryAreaFraction : ["Basin0", "Basin1",...]. Each entry contains
            ##      the precent basin area, normalized to the total seafloor area (including
            ##      the high latitude area).
            self.basins.calculateBasinParameters()

            # Apply bathymetry/VOC correction for present-day bathymetry model basin-by-basin
            
            ## Define basin count
            self.basinCnt = len(np.unique( self.basins.BasinIDA[~np.isnan(self.basins.BasinIDA)] ))

            ## Define basin area and volume fractions
            volBasin    = np.empty(self.basinCnt);
            areaBasin   = np.empty(self.basinCnt);
            for i in range(self.basinCnt):
                areaBasin[i]    = np.sum( self.areaWeights[self.basins.BasinIDA==i] );
                volBasin[i]     = np.sum( bathymetry[self.basins.BasinIDA==i]*self.areaWeights[self.basins.BasinIDA==i] );
            
            ## Use volume fraction of ocean to determine the fraction of ocean volume correction that should be applied to each basin
            totalVolRes = (np.sum(volBasin)-VOCTarget)

            ## Total basin volume/area fractions
            volBasinFrac = volBasin/np.sum(volBasin);
            
            ## Calculate percents to add to each bin then plot
            sxbin_p = 100*(self.bathymetryAreaDist/np.sum(self.bathymetryAreaDist)-self.bathymetryAreaDistEtopo/np.sum(self.bathymetryAreaDistEtopo));

            ## Tracks the amount of ocean volume removed
            basinVolumeRemoved = 0.0;
            
            ## Create logical to represent all values of depth_in in each bin (depthBinLogical)
            randomDepthIndBin = {};

            for i in range(self.basinCnt):
                 
                for j in range(len(self.binEdges[1:])):

                    # Note that depthBinLogical represents an (nx2) array with rows
                    # representing indecies of points that lie within basini and 
                    # self.binEdges[j] and self.binEdges[j+1] depths.
                    # Note that self.binEdges values are in units of km.
                    # Note that bathymetry is positive and in units of m.
                    if j < len(self.binEdges):      # Contains depths between current bin and next bin
                        depthBinLogical = np.argwhere( ((1e-3)*bathymetry>=self.binEdges[j])&((1e-3)*bathymetry<self.binEdges[j+1])&(self.basins.BasinIDA==i) );
                    else:                        # The last bin which contains depths below bins_in(end)
                        depthBinLogical = np.argwhere( ((1e-3)*bathymetry>=self.binEdges[j])&(self.basins.BasinIDA==i) );
                    
                    verbose2 = False
                    if verbose2:
                        cnt = 0;
                        for pointi in depthBinLogical:
                            if cnt < 10:
                                print("Basin-{0}_Bin-{1}_to_{2}".format(i, self.binEdges[j], self.binEdges[j+1]),
                                    bathymetry[pointi[0], pointi[1]] )
                            else:
                                break
                            cnt+=1;
                        print("\n")
                    
                    # Random_depth_ind represents random depths to modify
                    # This has a very minor effect on bathymetry distributions input
                    # into LOSCAR.
                    # randomDepthIndBin is a dictionary that contains the entries for each
                    # basin and bathymetry bin, specifically the corresponding indices
                    # within the bathymetry array.
                    randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(i, self.binEdges[j], self.binEdges[j+1])] = depthBinLogical
                    np.random.shuffle(randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(i, self.binEdges[j], self.binEdges[j+1])])                

            ## Factors which dictate the percentage of ocean volume movement
            ## from bins with excess ocean volume
            bin_redis_faction = cp.deepcopy(sxbin_p);
            bin_redis_faction[bin_redis_faction<0]=0;
            bin_redis_faction = bin_redis_faction/np.sum(bin_redis_faction);
            
            ## Iterate over basins
            for basini in range(self.basinCnt):

                ## Find values to be changed in basini
                logicalBasini = (self.basins.BasinIDA==basini);
            
                ## Define the volume reduction required in basini.
                volReduction = totalVolRes*volBasin[basini]
            
                ## Iterate over bin depths
                if verbose:
                    print("Basin ", basini)
                for j in range(len(self.binEdges[1:])):

                    ## Find the amount of ocean volume to be transfer between bin j (bins with
                    ## excess ocean volume/area) and other bins (bins lacking ocean area representation).
                    if sxbin_p[j] > 0:
                        ## If bin has excess ocean area/volume

                        ## Ocean volume to be removed through adding shallower depths than the current bin.
                        sxbin_p_rm = sxbin_p[0:j];
                        bini_redis_faction1 = ( sxbin_p_rm/np.sum( sxbin_p_rm[sxbin_p_rm<0] ) )/\
                            (np.abs(np.sum( sxbin_p_rm[sxbin_p_rm<0])/sxbin_p[j]));
                        ## Ocean volume to be added through adding deeper depths than the current bin.
                        sxbin_p_add = sxbin_p[j+1:];
                        bini_redis_faction2 = ( sxbin_p_add/np.sum( sxbin_p_add[sxbin_p_add<0] ) )/\
                            np.abs(np.sum( sxbin_p_add[sxbin_p_add<0])/sxbin_p[j]);
                        ## Ocean volume to be removed/added through adding shallower/deeper bins than the current bin
                        bini_redis_faction = np.append(np.append(bini_redis_faction1,np.array(0.0)),bini_redis_faction2);

                        ## Set maximum percentage of indices that can be changed in a bin that will have bathymetry
                        ## removed from it.
                        maxBin_p_Change = sxbin_p[j];

                        if verbose:
                            print('\n\n\n',j, "Bin depth", self.binEdges[1+j],"; bini_redis_faction ", bini_redis_faction)

                        ## Apply volume correction to each basini bin i.                       
                        binVolume = 0.0;           # (re)set binVolume each time we change start bin (every j interation)
                        jjj=0;                     # (re)set jjj each time we change start bin (every j interation)
                        areaChangeOut = 0.0;
                        for jj in range(len(bini_redis_faction)):
                            if bini_redis_faction[jj] > 0:
                                # Calculate how much ocean volume residual is
                                # represented in the lacking bin (bini_vol_residual)
                                bini_vol_residual = totalVolRes*volBasinFrac[basini]*bin_redis_faction[j]*bini_redis_faction[jj];
                                volumeChange = 0.0;
                                areaChangeIn = 0.0;

                                # Note1: this chooses the option for how the bathymetry
                                # should be redistrbuted to other bins.
                                # Note2: Also, this code will not excute unless the
                                # residual in bin (jj) constitutes <1% of the total
                                # basin's residual between constant ocean volume and
                                # model.
                                opt_max_vol_trans = True;
                                if opt_max_vol_trans & (bini_vol_residual>totalVolRes*volBasinFrac[basini]*1e-2):
                                    max_ind = np.shape(randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(basini, self.binEdges[j], self.binEdges[j+1])])[0];

                                    while volumeChange < bini_vol_residual:
                                        # Reached ~100% of ocean volume at studied time or max indices at depth were move to other depths 
                                        opt_redis_cutoff = False;
                                        if opt_redis_cutoff & (np.nansum(bathymetry*self.areaWeights) < VOCTarget):
                                            # print("Target global ocean volume is now predicted to be accurate for time. Some basins might be lacking basin volume representation compared to others.")
                                            # print("exit 1")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif jjj == (max_ind-1):
                                            # print("Max indices were reached. No more redistributions can be done to reconcile ocean volume.")
                                            # print("exit 2")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif areaBasin[basini]*(maxBin_p_Change/100) < areaChangeOut:
                                            # print("Max bathymetry area was reached. Such that further removal of bathymetry from bin {0}-{1} would lead to an underestimation of bin {0}-{1} area".format(self.binEdges[j], self.binEdges[j+1]))
                                            # print("exit 4")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif areaBasin[basini]*(np.abs(sxbin_p[jj])/100) < areaChangeIn:
                                            # print("Max bathymetry area was reached in adding bin. Such that further adding of bathymetry to bin {0}-{1} would lead to an underestimation of bin {0}-{1} area".format(self.binEdges[jj], self.binEdges[jj+1]))
                                            # print("exit 5")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        
                                        # Select random ind from random ind vector - This value will then be changed
                                        #   Note that ind is only random since randomDepthIndBin dictionary indices are
                                        #   randomly ordered.
                                        ind = randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(basini, self.binEdges[j], self.binEdges[j+1])][jjj];

                                        # New depth - Choose depth randomly within bin or at maximum depth
                                        bins_in_cal = np.hstack((1e3*self.binEdges, 1e3*self.binEdges[-1]-1))
                                        new_depth   = np.abs(np.array([bins_in_cal[jj+1]-1, bins_in_cal[jj]+1]));
                                        
                                        # Find the min and max volume changes to new bin
                                        delta_vol = np.abs(self.areaWeights[ind[0],ind[1]]*(np.abs(bathymetry[ind[0],ind[1]])-new_depth));

                                        # Set new depth such that ocean volume is minimized
                                        if new_depth[0]<np.abs(bathymetry[ind[0],ind[1]]):
                                            # New depth is shallower than old depth
                                            new_depth = new_depth[np.max(delta_vol)==delta_vol];
                                        if new_depth[0]>np.abs(bathymetry[ind[0],ind[1]]):
                                            # New depth is deeper than old depth
                                            new_depth = new_depth[np.min(delta_vol)==delta_vol];

                                        # Report
                                        if volumeChange==0:
                                            print("Old depth {}, New depth {}, volumeChange {}, np.max(delta_vol) {}, bini_vol_residual {}".format(bathymetry[ind[0],ind[1]],
                                                                                                                                                new_depth,
                                                                                                                                                volumeChange,
                                                                                                                                                np.max(delta_vol),
                                                                                                                                                bini_vol_residual))

                                        # Use the maximum volume change to new bin as new depth
                                        if volumeChange+np.max(delta_vol)<=bini_vol_residual:
                                            # Append the volume change to lacking bin
                                            volumeChange = volumeChange+np.max(delta_vol);
                                            # Append the area changed
                                            areaChangeOut   = areaChangeOut+self.areaWeights[ind[0],ind[1]];
                                            areaChangeIn    = areaChangeIn+self.areaWeights[ind[0],ind[1]];
                                            # Append the depth to that of the lacking bin
                                            bathymetry[ind[0],ind[1]]= np.abs(new_depth);
                                            # Move to next random index of the over represented ocean volume bin
                                            jjj=jjj+1;
                                        else:
                                            # Scenario where depth cannot be added with
                                            # out exceeding the residual volume
                                            # correction for that bin
                                            # print("exit 3")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break

                                # Tracks the amount of ocean volume removed
                                basinVolumeRemoved += volumeChange;
                                binVolume += bini_vol_residual;


            # Report
            if verbose:
                # Create bathymetry histogram (for model at 0 Ma) that includes VOC corrections
                self.bathymetryAreaDistVOCcorrection, self.bathymetryAreaDist_wHighlatVOCcorrection, self.binEdges = calculateBathymetryDistributionGlobal(bathymetry, self.lat, 90, self.areaWeights, binEdges = None, verbose=False);

                blues_cm = mpl.colormaps['Blues'].resampled(100)
                plotFnc2(self.lat, self.lon, bathymetry,
                         np.nansum(self.areaWeights[~np.isnan(bathymetry)]),
                         np.nansum(self.areaWeights[~np.isnan(self.etopobathymetry)]),
                         np.nansum(bathymetry*self.areaWeights),
                         np.nansum(self.etopobathymetry*self.areaWeights),
                         self.binEdges, self.bathymetryAreaDistVOCcorrection, self.bathymetryAreaDistEtopo,
                         outputDir = os.getcwd(),
                         fidName = "plotGlobal-VOCCorrection.png",
                         cmapOpts={"cmap":blues_cm,
                                   "cbar-title":"cbar-title",
                                   "cbar-range":[np.nanmin(np.nanmin(bathymetry)),np.nanmean(bathymetry)+2*np.nanstd(bathymetry)]},
                         pltOpts={"valueType": "Bathymetry",
                                  "valueUnits": "m",
                                  "plotTitle":"",
                                  "plotZeroContour":False},
                         saveSVG=False,
                         savePNG=True);

        else:
            # If age is not present-day then use previously calculated sxbin_p
            # to apply basin volume correction.

            # Calculate basin parameters. This produces the following useful values
            ## self.basins.BasinIDA : Global array with values corresponding to basin ID
            ## self.basins.bathymetryAreaDistBasin : ["Basin0", "Basin1",...].  Note that
            ##      this distribution is calculated with the exclusion of high latitude
            ##      distribution of seafloor depths. This is what is normally inputted into
            ##      the LOSCAR carbon cycle model.
            ## self.basins.bathymetryVolFraction : ["Basin0", "Basin1",...]. Each entry contains
            ##      the precent basin volume, normalized to the volume of all ocean basins
            ##      (excluding the high latitude ocean volume).
            ## self.basins.bathymetryAreaFraction : ["Basin0", "Basin1",...]. Each entry contains
            ##      the precent basin area, normalized to the total seafloor area (including
            ##      the high latitude area).
            self.basins.calculateBasinParameters()

            # Apply bathymetry/VOC correction for present-day bathymetry model basin-by-basin
            
            
            ## Define basin count
            self.basinCnt = len(np.unique( self.basins.BasinIDA[~np.isnan(self.basins.BasinIDA)] ))

            ## Define basin area and volume fractions
            volBasin    = np.empty(self.basinCnt);
            areaBasin   = np.empty(self.basinCnt);
            for i in range(self.basinCnt):
                areaBasin[i]    = np.sum( self.areaWeights[self.basins.BasinIDA==i] );
                volBasin[i]     = np.sum( bathymetry[self.basins.BasinIDA==i]*self.areaWeights[self.basins.BasinIDA==i] );
            
            ## Use volume fraction of ocean to determine the fraction of ocean volume correction that should be applied to each basin
            totalVolRes = (np.sum(volBasin)-VOCTarget)

            ## Total basin volume fractions
            volBasinFrac = volBasin/np.sum(volBasin);

            ## Calculate percents to add to each bin then plot
            sxbin_p = self.sxbin_p;
            
            ## Tracks the amount of ocean volume removed
            basinVolumeRemoved = 0.0;
            
            ## Create logical to represent all values of depth_in in each bin (depthBinLogical)
            randomDepthIndBin = {};

            for i in range(self.basinCnt):
                 
                for j in range(len(self.binEdges[1:])):

                    # Note that depthBinLogical represents an (nx2) array with rows
                    # representing indecies of points that lie within basini and 
                    # self.binEdges[j] and self.binEdges[j+1] depths.
                    # Note that self.binEdges values are in units of km.
                    # Note that bathymetry is positive and in units of m.
                    if j < len(self.binEdges):      # Contains depths between current bin and next bin
                        depthBinLogical = np.argwhere( ((1e-3)*bathymetry>=self.binEdges[j])&((1e-3)*bathymetry<self.binEdges[j+1])&(self.basins.BasinIDA==i) );
                    else:                        # The last bin which contains depths below bins_in(end)
                        depthBinLogical = np.argwhere( ((1e-3)*bathymetry>=self.binEdges[j])&(self.basins.BasinIDA==i) );
                    
                    verbose2 = False
                    if verbose2:
                        cnt = 0;
                        for pointi in depthBinLogical:
                            if cnt < 10:
                                print("Basin-{0}_Bin-{1}_to_{2}".format(i, self.binEdges[j], self.binEdges[j+1]),
                                    bathymetry[pointi[0], pointi[1]] )
                            else:
                                break
                            cnt+=1;
                        print("\n")
                    
                    # Random_depth_ind represents random depths to modify
                    # This has a very minor effect on bathymetry distributions input
                    # into LOSCAR.
                    # randomDepthIndBin is a dictionary that contains the entries for each
                    # basin and bathymetry bin, specifically the corresponding indices
                    # within the bathymetry array.
                    randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(i, self.binEdges[j], self.binEdges[j+1])] = depthBinLogical
                    np.random.shuffle(randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(i, self.binEdges[j], self.binEdges[j+1])])                

            ## Factors which dictate the percentage of ocean volume movement
            ## from bins with excess ocean volume
            bin_redis_faction = cp.deepcopy(sxbin_p);
            bin_redis_faction[bin_redis_faction<0]=0;
            bin_redis_faction = bin_redis_faction/np.sum(bin_redis_faction);
            
            ## Iterate over basins
            for basini in range(self.basinCnt):

                ## Find values to be changed in basini
                logicalBasini = (self.basins.BasinIDA==basini);
            
                ## Define the volume reduction required in basini.
                volReduction = totalVolRes*volBasin[basini]
            
                ## Iterate over bin depths
                if verbose:
                    print("Basin ", basini)
                for j in range(len(self.binEdges[1:])):
                        ## If bin has excess ocean area/volume

                        ## Ocean volume to be removed through adding shallower depths than the current bin.
                        sxbin_p_rm = sxbin_p[0:j];
                        bini_redis_faction1 = ( sxbin_p_rm/np.sum( sxbin_p_rm[sxbin_p_rm<0] ) )/\
                            (np.abs(np.sum( sxbin_p_rm[sxbin_p_rm<0])/sxbin_p[j]));
                        ## Ocean volume to be added through adding deeper depths than the current bin.
                        sxbin_p_add = sxbin_p[j+1:];
                        bini_redis_faction2 = ( sxbin_p_add/np.sum( sxbin_p_add[sxbin_p_add<0] ) )/\
                            np.abs(np.sum( sxbin_p_add[sxbin_p_add<0])/sxbin_p[j]);
                        ## Ocean volume to be removed/added through adding shallower/deeper bins than the current bin
                        bini_redis_faction = np.append(np.append(bini_redis_faction1,np.array(0.0)),bini_redis_faction2);

                        ## Set maximum percentage of indices that can be changed in a bin that will have bathymetry
                        ## removed from it.
                        maxBin_p_Change = sxbin_p[j];

                        if verbose:
                            print('\n\n\n',j, "Bin depth", self.binEdges[1+j],"; bini_redis_faction ", bini_redis_faction)

                        ## Apply volume correction to each basini bin i.                       
                        binVolume = 0.0;           # (re)set binVolume each time we change start bin (every j interation)
                        jjj=0;                     # (re)set jjj each time we change start bin (every j interation)
                        areaChangeOut = 0.0;
                        for jj in range(len(bini_redis_faction)):
                            if bini_redis_faction[jj] > 0:
                                # Calculate how much ocean volume residual is
                                # represented in the lacking bin (bini_vol_residual)
                                bini_vol_residual = totalVolRes*volBasinFrac[basini]*bin_redis_faction[j]*bini_redis_faction[jj];
                                volumeChange = 0.0;
                                areaChangeIn = 0.0;
                                
                                # Note1: this chooses the option for how the bathymetry
                                # should be redistrbuted to other bins.
                                # Note2: Also, this code will not excute unless the
                                # residual in bin (jj) constitutes <1% of the total
                                # residual between constant ocean volume and model.
                                opt_max_vol_trans = True;
                                if opt_max_vol_trans & (bini_vol_residual>totalVolRes*volBasinFrac[basini]*1e-2):
                                    max_ind = np.shape(randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(basini, self.binEdges[j], self.binEdges[j+1])])[0];

                                    while volumeChange < bini_vol_residual:
                                        # Reached ~100% of ocean volume at studied time or max indices at depth were move to other depths 
                                        opt_redis_cutoff = False;
                                        opt_redis_cutoff = False;
                                        if opt_redis_cutoff & (np.nansum(bathymetry*self.areaWeights) < VOCTarget):
                                            # print("Target global ocean volume is now predicted to be accurate for time. Some basins might be lacking basin volume representation compared to others.")
                                            # print("exit 1")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif jjj == (max_ind-1):
                                            # print("Max indices were reached. No more redistributions can be done to reconcile ocean volume.")
                                            # print("exit 2")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif areaBasin[basini]*(maxBin_p_Change/100) < areaChangeOut:
                                            # print("Max bathymetry area was reached. Such that further removal of bathymetry from bin {0}-{1} would lead to an underestimation of bin {0}-{1} area".format(self.binEdges[j], self.binEdges[j+1]))
                                            # print("exit 4")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        elif areaBasin[basini]*(np.abs(sxbin_p[jj])/100) < areaChangeIn:
                                            # print("Max bathymetry area was reached in adding bin. Such that further adding of bathymetry to bin {0}-{1} would lead to an underestimation of bin {0}-{1} area".format(self.binEdges[jj], self.binEdges[jj+1]))
                                            # print("exit 5")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break
                                        
                                        # Select random ind from random ind vector - This value will then be changed
                                        #   Note that ind is only random since randomDepthIndBin dictionary indices are
                                        #   randomly ordered.
                                        ind = randomDepthIndBin["Basin-{0:0.0f}_Bin-{1:0.0f}_to_{2:0.0f}".format(basini, self.binEdges[j], self.binEdges[j+1])][jjj];

                                        # New depth - Choose depth randomly within bin or at maximum depth
                                        bins_in_cal = np.hstack((1e3*self.binEdges, 1e3*self.binEdges[-1]-1))
                                        new_depth   = np.abs(np.array([bins_in_cal[jj+1]-1, bins_in_cal[jj]+1]));
                                        
                                        # Find the min and max volume changes to new bin
                                        delta_vol = np.abs(self.areaWeights[ind[0],ind[1]]*(np.abs(bathymetry[ind[0],ind[1]])-new_depth));

                                        # Set new depth such that ocean volume is minimized
                                        if new_depth[0]<np.abs(bathymetry[ind[0],ind[1]]):
                                            # New depth is shallower than old depth
                                            new_depth = new_depth[np.max(delta_vol)==delta_vol];
                                        if new_depth[0]>np.abs(bathymetry[ind[0],ind[1]]):
                                            # New depth is deeper than old depth
                                            new_depth = new_depth[np.min(delta_vol)==delta_vol];

                                        # Report
                                        if volumeChange==0:
                                            print("Old depth {}, New depth {}, volumeChange {}, np.max(delta_vol) {}, bini_vol_residual {}".format(bathymetry[ind[0],ind[1]],
                                                                                                                                                new_depth,
                                                                                                                                                volumeChange,
                                                                                                                                                np.max(delta_vol),
                                                                                                                                                bini_vol_residual))

                                        # Use the maximum volume change to new bin as new depth
                                        if volumeChange+np.max(delta_vol)<=bini_vol_residual:
                                            # Append the volume change to lacking bin
                                            volumeChange = volumeChange+np.max(delta_vol);
                                            # Append the area changed
                                            areaChangeOut   = areaChangeOut+self.areaWeights[ind[0],ind[1]];
                                            areaChangeIn    = areaChangeIn+self.areaWeights[ind[0],ind[1]];
                                            # Append the depth to that of the lacking bin
                                            bathymetry[ind[0],ind[1]]= np.abs(new_depth);
                                            # Move to next random index of the over represented ocean volume bin
                                            jjj=jjj+1;
                                        else:
                                            # Scenario where depth cannot be added with
                                            # out exceeding the residual volume
                                            # correction for that bin
                                            # print("exit 3")
                                            # print("np.nansum(bathymetry*self.areaWeights)", np.nansum(bathymetry*self.areaWeights))
                                            # print("VOCTarget", VOCTarget)
                                            break

                                # Tracks the amount of ocean volume removed
                                basinVolumeRemoved += volumeChange;
                                binVolume += bini_vol_residual;
            # Report
            if verbose:
                # Create bathymetry histogram (for model at 0 Ma) that includes VOC corrections
                self.bathymetryAreaDistVOCcorrection, self.bathymetryAreaDist_wHighlatVOCcorrection, self.binEdges = calculateBathymetryDistributionGlobal(bathymetry, self.lat, 90, self.areaWeights, binEdges = None, verbose=False);
                # Create bathymetry histogram (for etopo)
                self.bathymetryAreaDistEtopo, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistributionGlobal(self.etopobathymetry, self.lat, 90, self.areaWeights, binEdges = None, verbose=False);


                blues_cm = mpl.colormaps['Blues'].resampled(100)
                plotFnc2(self.lat, self.lon, bathymetry,
                         np.nansum(self.areaWeights[~np.isnan(bathymetry)]),
                         np.nansum(self.areaWeights[~np.isnan(self.etopobathymetry)]),
                         np.nansum(bathymetry*self.areaWeights),
                         np.nansum(self.etopobathymetry*self.areaWeights),
                         self.binEdges, self.bathymetryAreaDistVOCcorrection, self.bathymetryAreaDistEtopo,
                         outputDir = os.getcwd(),
                         fidName = "plotGlobal-VOCCorrection.png",
                         cmapOpts={"cmap":blues_cm,
                                   "cbar-title":"cbar-title",
                                   "cbar-range":[np.nanmin(np.nanmin(bathymetry)),np.nanmean(bathymetry)+2*np.nanstd(bathymetry)]},
                         pltOpts={"valueType": "Bathymetry",
                                  "valueUnits": "m",
                                  "plotTitle":"",
                                  "plotZeroContour":False},
                         saveSVG=False,
                         savePNG=True);

        return bathymetry, sxbin_p

        print('working progress')

    def saveNetCDF4(self):
        """
        Placeholder for writing additional consolidated outputs.

        Notes
        -----
        Not yet implemented.
        """

    def compareLithosphereAndPaleoDEMRecon(self, age, resolution, fuzzyAge=False):
        """
        Quick visual check of coverage and consistency between lithosphere-age
        grids and paleoDEMs at a given time.

        Parameters
        ----------
        age : int
            Reconstruction age (Ma).
        resolution : float
            Grid spacing (deg).
        fuzzyAge : bool, optional
            If True, allows nearest match in available products.

        Returns
        -------
        None

        Side Effects
        ------------
        Produces a contour plot of lithosphere ages with paleoDEM coastlines.
        """
        import matplotlib.pyplot as plt

        # Read the ocean lithosphere ages and paleoDEM for the reconstruction
        # period
        self.getDEM(age, resolution, fuzzyAge=fuzzyAge);
        self.getOceanLithosphereAgeGrid(age, resolution, fuzzyAge=fuzzyAge);

        # Plot lithosphere ages overlain by coastlines from the paleoDEMS.
        XX, YY = np.meshgrid(self.oceanLithAge['lon'][:], self.oceanLithAge['lat'][:])
        #plt.contourf(XX, YY, self.paleoDEM['z'][:].data)
        plt.contourf(XX, YY, self.oceanLithAge['z'][:].data)
        plt.contour(XX, YY, self.paleoDEM['z'][:].data, levels=[0], cmap='Set1');


class BathySynthBogumil24():
    """
    Build synthetic, basin-agnostic bathymetry distributions following
    the design in Bogumil et al. (2024).

    Unlike spatial reconstructions, this class does **not** operate on
    latitude/longitude grids. Instead, it synthesizes global depth-bin
    distributions and formats them using ``BasinsSynth`` so they can be
    consumed by LOSCAR-style carbon cycle models.

    Concept
    -------
    1. Basins share identical depth distributions (contrast with geologic
       reconstructions where basins differ).
    2. Deep bathymetry (> 1 km) follows a Gaussian-like 5-bin pattern
       centered on a chosen bin (σ ≈ 250–500 m depending on bin width).
    3. Shallow bins (≤ 600 m) are allocated uniformly and varied in area
       within plausible ranges over 0–80 Ma.

    Distinctions vs. Bogumil et al. (2024)
    --------------------------------------
    1. In Bogumil et al., LOSCAR mixing parameters were fixed at PD values;
       here, *connectivity bathymetry* (not mixing coefficients) is
       defined from the synthetic Gaussian distribution. Other classes may
       choose whether to drive mixing by these synthetic distributions or
       keep LOSCAR defaults.
    2. VOC is computed directly as ``(total area) × (depth distribution)``.
    3. Basin VOC partitions are derived from basin area fractions.

    Parameters
    ----------
    optSynthModel : {'Model1'}, optional
        Synthetic recipe to use. Currently only 'Model1' is implemented:
        Gaussian deep bathymetry + variable shallow fraction. Default 'Model1'.
    ModelName : str, optional
        Name used for on-disk outputs (subdirectory and file prefixes).
        Default 'myModel'.

    Attributes
    ----------
    LOSCARValues : dict
        Convenience constants matching LOSCAR conventions:
        - ``AOC`` : float
            Global seafloor area [m²], default 3.49e14.
        - ``hb10`` : float
            High-latitude *surface box depth* [m] (not seafloor depth), default 250.
        - ``fanoc`` : (4, 1) ndarray
            Area fractions for Atlantic, Indian, Pacific, High-lat boxes (decimal).
        - ``binEdges`` : (N,) ndarray
            Upper bounds of bathymetry bins [m]: 0, 100, 600, 1000, …, 6500.
    optSynthModel : str
        The selected synthetic model key (e.g., 'Model1').
    ModelName : str
        Logical name for the model family; used in output paths.
    AOC_factor : ndarray of shape (3,)
        For 'Model1': ``[min_factor, max_factor, n_steps]`` controlling
        how total AOC is scaled relative to LOSCAR present-day.
        Default ``[0.96, 1.2, 13]`` → 13 evenly spaced factors in [0.96, 1.20].
    synth_model_para : dict
        Free parameters controlling the synthetic shape. Currently:
        - ``'shallow_bathy_percent'`` : float
            Fraction of global seafloor area assigned to the two shallowest
            non-zero bins (split evenly). Default 0.025 (2.5%).
    directories : dict
        Output directory roots. ``directories['output']`` points to
        ``./bathymetries/{optSynthModel}/{ModelName}``.
    BathymetryParms : dict
        Populated by :meth:`makeSynthModels`. A mapping:
        ``model_key -> {'VOC', 'AOC', 'fanoc', 'fdvol', 'hb10', 'distribution'}``,
        where:
        - ``VOC`` : float, total ocean volume [m³].
        - ``AOC`` : float, total ocean area [m²].
        - ``fanoc`` : (4,) ndarray, basin/HL area fractions (decimal).
        - ``fdvol`` : (4,) ndarray, basin/HL volume fractional partition (decimal),
          normalized to the sum over the **three** ocean basins (HL excluded in norm).
        - ``hb10`` : float, high-lat surface box depth [m].
        - ``distribution`` : (len(binEdges)-1,) ndarray,
          global % area per depth bin (decimal, excludes the 0 m edge).

    Notes
    -----
    - The Gaussian “deep” vector in *Model1* is applied across five
      adjacent 500 m-spaced bins centered on a chosen bin index and
      renormalized to honor the shallow share and AOC scaling.
    - Land is not represented—distributions are purely oceanic (% over AOC).
    - Outputs are formatted with :class:`utils.BasinsSynth` for downstream use.

    References
    ----------
    Bogumil, T. R., et al. (2024), *PNAS*, doi:10.1073/pnas.2400232121
    LOSCAR documentation (area partitions and bin definitions).
    """

    def __init__(self, optSynthModel='Model1', ModelName='myModel'):
        """
        Initialize a synthetic bathymetry generator and set LOSCAR defaults.

        Parameters
        ----------
        optSynthModel : {'Model1'}, optional
            Synthetic recipe key. Currently only 'Model1' is implemented.
        ModelName : str, optional
            Human-readable model family name used in filenames/paths.

        Side Effects
        ------------
        - Creates ``LOSCARValues`` with canonical area, binning, and fractions.
        - Defines AOC scaling sweep in :attr:`AOC_factor`.
        - Sets the output directory root in :attr:`directories`.
        """
        # Define Some Default LOSCAR variables
        self.LOSCARValues = {};
        self.LOSCARValues['AOC'] = 3.49e14; # Seafloor area as defined in LOSCAR [m2]
        self.LOSCARValues['hb10'] = 250; # Depth of the high latitude surface box [m], note that this is note the depth of the high latitude seafloor.
        self.LOSCARValues['fanoc'] = np.array(([.26],[.18],[.46],[.10])); # A, I, P, and Highlat surface area fraction [%]
        self.LOSCARValues['binEdges'] = np.array(([0.],[.1],[.6],[1.],[1.5],[2.],[2.5],[3.],[3.5],[4.],[4.5],[5.],[5.5],[6.5]))*1e3; # Bathymetry distribution bin edges as defined in LOSCAR [m]
        
        # Define synthetic model parameters (Deep bathymetry gaussian sigma, Shallow bathymetry extent)
        self.optSynthModel = optSynthModel;
        self.ModelName = ModelName;
        if self.optSynthModel == 'Model1':
            # Factor to vary seafloor area by with respect to present-day (LOSCAR seafloor area)
            self.AOC_factor = np.array([.96,1.2,13]);
        
        shallow_bathy_percent = .025;
        self.synth_model_para = {'shallow_bathy_percent':shallow_bathy_percent};

        # Define some directories
        self.directories = {}
        self.directories['output'] = os.getcwd()+'/bathymetries/{0}/{1}'.format(optSynthModel, ModelName)

    def makeSynthModels(self):
        """
        Build a suite of synthetic depth distributions and cache parameters.

        For *Model1*, each realization combines:
        - a 5-bin Gaussian-like deep pattern centered successively on
          eligible bin centers between ~1–5 km, and
        - a uniform shallow allocation split between the two shallow bins,
          sized by ``synth_model_para['shallow_bathy_percent']``.

        For each AOC scale factor in ``np.linspace(AOC_factor[0],
        AOC_factor[1], int(AOC_factor[2]))`` and each valid deep-bin center:

        - Compute distribution (% per bin, decimal).
        - Compute basin/HL volume split ``fdvol`` using LOSCAR
          area fractions (``fanoc``) and the distribution.
        - Compute target VOC as the area-weighted integral of depths.
        - Store all quantities to :attr:`BathymetryParms` under a model key
          like ``aF{factor%}_U{bin_edge}m``.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The “Gaussian” deep vector used is
          ``[(100 - 34.1*2 - 13.6*2)/2, 13.6, 34.1*2, 13.6, (100 - 34.1*2 - 13.6*2)/2] * 1e-2``,
          which approximates (~13.6%, 68.2%, 13.6%) tails/center mass over five bins.
        - The shallow share is removed from the deep mass and split evenly
          into bins at 100 m and 600 m (following LOSCAR bin edges).
        - All percentages are maintained in **decimal** form internally.
        """
        if self.optSynthModel == 'Model1':
            # Gaussian distributed deep bathymetry with varying shallow seafloor area.
            # Description of the distribution used for the synthetic bathymetry models
            # (..., 13.6, 68.2, 13.6, ...) normal distribution w/ mu=bin-250 m, sigma = 250 m
            # Note that mu and sigma only hold for distributions applied to bins which are spaced in 500 m intervals. (i.e. mu = 2000 m,..., 4500 m)
            deep_dist = np.array([(100-34.1*2 -13.6*2)/2, 13.6, 34.1*2, 13.6, (100-34.1*2 -13.6*2)/2])*1e-2;
            
            # Placeholder make distributions
            self.LOSCARValues['binEdges'][5:-3]
            
            # Dictionary to hold dictionaries of synthetic bathymetry parameters
            self.BathymetryParms = {};

            # 
            interation = 0;
            for areaFactori in np.linspace(self.AOC_factor[0],self.AOC_factor[1],int(self.AOC_factor[2])):
                areaFactori = round(areaFactori,3);
                for bin_idx in range(2,len(self.LOSCARValues['binEdges'])-7):
                    # Iterates over binEdges at which gaussian distributed bathymetry is maximum.
                    #   I.e., we create bathymetry models with seafloor at depths defined in the binEdges.
                    
                    # Create distribution vector
                    distribution = np.zeros((1,len(self.LOSCARValues['binEdges'])));
                    # Set deep bathymetry fraction for distribution
                    distribution[0][bin_idx+2:bin_idx+7] = deep_dist*( 1 - self.synth_model_para['shallow_bathy_percent']*2.0 )/areaFactori;
                    
                    # Set shallow bathymetry fraction for distribution
                    distribution[0][1] = (1- np.sum(distribution[0][bin_idx+2:bin_idx+7]) )/2; # (1-2*(areaFactori-1));
                    distribution[0][2] = (1- np.sum(distribution[0][bin_idx+2:bin_idx+7]) )/2; # (1-2*(areaFactori-1));

                    # Calculate ocean volume per basin using AOC and distribution + highlat depth and area [m3]
                    ## Define fdvol as the volume, m3, of each basin and the high latitude box.
                    fdvol   = areaFactori*self.LOSCARValues['AOC'] * (distribution * self.LOSCARValues['binEdges'].transpose()) * (self.LOSCARValues['fanoc']);
                    ## Redefine fdvol as a vector of ocean basin volume, in decimal percentage
                    ## of total ocean basin volume. Note that the high latitude box component
                    ## volume is removed in these percentages. 
                    fdvol   = np.sum(fdvol[:-1],1)/np.sum(np.sum(fdvol[:-1],1)); 

                    # Calculate VOC using AOC and distribution + highlat depth and area [m3]
                    #VOC = self.LOSCARValues['hb10']*areaFactori*self.LOSCARValues['AOC']*(self.LOSCARValues['fanoc'][3]) + np.sum( areaFactori*self.LOSCARValues['AOC'] * (distribution * self.LOSCARValues['binEdges'].transpose()) * (self.LOSCARValues['fanoc'][0:3]*1e-2) ); # [meters^3]
                    VOC     = np.sum( areaFactori*self.LOSCARValues['AOC'] * (distribution * self.LOSCARValues['binEdges'].transpose()) * (self.LOSCARValues['fanoc']) ); # m3.

                    # Constant high latitude area
                    #fanoc = (1/(self.LOSCARValues['fanoc'][-1]*areaFactori))
                    #fanoc = np.append( fanoc, self.LOSCARValues['fanoc'][0:3]*((1-fanoc)/.9) )
                    
                    
                    # Stack varibles VOC, AOC, fanoc, hb10, distribution                    
                    outputi = np.hstack( (VOC,self.LOSCARValues['AOC']*areaFactori) )
                    outputi = np.hstack( (outputi, self.LOSCARValues['fanoc'].reshape((4,))) )
                    outputi = np.hstack( (outputi, self.LOSCARValues['hb10']) )
                    
                    for i in range(3):
                        outputi = np.hstack( (outputi, distribution[0]) )

                    modeliString = 'aF{}_U{}m'.format(str(1e2*areaFactori).replace('.',','), str(int(self.LOSCARValues['binEdges'][bin_idx])));
                    self.BathymetryParms[modeliString] = {};
                    self.BathymetryParms[modeliString]['VOC']           = VOC;
                    self.BathymetryParms[modeliString]['AOC']           = self.LOSCARValues['AOC']*areaFactori;
                    self.BathymetryParms[modeliString]['fanoc']         = self.LOSCARValues['fanoc'].reshape((4,));
                    self.BathymetryParms[modeliString]['fdvol']         = fdvol
                    self.BathymetryParms[modeliString]['hb10']          = self.LOSCARValues['hb10'];
                    self.BathymetryParms[modeliString]['distribution']  = distribution[0][1:];
                    
    def saveSynthModels(self, verbose=True):
        """
        Write synthetic distributions to standardized netCDF via BasinsSynth.

        This method iterates the same (AOC factor × deep-center) grid used
        in :meth:`makeSynthModels`, pulls the cached parameters from
        :attr:`BathymetryParms`, and writes one ``bathymetry_*_wBasins.nc``
        per realization using :class:`utils.BasinsSynth`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress and let downstream utilities be chatty.
            Default True.

        Returns
        -------
        None

        Output
        ------
        ``bathymetries/{ModelName}/bathymetry_{key}_wBasins.nc`` files containing:
        - LOSCAR-compatible bin edges (km),
        - global depth distribution (% per bin),
        - AOC, VOC, basin/HL area fractions (fanoc), basin volume fractions (fdvol),
        - and any auxiliary metadata written by ``BasinsSynth``.
        """
        # Create the directory to store synthetic bathymetry models within
        utils.create_file_structure(["/bathymetries/{}".format(self.ModelName)], root=False, verbose=False);

        # Iterate over changes in deep and shallow bathymetry distributions
        for areaFactori in np.linspace(self.AOC_factor[0],self.AOC_factor[1],int(self.AOC_factor[2])):
            areaFactori = round(areaFactori,3);
            for bin_idx in range(2,len(self.LOSCARValues['binEdges'])-7):
                # Iterates over binEdges at which gaussian distributed bathymetry is maximum.
                #   I.e., we create bathymetry models with seafloor at depths defined in the binEdges.

                modeliString = 'aF{}_U{}m'.format(str(1e2*areaFactori).replace('.',','), str(int(self.LOSCARValues['binEdges'][bin_idx])));
                self.BathymetryParms[modeliString]['VOC']
                self.BathymetryParms[modeliString]['AOC']
                self.BathymetryParms[modeliString]['fanoc']
                self.BathymetryParms[modeliString]['hb10']


                # Create object to hold bathymery components for a single synthetic bathymetry model.
                BasinSynthi = utils.BasinsSynth(dataDir=os.getcwd()+"/bathymetries/{}".format(self.ModelName), filename='bathymetry_{}_wBasins.nc'.format(modeliString), radius=6371e3)

                # Set bathymetry distributions for basins
                BasinSynthi.defineBasinParameters(
                    BasinCnt = 3,
                    Distribution = self.BathymetryParms[modeliString]['distribution']*1e2,
                    binEdges = self.LOSCARValues['binEdges']*1e-3,
                    AOC = self.BathymetryParms[modeliString]['AOC'],
                    VOC = self.BathymetryParms[modeliString]['VOC'],
                    fanoc=self.BathymetryParms[modeliString]['fanoc'],
                    fdvol=self.BathymetryParms[modeliString]['fdvol'],
                    verbose=verbose);
                
                # Set basin connectivity parameters
                # FIXME: Add for future analysis
                #BasinSynthi.defineBasinConnectivityParameters()

                # Write bathymetry distributions to netCDF4s
                BasinSynthi.saveCcycleParameter(verbose=verbose)
                

#######################################################################
############# ExoCcycle Calculate Ccycle Bathymetry Params ############
#######################################################################
# Define some methods to calculate bathymetry attributes from 
def calculateHighLatA(bathymetry, latitudes, areaWeights, highlatP, verbose=True):
    """
    Find the latitude cutoff that yields a target fraction of high-latitude ocean area,
    and return both the cutoff and the corresponding area.

    The routine treats the two hemispheres symmetrically. Starting from the poles
    (±90°), it steps the absolute-latitude cutoff equatorward in 0.1° increments
    until the cumulative ocean area poleward of that cutoff reaches or exceeds
    ``highlatP`` of the total ocean area (AOC). Ocean is defined where
    ``bathymetry > 0`` (m), land/undefined where values are 0 or NaN.

    Parameters
    ----------
    bathymetry : numpy.ndarray
        2-D array (ny, nx) of seafloor depths in meters. Positive values denote
        ocean; 0 or NaN are treated as non-ocean and excluded from area totals.
    latitudes : numpy.ndarray
        2-D array (ny, nx) of cell-center latitudes in degrees, spanning [-90, 90],
        aligned with ``bathymetry``.
    areaWeights : numpy.ndarray
        2-D array (ny, nx) of cell areas in m² (or latitude-only weights broadcast
        to 2-D). Sum over all cells should approximate ``4πR²`` for the planet at
        the grid resolution used.
    highlatP : float
        Target fraction of *ocean* surface to include in the high-latitude box
        (decimal in [0, 1]). For example, ``0.10`` requests ~10% of total ocean area.
    verbose : bool, optional
        If True, print the requested fraction, the resulting cutoff latitude,
        and the computed high-latitude area. Default True.

    Returns
    -------
    highlatlat : float
        The absolute latitude (degrees, 0–90) that defines the high-latitude box:
        cells with ``|lat| > highlatlat`` are counted as high-latitude ocean.
    highlatA : float
        Total high-latitude ocean area in m² using that cutoff.

    Notes
    -----
    * Total ocean area (AOC) is computed as the sum of ``areaWeights`` over
      cells where ``bathymetry > 0`` (and not NaN).
    * The search decreases the cutoff from 90° toward 0° in steps of 0.1° until
      the poleward ocean-area fraction reaches ``highlatP``; the first cutoff
      meeting/exceeding the target is returned.
    * If ``highlatP > 1``, a message is printed and the function returns ``None``.
      Consider validating inputs upstream if you need stricter error handling.

    Examples
    --------
    >>> highlatlat, highlatA = calculateHighLatA(bathy, lat, weights, highlatP=0.10)
    >>> highlatlat
    63.4
    >>> highlatA / np.nansum(weights[bathy > 0])
    0.1003  # ~10% of ocean area

    See Also
    --------
    calculateBathymetryDistributionGlobal : Compute depth-binned global bathymetry distributions.
    """
    # Calculate the total seafloor area (AOC)
    AOC = np.nansum(np.nansum( areaWeights[~np.isnan(bathymetry) & ~(bathymetry==0)] ));

    # Check that high latitude box is at most 100 % of the ocean
    if highlatP > 1.0:
        print("The high latitude box was defined to be too large (i.e., it is greater than the size of the ocean). Change self.highlatP.")
        return

    # Iterate until percentArea reaches at least the size of the 
    # high latitude box in precent area, self.highlatP.
    highlatlat = 90;
    percentArea = 0;
    while (percentArea < highlatP) and not (highlatlat == 0):
        highlatlat -= .1;
        percentArea = np.nansum(np.nansum( areaWeights[(np.abs(latitudes)>highlatlat) & ~np.isnan(bathymetry) & ~(bathymetry==0)]))/AOC;

    # Define bathymetry parameter
    highlatA = np.sum(np.sum( areaWeights[(np.abs(latitudes)>highlatlat) & ~np.isnan(bathymetry) & ~(bathymetry==0)] ));

    # Report
    if verbose:
        print("The input high latitude area should cover {:2.0f}% of seafloor area.".format(highlatP));
        print("The high latitude cutoff is {:2.1f} degrees.".format(highlatlat));
        print("The high latitude area is {:2.0f} m2.".format(highlatA));

    return highlatlat, highlatA


def calculateBathymetryDistributionGlobal(bathymetry, latitudes, highlatlat, areaWeights,
                                          binEdges=None, verbose=True):
    """
    Compute global depth-binned bathymetry area distributions (with and without
    high-latitude ocean), weighted by cell area.

    The routine forms two histograms of ocean depth (km):
    1) Including all ocean cells (global, pole-to-pole).
    2) Excluding high-latitude ocean, i.e., using only cells with
       ``|lat| <= highlatlat`` (degrees).

    Both histograms are weighted by ``areaWeights`` and normalized to sum to 100%
    (percent of ocean area). Land/undefined cells are ignored.

    Parameters
    ----------
    bathymetry : numpy.ndarray
        2-D array (ny, nx) of seafloor depths in meters. Positive values denote
        ocean; 0 or NaN are treated as non-ocean and excluded.
    latitudes : numpy.ndarray
        2-D array (ny, nx) of cell-center latitudes in degrees, spanning [-90, 90],
        aligned with ``bathymetry``.
    highlatlat : float
        Absolute-latitude cutoff in degrees. Cells with ``|lat| > highlatlat`` are
        considered “high-latitude” and are excluded from the low-latitude
        distribution.
    areaWeights : numpy.ndarray
        2-D array (ny, nx) of cell areas in m² (or latitude-only weights broadcast
        to 2-D). Values are used as histogram weights.
    binEdges : numpy.ndarray, optional
        1-D array of bin edges in kilometers used for depth binning. Anything deeper
        than the last edge falls into the last bin. Default is::
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5])
    verbose : bool, optional
        If True, plots side-by-side bar charts (all ocean vs. no-high-lat).
        Default True.

    Returns
    -------
    bathymetryAreaDist : numpy.ndarray
        Percent area per depth bin **excluding** high-latitude ocean
        (i.e., using cells with ``|lat| <= highlatlat``). Sums to ~100.
    bathymetryAreaDist_wHighlat : numpy.ndarray
        Percent area per depth bin **including** all ocean (global). Sums to ~100.
    binEdges : numpy.ndarray
        The bin edges used (km).

    Notes
    -----
    * Depths are converted from meters to kilometers before binning.
    * Each histogram uses normalized weights:
      ``weights = areaWeights[mask] / sum(areaWeights[mask])``, then scaled
      to percentages (×100).
    * Ensure that ``bathymetry`` has positive values over ocean; zeros/NaNs are
      excluded from the statistics.

    Examples
    --------
    >>> bins_km = np.array([0, 1, 2, 3, 4, 5, 6.5])
    >>> dist_lowlat, dist_global, used_bins = calculateBathymetryDistributionGlobal(
    ...     bathy_m, lat_deg, highlatlat=60.0, areaWeights=cell_area_m2,
    ...     binEdges=bins_km, verbose=False)
    >>> dist_lowlat.sum(), dist_global.sum()
    (100.0, 100.0)
    """
    # Set bins default array.
    if binEdges is None:
        binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);

    # Calculate bathymetry distribution of global bathymetry (including high
    # latitude areas).
    logical1 = ~np.isnan(bathymetry);
    bathy1   = (1e-3)*bathymetry[logical1];
    weights1 = areaWeights[logical1]/np.sum(areaWeights[logical1]);

    bathymetryAreaDist_wHighlat, binEdges = np.histogram(bathy1, bins=binEdges, weights=weights1);
    bathymetryAreaDist_wHighlat = 100*(bathymetryAreaDist_wHighlat/np.sum(bathymetryAreaDist_wHighlat));

    # Calculate bathymetry distribution of global bathymetry (excluding high
    # latitude areas).
    logical2 = (np.abs(latitudes)<=highlatlat) & ~np.isnan(bathymetry);
    bathy2   = (1e-3)*bathymetry[logical2];
    weights2 = areaWeights[logical2]/np.sum(areaWeights[logical2]);

    bathymetryAreaDist, binEdges = np.histogram(bathy2, bins=binEdges, weights=weights2);
    bathymetryAreaDist = 100*(bathymetryAreaDist/np.sum(bathymetryAreaDist));

    # Report
    if verbose:
        # print("Bin edges used:\n", binEdges)
        # print("Bathymetry area distribution including high latitude bathymetry:\n",bathymetryAreaDist_wHighlat);
        # print("Bathymetry area distribution excluding high latitude bathymetry:\n",bathymetryAreaDist);

        fig = plt.figure(figsize=(8,4))

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

    return bathymetryAreaDist, bathymetryAreaDist_wHighlat, binEdges


def calculateBathymetryDistributionBasin(bathymetry, latitudes, longitudes, basinIDA,
                                         highlatlat, areaWeights, binEdges=None,
                                         fldName=os.getcwd(), verbose=True):
    """
    Compute per-basin bathymetry area and volume distributions, excluding and
    including high-latitude ocean, with global reference histograms.

    The function iterates over all unique basin IDs in ``basinIDA`` to calculate:
      * area-weighted depth distributions (% per bin),
      * fractional basin area (with/without high-lat region),
      * fractional basin volume (relative to total non-high-lat volume),
      * and global aggregate distributions for comparison.

    Optionally, when ``verbose=True``, it generates and saves:
      * a global basin map (Mollweide projection) and
      * a stacked bar plot of basin bathymetry distributions.

    Parameters
    ----------
    bathymetry : numpy.ndarray
        2-D array (ny, nx) of seafloor depth values in meters. Positive depths
        represent ocean; zeros or NaNs are treated as land or undefined.
    latitudes : numpy.ndarray
        2-D array (ny, nx) of cell-center latitudes in degrees, ranging [-90, 90],
        aligned with ``bathymetry``.
    longitudes : numpy.ndarray
        2-D array (ny, nx) of cell-center longitudes in degrees, ranging [-180, 180],
        aligned with ``bathymetry``.
    basinIDA : numpy.ndarray
        2-D array (ny, nx) of integer basin identifiers. NaNs indicate unclassified
        or land cells.
    highlatlat : float
        Absolute-latitude cutoff (degrees) separating low-latitude ocean
        (|lat| ≤ highlatlat) from the high-latitude region.
    areaWeights : numpy.ndarray
        2-D array (ny, nx) of cell surface areas in m². The sum over all valid
        ocean cells should approximate ``4πR²`` for the Earth (or the model sphere).
    binEdges : numpy.ndarray, optional
        1-D array of depth bin edges in kilometers. Anything deeper than the last
        edge falls into the last bin. Default::
            np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5])
    fldName : str, optional
        Directory path for figure output when ``verbose=True``. Default: current working directory.
    verbose : bool, optional
        If True, print basin counts and generate basin map and histogram figures
        saved as ``Basin_Distributions.png`` and ``.svg`` in ``fldName``. Default True.

    Returns
    -------
    bathymetryAreaDist : dict[str, np.ndarray]
        Per-basin depth distributions (% area per bin) **excluding** high-latitude
        regions. Keys are 'Basin0', 'Basin1', etc.
    bathymetryVolFrac : dict[str, float]
        Fractional basin volume (decimal) normalized to the sum of all
        **non-high-latitude** ocean volumes.
    bathymetryAreaFrac : dict[str, float]
        Fractional basin area (decimal) excluding the high-lat region, normalized
        to total ocean area (including high-lat).
    bathymetryAreaFracG : dict[str, float]
        Fractional basin area (decimal) including all ocean area (high-lat included),
        normalized to total seafloor area.
    bathymetryAreaDist_wHighlatG : np.ndarray
        Global bathymetry distribution (% area per bin) **including** high-latitude
        ocean (pole-to-pole).
    bathymetryAreaDistG : np.ndarray
        Global bathymetry distribution (% area per bin) **excluding** high-latitude
        ocean (|lat| ≤ highlatlat).
    binEdges : np.ndarray
        Bin edges (km) used for all histograms.

    Notes
    -----
    * Depths are converted from meters to kilometers prior to binning.
    * Basin-wise histograms are normalized such that each sums to 100% area within
      its respective basin domain.
    * Area and volume fractions are normalized to global totals using ``areaWeights``.
    * Basin IDs are taken as unique non-NaN entries in ``basinIDA``; their order
      corresponds to sorted unique IDs.
    * The resulting basin-level distributions can be used directly as LOSCAR
      bathymetry inputs.

    Visualization
    -------------
    When ``verbose=True``, this routine:
      - Creates a Mollweide map of basin partitions over ``longitudes``/``latitudes``.
      - Plots bathymetry distributions per basin (stacked bar chart) with labeled bins.
      - Saves both figures to ``fldName/Basin_Distributions.{png,svg}``.

    Examples
    --------
    >>> dist, volFrac, areaFrac, areaFracG, distGH, distG, bins = calculateBathymetryDistributionBasin(
    ...     bathy, lat, lon, basinID, highlatlat=60, areaWeights=weights, verbose=False)
    >>> list(dist.keys())
    ['Basin0', 'Basin1', 'Basin2']
    >>> np.sum(list(volFrac.values()))
    1.0
    """
    # Set bins default array.
    if binEdges is None:
        binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    
    # Setup dictionaries to hold outputs (basin distributions, area fractions, and volume fractions)
    bathymetryAreaDist = {};
    bathymetryAreaFrac = {};
    bathymetryAreaFracG = {};
    bathymetryVolFrac  = {};
    
    basinIDs = np.unique(basinIDA[~np.isnan(basinIDA)]);

    # Iterate over basins
    for i in range(len(basinIDs)):
        # Set basin ID and define basin i only bathymetry
        basinIDi = basinIDs[i];
        
        # Calculate bathymetry distribution of basin bathymetry (excluding high
        # latitude areas).
        # High latitude constraint
        logical1 = (np.abs(latitudes)<=highlatlat) & ~np.isnan(bathymetry);
        logical2 = (basinIDA==basinIDi);
        bathy2   = bathymetry[logical1 & logical2];
        weights2 = areaWeights[logical1 & logical2]/np.sum(areaWeights[logical1 & logical2]);

        bathymetryAreaDisti, binEdges = np.histogram((1e-3)*bathy2, bins=binEdges, weights=weights2);
        bathymetryAreaDist['Basin{:0.0f}'.format(basinIDi)] = 100*(bathymetryAreaDisti/np.sum(bathymetryAreaDisti));
        bathymetryAreaFracG['Basin{:0.0f}'.format(basinIDi)] = np.sum(areaWeights[logical2])/np.nansum(areaWeights[~np.isnan(bathymetry) & ~(bathymetry==0)])
        bathymetryAreaFrac['Basin{:0.0f}'.format(basinIDi)] = np.sum(areaWeights[logical1 & logical2])/np.nansum(areaWeights[~np.isnan(bathymetry) & ~(bathymetry==0)])
        bathymetryVolFrac['Basin{:0.0f}'.format(basinIDi)]  = np.sum(bathymetry[logical1 & logical2]*areaWeights[logical1 & logical2]) / np.sum(bathymetry[logical1]*areaWeights[logical1]);

    # Calculate Global values
    ## Calculate bathymetry distribution of global bathymetry (including high
    ## latitude areas).
    logical1 = ~np.isnan(bathymetry);
    bathy1   = (1e-3)*bathymetry[logical1];
    weights1 = areaWeights[logical1]/np.sum(areaWeights[logical1]);

    bathymetryAreaDist_wHighlatG, binEdges = np.histogram(bathy1, bins=binEdges, weights=weights1);
    bathymetryAreaDist_wHighlatG = 100*(bathymetryAreaDist_wHighlatG/np.sum(bathymetryAreaDist_wHighlatG));

    ## Calculate bathymetry distribution of global bathymetry (excluding high
    ## latitude areas).
    logical2 = (np.abs(latitudes)<=highlatlat) & ~np.isnan(bathymetry);
    bathy2   = bathymetry[logical2];
    weights2 = areaWeights[logical2]/np.sum(areaWeights[logical2]);

    bathymetryAreaDistG, binEdges = np.histogram((1e-3)*bathy2, bins=binEdges, weights=weights2);
    bathymetryAreaDistG = 100*(bathymetryAreaDistG/np.sum(bathymetryAreaDistG));

    # Report
    if verbose:
        # print("Bin edges used:\n", binEdges);
        # print("Bathymetry area distribution excluding high latitude bathymetry:\n");
        # for basinIDi in range(len(bathymetryAreaDist)):
        #     print(bathymetryAreaDist['Basin{:0.0f}'.format(basinIDi)]);

        #fig = plt.figure(figsize=(8,4))

        # Define the number of basins
        cnt = len(bathymetryAreaDist);

        # Create colormap
        ## Set colormap
        #cmap = plt.get_cmap("Pastel1")
        cmap = plt.get_cmap("Set1")

        ## Extract basinCnt colors from the colormap
        colors_rgb = [cmap(i) for i in range(cnt)]
        ## Convert RGB to hex
        colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_rgb]
        ## Create a custom colormap from the list of colors
        custom_cmap = LinearSegmentedColormap.from_list("custom_pastel", colors, N=256)

        # Set up the Mollweide projection
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1]);  # 2 rows, 1 column, with both row heights equal.

        ax1 = fig.add_subplot(gs[0], projection=ccrs.Mollweide());

        # Plot basin contourf and coastlines

        ## Add the plot using pcolormesh
        mesh = ax1.pcolormesh(longitudes, latitudes, basinIDA, cmap=custom_cmap, transform=ccrs.PlateCarree())

        ## Add coastlines
        ### Set any np.nan values to 0.
        bathymetry[np.isnan(bathymetry)] = 0;
        ### Plot coastlines.
        zeroContour = ax1.contour(longitudes, latitudes, bathymetry,levels=[0], colors='black', transform=ccrs.PlateCarree())


        # Make bathymetry distribution plot
        ax2 = fig.add_subplot(gs[1]);

        ## Define factors for plotting
        factor1 = .1
        factor2 = .25
        if len(bathymetryAreaDist)%2:
            factor3 = 0.5;
        else:
            factor3 = 0;
        
        ## Iteratively plot basin bathymetry distributions
        for i in range(len(bathymetryAreaDist)):
            plt.bar(x=binEdges[1:]-(factor2/2)*(cnt/2 - i -factor3)*np.diff(binEdges),
                    height=bathymetryAreaDist['Basin{:0.0f}'.format(i)],
                    width=factor1*np.diff(binEdges),
                    label= "Basin {:0.0f}".format(i),
                    color=colors[i])
        ## ticks
        plt.xticks(binEdges[1:]);
        plt.yticks(np.arange(0,35,5));

        ## Labels
        plt.legend();
        plt.title("Planet's Bathymetry Distribution")
        plt.xlabel("Bathymetry Bins [km]");
        plt.ylabel("Seafloor Area [%]");

        ## figure format
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)


        plt.savefig(fldName+"/Basin_Distributions.png", dpi=600, transparent=True)
        plt.savefig(fldName+"/Basin_Distributions.svg", transparent=True)

    return bathymetryAreaDist, bathymetryVolFrac, bathymetryAreaFrac, bathymetryAreaFracG, bathymetryAreaDist_wHighlatG, bathymetryAreaDistG, binEdges

