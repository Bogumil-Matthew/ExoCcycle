#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:20:00 2024

@author: bogumilmatt-21
"""

#######################################################################
############################### Imports ###############################
#######################################################################
from ExoCcycle import utils # type: ignore
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
    '''
    Measured topography (~bathymetry) Venus/Mars/Moon/present-day Earth.

    getTopo is a method to download different topography models.
        getTopo(self, body = {'Mars':'True', 'Venus':'False', 'Moon':'False', 'Earth':'False'}):

        1. Moon:    https://pgda.gsfc.nasa.gov/products/54 or https://pgda.gsfc.nasa.gov/products/95 (include above 60 degrees)
        2. Mars:    !wget https://github.com/andrebelem/PlanetaryMaps/raw/v1.0/mola32.nc
        3. Venus:   https://astrogeology.usgs.gov/search/map/venus_magellan_global_c3_mdir_colorized_topographic_mosaic_6600m
        4. Earth:   etopo
    
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
        getTopo is a meethod used to download solar system topography
        models that can be used in further analysis

        Parameters
        ----------
        data_dir : STRING
            A directory which you store local data within. Note that this
            function will download directories [data_dir]/topographies
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        -------
        None.

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
            if os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif"):
                os.system("wget -O {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif https://planetarymaps.usgs.gov/mosaic/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif".format(data_dir, self.model));
            if os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif".format(data_dir, self.model)):
                os.system("export PROJ_IGNORE_CELESTIAL_BODY=YES &&\
                        gdalwarp -t_srs EPSG:4326 {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.tif {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif".format(data_dir, self.model));
            if os.path.exists("{0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc".format(data_dir, self.model)):
                os.system("gdal_translate -of NetCDF {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m_reprojected.tif {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc".format(data_dir, self.model));
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/Venus_Magellan_C3-MDIR_ClrTopo_Global_Mosaic_6600m.nc -JN0/5i -Crelief -P -K > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Venus":
            if os.path.exists("{0}/topographies/{1}/topogrd.img"):
                os.system("wget -O {0}/topographies/{1}/topogrd.img https://pds-geosciences.wustl.edu/mgn/mgn-v-rss-5-gravity-l2-v1/mg_5201/images/topogrd.img".format(data_dir, self.model));
            if os.path.exists("{0}/topographies/{1}/topogrd.nc"):
                # Write netCDF file
                ## Read .img
                fid = open("{0}/topographies/{1}/topogrd.img".format(data_dir, self.model), 'rb');
                elevmodel = np.fromfile(fid, dtype=np.uint8);
                lonmodel, latmodel = np.meshgrid(np.arange(-180, 180, 1), np.arange(90-1/2, -90, -1) )
                elevmodel = elevmodel.reshape(180,360);
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
                os.system("gmt grdimage {0}/topographies/{1}/topogrd.nc -JN0/5i -Crelief -P -K > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Earth":
            if os.path.exists("{0}/topographies/Earth/ETOPO1_Ice_c_gdal.grd.gz"):
                os.system("wget -O {0}/topographies/Earth/ETOPO1_Ice_c_gdal.grd.gz https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/cell_registered/netcdf/ETOPO1_Ice_c_gdal.grd.gz".format(data_dir))
            if os.path.exists("{0}/topographies/Earth/ETOPO1_Ice_c_gdal.grd"):
                os.system("yes N | gzip -k -d {0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd.gz".format(data_dir, self.model));
            if os.path.exists("{0}/topographies/Earth/ETOPO1_Ice_c_gdal.nc"):
                os.system("gmt grdconvert {0}/topographies/{1}/ETOPO1_Ice_c_gdal.grd {0}/topographies/{1}/ETOPO1_Ice_c_gdal.nc -fg".format(data_dir, self.model))
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/ETOPO1_Ice_c_gdal.nc -JN0/5i -Crelief -P -K > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Mars":
            if os.path.exists("{0}/topographies/{1}/mola32.nc"):
                os.system("wget -O {0}/topographies/{1}/mola32.nc https://github.com/andrebelem/PlanetaryMaps/raw/v1.0/mola32.nc".format(data_dir, self.model))
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/mola32.nc -JN0/5i -Crelief -P -K > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));
        elif self.model == "Moon":
            if os.path.exists("{0}/topographies/{1}/LDEM64_PA_pixel_202405.grd"):
                os.system("wget -O {0}/topographies/{1}/LDEM64_PA_pixel_202405.grd https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_pixel_202405.grd".format(data_dir, self.model));
            if os.path.exists("{0}/topographies/{1}/LDEM64_PA_pixel_202405.nc"):
                os.system("gmt grdconvert {0}/topographies/{1}/LDEM64_PA_pixel_202405.grd {0}/topographies/{1}/LDEM64_PA_pixel_202405.nc -fg".format(data_dir, self.model));
            if verbose:
                os.system("gmt grdimage {0}/topographies/{1}/LDEM64_PA_pixel_202405.nc -JN0/5i -Crelief -P -K > {0}/topographies/{1}/{1}.ps".format(data_dir, self.model));

    def readTopo(self, data_dir, new_resolution = 1, verbose=True):
        """
        readTopo method reads downloaded topography models. They are then
        interpolated and saved to the same input directory. If the topography
        model was previously written in the chosen resolution then this model
        will read that file instead instead.

        Parameters
        ----------
        data_dir : STRING
            A directory which you store local data within. Note that this
            function will download directories [data_dir]/topographies
        new_resolution : INT
            The resolution, in degrees, of the output topography. Note that this
            should often be lower than input resolution. For Earth, Moon, Mars,
            and Venus 0.5 degrees is acceptable. The default is 1.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Defines
        -------
        self.elev : NUMPY ARRAY
            nx2n array representing cell registered topography, in m.
        self.lon : NUMPY ARRAY
            nx2n array representing cell registered longitudes, in deg,
            ranging from [-180, 180]. Longitudes change from column to column.
        self.lat : NUMPY ARRAY
            nx2n array representing cell registered latitudes, in deg,
            ranging from [-90, 90]. Latitudes change from row to row.
        self.resolution : INT
            The resolution, in degrees, of the output topography. Note
            that this value is set to the new_resolution input value.
        self.data_dir : STRING
            A directory which you store local data within.

        Returns
        -------
        None.

        """
        # Set the object's resolution for topography/bathymetry calculations
        self.resolution = new_resolution;

        # Set the location of data storage
        self.data_dir = data_dir;

        # Define the resampled topography output file name.
        resampledTopoPath = "{0}/topographies/{1}/{1}_resampled_{2:0.0f}deg.nc".format(data_dir,  self.model, new_resolution);

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
            utils.plotGlobal(self.lat, self.lon, self.elev,
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
                os.system("gmt grdsample {0} -Rd -I1d -rp -G{1}".format(TopoPath, TopoPath.replace(".nc", "_resampled.nc")))
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
            self.lon, self.lat = np.meshgrid(np.arange(-180+new_resolution/2, 180+new_resolution/2, new_resolution), np.arange(90-new_resolution/2, -90-new_resolution/2, -new_resolution) )

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
             
    def setSeaLevel(self, basinVolume = {"on":True, 'uncompactedVol':None}, oceanArea = {"on":True, "area":0.7}, isostaticCompensation = {"on":False}, verbose=True):
        """
        setSetLevel method is used to define a bathymetry model from a
        topo model. The topography model can be defined by running the
        method readTopo.

        
        Parameters
        ----------
        basinVolume : DICTIONARY
            Option to define bathymetry by flooding topography with
            basinVolume['uncompactedVol'] amount of ocean water, in m3.
        oceanArea : DICTIONARY
            Option to define bathymetry by flooding topography until
            oceanArea['area'], decimal percent, of global area is covered
            with oceans.
        isostaticCompensation : DICTIONARY
            An option to apply isostatic compensation to for ocean loading
            on the topography. Option assumes a uniform physical properties
            of the lithosphere.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

            
        Defines
        -------
        self.bathymetry : NUMPY ARRAY
            nx2n array representing seafloor depth, in m, with positive values.
        self.bathymetryAreaDist : NUMPY LIST
            A histogram of seafloor bathymetry with using the following bin edges:
            0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
            that this distribution is calculated with the exclusion of high latitude
            distribution of seafloor depths. This is what is normally inputted into
            the LOSCAR carbon cycle model.
        self.bathymetryAreaDist_wHighlat : NUMPY LIST
            This is the same as bathymetryAreaDist, but includes the high latitude
            seafloor distribution of seafloor depths.
        self.AOC : FLOAT
            Total sea surface area, in m2.
        self.VOC : FLOAT
            Total basin volume, in m3.
        self.highlatA : FLOAT
            Total high latitude ocean area, in m2. Note that this is not the
            hb[10] value in the LOSCAR earth system model. self.highlatP is hb[10].
        self.highlatlat : FLOAT
            The lowest latitude of the high latitude cut off, in degree.
        self.areaWeights : NUMPY ARRAY
            An array of global degree to area weights. The size is dependent on
            input resolution. The sum of the array equals 4 pi radius^2 for 
            sufficiently high resolution, in m2.
        self.binEdges : NUMPY ARRAY
            A numpy list of bin edges, in km, to calculate the bathymetry distribution
            over. Note that anything deeper than the last bin edge will be defined within
            the last bin.

        Returns
        -------
        None.
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
        areaWeights, longitudes, latitudes, totalArea, totalAreaCalculated = utils.areaWeights(resolution = 1, radius = self.radiuskm*1e3, verbose=False);
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

    def saveBathymetry(self, verbose = True):
        """
        saveBathymetry is a method used to save bathymetry models
        created with setSeaLevel. Note that models will be saved
        under the same root folder that was supplied to the readTopo(...)
        method.

        Dimensions are as follows:

        lat     : latitude in degrees, ranging from -180,180.
        lon     : longitude in degrees, ranging from -90,90.
        binEdges: upper bound bin edges for bathymetry distributions, in km.

        Values saved to the output netCDF4 are as follows:

        bathymetry : NUMPY ARRAY
            nx2n array representing seafloor depth, in m, with
            negative values, and topography with positive values.
        lat : NUMPY VECTOR
            A vector of cell-registered latitudes range the entire
            planets surface, -90,90 degrees.
        lon : NUMPY VECTOR
            A vector of cell-registered longitudes range the entire
            planets surface, -180,180 degrees.
        areaWeights : NUMPY VECTOR
            An array of global degree to area weights. The size is dependent on
            input resolution. The sum of the array equals 4 pi radius^2 for 
            sufficiently high resolution, in m2.
        binEdges : NUMPY VECTOR
            A numpy list of bin edges, in km, used in calculating
            the bathymetry distribution. Note that 1) anything deeper
            than the last bin edge will be defined within the last
            bin and 2) the first bin edge corresponds to the first
            non-zero bin edge.
        bathymetry-distribution-G : NUMPY LIST
            A histogram of seafloor bathymetry within binEdges bins.
            Note that this distribution is calculated with the exclusion
            of high latitude distribution of seafloor depths. This is
            what is normally inputted into the LOSCAR carbon cycle model.
        bathymetry-distribution-whighlat-G : NUMPY LIST
            This is the same as bathymetry-distribution-G, but includes
            the high latitude seafloor distribution of seafloor depths.
        highlatA : FLOAT
            Total high latitude ocean area, in m2. Note that this is
            not the hb[10] value in the LOSCAR earth system model.
        highlatlat : FLOAT
            The lowest latitude of the high latitude cutoff, in degree.
            This value is define by the constraint that some (user define
            portion of the ocean surface should be within a high latitude
            ocean basin, faciliating ocean turnover in carbon cycle models).
        AOC : FLOAT
            Total sea surface area, in m2.
        VOC : FLOAT
            Total basin volume, in m3.

        Parameters
        ----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        -------
        None.
        
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
        BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.0f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        
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

    def readBathymetry(self, verbose = True):
        """
        readBathymetry is a method used to read bathymetry models
        created that are written with the readBathymetry method. 


        Parameters
        ----------
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        -------
        None.

        """

        BathyPath = "{0}/bathymetries/{1}/{1}_resampled_{2:0.0f}deg.nc".format(self.data_dir,  self.model, self.resolution);
        
        # Make new .nc file
        self.bathync = Dataset(BathyPath, mode='r', format='NETCDF4_CLASSIC') 


        

class BathyRecon():
    '''
    BathyRecon is a class used to create Earth bathymetry reconstructions
    from a set of models and methods.

    Bathymetry reconstructions are created as desribed in Bogumil et al.
    (2024) https://doi.org/10.1073/pnas.2400232121. The steps to their
    creation are as follows:
    
    1. We use the age-depth relationship of Crosby and McKenzie to convert
    the age of the oceanic lithosphere from paleo-isochrons to compute the
    thermal evolution and subsidence of the oceanic lithosphere.
    
    2. Next they are modified by adding isostatically compensated global
    paleodeep marine sediment distributions using a relationship based on
    present-day sediment thickness and ocean crust age.
    
    3. Continental shelves, flooded continental crust, and deep marine not
    represented in the plate motion model reconstructions were supplemented
    by paleo digital elevation maps.
    
    4. Bathymetry was then corrected for eustatic sea-level using the long-
    term Haq87 sea level curve, which is broadly consistent with paleoDEMs
    shoreline reconstructions.
    
    5. Last, the evolved global bathymetry distributions were deformed based
    on present-day model uncertainty to reproduce accurate global ocean
    container volumes throughout the reconstruction period.
    '''

    def __init__(self, directories):
        '''
        
        Parameters
        -----------
        directories : DICTIONARY
            A dictionary containing all necessary directories for the bathymetry
            reconstruction and analysis. User must define the following directories:
            (paleoDEMs and oceanLith).
            The files stored in these directories must follow the below naming convection  
                paleoDEMs -> [prefix]_[float/int]Ma.nc
                oceanLith -> [prefix]-[float/int].nc
            
        Defines
        --------
        self.ESL : NUMPY ARRAY
            2xn array of sea-level, in m, with respect to present-day in the
            first row and age in Ma in the second row.
        self.radiuskm : FLOAT
            Earth's radius in km.

        '''

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
        print('test')
        
        # Set the radius of planet
        self.radiuskm = 6371.0;
    
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


    
    def run(self, startMa=80, endMa=0, deltaMyr=5, resolution=1, verbose=True):
        '''
        run will make a netCDF4 file which contains
        bathymetry modeled with thermal subsidence,
        isostatically compensated sediments, supplemented
        paleo digital elevation maps, and volume corrected
        basin bathymetry as outline in Bogumil et al. (2024)
        https://doi.org/10.1073/pnas.2400232121.

        Parameters
        -----------
        start : INT
            Oldest bathymetry model to create, in Ma.
        end : INT
            Youngest bathymetry model to create, in Ma.
        deltaTime : INT
            Temporal resolution of bathymetry models, in
            Myr.
        resolution : Float
            Spatial resolution of bathymetry model, in
            degrees.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.


        (Re)defines
        ------------

        
        '''
        print('working progress')

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
            self.topography = self.addThermalSub(self.topography, self.oceanLithAge['z'][:].data, self.lat, verbose=False);
            
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
            self.topography, ESLi = self.getESL(self.topography, reconAge, factor=1, verbose=False);

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

            # 9. Save bathymetry model w/o the ocean volume corrections
            # FIXME: Need to define this at a higher level.
            self.data_dir = os.getcwd();
            self.resolution = resolution;
            self.model = "EarthRecon3Basins"
            #self.model = "EarthRecon3_4Basins"

            self.saveBathymetry(reconAge, verbose=True);

            # 10. Find basins (Note that this is a partially manual process)
            ## Define basins class for finding basins
            def mergeBasins(basins, reconAge):
                if reconAge == 0:
                    # Merge basins north of Atlantic
                    basins.mergeBasins(0,[1,2,3,4,6,8,12,25,29,30,31,35,40,43,44,45,46,47,51,54,57,62,63,64,65,66,68,69,70], write=False)
                    # Merge basins north of Atlantic with Atlantic.
                    # Note the new basinIDs
                    basins.mergeBasins(1,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,22,23,24,25,26], write=False)
                    # Merge basins for Pacific ocean.
                    # Note the new basinIDs
                    basins.mergeBasins(2,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], write=False)
                elif reconAge == 5:
                    # Merge basins north of Atlantic
                    basins.mergeBasins(0,[1,2,3,4,5,6,7,23,27,29,30,36,38,40,42,43,44,45,46,47,52,55,56,58,60,62,63,64,65,67], write=False)
                    # Merge basins north of Atlantic with Atlantic.
                    # Note the new basinIDs
                    basins.mergeBasins(1,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,20,21,22,23,25], write=False)
                    # Merge basins for Pacific ocean.
                    # Note the new basinIDs
                    basins.mergeBasins(2,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], write=False)
                elif reconAge == 10:
                    # Merge basins north of Atlantic
                    basins.mergeBasins(0,[1,2,3,4,5,6,7,14,30,33,35,36,37,38,43,47,49,50,52,53,54,55,56,57,58,59,60,64,65,68,69,70,73,74,75,76,77,78,79,80,82,83,84,85], write=False)
                    # Merge basins north of Atlantic with Atlantic.
                    # Note the new basinIDs
                    basins.mergeBasins(1,[2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20,23,24,27,28], write=False)
                    # Merge basins for Pacific ocean.
                    # Note the new basinIDs
                    basins.mergeBasins(2,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], write=False)
                    pass
                elif reconAge == 15:
                    pass
                elif reconAge == 20:
                    pass
                elif reconAge == 25:
                    pass
                elif reconAge == 30:
                    pass
                elif reconAge == 35:
                    pass
                elif reconAge == 40:
                    pass
                elif reconAge == 45:
                    pass
                elif reconAge == 50:
                    pass
                elif reconAge == 55:
                    pass
                elif reconAge == 60:
                    pass
                elif reconAge == 65:
                    pass
                elif reconAge == 70:
                    pass
                elif reconAge == 75:
                    pass
                elif reconAge == 80:
                    pass
                print("working progress")

                # Report
                blues_cm = mpl.colormaps['Blues'].resampled(100)
                basins.visualizeCommunities( cmapOpts={"cmap":blues_cm,
                                                    "cbar-title":"cbar-title",
                                                    "cbar-range":[np.nanmin(np.nanmin(basins.bathymetry)),
                                                                    np.nanmean(basins.bathymetry)+2*np.nanstd(basins.bathymetry)]},
                                            pltOpts={"valueType": "Bathymetry",
                                                    "valueUnits": "m",
                                                    "plotTitle":"{}".format(basins.body),
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
                
                # Return
                return basins

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
            
            if self.model == "EarthRecon3Basins":
                basins = mergeBasins(basins, reconAge)

            # Assign basins as a BathyRecon class attribute.
            self.basins = basins;

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
            

            # 12. Save bathymetry model w/ the ocean volume corrections

            # Report
            '''
            if verbose:
                blues_cm = mpl.colormaps['Blues'].resampled(100)
                self.highlatlat = 90
                print("FIXME:")
                # FIXME: Need to change the plotted values from bathymetryAreaDist_wHighlat to bathymetryAreaDist_wHighlat
                utils.plotGlobalwHist(self.lat, self.lon, self.bathymetry,
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
            '''

    def getVOCi(self, age):
        """
        getVOCi returns the expected ocean basin volume at some period
        in Earth history.

        Parameters
        -----------
        age : INT
            The age of at which to return an expected global
            ocean basin volume.

        Return
        -------
        VOC : FLOAT
            The volume of the global ocean basin system, in m3, at input
            time age.

        """
        if len(self.VOCValues) == 1:
            return self.VOCValues[0];
        else:
            return self.VOCValues[self.VOCAgeValues == age];

    def saveBathymetry(self, reconAge, verbose = True):
        """
        saveBathymetry is a method used to save reconstructed bathymetry
        models. Note that models will be saved under the same root folder
        that was supplied to the readTopo(...) method.

        Dimensions are as follows:

        lat     : latitude in degrees, ranging from -180,180.
        lon     : longitude in degrees, ranging from -90,90.
        binEdges: upper bound bin edges for bathymetry distributions, in km.

        Values saved to the output netCDF4 are as follows:

        bathymetry : NUMPY ARRAY
            nx2n array representing seafloor depth, in m, with
            negative values, and topography with positive values.
        lat : NUMPY VECTOR
            A vector of cell-registered latitudes range the entire
            planets surface, -90,90 degrees.
        lon : NUMPY VECTOR
            A vector of cell-registered longitudes range the entire
            planets surface, -180,180 degrees.
        areaWeights : NUMPY VECTOR
            An array of global degree to area weights. The size is dependent on
            input resolution. The sum of the array equals 4 pi radius^2 for 
            sufficiently high resolution, in m2.
        binEdges : NUMPY VECTOR
            A numpy list of bin edges, in km, used in calculating
            the bathymetry distribution. Note that 1) anything deeper
            than the last bin edge will be defined within the last
            bin and 2) the first bin edge corresponds to the first
            non-zero bin edge.
        bathymetry-distribution-G : NUMPY LIST
            A histogram of seafloor bathymetry within binEdges bins.
            Note that this distribution is calculated with the exclusion
            of high latitude distribution of seafloor depths. This is
            what is normally inputted into the LOSCAR carbon cycle model.
        bathymetry-distribution-whighlat-G : NUMPY LIST
            This is the same as bathymetry-distribution-G, but includes
            the high latitude seafloor distribution of seafloor depths.
        highlatA : FLOAT
            Total high latitude ocean area, in m2. Note that this is
            not the hb[10] value in the LOSCAR earth system model.
        highlatlat : FLOAT
            The lowest latitude of the high latitude cutoff, in degree.
            This value is define by the constraint that some (user define
            portion of the ocean surface should be within a high latitude
            ocean basin, faciliating ocean turnover in carbon cycle models).
        AOC : FLOAT
            Total sea surface area, in m2.
        VOC : FLOAT
            Total basin volume, in m3.

        Parameters
        ----------
        reconAge : INT
            Age of the reconstruction, in Ma.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        -------
        None.
        
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
        '''
        getDEM is a method used to read in a topography model
        from Scotese and Wright (2018) digitized paleoDEMS.
        A resolution can be specified, but note that the maximum
        resolution of accompanying DEMs are 1 degree.


        Parameters
        -----------
        age : INT
            The age, in Myr, at which that user wants to read a
            paleoDEM from.
        resolution : FLOAT
            The resolution, in degree, at which the user wants to
            read the paleoDEM at.
        fuzzyAge : BOOLEAN, optional
            An option to read paleoDEMs with similar, but not exact
            ages as represented in age. For example, this allows
            a user to read in a plaeoDEM marked with ages that are
            +-0.5 Myr. The default is False.

        Defines
        --------
        self.paleoDEM : NUMPY ARRAY
            nx2n global array of topography, in m, that now includes
            thermal subsidence of seafloor subsidence.        
        
        '''
        # Find the paleoDEM file most closely related to the
        # reconstruction period.
        paleoDEMfidi = self.directories['paleoDEMs']+"/"+self.paleoDEMsfids[ np.min(np.abs(self.paleoDEMsAges - age))==np.abs(self.paleoDEMsAges - age) ][0]

        # Use gmt to copy the paleoDEM, resampling into the user
        # defined resolution. Note that this code also converts
        # the paleoDEM from grid line to cell registered.
        os.system("gmt grdsample {0} -G{1} -I{2} -Rd -T".format(paleoDEMfidi,
                                                                os.getcwd()+'/tempPaleoDEMi.nc',
                                                                resolution))
        
        # Read paleoDEM
        self.paleoDEM = Dataset(os.getcwd()+'/tempPaleoDEMi.nc');

        # Delete paleoDEM
        os.system("rm {}".format(os.getcwd()+'/tempPaleoDEMi.nc'))

    def getOceanLithosphereAgeGrid(self, age, resolution, fuzzyAge=False):
        '''
        getOceanLithosphereAgeGrid is a method used to read in a
        ocean lithosphere age models (e.g., Muller et al. 2019).
        A resolution can be specified, but note that there is a
        maximum resolution of accompanying your defined reconstructions.


        Parameters
        -----------
        age : INT
            The age, in Myr, at which that user wants to read a
            paleoDEM from.
        resolution : FLOAT
            The resolution, in degree, at which the user wants to
            read the ocean lithospheric age grid at.
        fuzzyAge : BOOLEAN, optional
            An option to read paleoDEMs with similar, but not exact
            ages as represented in age. For example, this allows
            a user to read in a plaeoDEM marked with ages that are
            +-0.5 Myr. The default is False.

        Returns
        --------
        ageGrid : NUMPY ARRAY
            nx2n global array of topography, in m, that now includes
            thermal subsidence of seafloor subsidence.        
        
        '''
        # Find the paleoDEM file most closely related to the
        # reconstruction period.
        oceanLithAgefidi = self.directories['oceanLith']+"/"+self.oceanLithfids[ np.min(np.abs(self.oceanLithReconAges - age))==np.abs(self.oceanLithReconAges - age) ][0]

        # Use gmt to copy the paleoDEM, resampling into the user
        # defined resolution. Note that this code also converts
        # the paleoDEM from grid line to cell registered.
        os.system("gmt grdsample {0} -G{1} -I{2} -Rd -T".format(oceanLithAgefidi,
                                                                os.getcwd()+'/tempOecanLithAgei.nc',
                                                                resolution))
        
        # Read paleoDEM
        self.oceanLithAge = Dataset(os.getcwd()+'/tempOecanLithAgei.nc');

        # Delete paleoDEM
        os.system("rm {}".format(os.getcwd()+'/tempOecanLithAgei.nc'))

    def addThermalSub(self, topography, seafloorAge, latitude, verbose=True):
        '''
        addThermalSub is a method that calculates and adds first order
        seafloor depth from thermal subsidence of oceanic lithosphere
        to a topography model.

        Thermal subsidence is calculated according to Crosby and McKenzie's
        (2009) relationship (eq. 4) between present-day seafloor depths and
        ages. https://doi.org/10.1111/j.1365-246X.2009.04224.x 


        Parameters
        -----------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m,  or array of zeros
            to add seafloor subsidence to.
        seafloorAge : NUMPY ARRAY
            nx2n global array of seafloor ages, in Myr. Any area where
            seafloor ages are not represented -either due to lack of
            data, the non-existance of seafloor, or non-thermally
            subsiding seafloor- should be represented with np.nan values. 
        latitude : NUMPY ARRAY
            nx2n global array of latitudes corresponding to element locations
            in input topography and seafloorAge arrays.  
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Returns
        --------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m, that now includes
            thermal subsidence of seafloor subsidence.
        '''
                
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


        topography = depth;
        
        ## Return depth
        return topography

    def getESL(self, topography, age, factor=1, verbose=True):
        '''
        getESL method is used to a obtain the eustatic sealevel
        change with respect to present-day.


        Parameters
        -----------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m, modified with the
            eustatic change in sealevel.
        age : FLOAT
            The age, in Myr, at which that user wants to read a
            ESL from.
        factor : FLOAT, optional
            A factor, ranging from 0-1, that describes how much of the
            eustatic sea-level that will be applied to the reconstruction.
            This value might be changed from 1 such that the resulting
            continental area is appropriately flooded. The default is 1.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Return
        -------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m, modified with the
            eustatic change in sealevel.
        Haq_87_SL_temp : FLOAT
            The eustatic change in sealevel, in m, with respect to
            present day.
        '''
        # Resolution of input (read) eustatic sealevel curve, in myr.
        resolution = .09;
        
        ## Apply Haq-87 sea-level curve to ocean regions only (Haq_87_SL_temp=0 if opt_Haq87_SL==False)
        # Inceased sea level is added to depth since depth is positive
        Haq_87_SL_temp = self.ESL.loc[np.abs(self.ESL['Ma']-age)<resolution]['m'].values[0]

        # Modify the topography with sealevel
        topography[~np.isnan(topography)] -= Haq_87_SL_temp*factor;

        # Report
        if verbose:
            print("The eustatic sea-level at {0:.0f} Ma with respect to present-day is {1:0.1f} m.".format(age, Haq_87_SL_temp))

        return topography, Haq_87_SL_temp  

    def getSed(self, seafloorAge, latitude):
        '''
        getSed method finds sediment thicknesses as described in
        Straume et al. (2019), eq (2a). The function is a 2nd order
        poly. w/ inverted coefficents for present-day sediment
        thickness where seafloor age is younger 82 Ma and below 72
        degrees latitude.


        Parameters
        -----------
        seafloorAge : NUMPY ARRAY
            nx2n grid of seafloor ages, in Myr. Elements with np.nan
            entries represent either no available data or continental
            area.  
        latitude : NUMPY ARRAY
            nx2n grid of latitudes, not colatitudes,
            in degrees.

        Return
        -------
        sedThick : NUMPY ARRAY
            Seafloor sediment thickness, in meters.
        
        '''
        import copy as cp
        
        ## Calculate sediment thickness w/ Straume et al. (2019) age-sediment thickness relationship
        sedThick = cp.deepcopy(seafloorAge);
        sedThick[seafloorAge<0]=np.nan;
        sedThick = np.sqrt(seafloorAge)*(52-2.46*np.abs(latitude)+0.045*np.square(np.abs(latitude)));
        
        return sedThick

    def getIsostaticCorrection(self, topography, seafloorAge, latitude, longitude, verbose=True):
        '''
        getIsostaticCorrection method is used to add the seafloor
        sediment and the accompanying isostatic correction term for
        overlaying seafloor sediment and seawater.
        

        Parameters
        -----------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m. This should include
            bathymetry modified with any methods besides addVOCCorrection.
        seafloorAge : NUMPY ARRAY
            nx2n global array of seafloor ages, in Myr. Any area where
            seafloor ages are not represented -either due to lack of
            data, the non-existance of seafloor, or non-thermally
            subsiding seafloor- should be represented with np.nan values. 
        latitude : NUMPY ARRAY
            nx2n global array of latitudes corresponding to element locations
            in input seafloorAge arrays.
        latitude : NUMPY ARRAY
            nx2n global array of longitudes corresponding to element locations
            in input seafloorAge arrays.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.

        Return
        -------
        topography : NUMPY ARRAY
            nx2n global array of topography, in m, modified with the
            seafloor sediment thickness and isostatic compensation
            correction terms.
        '''
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
        '''
        addVOCCorrection 
        FIXME: add appropriate description.
        
        Parameters
        -----------
        age : INT
            Period of the reconstruction, in Ma. If age is 0 (present-day)
            then the correction distribution will be calculated as well.
            Note that the correction distribution needs to be calculated
            for the VOC correction to be applied at other periods beyond
            present-day.
        bathymetry : NUMPY ARRAY
            nx2n global array of bathymetry, in m.
        VOCTarget : FLOAT
            The target ocean basin volume to reconstruct. Bathymetry
            distributions will be modified based on present-day mismatch
            in modeling the bathtmetry distribution.
        resolution : FLOAT
            The resolution, in degree, at which the user wants to
            read the present-day bathymetry model at.
        verbose : BOOLEAN, optional
            Reports more information about process. The default is True.
        
        '''
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
                    

        def applyVOCCorrection(correctionDist, bathymetry, VOCTarget, VOCi):
            """
            applyVOCCorrection will use a distribution (correctionDist) to
            determine what ocean depths should be (removed or added) from
            a 2D global array of bathymetry. bathymetry will change such
            that the total basin volume VOC is equal to VOCtarget.

            Parameters
            -----------
            bathymetry : NUMPY ARRAY
                nx2n global array of bathymetry, in m.
            VOCTarget : FLOAT
                The target ocean basin volume to reconstruct. Bathymetry
                distributions will be modified based on present-day mismatch
                in modeling the bathtmetry distribution.

            Returns
            --------
            bathymetry : NUMPY ARRAY
                nx2n global array of bathymetry, in m.

            
            """

        # If the bathymetry is representing present day then calculate
        # the correction distribution to apply to all later paleo
        # bathymetry reconstructions.
        if age == 0:
            # Create copy of the measured present-day topography that is
            # at the same resolution as the reconstruction models.
            os.system("gmt grdsample {0} -G{1} -I{2} -Rd".format(self.directories['etopo']+"/"+self.etopofid,
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
                                            # FIXME: Check if this value should be positive of negative.
                                            #bathymetry[self.basins.BasinIDA==basini][ind[0],ind[1]]= np.abs(new_depth);
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
                                            # FIXME: Check if this value should be positive of negative.
                                            #bathymetry[self.basins.BasinIDA==basini][ind[0],ind[1]]= np.abs(new_depth);
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
        print('working progress')

    def compareLithosphereAndPaleoDEMRecon(self, age, resolution, fuzzyAge=False):
        '''
        compareLithosphereAndPaleoDEMRecon is used to compare the locations
        where lithosphere ages and paleoDEMS are defined.


        Parameters
        -----------
        age : INT
            The age, in Myr, at which that user wants to read a
            paleoDEM from.
        resolution : FLOAT
            The resolution, in degree, at which the user wants to
            read the paleoDEM at.
        fuzzyAge : BOOLEAN, optional
            An option to read paleoDEMs with similar, but not exact
            ages as represented in age. For example, this allows
            a user to read in a plaeoDEM marked with ages that are
            +-0.5 Myr. The default is False.
        
        '''
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








class BathySynthetic():
    '''
    Bathy calculated with from mantle convection models / statistical analysis of planetary topography.
    '''

    def __init__(self):
        self.x = 1;
















#######################################################################
############# ExoCcycle Calculate Ccycle Bathymetry Params ############
#######################################################################
# Define some methods to calculate bathymetry attributes from 
def calculateHighLatA(bathymetry, latitudes, areaWeights, highlatP, verbose=True):
    """
    calculateHighLatA is a function used to calculate the high latitude 
    region of an input bathymetry model.

    Parameters
    ----------
    bathymetry : NUMPY ARRAY
        nx2n array representing seafloor depth, in m, with positive values.
    latitudes : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    areaWeights : NUMPY ARRAY
        An array of global degree to area weights. The size is dependent on
        input resolution. The sum of the array equals 4 pi radius^2 for 
        sufficiently high resolution, in m2.
    highlatP : FLOAT
        Value representing the amount of ocean surface area, in decimal percent,
        that should be included in the high latitude box.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Return
    -------
    self.highlatA : FLOAT
        Total high latitude ocean area, in m2. Note that this is not
        the hb[10] value in the LOSCAR earth system model. self.highlatP
        is hb[10].
    self.highlatlat : FLOAT
        The lowest latitude of the high latitude cutoff, in degree.    
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


def calculateBathymetryDistributionGlobal(bathymetry, latitudes, highlatlat, areaWeights, binEdges=None, verbose=True):
    """
    calculateBathymetryDistributionGlobal function calculates the bathymetry 
    distribution of a global or basin (depending on input) bathymetry
    model. These distrbutions can be useful as inputs for carbon cycle
    models.


    Parameters
    ----------
    bathymetry : NUMPY ARRAY
        nx2n array representing seafloor depth, in m, with positive values.
    latitudes : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    highlatlat : FLOAT
        The lowest latitude of the high latitude cutoff, in degree.
    areaWeights : NUMPY ARRAY
        An array of global degree to area weights. The size is dependent on
        input resolution. The sum of the array equals 4 pi radius^2 for 
        sufficiently high resolution, in m2.
    binEdges : NUMPY LIST, optional
        A numpy list of bin edges, in km, to calculate the bathymetry distribution
        over. Note that anything deeper than the last bin edge will be defined within
        the last bin. The default is None, but this is modified to 
        np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
        the code.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.
    
    Return
    -------
    bathymetryAreaDist_wHighlat : NUMPY ARRAY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        a histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the inclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    bathymetryAreaDist : NUMPY ARRAY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        a histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the exclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    binEdges : NUMPY LIST, optional
        A numpy list of bin edges, in km, to calculate the bathymetry distribution
        over. Note that anything deeper than the last bin edge will be defined within
        the last bin.
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



def calculateBathymetryDistributionBasin(bathymetry, latitudes, longitudes, basinIDA, highlatlat, areaWeights, binEdges=None, verbose=True):
    """
    calculateBathymetryDistributionBasin function calculates the bathymetry 
    distribution of a global or basin (depending on input) bathymetry
    model. These distrbutions can be useful as inputs for carbon cycle
    models.


    Parameters
    ----------
    bathymetry : NUMPY ARRAY
        nx2n array representing seafloor depth, in m, with positive values.
    latitudes : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-90, 90]. Latitudes change from row to row.
    longitudes : NUMPY ARRAY
        nx2n array representing cell registered latitudes, in deg,
        ranging from [-180, 180]. Latitudes change from row to row.
    basinIDA : NUMPY ARRAY
        nx2n array representing cell registered BasinID.
    highlatlat : FLOAT
        The lowest latitude of the high latitude cutoff, in degree.
    areaWeights : NUMPY ARRAY
        An array of global degree to area weights. The size is dependent on
        input resolution. The sum of the array equals 4 pi radius^2 for 
        sufficiently high resolution, in m2.
    binEdges : NUMPY LIST, optional
        A numpy list of bin edges, in km, to calculate the bathymetry distribution
        over. Note that anything deeper than the last bin edge will be defined within
        the last bin. The default is None, but this is modified to 
        np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
        the code.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.
    
    Return
    -------
    bathymetryAreaDist : DICTIONARY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        a histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the exclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    bathymetryVolFrac : DICTIONARY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        the precent basin volume, normalized to the volume of all ocean basins
        (excluding the high latitude ocean volume).
    bathymetryAreaFrac : DICTIONARY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        the precent basin area, normalized to the total seafloor area (including
        the high latitude area).
    bathymetryAreaDist_wHighlatG : NUMPY ARRAY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        a histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the inclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    bathymetryAreaDistG : NUMPY ARRAY
        A dictionary with entries ["Basin0", "Basin1",...]. Each entry contains
        a histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the exclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    binEdges : NUMPY LIST, optional
        A numpy list of bin edges, in km, to calculate the bathymetry distribution
        over. Note that anything deeper than the last bin edge will be defined within
        the last bin.
    """
    # Set bins default array.
    if binEdges is None:
        binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);
    
    # Setup dictionaries to hold outputs (basin distributions, area fractions, and volume fractions)
    bathymetryAreaDist = {};
    bathymetryAreaFrac = {};
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
        cmap = plt.get_cmap("Pastel1")
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

    return bathymetryAreaDist, bathymetryVolFrac, bathymetryAreaFrac, bathymetryAreaDist_wHighlatG, bathymetryAreaDistG, binEdges

























#######################################################################
############## ExoCcycle Define Ccycle Bathymetry Params ##############
#######################################################################
class CalculateLOSCARParam():
    '''
    CalculateLOSCARParam class is used to calculate bathymetry parameters used within the LOSCAR
    carbon cycle model.
    '''

    def __init__(self):
        self.x = 1;