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
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt

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
        self.x = 1;

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
                self.initiallykm = False;
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
            sufficiently high resolution.
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
                sufficiently high resolution.
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
                sufficiently high resolution.
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
        self.bathymetryAreaDist, self.bathymetryAreaDist_wHighlat, self.binEdges = calculateBathymetryDistribution(self.bathymetry, self.lat, self.highlatlat, areaWeights, binEdges = None, verbose=True);






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
            sufficiently high resolution.
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
        distribution_whighlat.units = 'kernal distribution'
        distribution_whighlat.standard_name = 'bathymetry-distribution-whighlat-G'

        distribution = ncfile.createVariable('bathymetry-distribution-G', np.float64, ('binEdges',))
        distribution.units = 'kernal distribution'
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
    Used for analogs of planets with active plate tectonics (using Earth reconstructions as proxy).
    '''

    def __init__(self):
        self.x = 1;


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
        sufficiently high resolution.
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
    AOC = np.nansum(np.nansum( areaWeights[~np.isnan(bathymetry)] ));

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
        percentArea = np.nansum(np.nansum( areaWeights[latitudes>highlatlat] ))/AOC;

    # Define bathymetry parameter
    highlatA = np.sum(np.sum( areaWeights[latitudes>highlatlat] ));

    # Report
    if verbose:
        print("The input high latitude area should cover {:2.0f}% of seafloor area.".format(highlatP));
        print("The high latitude cutoff is {:2.1f} degrees.".format(highlatlat));
        print("The high latitude area is {:2.0f} m2.".format(highlatA));

    return highlatlat, highlatA


def calculateBathymetryDistribution(bathymetry, latitudes, highlatlat, areaWeights, binEdges=None, verbose=True):
    """
    calculateBathymetryDistribution function calculates the bathymetry 
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
        sufficiently high resolution.
    binEdges : NUMPY LIST
        A numpy list of bin edges, in km, to calculate the bathymetry distribution
        over. Note that anything deeper than the last bin edge will be defined within
        the last bin. The default is None, but this is modified to 
        np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]) within
        the code.
    verbose : BOOLEAN, optional
        Reports more information about process. The default is True.

    Define
    -------
    self.bathymetryAreaDist : NUMPY LIST
        A histogram of seafloor bathymetry with using the following bin edges:
        0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, in km. Note
        that this distribution is calculated with the exclusion of high latitude
        distribution of seafloor depths. This is what is normally inputted into
        the LOSCAR carbon cycle model.
    self.bathymetryAreaDist_wHighlat : NUMPY LIST
        This is the same as bathymetryAreaDist, but includes the high latitude
        seafloor distribution of seafloor depths.
    
    Return
    -------
    None.
    """
    # Set bins default array.
    if binEdges is None:
        binEdges = np.array([0, 0.1, 0.6, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]);

    # Calculate bathymetry distribution of global bathymetry (including high
    # latitude areas).
    logical1 = ~np.isnan(bathymetry);
    bathy1   = (1e-3)*bathymetry[logical1];
    weights1 = areaWeights[logical1]/np.sum(areaWeights[logical1]);

    bathymetryAreaDist_wHighlat, binEdges = np.histogram(bathy1, bins=binEdges, density=True, weights=weights1);
    bathymetryAreaDist_wHighlat = 100*bathymetryAreaDist_wHighlat;

    # Calculate bathymetry distribution of global bathymetry (excluding high
    # latitude areas).
    logical2 = (latitudes<=highlatlat) & ~np.isnan(bathymetry);
    bathy2   = (1e-3)*bathymetry[logical2];
    weights2 = areaWeights[logical2]/np.sum(areaWeights[logical2]);

    bathymetryAreaDist, binEdges = np.histogram(bathy2, bins=binEdges, density=True, weights=weights2);
    bathymetryAreaDist = 100*bathymetryAreaDist;

    # Report
    if verbose:
        print("Bin edges used:\n", binEdges)
        print("Bathymetry area distribution including high latitude bathymetry:\n",bathymetryAreaDist_wHighlat);
        print("Bathymetry area distribution excluding high latitude bathymetry:\n",bathymetryAreaDist);

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
        plt.yticks(np.arange(0,110,10));

        # Labels
        plt.legend();
        plt.title("Planet's Bathymetry Distribution")
        plt.xlabel("Bathymetry Bins [km]");
        plt.ylabel("Seafloor Area [%]");

        # figure format
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    return bathymetryAreaDist, bathymetryAreaDist_wHighlat, binEdges

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