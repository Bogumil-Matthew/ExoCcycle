#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 7 2025

@author: Matthew Bogumil
"""
#######################################################################
############################### Imports ###############################
#######################################################################
# Import general libraries
import os
import copy as cp

# Import analysis libraries
import numpy as np
import pandas as pd

# Import plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec

# Import other modules
from ExoCcycle import Bathymetry    # type: ignore
from ExoCcycle import utils         # type: ignore


########################################################################
################ Plotting functions (Global-Regional)  #################
########################################################################
def plotGlobal(
    lat,
    lon,
    values,
    outputDir="",
    fidName="plotGlobal.png",
    cmapOpts={"cmap": "viridis",
              "cbar-title": "cbar-title",
              "cbar-range": [0, 1],
              "cbar-levels": [0, 0.5, 1]},
    pltOpts={"valueType": "Bathymetry",
             "valueUnits": "m",
             "plotTitle": "",
             "plotZeroContour": False,
             "plotIntegerContours": False,
             "transparent": False,
             "region": [-180, 180, -90, 90],
             "projection": ccrs.Mollweide(),
             "regionalZBoundaries": False},
    saveSVG=False,
    savePNG=False,
):
    """
    Plot a global (or regional) gridded field on a map projection using Cartopy.

    The function renders a latitude/longitude gridded dataset with optional mesh
    shading, coastlines, contours (zero or integer/label-like boundaries),
    regional masking and recoloring, and a horizontally oriented colorbar.
    Saving to PNG and/or SVG is supported.

    Args:
        lat (numpy.ndarray):  
            2-D array of **cell-centered** latitudes in degrees (same shape as ``values``),
            typically spanning ``[-90, 90]`` row-wise.
        lon (numpy.ndarray):  
            2-D array of **cell-centered** longitudes in degrees (same shape as ``values``),
            typically spanning ``[-180, 180]`` column-wise.
        values (numpy.ndarray):  
            2-D array of gridded data (same shape as ``lat``/``lon``). May include ``NaN``.
        outputDir (str, optional):  
            Directory to which the figure is saved. Defaults to ``""`` (current behavior
            depends on Matplotlib’s savefig path).
        fidName (str, optional):  
            Base filename for outputs (PNG and/or SVG). Defaults to ``"plotGlobal.png"``.
        cmapOpts (dict, optional):  
            Colormap and colorbar configuration. Recognized keys:
            - ``"cmap"`` (str or Colormap): Matplotlib colormap name or object.  
            - ``"cbar-title"`` (str): Title text for the colorbar (not always used).  
            - ``"cbar-range"`` (list[float, float]): ``[vmin, vmax]`` for color scaling.  
            - ``"cbar-levels"`` (list[float]): Optional levels.  
              (Note: passed to ``pcolormesh`` in a guarded try/except.)
        pltOpts (dict, optional):  
            Plot behavior and aesthetics. Recognized keys:
            - ``"valueType"`` (str): Label for data type, used in colorbar label.  
            - ``"valueUnits"`` (str): Units string for colorbar label.  
            - ``"plotTitle"`` (str): Figure title.  
            - ``"plotZeroContour"`` (bool): If True, overlay a contour at value 0.  
            - ``"plotIntegerContours"`` (bool): If True, draw boundaries around integer-valued regions.  
            - ``"transparent"`` (bool): If True, save figures with transparent background.  
            - ``"region"`` (list[float, float, float, float]): ``[lon_min, lon_max, lat_min, lat_max]``  
              extent in degrees. Full-globe default is ``[-180, 180, -90, 90]``.  
            - ``"projection"`` (cartopy.crs.Projection): Target map projection (default Mollweide).  
            - ``"regionalZBoundaries"`` (bool): If True, mask outside ``region`` and remap
              in-range values to consecutive integers starting at 0, updating the colorbar range.  
            - ``"mesh"`` (bool): If True, draw shaded mesh with ``pcolormesh`` (default True).  
            - ``"nanSolidPoly"`` (bool): If True, render NaN cells as solid polygons.  
            - ``"nanSolidPolyOutline"`` (bool): If True, outline contiguous NaN regions.  
            - ``"coastlines"`` (bool): If True, draw coastlines (blue, 1 px).
        saveSVG (bool, optional):  
            If True, saves an SVG copy (filename derived from ``fidName``). Defaults to False.
        savePNG (bool, optional):  
            If True, saves a PNG copy (uses ``fidName``). Defaults to False.

    Returns:
        None:  
            Writes a figure to disk if ``savePNG`` and/or ``saveSVG`` is True. Does not return a value.

    Raises:
        ValueError:  
            If array shapes of ``lat``, ``lon``, and ``values`` are inconsistent. *(Not enforced here; caller responsibility.)*
        FileNotFoundError:  
            If ``outputDir`` does not exist or is not writable. *(Depends on Matplotlib/OS behavior.)*
        Exception:  
            Any Matplotlib/Cartopy errors encountered during plotting or saving.

    Notes:
        - ``pltOpts["regionalZBoundaries"]`` masks values outside the ``region`` and then
          renumbers unique in-region values to contiguous integers starting at 0, updating
          ``cmapOpts["cbar-range"]`` accordingly.  
        - If ``pltOpts["plotZeroContour"]`` is True, NaNs in ``values`` are set to 0 for contouring.  
        - The colorbar tick spacing is heuristically derived from ``"cbar-range"`` to produce
          ~4 ticks.  
        - Gridlines are labeled for regional plots and unlabeled for full-globe plots.  
        - ``pcolormesh`` is called with ``transform=ccrs.PlateCarree()`` to map lon/lat to the projection.

    Example:
        Render a global field in a Mollweide projection and save a PNG:

        ```python
        cmapOpts = {"cmap": "viridis", "cbar-range": [-6000, 6000], "cbar-levels": [-6000, 0, 6000]}
        pltOpts = {"valueType": "Topography", "valueUnits": "m", "plotTitle": "ETOPO1",
                   "projection": ccrs.Mollweide(), "region": [-180, 180, -90, 90],
                   "plotZeroContour": True, "coastlines": True}
        plotGlobal(lat, lon, z, outputDir="figs", fidName="etopo1.png",
                   cmapOpts=cmapOpts, pltOpts=pltOpts, savePNG=True)
        ```
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

def plotGlobalwBoundaries(
    lat,
    lon,
    values,
    BasinIDA,
    outputDir="",
    fidName="plotGlobal_wBoundaries.png",
    cmapOpts={"cmap": "jet",
              "cbar-title": "cbar-title",
              "cbar-range": [0, 1]},
    pltOpts={"valueType": "Silhouette Structure",
             "valueUnits": "-",
             "plotTitle": "",
             "plotZeroContour": False,
             "nanSolidPoly": True,
             "nanSolidPolyOutline": True,
             "plotIntegerContours": True,
             "regionalZBoundaries": False,
             "region": [-180, 180, -90, 90],
             "projection": ccrs.Mollweide(),
             "transparent": True},
    saveSVG=False,
    savePNG=False,
):
    """
    Plot a global (or regional) scalar field with overlaid basin boundaries.

    Renders a gridded field (``values``) on a Cartopy map projection with optional
    masking to a region, solid polygons/contours to visualize NaN areas, and
    integer-boundary contours derived from a basin-ID array (``BasinIDA``). A
    horizontal colorbar is added when mesh shading is enabled. PNG and SVG export
    are supported.

    Args:
        lat (numpy.ndarray):  
            2-D array of **cell-centered** latitudes (degrees), same shape as ``values``.
        lon (numpy.ndarray):  
            2-D array of **cell-centered** longitudes (degrees), same shape as ``values``.
        values (numpy.ndarray):  
            2-D array of scalar data to shade (same shape as ``lat``/``lon``). May include ``NaN``.
        BasinIDA (numpy.ndarray):  
            2-D array of integer (or categorical) basin IDs aligned with ``values`` for
            boundary visualization (integer-step contouring and NaN overlays).
        outputDir (str, optional):  
            Directory to which the figure is saved. Defaults to ``""`` (current behavior
            depends on Matplotlib’s savefig path).
        fidName (str, optional):  
            Output filename for PNG; used as base to derive SVG name. Defaults to
            ``"plotGlobal_wBoundaries.png"``.
        cmapOpts (dict, optional):  
            Colormap and colorbar configuration. Recognized keys:
            - ``"cmap"`` (str): Colormap name (resolved via ``plt.cm.get_cmap``).  
            - ``"cbar-title"`` (str): Title for the colorbar (not always used).  
            - ``"cbar-range"`` (list[float, float]): ``[vmin, vmax]``.
        pltOpts (dict, optional):  
            Plot behavior and aesthetics. Recognized keys:
            - ``"valueType"`` (str): Label for data type (used in colorbar label).  
            - ``"valueUnits"`` (str): Units string for colorbar label.  
            - ``"plotTitle"`` (str): Figure title text.  
            - ``"plotZeroContour"`` (bool): If True, contour a zero isoline of ``values``.  
            - ``"nanSolidPoly"`` (bool): If True, shade NaN regions (from ``BasinIDA``) as solid polygons.  
            - ``"nanSolidPolyOutline"`` (bool): If True, outline contiguous NaN regions (from ``BasinIDA``).  
            - ``"plotIntegerContours"`` (bool): If True, draw boundaries for each integer basin ID.  
            - ``"regionalZBoundaries"`` (bool): If True, mask ``values`` outside ``region`` and remap
              surviving unique values to 0..N-1; updates ``cmapOpts["cbar-range"]``.  
            - ``"region"`` (list[float, float, float, float]): ``[lon_min, lon_max, lat_min, lat_max]`` extent.  
            - ``"projection"`` (cartopy.crs.Projection): Target map projection (default Mollweide).  
            - ``"transparent"`` (bool): If True, save with transparent background.  
            - ``"boundaryColor"`` (str, optional): Color for integer boundary contours (default ``'k'``).  
            - ``"boundaryLinewidth"`` (float, optional): Line width for boundary contours (default ``1``).  
            - ``"mesh"`` (bool, optional): If True, render the shaded mesh (default True).
        saveSVG (bool, optional):  
            If True, also save an SVG (name derived from ``fidName``). Defaults to False.
        savePNG (bool, optional):  
            If True, save a PNG using ``fidName``. Defaults to False.

    Returns:
        None:  
            Writes a figure to disk when ``savePNG`` and/or ``saveSVG`` is True.

    Raises:
        ValueError:  
            If input array shapes are inconsistent. *(Not validated here; caller responsibility.)*
        FileNotFoundError:  
            If ``outputDir`` is invalid/unwritable. *(Depends on Matplotlib/OS behavior.)*
        Exception:  
            Any Matplotlib/Cartopy errors encountered during plotting or saving.

    Notes:
        - ``regionalZBoundaries``: values outside the extent are set to ``NaN``; remaining unique
          values are renumbered sequentially starting at 0 to create a compact categorical range.  
        - Integer boundary contours are derived from ``BasinIDA`` by thresholding each ID and
          contouring at 0.5.  
        - NaN overlays/contours are also computed from ``BasinIDA``’s NaN mask (not ``values``)
          to highlight missing/invalid basins.  
        - Colormap is resolved via ``plt.cm.get_cmap(cmapOpts["cmap"])``.  
        - Gridlines are labeled for regional plots and hidden for full-globe plots.  
        - ``pcolormesh`` uses ``transform=ccrs.PlateCarree()`` to project lon/lat to the target projection.

    Example:
        Plot silhouette values with basin boundaries and save a transparent PNG:

        ```python
        pltOpts = {
            "valueType": "Silhouette",
            "valueUnits": "-",
            "plotTitle": "Basins & Silhouette",
            "projection": ccrs.Mollweide(),
            "region": [-180, 180, -90, 90],
            "plotIntegerContours": True,
            "transparent": True
        }
        cmapOpts = {"cmap": "viridis", "cbar-range": [0, 1]}
        plotGlobalwBoundaries(lat, lon, silhouette, basin_ids,
                              outputDir="figs",
                              fidName="silhouette_basins.png",
                              cmapOpts=cmapOpts, pltOpts=pltOpts,
                              savePNG=True)
        ```
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
                                vmin=cmapOpts['cbar-range'][0],
                                vmax=cmapOpts['cbar-range'][1],
                                zorder=0)
        except:
            mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmapOpts["cmap"],
                                vmin=cmapOpts['cbar-range'][0],
                                vmax=cmapOpts['cbar-range'][1],
                                zorder=0)

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
                       colors=pltOpts.get("boundaryColor", 'k'),
                       linewidths=pltOpts.get("boundaryLinewidth", 1),
                       transform=ccrs.PlateCarree(),
                       zorder=1)

            
            
    ## Add coastlines
    if coastlinesOpt:
        ax.coastlines(color='blue', linewidth=1, zorder=10)

    ## Add a colorbar
    if meshOpt:
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.7)

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

def plotGlobalSilhouette(
    lat,
    lon,
    values,
    BasinIDA,
    outputDir="",
    fidName="plotGlobal_silhouette.png",
    cmapOpts={"cmap": "jet",
              "cbar-title": "cbar-title",
              "cbar-range": [0, 1],
              "cbar-levels": np.array([0, .25, .5, .7, 1]),
              "cbar-intervals": np.array([0, .25, .5, .7, 1]),
              "cbar-level-names": ["No", "Weak", "Medium", "Strong"]},
    pltOpts={"valueType": "Silhouette Structure",
             "valueUnits": "-",
             "plotTitle": "",
             "plotZeroContour": False,
             "nanSolidPoly": True,
             "nanSolidPolyOutline": True,
             "plotIntegerContours": True,
             "regionalZBoundaries": False,
             "region": [-180, 180, -90, 90],
             "projection": ccrs.Mollweide(),
             "transparent": True},
    saveSVG=False,
    savePNG=False
):
    """
    Plot a global (or regional) **silhouette** field with discrete color levels and basin boundaries.

    Renders a gridded silhouette metric (``values``) on a Cartopy map using a **discrete**
    colormap defined by ``cmapOpts["cbar-levels"]`` and ``BoundaryNorm``. Optionally
    overlays basin boundaries derived from ``BasinIDA``, draws NaN masks, and labels a
    horizontal colorbar with category names supplied via ``"cbar-level-names"``.

    Args:
        lat (numpy.ndarray):  
            2-D array of cell-centered latitudes in degrees (same shape as ``values``).
        lon (numpy.ndarray):  
            2-D array of cell-centered longitudes in degrees (same shape as ``values``).
        values (numpy.ndarray):  
            2-D array of silhouette values to shade; may include ``NaN``.
        BasinIDA (numpy.ndarray):  
            2-D array of integer/categorical basin IDs aligned with ``values`` for
            boundary overlays and NaN masks.
        outputDir (str, optional):  
            Directory to which the figure is saved. Defaults to ``""`` (current behavior
            depends on Matplotlib’s savefig path).
        fidName (str, optional):  
            Filename for PNG output. The SVG name is derived by replacing ``.png`` with ``.svg``.
        cmapOpts (dict, optional):  
            Discrete colormap and colorbar settings:
            - ``"cmap"`` (str): Matplotlib colormap name (resolved via ``plt.cm.get_cmap``).  
            - ``"cbar-title"`` (str): Title for the colorbar (not always used).  
            - ``"cbar-range"`` (list[float, float]): Overall numeric range (may be unused with discrete norm).  
            - ``"cbar-levels"`` (array-like): Monotonic sequence of bin edges for discrete coloring.  
            - ``"cbar-intervals"`` (array-like): Provided but unused here (kept for API symmetry).  
            - ``"cbar-level-names"`` (list[str]): Labels for colorbar ticks; one fewer or equal to the number
              of edges depending on labeling strategy (this function labels mid-bin positions).
        pltOpts (dict, optional):  
            Plot behavior and aesthetics:
            - ``"valueType"`` (str): Label for data type (used in colorbar label).  
            - ``"valueUnits"`` (str): Units for colorbar label.  
            - ``"plotTitle"`` (str): Figure title.  
            - ``"plotZeroContour"`` (bool): If True, overlays a zero contour of ``values``.  
            - ``"nanSolidPoly"`` (bool): If True, fills NaN regions derived from ``BasinIDA``.  
            - ``"nanSolidPolyOutline"`` (bool): If True, outlines contiguous NaN regions.  
            - ``"plotIntegerContours"`` (bool): If True, draws per-basin boundary contours from ``BasinIDA``.  
            - ``"regionalZBoundaries"`` (bool): If True, masks ``values`` outside ``"region"`` and remaps
              unique in-range values to consecutive integers starting at 0; updates colorbar range.  
            - ``"region"`` (list[float, float, float, float]): Extent ``[lon_min, lon_max, lat_min, lat_max]``.  
            - ``"projection"`` (cartopy.crs.Projection): Target projection (default Mollweide).  
            - ``"transparent"`` (bool): If True, saves with transparent background.
        saveSVG (bool, optional):  
            Save an SVG alongside PNG. Defaults to ``False``.
        savePNG (bool, optional):  
            Save a PNG using ``fidName``. Defaults to ``False``.

    Returns:
        None:  
            Writes figure(s) to disk if requested; no value is returned.

    Raises:
        ValueError:  
            If input array shapes are inconsistent. *(Not explicitly validated here.)*  
        FileNotFoundError:  
            If ``outputDir`` is invalid/unwritable. *(Depends on Matplotlib/OS behavior.)*  
        Exception:  
            Any Matplotlib/Cartopy runtime errors encountered during plotting or saving.

    Notes:
        - **Discrete colormap:** Coloring uses ``mpl.colors.BoundaryNorm(cbar-levels, cmap.N)`` so bins are
          closed-open intervals aligned to your silhouette thresholds.  
        - **Colorbar ticks:** Tick positions are set at bin midpoints and labeled with ``"cbar-level-names"``.  
        - **Boundaries:** Basin boundaries are drawn by thresholding each integer ID in ``BasinIDA`` and
          contouring at 0.5. NaN masks (fill/outline) are derived from ``BasinIDA`` NaNs.  
        - **Regional masking:** With ``regionalZBoundaries=True``, values outside ``region`` are set to NaN
          and remaining unique values are renumbered 0..N-1, updating the colorbar range accordingly.  
        - ``pcolormesh`` uses ``transform=ccrs.PlateCarree()`` to map lon/lat into the selected projection.

    Example:
        Plot silhouette strength categories with custom labels:

        ```python
        cmapOpts = {
            "cmap": "viridis",
            "cbar-levels": np.array([0.0, 0.25, 0.5, 0.7, 1.0]),
            "cbar-level-names": ["No", "Weak", "Medium", "Strong"]
        }
        pltOpts = {
            "valueType": "Silhouette",
            "valueUnits": "-",
            "plotTitle": "Silhouette categories",
            "projection": ccrs.Mollweide(),
            "plotIntegerContours": True,
            "transparent": True
        }
        plotGlobalSilhouette(lat, lon, sil_values, basin_ids,
                             outputDir="figs", fidName="silhouette.png",
                             cmapOpts=cmapOpts, pltOpts=pltOpts, savePNG=True)
        ```
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

def plotGlobalwHist(
    lat,
    lon,
    values,
    binEdges,
    bathymetryAreaDist_wHighlat,
    bathymetryAreaDist,
    highlatlat,
    outputDir="",
    fidName="plotGlobal.png",
    cmapOpts={"cmap": "viridis",
              "cbar-title": "cbar-title",
              "cbar-range": [0, 1]},
    pltOpts={"valueType": "Bathymetry",
             "valueUnits": "m",
             "plotTitle": "",
             "plotZeroContour": False},
    saveSVG=False,
    savePNG=False
):
    """
    Plot a global (Mollweide) gridded map with a stacked histogram underneath.

    The top panel renders a latitude/longitude gridded field (``values``) on a
    Cartopy Mollweide projection with an accompanying horizontal colorbar.
    The bottom panel draws a side-by-side bar chart comparing two bathymetry
    area distributions across common bins (``binEdges``).

    Args:
        lat (numpy.ndarray):  
            2-D array of **cell-centered** latitudes in degrees, same shape as ``values``.
        lon (numpy.ndarray):  
            2-D array of **cell-centered** longitudes in degrees, same shape as ``values``.
        values (numpy.ndarray):  
            2-D array of gridded data to shade (same shape as ``lat``/``lon``). May contain ``NaN``.
        binEdges (array-like):  
            Monotonically increasing 1-D array of bin edges (length ``N+1``) for the histogram x-axis.
        bathymetryAreaDist_wHighlat (array-like):  
            Length-``N`` array of area percentages (or other weights) for **all** bathymetry.
        bathymetryAreaDist (array-like):  
            Length-``N`` array of area percentages for the bathymetry **excluding** high-latitude data.
        highlatlat (float):  
            Latitude threshold (in degrees) used to define the “high-latitude” removal, for legend text.
        outputDir (str, optional):  
            Directory to which the figure is saved. Defaults to ``""`` (current behavior
            depends on Matplotlib’s savefig path).
        fidName (str, optional):  
            PNG output filename (also used to derive SVG name). Defaults to ``"plotGlobal.png"``.
        cmapOpts (dict, optional):  
            Colormap configuration:
            - ``"cmap"`` (str or Colormap): Name/object passed to ``pcolormesh``.  
            - ``"cbar-title"`` (str): (Not directly used here; label is derived from ``pltOpts``).  
            - ``"cbar-range"`` (list[float, float]): ``[vmin, vmax]`` for shading.
        pltOpts (dict, optional):  
            Plot appearance and behavior:
            - ``"valueType"`` (str): Label for colorbar (e.g., *Bathymetry*).  
            - ``"valueUnits"`` (str): Units label (e.g., *m*).  
            - ``"plotTitle"`` (str): Overall figure title.  
            - ``"plotZeroContour"`` (bool): If True, overlay a 0-contour on the map panel.
        saveSVG (bool, optional):  
            If True, save an SVG copy (``fidName`` with ``.svg``). Defaults to False.
        savePNG (bool, optional):  
            If True, save a PNG using ``fidName``. Defaults to False.

    Returns:
        None:  
            Writes the composed figure to disk when requested; no value is returned.

    Raises:
        ValueError:  
            If the histogram inputs have incompatible lengths (caller responsibility; not checked here).
        FileNotFoundError:  
            If ``outputDir`` is invalid/unwritable. *(Depends on Matplotlib/OS behavior.)*
        Exception:  
            Any Cartopy/Matplotlib errors encountered during plotting or saving.

    Notes:
        - The upper map uses ``pcolormesh`` with ``transform=ccrs.PlateCarree()`` onto a Mollweide axes.  
        - The colorbar label is formed from ``pltOpts["valueType"]`` and ``pltOpts["valueUnits"]``.  
        - The histogram shows two bar series at each bin center, offset slightly to avoid overlap.  
        - Gridlines are drawn on the map via ``ax1.gridlines()`` with library defaults.

    Example:
        Render a bathymetry map and a bin-wise area comparison:

        ```python
        cmapOpts = {"cmap": "viridis", "cbar-range": [-6000, 6000]}
        pltOpts  = {"valueType": "Bathymetry", "valueUnits": "m", "plotTitle": "Global bathymetry",
                    "plotZeroContour": True}
        plotGlobalwHist(lat, lon, z, binEdges,
                        area_all, area_no_highlat, 60.0,
                        outputDir="figs", fidName="bathymetry_map_hist.png",
                        cmapOpts=cmapOpts, pltOpts=pltOpts, savePNG=True)
        ```
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

def plotLargeBasins(
    basins,
    fieldNum="Field1",
    percentage=0.05,
    fldName="",
    savePNG=False,
    verbose=True,
):
    """
    Plot only the basins whose area exceeds a given fraction of the domain.

    This function computes area-weighted sizes of each basin (using
    ``basins.areaWeights`` masked by ``basins.BasinIDA``), removes basins whose
    area is below ``percentage * total_area``, reindexes the remaining basins to
    consecutive integers starting at 0, and renders a categorical global map via
    :func:`plotGlobal`.

    Args:
        basins:  
            ExoCcycle-style object that provides at least the attributes:
            - ``lat`` (2-D array): cell-centered latitudes (deg)
            - ``lon`` (2-D array): cell-centered longitudes (deg)
            - ``BasinIDA`` (2-D array): integer/categorical basin IDs (``NaN`` for no data)
            - ``areaWeights`` (2-D array): area per cell (same shape as ``BasinIDA``)
            - ``Fields`` (dict-like): metadata; accessed as ``basins.Fields[fieldNum]``
        fieldNum (str, optional):  
            Key into ``basins.Fields`` used **only** to label the colorbar title as
            ``"BasinID divided by {parameterName}"``. Defaults to ``"Field1"``.
        percentage (float, optional):  
            Minimum fraction of the total represented area a basin must cover to be
            retained (e.g., ``0.05`` for 5%). Defaults to ``0.05``.
        fldName (str, optional):  
            Directory to save the resulting figure. Defaults to ``""`` (current behavior
            depends on :func:`plotGlobal`).
        savePNG (bool, optional):  
            If ``True``, writes a PNG to disk using :func:`plotGlobal`. Defaults to ``False``.
        verbose (bool, optional):  
            If ``True``, prints summary information about kept/removed basins. Defaults to ``True``.

    Returns:
        None:  
            Writes a figure to disk when ``savePNG`` is True; no value is returned.

    Raises:
        KeyError:  
            If ``fieldNum`` is not present in ``basins.Fields``. *(Not validated here.)*
        ValueError:  
            If ``percentage`` is outside ``[0, 1]``. *(Not validated here.)*
        Exception:  
            Any errors raised downstream by :func:`plotGlobal` or file I/O.

    Notes:
        - Basins whose area (sum of ``areaWeights`` within each basin ID) is
          below ``percentage * total_area`` are set to ``NaN`` and excluded.
        - Remaining basin IDs are **reindexed** to 0..N-1 prior to plotting so that
          colormap limits reflect the new categorical range.
        - The plotted colorbar label uses
          ``basins.Fields[fieldNum]['parameterName']`` for context.

    Example:
        ```python
        plotLargeBasins(basins, fieldNum="Field1", percentage=0.1,
                        fldName="figs", savePNG=True, verbose=True)
        ```
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


########################################################################
################ Plotting functions (Graph-Weighting)  #################
########################################################################
def plot_quantile_transform_distribution(x_values, qt_transformer, qt_diss=None, field_name=None,
                                         bins=20, show=True, save_path=None):
    """
    Plot quantile transformation distributions for data vs. Gaussian (QT) domain.

    This function visualizes how an input dataset maps between its native domain
    and the quantile-transformed (Gaussianized) domain used in statistical weighting.
    It generates two subplots:
        1. Histogram of transformed values (QT domain)
        2. Comparison between original and back-transformed (inverse QT) data

    Args:
        x_values (array-like):  
            Original data values to transform.
        qt_transformer (object):  
            A fitted scikit-learn quantile transformer (e.g., 
            ``sklearn.preprocessing.QuantileTransformer``) or similar object with
            ``.transform`` and ``.inverse_transform`` methods.
        qt_diss (array-like, optional):  
            Already-transformed data (QT domain). If None, it is computed from ``x_values``.
        field_name (str, optional):  
            Optional field label for plot legends/titles.
        bins (int, optional):  
            Number of bins for histograms. Defaults to 20.
        show (bool, optional):  
            Whether to immediately show the plots using ``plt.show()``. Defaults to True.
        save_path (str, optional):  
            Path to save the figure (e.g., ``"figures/qt_transform_plot.png"``). If None, the figure
            is not saved.

    Returns:
        matplotlib.figure.Figure:  
            The created matplotlib figure instance.

    Example:
        ```python
        from sklearn.preprocessing import QuantileTransformer
        import numpy as np

        qt = QuantileTransformer(output_distribution='normal')
        x = np.random.lognormal(mean=0, sigma=1, size=5000)
        qt.fit(x.reshape(-1, 1))
        plot_quantile_transform_distribution(x, qt)
        ```

    """
    # Ensure numpy array
    x_values = np.asarray(x_values)

    # Compute transformed data if not given
    if qt_diss is None:
        qt_diss = qt_transformer.transform(x_values.reshape(-1, 1))

    # Create equally spaced bins in the data domain
    bins_data = np.linspace(np.min(x_values), np.max(x_values), bins)
    bins_qt = qt_transformer.transform(bins_data.reshape(-1, 1))

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))

    # --- (1) QT-domain histogram ---
    plt.sca(axes[0])
    plt.hist(qt_diss, alpha=1, bins=np.arange(-6, 6, 0.1), label='Transformed (QT domain)', density=True)
    plt.vlines(x=bins_qt, ymin=0, ymax=0.4, colors='r', alpha=0.3)
    plt.title("Quantile-Transformed Distribution")
    plt.ylabel("Density")
    plt.legend()

    # --- (2) Original vs. inverse-QT histogram ---
    plt.sca(axes[1])
    hist = plt.hist(x_values, alpha=0.5, bins=bins_data, label='Original Data', density=True)
    plt.vlines(x=bins_data, ymin=0, ymax=np.max(hist[0]), colors='r', alpha=0.3)
    qtx_values = qt_transformer.inverse_transform(qt_diss)
    plt.hist(qtx_values, alpha=0.5, bins=bins_data, label='Back-Transformed (QT→Data)', density=True)
    plt.ylabel("Density")
    plt.xlabel("Data Values")
    plt.title("Original vs. Inverse QT Comparison")
    plt.legend()

    plt.tight_layout()

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig


def plot_pdf_vs_sigmoid(
    data_edge_diff_iqr_filtered,
    pdf,
    popt,
    sigmoid_fn,
    bins=30,
    title=None,
    show=True,
    save_path=None,
):
    """
    Plot a fitted sigmoid against an empirical PDF, plus a histogram of the data.

    This function mirrors your snippet by:
      1) Mirroring the input data as ``x_values = [+x, -x]`` to enforce symmetry,
      2) Plotting the **normalized** PDF and **normalized** sigmoid in the top subplot,
      3) Plotting the **raw** PDF, a histogram of the mirrored data, and the **raw** sigmoid
         in the bottom subplot.

    Parameters
    ----------
    data_edge_diff_iqr_filtered : array-like, shape (N,)
        One-sided data array (e.g., positive differences). The function mirrors this
        array to construct ``x_values = np.append(x, -x)`` for plotting.
    pdf : array-like, shape (2N,)
        Empirical probability density values corresponding **one-to-one** with the
        mirrored ``x_values`` ordering used here (``[x, -x]`` in that sequence).
    popt : sequence
        Optimized parameters for the passed ``sigmoid_fn`` (e.g., from ``scipy.optimize.curve_fit``).
    sigmoid_fn : callable
        A callable with signature ``sigmoid_fn(x, *popt)`` that evaluates the sigmoid
        at vector ``x``.
    bins : int, optional
        Number of histogram bins for the data panel. Default is 30.
    title : str, optional
        Figure-level title. Default None (no suptitle).
    show : bool, optional
        If True, call ``plt.show()`` at the end. Default True.
    save_path : str, optional
        If provided, save the figure to this path (e.g., ``"figs/pdf_sigmoid.png"``).

    Returns
    -------
    (fig, axes) : tuple
        The created Matplotlib figure and array of axes (length 2).

    Notes
    -----
    - The top panel normalizes both curves for a shape-only comparison:
      ``pdf / max(pdf)`` and ``sigmoid(x)/sigmoid(0)``.
    - The bottom panel compares raw magnitudes: empirical PDF, histogram,
      and raw sigmoid evaluated on the mirrored grid.
    - Ensure ``pdf`` is aligned with the constructed mirrored ``x_values``. If your
      original PDF is defined on a different grid or order, resample/reorder before use.

    Example
    -------
    >>> # Define a standard logistic sigmoid:
    >>> def sigmoid(x, a, b, c, d):
    ...     return a / (1.0 + np.exp(-b*(x - c))) + d
    ...
    >>> x_pos = np.linspace(0, 3, 200)
    >>> x_all = np.append(x_pos, -x_pos)
    >>> pdf = np.exp(-0.5*(x_all/0.8)**2)  # toy Gaussian-shaped PDF on mirrored grid
    >>> popt = (1.0, 2.0, 0.0, 0.0)        # toy params
    >>> fig, axes = plot_pdf_vs_sigmoid(x_pos, pdf, popt, sigmoid, bins=30, show=False)
    >>> plt.show()
    """
    # 1) Build mirrored x-values
    x_values = np.append(np.asarray(data_edge_diff_iqr_filtered),
                         -np.asarray(data_edge_diff_iqr_filtered))

    # Sanity checks (lightweight)
    pdf = np.asarray(pdf)
    if pdf.shape != x_values.shape:
        raise ValueError(
            f"`pdf` must have the same shape as mirrored data x_values: "
            f"{pdf.shape} != {x_values.shape}"
        )

    # 2) Prepare figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    # --- Top subplot: normalized comparison (shape-only) ---
    plt.sca(axes[0])
    pdf_max = np.nanmax(pdf) if np.size(pdf) else 1.0
    pdf_max = pdf_max if pdf_max not in (0.0, np.nan) else 1.0

    # Normalize pdf by its max
    pdf_norm = pdf / pdf_max

    # Normalize sigmoid by its value at x=0 (avoid div-by-zero)
    s0 = sigmoid_fn(0.0, *popt)
    s0 = s0 if (isinstance(s0, (int, float, np.floating)) and s0 != 0) else 1.0
    sig_norm = sigmoid_fn(x_values, *popt) / s0

    plt.plot(x_values, pdf_norm, 'b.', alpha=1.0, label='pdf (normalized)')
    plt.plot(x_values, sig_norm, 'r.', alpha=1.0, label='sigmoid (normalized)')
    plt.ylabel("Normalized weight")
    plt.legend(loc="best")

    # --- Bottom subplot: raw comparison + histogram ---
    plt.sca(axes[1])
    plt.plot(x_values, pdf, 'b.', alpha=1.0, label='pdf (raw)')
    # Histogram of the mirrored data
    plt.hist(x_values, alpha=0.4,
             bins=np.linspace(np.min(x_values), np.max(x_values), bins),
             label='Data', density=True)
    plt.plot(x_values, sigmoid_fn(x_values, *popt), 'r.', alpha=1.0, label='sigmoid (raw)')
    plt.ylabel("Unnormalized weight")
    plt.xlabel("Difference in Node Property")
    plt.legend(loc="best")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig, axes



