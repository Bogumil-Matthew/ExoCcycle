#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:20:00 2024

@author: Matthew Bogumil
"""
#######################################################################
############################### Imports ###############################
#######################################################################
# Import general libraries
import os
import copy as cp
import multiprocessing

# Import analysis libraries
import numpy as np
import pandas as pd
from netCDF4 import Dataset 
import itertools

# Import plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cartopy.crs as ccrs # type: ignore
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Graph tools
import networkx as nx # type: ignore
import ctypes
import igraph as ig
import leidenalg
import louvain

# For progress bars
from tqdm.auto import tqdm # used for progress bar

# Import other modules
from ExoCcycle import Bathymetry # type: ignore
from ExoCcycle import plotHelper # type: ignore

################################################################
###################### Functions to Sort #######################
################################################################
def combine_lists(*lists):
    """
    Compute the Cartesian product of multiple input lists.

    Returns a NumPy array of shape ``(number_of_combinations, N)``, where each
    row is one unique combination (i.e., the Cartesian product) of the input lists.

    Parameters
    ----------
    *lists : sequence of array_like
        One or more input sequences. Each sequence should contain only numeric
        values (``int`` or ``float``). Lists need not be the same length.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape ``(number_of_combinations, N)``, containing all
        unique combinations from the input lists.

    Raises
    ------
    ValueError
        If no input lists are provided.

    Notes
    -----
    If inputs contain mixed data types or non-numeric values, NumPy may upcast
    the array to ``dtype=object``. This function is intended for numeric inputs.

    See Also
    --------
    itertools.product : Python standard library function that computes Cartesian products.

    Examples
    --------
    Build combinations of file IDs and ensemble sizes.

    >>> import numpy as np
    >>> import itertools
    >>> file_names = ["file1.nc", "file2.nc"]
    >>> file_name_ids = [0, 1]
    >>> ensemble_sizes = [10, 50]
    >>> A = combine_lists(file_name_ids, ensemble_sizes)
    >>> for file_id, ens in A:
    ...     print("values:", file_id, ens, file_names[file_id])
    values: 0 10 file1.nc
    values: 0 50 file1.nc
    values: 1 10 file2.nc
    values: 1 50 file2.nc
    """
    # Generate all combinations (cartesian product)
    combinations = list(itertools.product(*lists))
    
    # Convert to numpy array
    A = np.array(combinations)
    
    return A


###############################################################################################
###################### Helper Functions (Download & Directory Creation) #######################
###############################################################################################
def create_file_structure(list_of_directories, root=False, verbose=True):
    """
    Create directories from a provided list of paths.

    Each path in ``list_of_directories`` is created if it does not already exist.
    When ``root`` is ``False`` (default), each provided path is treated as
    relative to the current working directory (``os.getcwd()``) and is
    **prepended** with it. When ``root`` is ``True``, the paths are used as-is
    (e.g., absolute paths like ``/data/...``).

    Args:
        list_of_directories (list[str]):  
            List of directory paths to create, e.g.:

            ``["/data",
            "/data/folder1", "/data/folder1/folder1",
            "/data/folder2", "/data/folder2/folder1",
            "/data/folder3"]``

            Note: This function creates exactly the directories provided in this
            list; it does **not** automatically create missing parents unless
            they are also included in the list and appear earlier.
        root (bool, optional):  
            If ``True``, use each path exactly as given.  
            If ``False``, prefix each path with ``os.getcwd()``. Defaults to ``False``.
        verbose (bool, optional):  
            If ``True``, prints a message listing any directories that already
            existed. Defaults to ``True``.

    Returns:
        None:  
            Writes directories to the filesystem; no return value.

    Raises:
        PermissionError:  
            If the process lacks permission to create one or more directories.
        FileNotFoundError:  
            If a parent directory does not exist and was not provided earlier
            in ``list_of_directories``.
        OSError:  
            For other OS-level errors triggered by ``os.mkdir``.

    Notes:
        - Paths are created with ``os.mkdir`` (single-level). Provide parent
          directories first if they do not already exist.
        - Existing directories are collected and, when ``verbose`` is ``True``,
          reported at the end of execution.
        - This function uses simple string concatenation when ``root`` is
          ``False`` (``os.getcwd() + path``). Ensure your provided paths begin
          with a path separator (e.g., ``"/data"``) for the intended result.

    Example:
        Create a small tree under the current working directory:

        ```python
        create_file_structure(
            ["/data", "/data/raw", "/data/processed"],
            root=False,
            verbose=True
        )
        ```
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
    Create a uniquely named folder in a numbered series.

    The function attempts to create a new folder following the pattern
    ``"{fldBase}_{i}"``, where ``i`` starts at 0 and increments until
    a non-existing folder name is found or ``maxFolders`` is reached.
    Once a folder is successfully created, its name (including the
    numeric suffix) is returned.

    Args:
        fldBase (str, optional):  
            The base name for the folder series (e.g., ``"run"`` will produce
            ``run_0``, ``run_1``, ...). Defaults to ``"fldBase"``.
        maxFolders (int, optional):  
            The maximum number of folders to attempt before giving up.
            Defaults to ``1000``.

    Returns:
        str:  
            The name (path) of the folder successfully created, e.g. ``"fldBase_3"``.

    Raises:
        FileExistsError:  
            If all possible folder names up to ``maxFolders - 1`` already exist.
        PermissionError:  
            If the process lacks permission to create folders in the current directory.
        OSError:  
            For other OS-level errors during directory creation.

    Notes:
        - The function stops and returns as soon as a new folder is successfully created.  
        - If all ``maxFolders`` already exist, the function raises ``FileExistsError``.  
        - Folders are created in the current working directory unless ``fldBase`` includes a path.  

    Example:
        Create the first available folder in a series named ``simulation_0``, ``simulation_1``, etc.:

        ```python
        folder_path = makeFolderSeries("simulation", maxFolders=100)
        print(f"Created folder: {folder_path}")
        ```
    """
    # initialize counter
    i = 0
    
    # Loop over maxFolders
    for i in range(maxFolders):
        # Try to make folder and break if folder is created.
        try:
            os.mkdir('{0}_{1}'.format(fldBase, i))
            # folder was created successfully, return path/name
            return '{0}_{1}'.format(fldBase, i)
        except FileExistsError:
            # folder already exists, continue
            continue
        except Exception as e:
            # other errors: re-raise for visibility
            raise e
    # if loop completes without success
    raise FileExistsError(
        f"All possible folder names ({fldBase}_0 ... {fldBase}_{maxFolders-1}) already exist."
    )

def downloadSolarSystemBodies(data_dir):
    """
    Download and process standardized topography models for key Solar System bodies.

    This function downloads and prepares global topography models for all
    planetary bodies integrated into **ExoCcycle**, currently including
    *Venus*, *Earth*, *Mars*, and the *Moon*.  
    For each body, the function:
    1. Downloads raw topography data.  
    2. Creates a NetCDF file at 1° resolution.  
    3. Re-runs the postprocessing routine to generate GMT-compatible outputs.

    Args:
        data_dir (str):  
            Path to the base directory where local data are stored.  
            The function will create or use the subdirectory  
            ``[data_dir]/topographies`` to store downloaded files.

    Returns:
        None:  
            The function writes new NetCDF and GMT-compatible topography files
            to disk but does not return a value.

    Raises:
        FileNotFoundError:  
            If ``data_dir`` does not exist or is inaccessible.
        ImportError:  
            If the required ``Bathymetry`` module is not available.
        Exception:  
            For unexpected download, read, or file I/O failures.

    Notes:
        - The function uses :class:`Bathymetry.BathyMeasured` from the
          **ExoCcycle** package to handle downloads and resampling.  
        - Each planetary body is processed at a default resolution of 1°.  
        - If files already exist in ``data_dir/topographies``, they may be
          reused or overwritten depending on implementation of
          ``Bathymetry.BathyMeasured``.

    Example:
        Download topography datasets for all supported Solar System bodies
        into a local ``data`` directory:

        ```python
        from ExoCcycle import Bathymetry
        downloadSolarSystemBodies("./data")
        ```

        This will populate:
        ```
        ./data/topographies/venus_1deg.nc
        ./data/topographies/earth_1deg.nc
        ./data/topographies/mars_1deg.nc
        ./data/topographies/moon_1deg.nc
        ```
    """
    # Define the set of bodies to be downloaded.
    bodies = ["venus", "earth", "mars", "moon"]

    # Iterate over solar system bodies.
    for bodyi in bodies:
        # Create object for topography model.
        bodyBathymetry = Bathymetry.BathyMeasured(body=bodyi)

        # Download raw topography model.
        bodyBathymetry.getTopo(data_dir)

        # Create body topography netCDF with standard resolution.
        bodyBathymetry.readTopo(data_dir, new_resolution=1, verbose=False)

        # Run function again: will create a gmt post script of topography 
        # since the netCDF with standard resolution has already been created.
        bodyBathymetry.readTopo(data_dir, new_resolution=1, verbose=False)

#############################################################################################
###################### Helper Functions (Field creation & Statistics) #######################
#############################################################################################
def filterNc(options={"inputFile": None, "outputFile": None, "threshold": None, "lr": None, "gt": None},
             keepVars=['lat', 'lon', 'bathymetry']):
    """
    Filters and copies selected variables from a NetCDF file into a new one.

    This function opens an input NetCDF file, copies only the specified
    variables and their required dimensions, and optionally filters one
    variable by a numeric threshold (setting values to ``NaN``). The result
    is written to a new NetCDF file in ``NETCDF4_CLASSIC`` format while
    preserving global attributes.

    Args:
        options (dict, optional):  
            Configuration dictionary controlling file I/O and filtering.  
            Recognized keys include:

            - **inputFile** (`str` or `path-like`): Path to the source NetCDF file to read. **Required**.  
            - **outputFile** (`str` or `path-like`, optional): Path to the destination NetCDF file to create.  
              Defaults to ``os.getcwd() + "/filteredNc.nc"``.  
            - **threshold** (`float`, optional): Threshold used for filtering values in ``varName``.  
            - **le** (`bool`, optional): If True, sets values ``<= threshold`` in ``varName`` to ``NaN``.  
            - **gt** (`bool`, optional): If True, sets values ``> threshold`` in ``varName`` to ``NaN``.  
            - **varName** (`str`, optional): Name of the variable to which the threshold filtering applies.  

            If a key is not provided, a default value is assigned at runtime.  
            Both ``le`` and ``gt`` cannot be True simultaneously — an error will be raised if this occurs.

        keepVars (list of str, optional):  
            List of variable names to retain and copy into the output file.
            Only dimensions required by these variables are written.
            Defaults to ``['lat', 'lon', 'bathymetry']``.

    Raises:
        ValueError:  
            - If both ``le`` and ``gt`` are True (mutually exclusive filtering).  
            - If ``inputFile`` is not defined in the options dictionary.

    Returns:
        None:  
            The function writes a new NetCDF file to disk and does not return a value.

    Notes:
        - Filtering is applied only when the variable name matches ``options["varName"]``.  
        - Other listed variables are copied verbatim.  
        - The input file is never modified; a new file is always created.  
        - If the filtered variable has an integer dtype, assigning ``NaN`` may upcast
          the result when read back into NumPy.

    See Also:
        netCDF4.Dataset: Interface for reading and writing NetCDF files.  
        numpy.isnan: Check for NaN values.  
        os.getcwd: Used to determine the default output path.

    Example:
        Copy latitude, longitude, and bathymetry variables while filtering
        bathymetry values below sea level:

        ```python
        import os

        opts = {
            "inputFile": "in.nc",
            "outputFile": os.path.join(os.getcwd(), "out.nc"),
            "threshold": 0.0,
            "le": True,    # set values <= 0.0 to NaN
            "gt": False,   # do not filter values > threshold
            "varName": "bathymetry"
        }

        filterNc(opts, keepVars=["lat", "lon", "bathymetry"])  # doctest: +SKIP
        ```

        The resulting file ``out.nc`` will contain only the specified variables
        and their dimensions, with bathymetry filtered according to the given threshold.
    """
    # Check options have been defined, if not then assign a default
    optionsList = ["inputFile", "outputFile", "threshold", "le", "gt"]
    optionsListDefault = [None, os.getcwd() + "/filteredNc.nc", 0, None, None]
    for optionName, option in zip(optionsList, optionsListDefault):
        options[optionName] = options.get(optionName, option)

    # Throw errors based on inputs
    if (options["le"] is None) & (options["gt"] is None):
        raise ValueError(
            'Both "le" (less-than-or-equal) and "gt" (greater-than) options '
            'cannot be True simultaneously. Choose only one threshold direction.'
        )

    if options["inputFile"] is None:
        raise ValueError(
            '"inputFile" (path to input .nc file) option in options input '
            'was not defined.'
        )

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
                var = variable[:]
                if options["le"]:
                    var.data[var.data <= options["threshold"]] = np.nan
                if options["gt"]:
                    var.data[var.data > options["threshold"]] = np.nan
                # Copy modified variable data
                x[:] = var
            else:
                # Copy variable data
                x[:] = variable[:]

    # Copy global attributes
    dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

    # Close files
    src.close()
    dst.close()

def weightedAvgAndStd(values, weights):
    """
    Compute the weighted average and weighted standard deviation.

    The function calculates the mean and standard deviation of a numeric array
    while accounting for a set of weights. NaN values in ``values`` are masked,
    and the corresponding entries in ``weights`` are ignored.  
    The weights are normalized internally so that they sum to 1 (hence they
    must not all be zero or NaN).

    Args:
        values (numpy.ndarray):  
            Array of numeric data for which to compute the weighted average and
            standard deviation.
        weights (numpy.ndarray):  
            Array of non-negative weights corresponding to ``values``.
            Must have the same shape as ``values``.

    Returns:
        tuple[float, float]:  
            A tuple ``(average, standard_deviation)`` containing the weighted
            mean and weighted standard deviation.

    Raises:
        ValueError:  
            If ``values`` and ``weights`` do not have the same shape.
        ZeroDivisionError:  
            If all weights are zero or NaN (cannot normalize).
        TypeError:  
            If input arrays contain non-numeric types.

    Notes:
        - NaN values in ``values`` are automatically masked out.  
        - The function uses :func:`numpy.average` for mean and variance computation.  
        - Variance is computed efficiently as  
          ``np.average((values - mean)**2, weights=weights)``.  
        - Returned standard deviation is the square root of this weighted variance.

    Example:
        Compute a weighted mean and standard deviation:

        ```python
        import numpy as np
        values = np.array([1.0, 2.0, 3.0, np.nan])
        weights = np.array([0.2, 0.3, 0.5, np.nan])
        avg, std = weightedAvgAndStd(values, weights)
        print(f"Weighted mean: {avg:.2f}, Weighted std: {std:.2f}")
        ```
    """
    # Create masked values and weights
    masked_values  = np.ma.masked_array(values, np.isnan(values))
    masked_weights = np.ma.masked_array(weights, np.isnan(values))
    # Normalize weights so they sum to 1.
    masked_weights /= np.nansum(masked_weights)
    # Find weighted average
    average = np.average(masked_values, weights=masked_weights)
    # Fast and numerically precise:
    variance = np.average((masked_values - average)**2, weights=masked_weights)
    return (average, np.sqrt(variance))

def areaWeights(resolution=1, radius=6371e3, LonStEd=[-180, 180], LatStEd=[-90, 90]):
    """
    Compute a 2D array of spherical surface area weights based on latitude and longitude resolution.

    This function calculates grid-cell surface areas for a spherical body, returning
    area weights and coordinate arrays corresponding to the centers of each cell.
    The sum of the weights approaches the analytic surface area ``4πR²`` at sufficiently
    high spatial resolution.

    Args:
        resolution (float, optional):  
            Angular resolution of the grid in degrees.  
            Defines both the longitudinal and latitudinal spacing.  
            Defaults to ``1.0``.
        radius (float, optional):  
            Radius of the spherical body in meters.  
            Common planetary values include:
            - Earth: ``6371e3``  
            - Venus: ``6051e3``  
            - Mars: ``3389.5e3``  
            - Moon: ``1737.4e3``  
            Defaults to Earth's mean radius (``6371e3`` m).
        LonStEd (list[float], optional):  
            Two-element list specifying the starting and ending longitude bounds in degrees.  
            Defaults to ``[-180, 180]``.
        LatStEd (list[float], optional):  
            Two-element list specifying the starting and ending latitude bounds in degrees.  
            Defaults to ``[-90, 90]``.

    Returns:
        tuple:
            - **areaWeights** (`numpy.ndarray`): 2D array of grid-cell surface areas in square meters.  
            - **longitudes** (`numpy.ndarray`): 2D array of longitudes (cell centers).  
            - **latitudes** (`numpy.ndarray`): 2D array of latitudes (cell centers).  
            - **totalArea** (`float`): Analytic surface area ``4πR²`` in m².  
            - **totalAreaCalculated** (`float`): Numerically summed area from ``areaWeights`` in m².

    Raises:
        ValueError:  
            If resolution is non-positive or exceeds valid angular bounds.  
        TypeError:  
            If inputs are of incorrect type (e.g., non-numeric resolution).  

    Notes:
        - The grid is constructed cell-centered (e.g., ``0.5°``, ``1.5°``, ...).  
        - For symmetric domains (e.g., ``[-180, 180]``, ``[-90, 90]``), longitude and
          latitude arrays are adjusted to ensure full coverage.  
        - The function assumes a perfectly spherical body—ellipsoidal corrections are not applied.  
        - For each latitude band, surface area per cell is computed using
          :func:`cellAreaOnSphere`.

    Example:
        Compute 5° × 5° global area weights for Earth:

        ```python
        area, lon, lat, A_theoretical, A_calc = areaWeights(resolution=5)
        print(f"Theoretical surface area: {A_theoretical:.3e} m²")
        print(f"Numerical total area:     {A_calc:.3e} m²")
        print(f"Relative error: {(A_calc - A_theoretical) / A_theoretical:.3e}")
        ```
    """
    # Create vectors throughout domains and along dimensions
    Y = np.arange(LatStEd[1]-resolution/2, LatStEd[0]-resolution/2, -resolution)
    X = np.arange(LonStEd[0]+resolution/2, LonStEd[1]+resolution/2, resolution)

    if (Y[0] != -Y[-1]) & (LatStEd[1] == -1 * LatStEd[0]):
        Y = np.array([resolution/2, -resolution/2])
        while (Y[0] + resolution) < 90:
            Y = np.append(Y[0] + resolution, Y)
        while -90 < (Y[-1] - resolution):
            Y = np.append(Y, Y[-1] - resolution)

    if (X[0] != -X[-1]) & (LonStEd[1] == -1 * LonStEd[0]):
        X = np.array([-resolution/2, resolution/2])
        while -180 < (X[0] - resolution):
            X = np.append(X[0] - resolution, X)
        while (X[-1] + resolution) < 180:
            X = np.append(X, X[-1] + resolution)

    # Create meshgrid of latitude and longitudes.
    longitudes, latitudes = np.meshgrid(X, Y)

    # Total area (analytical)
    totalArea = 4 * np.pi * (radius ** 2)

    # Calculate the area weights for this resolution
    areaWeights = np.zeros(np.shape(longitudes))
    for i in range(len(latitudes)):
        areaWeights[i, :] = cellAreaOnSphere(latitudes[i], resolution=resolution, radius=radius)
    totalAreaCalculated = np.sum(np.sum(areaWeights))

    return areaWeights, longitudes, latitudes, totalArea, totalAreaCalculated

def cellAreaOnSphere(clat, resolution=1.0, radius=6371e3):
    """
    Compute the surface area of a latitude-longitude grid cell on a sphere.

    This function calculates the surface area of a single rectangular cell
    centered at latitude ``clat`` on a spherical body, using an approximation
    valid for small angular resolutions. The area is derived from the spherical
    surface differential:

    .. math::

        A = R^2 \, \Delta\lambda \, \Delta\phi \, \cos(\phi)

    where ``R`` is the sphere’s radius, ``Δλ`` and ``Δφ`` are cell widths in
    radians (longitude and latitude), and ``φ`` is the cell center latitude.

    Args:
        clat (float):  
            Center latitude of the cell, in **degrees**.
        resolution (float, optional):  
            Angular resolution of the cell in **degrees** (applied to both
            latitude and longitude). Defaults to ``1.0``.
        radius (float, optional):  
            Radius of the spherical body in **meters**. Defaults to Earth's mean
            radius (``6371e3``).

    Returns:
        float:  
            Cell surface area in **square meters**.

    Raises:
        ValueError:  
            If ``resolution`` is not positive or exceeds 180°.  
        TypeError:  
            If input arguments are not numeric.

    Notes:
        - The function assumes a perfectly spherical body.  
        - For fine resolutions (≤ 5°), this approximation agrees well with
          the spherical excess formula.  
        - Units: meters are used internally and for the output.

    Example:
        Compute the surface area of a 1° × 1° cell centered at 45°N on Earth:

        ```python
        import numpy as np
        area = cellAreaOnSphere(45, resolution=1.0)
        print(f"Cell area: {area/1e6:.2f} km²")
        ```
    """
    # Convert degrees to radians
    deltaLat = np.deg2rad(resolution)
    deltaLon = np.deg2rad(resolution)

    # Calculate cell surface area using small-angle spherical approximation
    area = radius * np.cos(np.deg2rad(clat)) * (deltaLon) * (radius * deltaLat)

    return area

#################################################################################
###################### Helper Functions (Transformations) #######################
#################################################################################
def lonlat2xyz(longitude, latitude, radius=1.0):
    """
    Convert geographic coordinates (longitude, latitude) to Cartesian (X, Y, Z) coordinates
    on a sphere.

    Parameters
    ----------
    longitude : float or array_like
        Longitude(s) in degrees. Positive east of Greenwich.
    latitude : float or array_like
        Latitude(s) in degrees. Positive north of the equator.
    radius : float, optional
        Radius of the sphere. Defaults to 1.0 (unit sphere).
        For Earth, use ~6_371_000 meters.

    Returns
    -------
    x, y, z : float or ndarray
        Cartesian coordinates corresponding to the input points, in the same
        units as ``radius``.

    Notes
    -----
    This transformation assumes a right-handed coordinate system:
    - The X-axis passes through longitude = 0°, latitude = 0°.
    - The Y-axis passes through longitude = 90°, latitude = 0°.
    - The Z-axis points toward the North Pole (latitude = +90°).

    Examples
    --------
    >>> lonlat2xyz(0, 0)
    (1.0, 0.0, 0.0)

    >>> lonlat2xyz(90, 0)
    (6.123233995736766e-17, 1.0, 0.0)

    >>> lonlat2xyz(0, 90)
    (0.0, 0.0, 1.0)
    """
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def xyz2lonlat(x, y, z, radius=1.0):
    """
    Convert Cartesian (X, Y, Z) coordinates to geographic coordinates
    (longitude, latitude) on a sphere.

    Parameters
    ----------
    x : float or array_like
        X coordinate(s), in the same units as ``radius``.
    y : float or array_like
        Y coordinate(s), in the same units as ``radius``.
    z : float or array_like
        Z coordinate(s), in the same units as ``radius``.
    radius : float, optional
        Radius of the sphere. Defaults to 1.0 (unit sphere).

    Returns
    -------
    longitude, latitude : float or ndarray
        Geographic coordinates in degrees.

    Notes
    -----
    - Longitude is computed from ``atan2(y, x)``, ranging from -180° to 180°.
    - Latitude is computed from ``atan2(z, sqrt(x² + y²))``, ranging from -90° to 90°.
    - The input coordinates need not be normalized by ``radius``; this function
      accounts for magnitude automatically.

    Examples
    --------
    >>> xyz2lonlat(1, 0, 0)
    (0.0, 0.0)

    >>> xyz2lonlat(0, 1, 0)
    (90.0, 0.0)

    >>> xyz2lonlat(0, 0, 1)
    (0.0, 90.0)
    """
    longitude = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    latitude = np.degrees(np.arctan2(z, hyp))
    return longitude, latitude

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

###############################################################
###################### Node Grid Creation #####################
###############################################################
class eaNodes():
    """
    Equal-area node generator and utilities for spherical grids.

    This class builds a grid of node centroids arranged as equal-area
    diamond-shaped regions on the surface of a sphere. It can optionally
    persist the generated grid, connectivity, and related metadata to a
    compressed ``.npz`` file, and later reload them. It also supports
    interpolating values from a gridded dataset onto the equal-area nodes.

    Parameters
    ----------
    inputs : dict, optional
        Initialization options. Recognized keys include:
        - ``"resolution"`` (float): Node spacing in **degrees**.
        - ``"dataGrid"`` (str): Path to an input data grid (used by user code).
        If ``{"undefined": True}`` is provided (default), the class sets
        internal defaults for resolution and paths.
    precalculated : bool, optional
        If ``True``, attempt to load a previously saved set of attributes
        from disk on init. Defaults to ``False``.
    precalculate : bool, optional
        If ``True``, write a saved set of attributes to disk after creating
        the grid. Defaults to ``False``.

    Attributes
    ----------
    inputs : dict
        Initialization dictionary (stored as provided or with defaults).
    resolution : float
        Grid spacing in degrees.
    dataGrid : str
        Path to an input grid (if provided).
    interpGrid : str
        Path to the equal-area node list in lon/lat.
    output : str
        Path to a default output file used in downstream processing.
    filename1 : str
        Path to text file containing node indices and XYZ coordinates.
    filename2 : str
        Path to text file containing node lon/lat coordinates.
    ealon, ealat : ndarray or None
        Node longitudes and latitudes (degrees), populated by ``makegrid`` or load.
    connectionNodeIDs : ndarray
        Array with columns: [node_id, nbr_1, nbr_2, nbr_3, nbr_4]. Populated by ``makegrid`` or load.
    connectionNodeDis : ndarray
        Array with columns: [node_id, d1, d2, d3, d4] (distances on unit sphere). Populated by ``makegrid`` or load.
    color, hist : any
        User-facing placeholders (e.g., colors for plotting).
    data : dict
        Mapping of interpolated variable name → values sampled at equal-area nodes.
    precalculated, precalculate : bool
        Flags controlling load/save behavior.

    Notes
    -----
    - All file artifacts are written under ``./Nodes`` (created if absent).
    - This class assumes a **unit sphere** for geometry; distances are in radians
      when using great-circle formulas (see internal helpers).
    """

    def __init__(self, inputs={"undefined": True}, precalculated=False, precalculate=False):
        '''
        Initialize the equal-area node container and (optionally) load a saved grid.

        Parameters
        ----------
        inputs : dict, optional
            See class docstring. If ``{"undefined": True}``, internal defaults
            for resolution and paths are assigned.
        precalculated : bool, optional
            If ``True``, attempt to load attributes from a saved ``.npz`` file.
        precalculate : bool, optional
            If ``True``, enable saving attributes to a ``.npz`` after grid creation.

        Returns
        -------
        None
        '''
        # (implementation unchanged)
        # ---------------------------------------------------------------------
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
        if not os.path.exists(os.getcwd()+"/Nodes"):
            os.mkdir(os.getcwd()+"/Nodes")

        # Assign more class attributes (filepaths)
        self.filename1  = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'
        self.filename2  = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt'
        self.interpGrid = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt';
        self.output     = os.getcwd()+"/Nodes/EASampled.txt";

        # Define all attributes assigned to object.
        self.color  = None
        self.hist   = None
        self.ealon = None
        self.ealat = None

        # Set read/write options
        self.precalculated = precalculated
        self.precalculate  = precalculate
        
        # Equal area node interpolated data
        self.data = {}

        # Try to read precalculated node grid 
        if self.precalculated:
            self.precalculated = self.save_or_load_attributes(mode="r")

    def save_or_load_attributes(self, mode="w"):
        """
        Save or load all class attributes to/from a compressed ``.npz`` file.

        Parameters
        ----------
        mode : {'w','r'}, optional
            - ``'w'``: write current attributes to disk
            - ``'r'``: read attributes from disk into this object

        File
        ----
        The file is created under ``./Nodes`` and named
        ``EA_Nodes_{resolution}.npz``.

        Returns
        -------
        bool
            ``True`` if the operation succeeded, ``False`` if reading failed
            due to a missing file.

        Notes
        -----
        - Paths like ``filename1``, ``filename2``, ``interpGrid``, and ``output``
          are **recomputed** on load to reflect the current working directory,
          rather than trusting serialized paths from another system.
        - Dict-like attributes are stored with ``allow_pickle=True``.
        """
        # (implementation unchanged)
        # ---------------------------------------------------------------------
        filename = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'.replace("_xyz.txt", ".npz")

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
            np.savez_compressed(filename, **attr_dict)
            print(f"Attributes saved to {filename}")
            return True

        elif mode == "r":
            if not os.path.exists(filename):
                print(f"File {filename} does not exist. Cannot load. {filename} will be created for current use and written (precalculated) for future use at {self.resolution:0.1f} degree spatial resolution.")
                return False
            
            print(f"File {filename} does exist. loading. {filename} will be read for current use.")

            npzfile = np.load(filename, allow_pickle=True)

            self.inputs = npzfile['inputs'].item()
            self.resolution = npzfile['resolution'].item()
            self.dataGrid = npzfile['dataGrid'].item()
            self.color = npzfile['color']
            self.hist = npzfile['hist'].item()
            self.ealon = npzfile['ealon']
            self.ealat = npzfile['ealat']
            self.data = npzfile['data'].item()
            self.connectionNodeIDs = npzfile['connectionNodeIDs']
            self.connectionNodeDis = npzfile['connectionNodeDis']

            # Recompute paths for current system
            self.filename1  = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_xyz.txt'
            self.filename2  = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt'
            self.interpGrid = os.getcwd()+"/Nodes/"+f'EA_Nodes_{self.resolution:0.1f}_LatLon.txt';
            self.output     = os.getcwd()+"/Nodes/EASampled.txt";

            print(f"Attributes loaded from {filename}")
            return True
        else:
            print("Mode must be 'w' or 'r'.")

    def rotate_around_vec_by_a(self, A, x, y, z):
        """
        Build a 3×3 rotation matrix for rotation by angle ``A`` about an arbitrary axis.

        Parameters
        ----------
        A : float
            Rotation angle in **radians**.
        x, y, z : float
            Components of the rotation axis vector (need not be unit length).

        Returns
        -------
        ndarray, shape (3, 3)
            Rotation matrix ``R`` such that ``R @ v`` rotates vector ``v`` by
            ``A`` about the axis ``(x, y, z)`` (right-hand rule).

        Notes
        -----
        Implements Rodrigues’ rotation formula after normalizing the axis.
        """
        # (implementation unchanged)
        # ---------------------------------------------------------------------
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
        """
        Rotate a vector about the axis defined by ``from_vec × to_vec``.

        Parameters
        ----------
        from_vec : array_like, shape (3,)
            Reference vector that defines the rotation axis with ``to_vec``.
        to_vec : array_like, shape (3,)
            Target vector that defines the rotation axis with ``from_vec``.
        by_angle : float
            Rotation angle in **radians**.

        Returns
        -------
        ndarray, shape (3,)
            Rotated copy of ``from_vec`` by ``by_angle`` around ``from_vec × to_vec``.

        Notes
        -----
        The axis is the cross product of the two input vectors. No normalization
        of the inputs is required; the axis is normalized internally.
        """
        axis_vec = np.cross(from_vec, to_vec)
        R = self.rotate_around_vec_by_a(by_angle, *axis_vec)
        return R @ from_vec

    def makegrid(self, plotq=0):
        '''
        Construct the equal-area diamond grid and nearest-neighbor connectivity.

        This method tessellates the sphere with 12 rotated diamond patches,
        generates node coordinates on a **unit sphere**, writes them to disk in
        XYZ and lon/lat formats, and builds up to 4 nearest-neighbor connections
        per node based on great-circle distances.

        Parameters
        ----------
        plotq : int, optional
            If non-zero, renders a 3D Plotly visualization of node positions.
            ``1`` plots a scatter of non-duplicate nodes. Defaults to ``0``.

        (Re)defined Attributes
        ----------------------
        ealat, ealon : ndarray
            Node latitudes/longitudes in degrees (read from written files).
        connectionNodeIDs : ndarray, shape (N, 5)
            Columns: [node_id, nbr1, nbr2, nbr3, nbr4], with ``None`` where absent.
        connectionNodeDis : ndarray, shape (N, 5)
            Columns: [node_id, d1, d2, d3, d4] (unit-sphere chord/arc distances).
        color : ndarray
            Colors used for plotting patches (flattened, duplicates removed).

        Side Effects
        ------------
        - Creates directory ``./Nodes`` if absent.
        - Writes:
            * ``EA_Nodes_{resolution}_xyz.txt`` (node ID + x, y, z)
            * ``EA_Nodes_{resolution}_LatLon.txt`` (lon, lat)
        - Optionally writes a compressed ``.npz`` (if ``self.precalculate`` is True).

        Returns
        -------
        None
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
                            lon, lat = xyz2lonlat(x, y, z);
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
        Interpolate values from a gridded file to the equal-area node locations.

        Uses GMT’s ``grdtrack`` to sample the provided grid at lon/lat points
        stored in ``self.filename2`` and stores the resulting values under
        ``self.data[name]``.

        Parameters
        ----------
        path : str
            Path to the input grid file to be sampled by GMT (e.g., NetCDF grid).
        name : str
            Key under which sampled values will be stored in ``self.data``.
        resolution : int, optional
            Unused placeholder (kept for API symmetry). Defaults to ``1``.

        (Re)defined Attributes
        ----------------------
        data : dict
            Adds/overwrites entry ``self.data[name]`` with sampled values
            (NumPy array of length equal to the number of nodes).

        Notes
        -----
        - The method runs:
          ``gmt grdtrack -R-181/181/-90/90 {filename2} -G{path} -N > temp.txt -Vq``  
          to ensure wrap-around at the dateline for node sampling.
        - A temporary ``temp.txt`` is produced and then removed.
        - Requires GMT to be installed and accessible on ``PATH``.

        Returns
        -------
        None
        """
        # Interpolate the values to node locations
        # Note that the region R must be set to -181/181 so that 
        # nodes at edges lon=-180=180 and lon=0 will have appropriately
        # interpolated values (i.e., not result in an nan value when
        # a value does exist).
        os.system("gmt grdtrack -R-181/181/-90/90 {0} -G{1} -N > {2} -Vq".format(self.filename2, path, 'temp.txt'))
 
        # Read interpolated values
        self.data[name] = np.loadtxt('temp.txt', delimiter='\t',usecols=[2])

        # Delete temporary file
        os.system('temp.txt');

class edllNodes():
    """
    Equally spaced latitude–longitude node generator (EDLL) with utilities.

    This class constructs a grid of points evenly spaced in *latitude* and
    *longitude* over a user-specified rectangular region. It mirrors the I/O
    and attribute structure of :class:`eaNodes` so downstream code can treat
    both grid types uniformly. The class can persist grid geometry and basic
    connectivity to a compressed ``.npz`` and reload it later. It also supports
    sampling gridded datasets onto the node locations via GMT ``grdtrack``.

    Parameters
    ----------
    inputs : dict, optional
        Initialization options. Recognized keys include:
        - ``"resolution"`` (float): Node spacing in **degrees** (Δlon = Δlat).
        - ``"dataGrid"`` (str): Path to an input data grid (for user workflows).
        If ``{"undefined": True}`` (default), internal defaults are assigned.
    precalculated : bool, optional
        If ``True``, attempt to load a previously saved set of attributes from
        disk upon initialization. Defaults to ``False``.
    precalculate : bool, optional
        If ``True``, write the current attributes to disk after grid creation.
        Defaults to ``False``.
    region : array_like or None, optional
        2×2 array-like defining the analysis extent as
        ``[[lon_min, lat_min], [lon_max, lat_max]]`` in **degrees**.
        If ``None``, a small default region is used: ``[[-100, 20], [-80, 31]]``.

    Attributes
    ----------
    inputs : dict
        Initialization dictionary (stored as provided or with defaults).
    resolution : float
        Grid spacing in degrees.
    region : ndarray, shape (2, 2)
        Region bounding box ``[[lon_min, lat_min], [lon_max, lat_max]]``.
    dataGrid : str
        Path to an input grid (if provided).
    interpGrid : str
        Path to the node list in lon/lat text format.
    output : str
        Default output path (for user workflows).
    filename1 : str
        File with node indices and XYZ coordinates (txt).
    filename2 : str
        File with node lon/lat coordinates (txt).
    ealon, ealat : ndarray or None
        Node longitudes/latitudes (degrees), defined after :meth:`makegrid`
        or when loading a saved grid.
    connectionNodeIDs : ndarray
        Array with columns: ``[node_id, west_id, east_id, north_id, south_id]``,
        where missing neighbors are ``None`` (set by :meth:`makegrid`).
    connectionNodeDis : ndarray
        Array with columns: ``[node_id, dW, dE, dN, dS]`` (unit-sphere distances),
        set by :meth:`makegrid`.
    color, hist : any
        Placeholders for symmetry with :class:`eaNodes` (no semantic meaning here).
    data : dict
        Mapping of sampled variable name → values at node locations.
    precalculated, precalculate : bool
        Flags controlling load/save behavior.

    Notes
    -----
    - All artifacts are written under ``./Nodes`` (created if missing).
    - Spherical calculations assume a **unit sphere** unless otherwise noted.
    """

    def __init__(self, inputs={"undefined":True}, precalculated=False, precalculate=False, region=None):
        '''
        Initialize the EDLL node container and optionally load a saved grid.

        Parameters
        ----------
        inputs : dict, optional
            See class docstring. If ``{"undefined": True}``, internal defaults
            for resolution and paths are assigned.
        precalculated : bool, optional
            If ``True``, attempt to load attributes from a saved ``.npz`` file.
        precalculate : bool, optional
            If ``True``, enable saving attributes to a ``.npz`` after grid creation.
        region : array_like or None, optional
            Analysis extent ``[[lon_min, lat_min], [lon_max, lat_max]]`` in degrees.

        Returns
        -------
        None
        '''
        # Assign inputs
        self.inputs = inputs

        # Assign class attributes
        try:
            if inputs["undefined"]:
                self.resolution = .1;
                self.dataGrid   = "/home/bogumil/Documents/data/Muller_etal_2019_Tectonics_v2.0_netCDF/Muller_etal_2019_Tectonics_v2.0_AgeGrid-0.nc";
                self.interpGrid = "EA_Nodes_{}_LatLon.txt".format(self.resolution);
                self.output     = "Muller_etal_2019_Tectonics_v2.0_AgeGrid-0_EASampled.nc";
        except:
            self.resolution = self.inputs["resolution"];
            self.dataGrid   = self.inputs["dataGrid"];
            self.region     = region;
        if region is None:
            self.region     = np.array([[-100, 20], [-80, 31]]); # [Lower left corner (lon, lat), Upper right corner (lon, lat)]

        # Create directory to store precalculated files within
        if not os.path.exists(os.getcwd()+"/Nodes"):
            os.mkdir(os.getcwd()+"/Nodes")

        # Assign more class attributes (filepaths)
        self.filename1  = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_xyz.txt'
        self.filename2  = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_LatLon.txt'
        self.interpGrid = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_LatLon.txt';
        self.output     = os.getcwd()+"/Nodes/EASampled.txt";

        # Define all attributes assigned to object.
        self.color  = None; # Holds the colors used to distinguish between regions points on a sphere [meaningless for edllNodes - has meaning in eaNodes]. 
        self.hist   = None; #
        self.ealon = None;  # node longitude
        self.ealat = None;  # node latitude
        
        # Set read/write options
        self.precalculated = precalculated;
        self.precalculate  = precalculate;
        
        # node interpolated data
        self.data = {};

        # Try to read precalculated node grid 
        if self.precalculated:
            self.precalculated = self.save_or_load_attributes(mode="r")

    def save_or_load_attributes(self, mode="w"):
        """
        Save or load all class attributes to/from a compressed ``.npz`` file.

        Parameters
        ----------
        mode : {'w', 'r'}, optional
            - ``'w'``: Write current attributes to disk.
            - ``'r'``: Read attributes from disk into this object.

        File
        ----
        Stored under ``./Nodes`` and named:
        ``EDLL_Nodes_{resolution}.npz``.

        Returns
        -------
        bool
            ``True`` if the operation succeeded, ``False`` if reading failed
            because the file was not found.

        Notes
        -----
        - Paths like ``filename1``, ``filename2``, ``interpGrid``, and ``output``
          are **recomputed** on load to reflect the current working directory,
          rather than trusting serialized paths from another system.
        - Dict-like attributes are stored with ``allow_pickle=True``.
        """
        # Create filename
        filename = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_xyz.txt'.replace("_xyz.txt", ".npz")

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

            # Recompute filepaths for current system
            self.filename1  = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_xyz.txt'
            self.filename2  = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_LatLon.txt'
            self.interpGrid = os.getcwd()+"/Nodes/"+f'EDLL_Nodes_{self.resolution:0.1f}_LatLon.txt';
            self.output     = os.getcwd()+"/Nodes/EASampled.txt";

            print(f"Attributes loaded from {filename}")

            return True
        else:
            print("Mode must be 'w' or 'r'.")

    def makegrid(self, plotq=0):
        '''
        Build the equally spaced lon–lat grid and 4-neighbor connectivity.

        This method creates a regular lattice over ``self.region`` with spacing
        ``self.resolution`` in both longitude and latitude, writes node
        coordinates to disk (XYZ and lon/lat), and constructs a 4-neighbor
        connectivity (west, east, north, south) with unit-sphere distances.

        Parameters
        ----------
        plotq : int, optional
            If ``1``, display a 3D Plotly visualization of the nodes on a sphere.
            Defaults to ``0``.

        (Re)defined Attributes
        ----------------------
        ealon, ealat : ndarray
            Node longitudes and latitudes (degrees), flattened meshgrid.
        connectionNodeIDs : ndarray, shape (N, 5)
            Node ID plus west/east/north/south neighbor IDs (``None`` at edges).
        connectionNodeDis : ndarray, shape (N, 5)
            Node ID plus corresponding distances to those neighbors (unit sphere).

        Side Effects
        ------------
        - Writes:
            * ``EDLL_Nodes_{resolution}_xyz.txt`` (node ID + x, y, z)
            * ``EDLL_Nodes_{resolution}_LatLon.txt`` (lon, lat)

        Returns
        -------
        None
        '''
        ###################
        #### Section 1 ####
        ###################
        # Define the coordinates of nodes within region
        if self.region[0][0] > self.region[1][0]: # region passes through -180/180 line
            Lon = np.arange(self.region[0][0], 360+self.region[1][0]+self.resolution/2, self.resolution); 
            Lon[Lon>=180] = Lon[Lon>=180]-360
        else:
            Lon = np.arange(self.region[0][0], self.region[1][0]+self.resolution/2, self.resolution);
        Lat = np.arange(self.region[0][1], self.region[1][1]+self.resolution/2, self.resolution);

        LonArray, LatArray = np.meshgrid(Lon, Lat); 

        self.ealon = LonArray.flatten()
        self.ealat = LatArray.flatten()        
        
        xyz = np.array(lonlat2xyz(self.ealon, self.ealat))
        # LonLat = np.array(xyz2lonlat(xyz[0],xyz[1],xyz[2]))
        
        # Create node ids
        nodeIDs = np.arange(0, np.size(LonArray), 1)
                                    
        ###################
        #### Section 3 ####
        ###################
        # Convert the grid back to spherical coords, and save the results
        # Saving the grid points (but don't save duplicates)
        with open(self.filename1, 'w') as f:
            for nodeID, x, y, z in np.vstack( (nodeIDs ,xyz) ).T:
                f.write(f'{int(nodeID)}, {x:.8e}, {y:.8e}, {z:.8e}\n');

        with open(self.filename2, 'w') as f:
            for lon, lat in zip(self.ealon, self.ealat):
                f.write(f'{lon:.8e}, {lat:.8e}\n');
                        
            
        ###################
        #### Section 3 ####
        ###################
        # Convert the grid back to spherical coords, and save the results
        # Saving the grid points (but don't save duplicates)
        with open(self.filename1, 'w') as f:
            for nodeID, x, y, z in np.vstack( (nodeIDs ,xyz) ).T:
                f.write(f'{int(nodeID)}, {x:.8e}, {y:.8e}, {z:.8e}\n');

        with open(self.filename2, 'w') as f:
            for lon, lat in zip(self.ealon, self.ealat):
                f.write(f'{lon:.8e}, {lat:.8e}\n');
            
        ##############################
        #### Values yet to define ####
        ##############################
        # self.connectionNodeIDs, self.connectionNodeDis (filled below)
        
        ###################
        #### Section 4 ####
        ###################
        maxEdgeConnections = 4
        # Define array to hold node connection ids
        self.connectionNodeIDs = np.zeros(shape=(len(self.ealon), 1+maxEdgeConnections));
        self.connectionNodeIDs[:,0] = np.arange(len(self.ealon)); # Set evaluation node IDs
        
        # Define array to hold node connection distances
        self.connectionNodeDis = np.zeros(shape=(len(self.ealon), 1+maxEdgeConnections));
        self.connectionNodeDis[:,0] = np.arange(len(self.ealon)); # Set evaluation node IDs
        
        # Make Node ID array
        IDArray = np.reshape(nodeIDs, np.shape(LonArray) )
        
        ## Find all neighboring nodes (i.e., fill out self.connectionNodeIDs array)
        ### Western node
        WNodeID  = np.roll(IDArray,   1, axis=1)        
        ### Eastern node
        ENodeID  = np.roll(IDArray,  -1, axis=1)
        ### Northern node
        NNodeID  = np.roll(IDArray,   1, axis=0)
        ### Southern node
        SNodeID  = np.roll(IDArray,  -1, axis=0)
        
        ## Set Node Western, Eastern, Northern, and Southern NodeIDs
        self.connectionNodeIDs[:, 1] = WNodeID.flatten()[nodeIDs]
        self.connectionNodeIDs[:, 2] = ENodeID.flatten()[nodeIDs]
        self.connectionNodeIDs[:, 3] = NNodeID.flatten()[nodeIDs]
        self.connectionNodeIDs[:, 4] = SNodeID.flatten()[nodeIDs]
        
        ## Remove all wrapped region node connections since the region should never be wrapped
        WNodeID = np.reshape(self.connectionNodeIDs[:, 1], np.shape(LonArray))
        WNodeID[:,0] = None
        self.connectionNodeIDs[:, 1] = WNodeID.flatten()
        
        ENodeID = np.reshape(self.connectionNodeIDs[:, 2], np.shape(LonArray))
        ENodeID[:,-1] = None
        self.connectionNodeIDs[:, 2] = ENodeID.flatten()
        
        NNodeID = np.reshape(self.connectionNodeIDs[:, 3], np.shape(LonArray))
        NNodeID[0,:] = None
        self.connectionNodeIDs[:, 3] = NNodeID.flatten()
        
        SNodeID = np.reshape(self.connectionNodeIDs[:, 4], np.shape(LonArray))
        SNodeID[-1,:] = None
        self.connectionNodeIDs[:, 4] = SNodeID.flatten()
        
        # Find distance between all neighboring nodes
        for nodeID in self.connectionNodeIDs[:, 0]:
            nodeID = int(nodeID)
            # Set neighboring node ids
            neighboringNodeIDs = self.connectionNodeIDs[nodeID, 1:][ ~np.isnan(self.connectionNodeIDs[nodeID, 1:]) ];
            neighboringLats    = LatArray.flatten()[neighboringNodeIDs.astype(int)]
            neighboringLons    = LonArray.flatten()[neighboringNodeIDs.astype(int)]
            
            # Find distance between evaluation node and neighboring nodes
            distance = haversine_distance(lat1=LatArray[nodeID==IDArray],
                                          lon1=LonArray[nodeID==IDArray],
                                          lat2=neighboringLats,
                                          lon2=neighboringLons,
                                          radius=1);
            
            # Set neighboring node distances
            self.connectionNodeDis[nodeID, 1:][ ~np.isnan(self.connectionNodeIDs[nodeID, 1:]) ] = distance
            
        ###################
        #### Section 5 ####
        ###################
        if plotq==1:
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

            # Add the Node location scatter plot
            fig.add_trace(go.Scatter3d(x=xyz[0,:],
                                    y=xyz[1,:],
                                    z=xyz[2,:],
                                    mode='markers',
                                    marker=dict(size=5)))
        if plotq:
            # Show the plot
            fig.show()

        # Write precalculated node grid 
        if self.precalculate:
            self.save_or_load_attributes(mode="w")

    def interp2IrregularGrid(self, path, name, resolution=1):
        """
        Interpolate gridded values to node locations via GMT ``grdtrack``.

        Samples the provided grid at the lon/lat node positions stored in
        ``self.filename2`` and writes the sampled values into
        ``self.data[name]``.

        Parameters
        ----------
        path : str
            Path to an input grid readable by GMT (e.g., NetCDF grid).
        name : str
            Key to store the sampled values in ``self.data``.
        resolution : int, optional
            Unused placeholder (kept for API symmetry). Defaults to ``1``.

        (Re)defined Attributes
        ----------------------
        data : dict
            Adds/overwrites entry ``self.data[name]`` with sampled values
            (NumPy array, length = number of nodes).

        Notes
        -----
        - Runs:
          ``gmt grdtrack -R-181/181/-90/90 {filename2} -G{path} -N > temp.txt -Vq``  
          using a slightly extended longitude range to handle wrap-around at the dateline.
        - A temporary ``temp.txt`` is produced and then removed.
        - Requires GMT to be installed and available on ``PATH``.

        Returns
        -------
        None
        """
        # Interpolate the values to node locations
        # Note that the region R must be set to -181/181 so that 
        # nodes at edges lon=-180=180 and lon=0 will have appropriately
        # interpolated values (i.e., not result in an nan value when
        # a value does exist).
        os.system("gmt grdtrack -R-181/181/-90/90 {0} -G{1} -N > {2} -Vq".format(self.filename2, path, 'temp.txt'))

        # Read interpolated values
        self.data[name] = np.loadtxt('temp.txt', delimiter='\t',usecols=[2])

        # Delete temporary file
        os.system('temp.txt');

#######################################################################################
######################## Helper Functions (Community Detection) #######################
#######################################################################################

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

def mergerPackages(package = '', verbose=True):
    """
    Return a predefined basin-merger configuration (or a pass-through default).

    This helper provides ready-made **basin merging strategies** for different
    reconstruction/modeling setups. Each strategy is encoded as a dictionary
    describing:
      1) how to **merge small basins** below one or more thresholds, and
      2) (optionally) how to **merge specific basin IDs** at designated
         time/epoch markers.

    Parameters
    ----------
    package : str, optional
        Name of the preset to load. Recognized values include (non-exhaustive):
        - ``"EarthRecon3Basins_CM2009"``
        - ``"Lite"``, ``"Lite1"``, ``"Lite2"``, ``"Lite3"``, ``"Lite4"``
        - ``"LiteShelf0,3degree"``
        - ``"None"``  (merge logic on, with zero threshold)
        - ``"EarthRecon3BasinsRK2021_H_2,10e-12"``,
          ``"EarthRecon3BasinsRK2021_H_4,80e-12"``,
          ``"EarthRecon3BasinsRK2021_H_8,00e-12"``
        Any other value (including the empty string) returns a conservative
        default that **does not merge** basins.
    verbose : bool, optional
        Currently not used inside this function, but included for API symmetry.
        Default is ``True``.

    Returns
    -------
    mergerPackage : dict
        A configuration dictionary. Common keys:

        ``"mergeSmallBasins"`` : dict  
            Controls threshold-based merging of very small basins.
            Sub-keys:
              - ``"on"`` : bool — enable/disable this step.
              - ``"threshold"`` : ndarray of float — one or more thresholds.
              - ``"thresholdMethod"`` : str — interpretation of thresholds
                (e.g., ``"%"`` means percent of total area).
              - ``"mergeMethod"`` : str — how to choose the target basin
                (e.g., ``"nearBasinEdge"``).
        ``"verbose"`` : bool  
            Optional flag for downstream verbosity.

        The following keys may be present in some presets:

        ``"mergerID"`` : ndarray of int  
            Time/epoch markers or other indices at which explicit mergers
            are defined.

        ``"mergersX"`` : dict  
            For a marker ``X`` in ``"mergerID"``, maps **basin label** (as a
            string, e.g., ``"0"``) to a **list of basin IDs** to merge with at
            that marker. Example:
            ``'mergers10': {'0':[0,6,7,9,10], '1':[1,4], '2':[2,3,4,5]}``.

        ``"arrangeX"`` : list of int  
            Optional permutation of basin display/order at marker ``X``.

    Notes
    -----
    - Thresholds in the ``"mergeSmallBasins"`` block are typically **decimal
      fractions** of total surface area; e.g., ``0.5`` means *0.5%* when
      ``thresholdMethod == "%"``.
    - Basin labels in the merger maps are **strings** (``"0"``, ``"1"``, ...),
      while the lists they map to contain **integer** basin IDs.
    - The return value is intended to be consumed by a higher-level routine
      that actually **applies** these merges to basin geometries/IDs.

    Examples
    --------
    >>> cfg = mergerPackages('Lite')
    >>> cfg['mergeSmallBasins']
    {'on': True, 'threshold': array([0.1 , 0.5]),
     'thresholdMethod': '%', 'mergeMethod': 'nearBasinEdge'}

    >>> cfg = mergerPackages('EarthRecon3Basins_CM2009')
    >>> 10 in cfg['mergerID']
    True
    >>> sorted(cfg.keys())[:3]
    ['arrange0', 'arrange15', 'arrange20']  # plus many others (varies by preset)
    """
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

#################################################################################
###################### Basin definition class (Data Fields) #####################
#################################################################################
class BasinsEA():
    """
    Basins is a class meant to construct basins and bathymetry properties
    given a bathymetry model netCDF4.
    """

    def __init__(self, dataDir, filename, body, region=np.array([[-180,-90],[180,90]])):
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
        region : NUMPY ARRAY
            2x2 array of [Lower left corner [lon, lat], Upper right
            corner [lon, lat]] that describes the analysis region.
        
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

        # Define region of analysis (default is global)
        self.region = region;

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
            self.bathymetry = field;
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
                "useGravity"
                    No user input needed.
                "useLogistic"
                    Choose lower and upper bound weights 'S_at_lower',
                    'S_at_upper' (between 0-1) and their correpsonding
                    data field values 'factor_at_lower', 'factor_at_upper'
                    (in units of standard deviation) that will be used
                    to construct the logistic-like weighting curve.
                "useQTGaussianSigmoid"
                    Working progress - useLogistic should produce similar
                    expected results.
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
                # initiation prompts the creation of a graph network (include node/edge weight calculations)


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
                if (self.region[0,0]==-180) & (self.region[1,0]==180) & (self.region[0,1]==-90) & (self.region[1,1]==90):
                    # Define equal area points
                    # eaPoint.lat, eaPoint.lon are created here
                    # Note that only one eaNodes object is need for multiple fields.
                    # Use the first used field to create the object
                    # self.eaPoint.precalculated is set to False is the grid has not been
                    # precalculated. Otherwise the precalculated grid is read in.
                    self.eaPoint = eaNodes(inputs = self.Fields[self.Fields['usedFields'][0]],
                                        precalculate=True,
                                        precalculated=True);
                else:
                    self.eaPoint = edllNodes(inputs = self.Fields[self.Fields['usedFields'][0]],
                                        precalculate=True,
                                        precalculated=True,
                                        region=self.region);


                # Creates
                # 1) Set of nodes that represent equal area quadrangles.
                # 2) Define the connects between all nodes (even to nodes
                # with missing data)
                if not self.eaPoint.precalculated:
                    self.eaPoint.makegrid(plotq=0);

                # Rounding to about 1.1 km resolution
                Spresolution = np.round( np.abs(np.diff(self.lat[:,0])[0]), 5)

                # Loop over all used fields to interpolate data to graph
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
                
                # Area covered by a node m2.
                # - Weight should be updated for regional analysis that span large latitude arcs
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
                    useGravity = False;
                    useLogistic = False;
                    useQTGaussianSigmoid = False;
                    useQTGaussianShiftedGaussianWeightDistribution = True;

                    # Set method and some default parameters
                    if edgeWeightMethod['method'] == "useGravity":
                        # Set method
                        useGravity=True;
                    
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

                    # Set inverse distance power weight & other weighting properties.  
                    if useGravity:
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

                        # Plot quantile transformation distributions for data vs. Gaussian (QT) domain.
                        verbose = True;
                        if verbose:
                            plotHelper.plot_quantile_transform_distribution(xValues, self.Fields[field]['weightMethodPara']['qt'])

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

                ## Iterate through each node to add edge weights
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


                            # elif useQTGaussianSigmoid:
                            #     # FIXME: Update for multiple fields
                            #     # Difference in properties at nodes
                            #     diff = np.abs(values1-values2);
                            #     # Transform from diff-space to gaussian-space
                            #     QTGdiff = qt.transform( np.reshape( np.array(diff), (1,1) ) );

                            #     # Apply stretch factor after QTGdiff is defined with a Guassian Transformer.
                            #     # This will give less weight to tail values of the distribution
                            #     QTGdiffStretch = 0.1; # Decimal percentage to stretch the QTGdiff value.
                            #     QTGdiff *= (1 + QTGdiffStretch);

                            #     # Use the logistic function to calculated the edge weight component S.
                            #     S = logisticAttributes["L"]*(1+np.exp(-logisticAttributes["k"]*QTGdiff+logisticAttributes["shift"]))**(-1)


                            if useGravity :
                                # Note that the gravity model represents node edges weights
                                # with (property1*property2)/(distanceV**2).

                                # Calculate the product of the property weights:
                                # The inverse distance squared is added with the
                                # later calculated nodeSpacingNormalizer.

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

                                        # Set current field
                                        field = self.Fields['usedFields'][cnt]

                                        Ss[cnt] = values1*values2;
                            
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

                                        # Set current field
                                        field = self.Fields['usedFields'][cnt]

                                        # Difference in properties at nodes
                                        diff = np.abs(value1-value2)

                                        # Use the logistic function to calculated the edge weight component S.
                                        Ss[cnt] = self.Fields[field]['weightMethodPara']['logisticAttributes']["L"]*(1+np.exp(-self.Fields[field]['weightMethodPara']['logisticAttributes']["k"]*diff+self.Fields[field]['weightMethodPara']['logisticAttributes']["shift"]))**(-1)

                                        # Move data field index
                                        cnt+=1

                            elif useQTGaussianShiftedGaussianWeightDistribution:
                                # This method does the following to calculate weights
                                # 1. Filter outliers from difference data (using IQR method)
                                # 2. Convert difference data into gaussian (using QT method)
                                # 3. Calculate z-score of difference data between nodei and nodej
                                # 4. Given the z-score from step 3) calculate a CDF (0-1) value on a
                                # new distribution centered at 1 sigma (from the first distribution)
                                # and with a std of sigma/a (from the first distribution).
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

                                        # Set current field
                                        field = self.Fields['usedFields'][cnt]

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

                            # Merge multiple field weights, default is mean
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
                            else:
                                # No MultiField method used
                                S = Ss;

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
                            method_name="consensus_ledien",
                            method_parameters={
                                "resolution_parameter": resolution,
                                "runs": ensembleSize,
                                "distance_threshold": distance_threshold}
                            )

        LGNClusters=NodeClustering(communities=self.communitiesFinal,
                            graph=self.G,
                            method_name="consensus_ledien",
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

            # Apply Girvan-Newman algorithm to the simplified community graph
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

            # Map each Girvan-Newman community to its reduced (Leiden/Louvain) community
            reduced_to_gn = {}
            for idx, comm in enumerate(GNcommunities):
                for c in comm:
                    reduced_to_gn[c] = idx
            
            # Map each original node to a Girvan-Newman community via its reduced (Leiden/Louvain) community
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

#############################################################################################
###################### Basin definition class (Synthetic Distributions) #####################
#############################################################################################
class BasinsSynth:
    """
    Build synthetic ocean basins and bathymetry parameters for carbon-cycle models.

    This class organizes inputs and derived properties for a **synthetic**
    bathymetry, including:
    - Global/basin bathymetry distributions over depth bins,
    - Area/volume fractions by basin and a high-latitude box,
    - Optional basin-to-basin connectivity distributions,
    - A writer to persist parameters to a netCDF4 file structure.

    Parameters
    ----------
    dataDir : str
        Directory where outputs will be written (e.g., the netCDF file).
    filename : str
        Output file name to create inside ``dataDir``.
    radius : float
        Planetary radius in meters used by the synthetic model.

    Attributes
    ----------
    dataDir : str
        Output directory (as passed to ``__init__``).
    filename : str
        Output file name (as passed to ``__init__``).
    radius : float
        Planet radius in meters.
    basinConnectionDefined : bool
        Flag indicating whether basin connectivity parameters have been defined.
    BasinParametersDefined : bool
        Flag indicating whether basin/bathymetry parameters have been defined.

    # Set by :meth:`defineBasinParameters`
    bathymetryVolFraction : dict
        ``{'Basin0': float, 'Basin1': float, ...}`` volume fractions (decimal)
        that sum to 1 across basins (excluding the high-latitude box).
    bathymetryAreaFraction : dict
        ``{'Basin0': float, 'Basin1': float, ...}`` area fractions (decimal)
        that sum to ``1 - highlatA/AOC`` across basins.
    bathymetryAreaDistBasin : dict
        Per-basin bathymetry histograms over ``binEdges``.
    bathymetryAreaDist_wHighlatG : array_like
        Global bathymetry histogram including the high-latitude region.
    bathymetryAreaDistG : array_like
        Global bathymetry histogram excluding the high-latitude region.
    binEdges : array_like
        Depth bin edges in kilometers (length ``n+1`` for ``n`` bins).
    AOC : float
        Total seafloor area (m²).
    VOC : float
        Total ocean volume (m³).
    highlatA : float
        Area of the high-latitude box (m²).
    highlatlat : float or None
        Latitude threshold used to define the high-latitude region; ``None`` for
        synthetic models.

    # Set by :meth:`defineBasinConnectivityParameters`
    bathymetryConDist : ndarray
        3-D array (``BasinCnt × BasinCnt × (len(binEdges)-1)``) of
        connectivity bathymetry histograms (%).

    Notes
    -----
    The class focuses on parameter bookkeeping for idealized/synthetic
    bathymetry rather than generating gridded fields.
    """

    def __init__(self, dataDir, filename, radius):
        """
        Initialize the synthetic basin container.

        Parameters
        ----------
        dataDir : str
            Directory where outputs will be written (e.g., netCDF file).
        filename : str
            Output file name to create within ``dataDir``.
        radius : float
            Planetary radius in meters.

        Returns
        -------
        None
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
        """
        Define the depth bin edges for bathymetry histograms.

        Parameters
        ----------
        bathymetryBins : array_like
            Monotonically increasing vector of bin edges (km) with length ``n+1``.
            Histograms produced later will have length ``n``.

        Returns
        -------
        None

        Notes
        -----
        This is a convenience setter; subsequent methods (e.g.,
        :meth:`defineBasinParameters`) expect compatible ``binEdges``.
        """

    def defineBasinParameters(
        self,
        BasinCnt=3,
        Distribution=None,
        binEdges=None,
        AOC=None,
        VOC=None,
        fanoc=np.array([.30, .30, .30, .10]),
        fdvol=np.array([.333, .333, .334]),
        verbose=True,
    ):
        """
        Define per-basin bathymetry distributions and area/volume partitions.

        Populates global and per-basin bathymetry histograms over ``binEdges``,
        sets basin area/volume fractions, and records high-latitude box area.

        Parameters
        ----------
        BasinCnt : int, optional
            Number of basins to define. Default is ``3``.
        Distribution : array_like, optional
            Length-``n`` global bathymetry histogram (%) over the ``n`` intervals
            defined by ``binEdges`` (sums to 100). Used to seed global and per-basin
            distributions for synthetic cases.
        binEdges : array_like, optional
            Length-``n+1`` depth bin edges (km). Values deeper than the last edge
            are assigned to the last bin.
        AOC : float, optional
            Total seafloor area (m²).
        VOC : float, optional
            Total ocean volume (m³).
        fanoc : array_like, optional
            Length ``BasinCnt+1`` decimal fractions for **area**: one per basin
            plus one high-latitude box fraction. Must sum to 1. Default
            ``[0.30, 0.30, 0.30, 0.10]``.
        fdvol : array_like, optional
            Length ``BasinCnt`` decimal fractions for **volume** per basin
            (excluding the high-latitude box). Must sum to 1. Default
            ``[1/3, 1/3, 1/3]``.
        verbose : bool, optional
            If ``True``, print status messages. Default ``True``.

        Returns
        -------
        None

        Sets
        ----
        bathymetryVolFraction : dict
        bathymetryAreaFraction : dict
        bathymetryAreaDistBasin : dict
        bathymetryAreaDist_wHighlatG : array_like
        bathymetryAreaDistG : array_like
        binEdges : array_like
        AOC : float
        VOC : float
        highlatA : float
        highlatlat : None
        BasinParametersDefined : bool

        Notes
        -----
        For synthetic cases, per-basin distributions are initially copied from the
        provided global ``Distribution``.
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

    def defineBasinConnectivityParameters(
        self,
        BasinCnt=3,
        Distribution=None,
        verbose=True,
    ):
        """
        Define basin-to-basin connectivity bathymetry distributions.

        For synthetic models, the connectivity distributions are initialized from
        a common global shape (``Distribution``) and stored in a 3-D array with
        basin-pair slices.

        Parameters
        ----------
        BasinCnt : int, optional
            Number of basins (sets the first two dimensions). Default ``3``.
        Distribution : array_like, optional
            Length-``n`` bathymetry histogram (%) over the ``n`` intervals defined by
            the previously established ``binEdges``. Sums to 100.
        verbose : bool, optional
            If ``True``, print status messages. Default ``True``.

        Returns
        -------
        None

        Sets
        ----
        bathymetryConDist : ndarray
            Shape ``(BasinCnt, BasinCnt, len(binEdges)-1)``.
        basinConnectionDefined : bool
            Flag set to ``True`` on success.

        Notes
        -----
        - Requires that ``self.binEdges`` and ``self.basinCnt`` (or equivalents) be
        consistent with the provided arguments.
        - Each basin-pair distribution should sum to 100 (%).
        """


        # Setup array to hold connective bathymetry distributions.
        self.bathymetryConDist = np.zeros( (self.basinCnt, self.basinCnt, len(binEdges)-1) ,dtype=float)

        # Populate the array with bathymetry distributions
        print("Working Progress - Not yet implemented")

        self.basinConnectionDefined = True;
    
    def saveCcycleParameter(self, verbose=True):
        """
        Write synthetic bathymetry parameters to a netCDF4 file.

        Creates groups and variables suitable for downstream carbon-cycle models.
        Content depends on which parameter sets have been defined.

        Output Structure
        ----------------
        Group ``/CycleParms`` (written if :attr:`BasinParametersDefined` is True):
            dimensions
                - ``binEdges``: ``len(self.binEdges[1:])``
                - ``BasinID``: number of basins
            variables
                - ``binEdges`` (km)
                - ``Global-whighlat`` (% over bins)
                - ``Global`` (% over bins)
                - ``basin-0``, ``basin-1``, ... (% over bins)
                - ``BasinID`` (IDs)
                - ``fdvol`` (volume fractions per basin, %)
                - ``fanoc`` (area fractions per basin, %)
                - ``highlatlat`` (degrees)
                - ``highlatA`` (m²)
                - ``AOC`` (m²)
                - ``VOC`` (m³)

        Group ``/basinConnections`` (written if :attr:`basinConnectionDefined` is True):
            dimensions
                - ``binEdges``: ``len(self.binEdges[1:])``
                - ``BasinID``: number of basins
            variables
                - ``basinConnectionBathymetry``: (BasinID, BasinID, binEdges)

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, prints a summary of what was written and a table of groups,
            variables, dimensions, and shapes. Default ``True``.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the output path cannot be created or the file cannot be written.
        ValueError
            If required parameters (e.g., ``binEdges``) are missing when attempting
            to write their dependent variables.

        Notes
        -----
        - File is written to ``{dataDir}/{filename}``.
        - Only the groups for which parameters have been defined are written.
        - The method reopens the file in read mode at the end (when ``verbose``)
        to print a concise content report.
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

############################################################
###################### Helper Function #####################
############################################################
def haversine_distance(lat1, lon1, lat2, lon2, radius):
    """
    Great-circle distance via the Haversine formula.

    Computes the central–angle distance between two points on a sphere and
    returns the arc length in the same units as ``radius``. Supports NumPy
    broadcasting, so ``lat2``/``lon2`` may be vectors/arrays.

    Parameters
    ----------
    lat1 : float or array_like
        Latitude of the first point in **degrees** (positive north).
    lon1 : float or array_like
        Longitude of the first point in **degrees** (positive east).
    lat2 : float or array_like
        Latitude of the second point in **degrees**. Can be a vector/array
        broadcastable against ``lat1``.
    lon2 : float or array_like
        Longitude of the second point in **degrees**. Can be a vector/array
        broadcastable against ``lon1``.
    radius : float
        Sphere radius (e.g., ``6371e3`` for Earth in meters). The returned
        distance uses the same units.

    Returns
    -------
    distance : float or ndarray
        Great-circle distance(s) between the points, in units of ``radius``.
        Shape follows NumPy broadcasting of the inputs.

    Notes
    -----
    - Formula: ``a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2)``,
      ``c = 2·atan2(√a, √(1−a))``, ``d = R·c``.
    - Inputs are converted to radians internally.
    - Works for scalar or array inputs; all arguments are broadcast together.

    Examples
    --------
    Distance (km) between two cities (approx.):

    >>> R_earth = 6371.0  # km
    >>> d = haversine_distance(37.7749, -122.4194, 34.0522, -118.2437, R_earth)
    >>> round(d, 1)  # San Francisco ↔ Los Angeles
    559.1

    Vectorized against a single origin:

    >>> lat1, lon1 = 0.0, 0.0
    >>> lat2 = [0.0, 0.0, 10.0]
    >>> lon2 = [0.0, 10.0, 0.0]
    >>> haversine_distance(lat1, lon1, lat2, lon2, R_earth).shape
    (3,)
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

#######################################################################
################ Process Global Ocean Physics Reanalysis ##############
#######################################################################
class GLORYS12V1_QT:
    """
    GLORYS12V1 downloader/formatter with quantile-transformed depth averaging.

    This helper class prepares GLORYS12V1 ocean reanalysis fields for use in
    ExoCcycle-style workflows. It can (optionally) download monthly NetCDF
    tiles, simplify them to lon/lat/``z`` variables, and produce a **quantile-
    transformed**, layer-thickness–weighted vertical average over a user-given
    depth interval. It can also average multiple monthly files with GMT.

    Parameters
    ----------
    options : dict, optional
        Configuration dictionary. Recognized keys:
        - ``"download"`` (bool): If ``True``, :meth:`download` should be called
          to retrieve the monthly NetCDFs. Default ``False``.
        - ``"dataDir"`` (str): Directory containing/receiving GLORYS files.
          Default ``os.getcwd() + "/GLORYS12V1"``.
        - ``"year"`` (list[int]): Years to process (e.g., ``[1994]``).
        - ``"data"`` (str): Variable to extract:
            * ``"bottomT"`` → ``sea_water_potential_temperature_at_sea_floor``
            * ``"thetao"`` → ``sea_water_potential_temperature`` (has depth)
            * ``"so"`` → ``sea_water_salinity`` (has depth)
        - ``"depthAve"`` (list[float, float]): Lower/upper bounds (meters) of
          the depth interval for vertical averaging when the variable has a
          depth dimension.

    Attributes
    ----------
    options : dict
        Stored configuration.
    areaWeightsA : any
        Placeholder for lazy initialization flag; the area-weight grid is
        created on first call to :meth:`simplifyNetCDF`.
    areaWeights, longitudes, latitudes : ndarray
        Area-weight grid and its coordinates (set by :meth:`makeAreaWeightGrid`).
    totalArea, totalAreaCalculated : float
        Sphere area (analytic and from the weights) returned by
        :func:`areaWeights`.
    netCDFGeneral : str
        Filename template for monthly tiles:
        ``"mercatorglorys12v1_gl12_mean_YEARMONTH.nc"``.
    ListOfNetCDFs, ListOfSimpNetCDFs : list[str]
        Populated by :meth:`averageModels` with monthly file paths.

    Notes
    -----
    - The depth-average in :meth:`simplifyNetCDF` applies a **quantile
      transformation** (to a normal distribution) *per depth layer* before
      thickness-weighting and averaging over the requested interval.
    - :meth:`averageModels` shells out to **GMT** via ``grdmath`` to average
      monthly simplified grids. GMT must be available on ``PATH``.
    """

    def __init__(self, options={"download": False, "dataDir": os.getcwd()+"/GLORYS12V1",
                                "year": [1994], "data": "bottomT", "depthAve": [0, 100]}):
        """
        Initialize the GLORYS12V1_QT helper.

        Parameters
        ----------
        options : dict, optional
            See class docstring for supported keys and defaults.

        Returns
        -------
        None
        """
        # Assign options to object
        self.options = options

        # Assign general name of netCDF file
        self.options["netCDFGeneral"] = "mercatorglorys12v1_gl12_mean_YEARMONTH.nc"

        # Define initial attributes
        self.areaWeightsA = None

    def averageModels(self):
        """
        Build a monthly file list, simplify each to lon/lat/``z``, then GMT-average.

        For each month of each requested year:
        1. Construct the monthly filename from the template.
        2. Call :meth:`simplifyNetCDF` to write a compact NetCDF with only
           longitude, latitude, and a single 2-D field named ``"z"``.
        3. Use GMT ``grdmath`` to compute the mean of all simplified files and
           write the result to ``{dataDir}/{data}_average_{z0}_{z1}m_QTAveraged.nc``,
           where ``z0``/``z1`` come from ``options['depthAve']``.

        Returns
        -------
        None

        Side Effects
        ------------
        - Populates :attr:`ListOfNetCDFs` and :attr:`ListOfSimpNetCDFs`.
        - Writes one simplified NetCDF per monthly file into ``dataDir``.
        - Writes a single averaged NetCDF to ``dataDir`` via GMT.

        Notes
        -----
        - Requires GMT to be installed and accessible on ``PATH``.
        - The averaging uses a straight arithmetic mean of the simplified 2-D
          fields.
        """
        # Create list of netCDFs to average
        self.ListOfNetCDFs = []
        for i in range(len(self.options['year'])):
            for month in range(12):
                readFile = self.options["netCDFGeneral"] \
                    .replace("YEAR", str(self.options['year'][i])) \
                    .replace("MONTH", f"{month+1:02d}")
                self.ListOfNetCDFs.append(readFile)

        # Iterate through all netCDF4 files, copying only the used variables
        self.ListOfSimpNetCDFs = []
        for i in range(len(self.ListOfNetCDFs)):
            self.simplifyNetCDF(
                inputPath=self.options["dataDir"] + "/" + self.ListOfNetCDFs[i],
                outputPath=self.options["dataDir"] + f"/file{i}.nc",
                variableList=['longitude', 'latitude', self.options['data']]
            )
            self.ListOfSimpNetCDFs.append(self.options["dataDir"] + f"/file{i}.nc")

        # Create a gmt command and use to gmt to average all netCDF4s.
        ## Create list of files to add
        SimpNetCDFs = " ".join(self.ListOfSimpNetCDFs)
        ## Create list of adds
        adds = []
        for _ in range(len(self.ListOfSimpNetCDFs) - 1):
            adds.append("ADD")
        adds = " ".join(adds)
        ## Define the command
        outputFileName = "{0}_average_{1}_{2}m_QTAveraged.nc".format(
            self.options["dataDir"] + "/" + self.options['data'],
            self.options['depthAve'][0],
            self.options['depthAve'][1]
        )
        GMTcommand = f"gmt grdmath {SimpNetCDFs} {adds} {len(self.ListOfSimpNetCDFs)} DIV = {outputFileName}"
        ## Use the command
        os.system(GMTcommand)
        ## Apply a mask to the averaged grid (FIXME: No longer need)
        # os.system("gmt grdmath {0} {1} OR = {1}".format(self.options["dataDir"]+"/"+self.options["ListOfSimpNetCDFs"][0], outputFileName)

    def makeAreaWeightGrid(self, step, latRange, lonRange):
        """
        Create and cache an area-weight grid covering the data domain.

        Parameters
        ----------
        step : float
            Grid resolution in degrees (assumed square cells: Δlat = Δlon = step).
        latRange : (float, float)
            ``(lat_min, lat_max)`` in degrees.
        lonRange : (float, float)
            ``(lon_min, lon_max)`` in degrees.

        Returns
        -------
        None

        Sets
        ----
        areaWeights : ndarray
            Cell areas for the requested grid (units scale with radius used
            inside :func:`areaWeights`, here 1).
        longitudes, latitudes : ndarray
            2-D arrays from ``np.meshgrid`` for the cell centers.
        totalArea, totalAreaCalculated : float
            Sphere surface area (analytic vs. sum of weights).
        """
        areaWeightsA, longitudes, latitudes, totalArea, totalAreaCalculated = areaWeights(
            resolution=step,
            radius=1,
            LonStEd=[lonRange[0]-step/2, lonRange[1]+step/2],
            LatStEd=[latRange[0]-step/2, latRange[1]+step/2]
        )
        self.areaWeights = areaWeightsA
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.totalArea = totalArea
        self.totalAreaCalculated = totalAreaCalculated

    def simplifyNetCDF(self, inputPath="path/file.nc", outputPath="~/file2.nc",
                       variableList=['longitude', 'latitude', 'variable']):
        """
        Write a compact NetCDF with lon/lat and a single 2-D field ``z``.

        If the source variable has a depth dimension, compute a **quantile-
        transformed**, layer-thickness–weighted average over
        ``options['depthAve']``. If it does **not** have a depth dimension,
        copy the 2-D field as-is into ``z``.

        Parameters
        ----------
        inputPath : str, optional
            Path to the input NetCDF file.
        outputPath : str, optional
            Path for the simplified NetCDF. ``~`` is expanded.
        variableList : list[str], optional
            Names of the longitude (x), latitude (y), and data (z) variables
            in the source file, e.g. ``['longitude', 'latitude', 'thetao']``.

        Returns
        -------
        None

        Side Effects
        ------------
        - On first use, builds and caches an area-weight grid via
          :meth:`makeAreaWeightGrid`.
        - Writes a new NetCDF in ``NETCDF4_CLASSIC`` format with:
            * dimensions copied for lon/lat,
            * variables: lon, lat, and a 2-D variable named ``"z"``.

        Notes
        -----
        - The depth-average path expects a ``depth`` coordinate in meters.
        - QuantileTransform is fitted on **non-masked** values per layer and
          applied to the full layer (masked values are ignored in the sum).
        """
        from sklearn.preprocessing import QuantileTransformer

        # Expand the user path (~) to an absolute path
        outputPath = os.path.expanduser(outputPath)

        # Open the original NetCDF file (file1.nc) in read mode
        with Dataset(inputPath, 'r') as src:
            # Define area weights if not already defined
            if self.areaWeightsA is None:
                # Defines: self.areaWeights, self.longitudes, self.latitudes, self.totalArea, self.totalAreaCalculated
                self.makeAreaWeightGrid(step=src['latitude'].step,
                                        latRange=[src['latitude'].valid_min, src['latitude'].valid_max],
                                        lonRange=[src['longitude'].valid_min, src['longitude'].valid_max])

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
                if variableList[0] in src.variables:
                    lat = src.variables[variableList[0]]
                    lat_dst_var = dst.createVariable(variableList[0], np.float32, lat.dimensions)
                    lat_dst_var[:] = lat[:]
                if variableList[1] in src.variables:
                    lon = src.variables[variableList[1]]
                    lon_dst_var = dst.createVariable(variableList[1], np.float32, lon.dimensions)
                    lon_dst_var[:] = lon[:]

                # Copy variable and rename it to 'z'
                if variableList[2] in src.variables:
                    z_var = src.variables[variableList[2]]
                    dst_z = dst.createVariable("z", np.float32, (z_var.dimensions[-2], z_var.dimensions[-1]))
                    if len(z_var.dimensions) == 3:
                        # Copy data (already 2-D)
                        dst_z[:] = z_var[:]
                    else:
                        # Thickness-weighted quantile-transformed average over depth
                        depth = src.variables['depth'][:]
                        layerThickness = np.diff(np.append(0, depth))
                        depthLogical = (depth > self.options['depthAve'][0]) & (depth < self.options['depthAve'][1])

                        LayerIdx = np.argwhere(depthLogical.data).T
                        if np.size(LayerIdx) > 1:
                            LayerIdx = LayerIdx[0]
                            topLayerIdx = np.argwhere(depthLogical.data)[0][0]
                        else:
                            topLayerIdx = np.argwhere(depthLogical.data)[0]
                        sum = z_var[:][0][topLayerIdx].data * 0
                        intervals = z_var[:][0][topLayerIdx].data * 0
                        layerThickness = layerThickness[depthLogical]
                        dataLayers = z_var[:][0][depthLogical]

                        for i in range(len(layerThickness)):
                            dataLayer = dataLayers[i]
                            if np.size(dataLayer.data[~dataLayer.mask]) == 0:
                                continue

                            qt = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='normal')
                            qt.fit_transform(np.reshape(dataLayer.data[~dataLayer.mask],
                                                        (np.size(dataLayer.data[~dataLayer.mask]), 1)))
                            dataLayerTransformed = qt.transform(np.reshape(dataLayer.data,
                                                                           (np.size(dataLayer.data), 1)))
                            dataLayerTransformed = np.reshape(dataLayerTransformed, np.shape(dataLayer.data))

                            sum[~dataLayer.mask] += dataLayerTransformed[~dataLayer.mask] * layerThickness[i]
                            intervals[~dataLayer.mask] += layerThickness[i]

                        dst_z[:] = sum / intervals

    def readnetCDF(self, year, month):
        """
        Open a single monthly GLORYS12V1 NetCDF for reading.

        Parameters
        ----------
        year : int
            Four-digit year (e.g., ``1994``).
        month : int
            Month number ``1..12``.

        Returns
        -------
        netCDF4.Dataset
            Opened dataset in read mode.

        Notes
        -----
        Builds the filename from ``options['netCDFGeneral']`` and
        ``options['dataDir']``.
        """
        # Define the file name for year and month
        readFile = self.options["dataDir"] + "/" + \
            self.options["netCDFGeneral"].replace("YEAR", str(year)) \
                                         .replace("MONTH", f"{month:02d}")

        # Read the netCDF file
        return Dataset(readFile, "r")

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

class GLORYS12V1:
    """
    GLORYS12V1 downloader/formatter for ExoCcycle workflows.

    This helper prepares GLORYS12V1 ocean reanalysis fields for use in
    ExoCcycle-style pipelines. It can (optionally) download monthly NetCDF
    tiles, simplify them to ``lon``/``lat``/``z`` variables, perform a
    thickness-weighted vertical average over a user-defined depth interval
    when applicable, and compute multi-month means via GMT.

    Parameters
    ----------
    options : dict, optional
        Configuration dictionary. Recognized keys:

        - ``download`` : bool
            If ``True``, :meth:`download` should be used to retrieve the data.
        - ``dataDir`` : str
            Directory containing/receiving GLORYS files. Default:
            ``os.getcwd() + "/GLORYS12V1"``.
        - ``year`` : list[int]
            Years to process (e.g., ``[1994]``).
        - ``data`` : str
            Variable to extract:

            * ``"bottomT"`` → ``sea_water_potential_temperature_at_sea_floor``
            * ``"thetao"`` → ``sea_water_potential_temperature`` (has depth)
            * ``"so"`` → ``sea_water_salinity`` (has depth)

        - ``depthAve`` : list[float, float]
            Two-element list ``[zmin, zmax]`` (meters) for the vertical
            averaging interval; used only for variables with a depth dimension.

    Attributes
    ----------
    options : dict
        Stored configuration, augmented with ``netCDFGeneral`` template.
    netCDFGeneral : str
        Filename template for monthly tiles:
        ``"mercatorglorys12v1_gl12_mean_YEARMONTH.nc"``.
    ListOfNetCDFs : list[str]
        Monthly file names constructed from the template (set by
        :meth:`averageModels`).
    ListOfSimpNetCDFs : list[str]
        Paths to simplified monthly files written by :meth:`simplifyNetCDF`.

    Notes
    -----
    - Vertical averaging expects a ``depth`` coordinate (meters) and uses
      simple layer-thickness weighting.
    - :meth:`averageModels` shells out to **GMT** (``grdmath``); GMT must be
      available on ``PATH``.
    """

    def __init__(self, options={"download": False, "dataDir": os.getcwd()+"/GLORYS12V1",
                                 "year": [1994], "data": "bottomT", "depthAve": [0, 100]}):
        """
        Initialize the GLORYS12V1 workflow helper.

        Parameters
        ----------
        options : dict, optional
            See class docstring for supported keys and defaults.

        Returns
        -------
        None
        """
        # Assign options to object
        self.options = options

        # Assign general name of netCDF file
        self.options["netCDFGeneral"] = "mercatorglorys12v1_gl12_mean_YEARMONTH.nc"

    def averageModels(self):
        """
        Create simplified monthly grids and compute their GMT average.

        Workflow
        --------
        1. For each month of each year in ``options['year']``, build the
           filename from the template and record it in :attr:`ListOfNetCDFs`.
        2. For each monthly file, call :meth:`simplifyNetCDF` to write a compact
           NetCDF with only longitude, latitude, and a single 2-D field ``z``.
        3. Use GMT ``grdmath`` to compute the arithmetic mean across all
           simplified files and write the result to:

           ``{dataDir}/{data}_average_{z0}_{z1}m.nc``

           where ``z0`` and ``z1`` come from ``options['depthAve']``.

        Returns
        -------
        None

        Side Effects
        ------------
        - Populates :attr:`ListOfNetCDFs` and :attr:`ListOfSimpNetCDFs`.
        - Writes simplified monthly NetCDFs into ``dataDir``.
        - Writes a single averaged NetCDF to ``dataDir`` via GMT.
        """
        # Create list of netCDFs to average
        self.ListOfNetCDFs = []
        for i in range(len(self.options['year'])):
            for month in range(12):
                readFile = self.options["netCDFGeneral"].replace("YEAR", str(self.options['year'][i])).replace("MONTH", "{}".format(month+1).zfill(2))
                self.ListOfNetCDFs.append(readFile)

        # Iterate through all netCDF4 files, copying only the used variables
        self.ListOfSimpNetCDFs = []
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
        adds = []
        for i in range(len(self.ListOfSimpNetCDFs)-1):
            adds.append("ADD")
        adds = " ".join(adds)
        ## Define the command
        outputFileName = "{0}_average_{1}_{2}m.nc".format(self.options["dataDir"]+"/"+self.options['data'], self.options['depthAve'][0], self.options['depthAve'][1])
        GMTcommand = "gmt grdmath {0} {1} {2} DIV = {3}".format(SimpNetCDFs, adds, len(self.ListOfSimpNetCDFs), outputFileName)
        ## Use the command
        os.system(GMTcommand)
        ## Apply a mask to the averaged grid (FIXME: No longer need)
        #os.system("gmt grdmath {0} {1} OR = {1}".format(self.options["dataDir"]+"/"+self.options["ListOfSimpNetCDFs"][0], outputFileName)

    def simplifyNetCDF(self, inputPath="path/file.nc", outputPath="~/file2.nc",
                       variableList=['longitude', 'latitude', 'variable']):
        """
        Write a compact NetCDF with lon/lat and a single 2-D field ``z``.

        If the source variable has a depth dimension, compute a thickness-
        weighted vertical average over the interval defined by
        ``options['depthAve']``. If it does **not** have a depth dimension,
        copy the 2-D field as-is into ``z``.

        Parameters
        ----------
        inputPath : str, optional
            Path to the input NetCDF file.
        outputPath : str, optional
            Destination path for the simplified NetCDF. ``~`` is expanded.
        variableList : list[str], optional
            Names of the longitude (x), latitude (y), and data (z) variables
            in the source file (e.g., ``['longitude', 'latitude', 'thetao']``).

        Returns
        -------
        None

        Notes
        -----
        - The depth-average path expects a ``depth`` coordinate in meters and
          uses simple layer-thickness weighting within the requested interval.
        - Output format is ``NETCDF4_CLASSIC``; the data variable is named
          ``"z"`` and has shape ``(y, x)``.
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
                if variableList[0] in src.variables:
                    lat = src.variables[variableList[0]]
                    lat_dst_var = dst.createVariable(variableList[0], np.float32, lat.dimensions)
                    lat_dst_var[:] = lat[:]  # Copy data
                if variableList[1] in src.variables:
                    lon = src.variables[variableList[1]]
                    lon_dst_var = dst.createVariable(variableList[1], np.float32, lon.dimensions)
                    lon_dst_var[:] = lon[:]  # Copy data

                # Copy variable and rename it to 'z'
                if variableList[2] in src.variables:
                    z_var = src.variables[variableList[2]]
                    dst_z = dst.createVariable("z", np.float32, (z_var.dimensions[-2], z_var.dimensions[-1]))
                    if len(z_var.dimensions) == 3:
                        # Copy data
                        dst_z[:] = z_var[:]
                    else:
                        # Define depth variable and logical defining the averaging window
                        depth = src.variables['depth'][:]
                        layerThickness = np.diff(np.append(0, depth))
                        depthLogical = (depth > self.options['depthAve'][0]) & (depth < self.options['depthAve'][1])

                        # Prepare accumulators (use top in-window layer for shape)
                        LayerIdx = np.argwhere(depthLogical.data).T
                        if np.size(LayerIdx) > 1:
                            LayerIdx = LayerIdx[0]
                            topLayerIdx = np.argwhere(depthLogical.data)[0][0]
                        else:
                            topLayerIdx = np.argwhere(depthLogical.data)[0]
                        sum = z_var[:][0][topLayerIdx].data * 0
                        intervals = z_var[:][0][topLayerIdx].data * 0
                        layerThickness = layerThickness[depthLogical]
                        dataLayers = z_var[:][0][depthLogical]

                        # Accumulate thickness-weighted values
                        for i in range(len(layerThickness)):
                            sum[~dataLayers[i].mask]       += dataLayers[i].data[~dataLayers[i].mask] * layerThickness[i]
                            intervals[~dataLayers[i].mask] += layerThickness[i]

                        # Assign averaged result
                        dst_z[:] = sum / intervals

    def readnetCDF(self, year, month):
        """
        Open a single monthly GLORYS12V1 NetCDF for reading.

        Parameters
        ----------
        year : int
            Four-digit year (e.g., ``1994``).
        month : int
            Month number ``1..12``.

        Returns
        -------
        netCDF4.Dataset
            Opened dataset in read mode.

        Notes
        -----
        The filename is constructed from :attr:`netCDFGeneral` and
        ``options['dataDir']``.
        """
        # Define the file name for year and month
        readFile = self.options["dataDir"]+"/"+self.options["netCDFGeneral"].replace("YEAR", str(year)).replace("MONTH", "{}".format(month).zfill(2))

        # Read the netCDF file
        return Dataset(readFile, "r")

#######################################################################
################# Extra Functions, might not be used ##################
#######################################################################
def polygonAreaOnSphere(vertices, radius=6371e3):
    """
    Compute the area of a spherical polygon via the spherical-excess method.

    Great-circle arcs are assumed between consecutive vertices. The polygon
    may be open or explicitly closed (i.e., repeating the first vertex at the
    end is optional).

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Sequence of ``(latitude, longitude)`` vertex coordinates in **degrees**.
        At least three vertices are required. Longitude may be any degree value
        (wrapping is handled modulo 360).
    radius : float, optional
        Sphere radius. The default is Earth's mean radius in meters
        (``6371e3``). The output units scale with ``radius**2``.

    Returns
    -------
    float
        Polygon area on the sphere, in the same squared units as ``radius``
        (e.g., m² when ``radius`` is in meters).

    Notes
    -----
    - Edges are geodesics (great-circle segments).
    - The algorithm uses a running sum of signed angles; the final area is
      based on the spherical excess :math:`E = \\left|\\sum \\alpha_i - (n-2)\\pi\\right|`.
    - Vertex ordering (clockwise vs. counter-clockwise) does not affect the
      magnitude of the returned area due to the absolute value, but self-
      intersecting polygons are not supported.
    - Numerical stability can degrade for very small polygons or for polygons
      spanning antipodal points.

    Examples
    --------
    >>> # Approximate area of a 1°x1° patch near the equator
    >>> verts = [(0, 0), (0, 1), (1, 1), (1, 0)]
    >>> area_m2 = polygonAreaOnSphere(verts)  # radius default in meters

    See Also
    --------
    haversine_distance : Great-circle distance between two points on a sphere.
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
        total_angle += np.arctan2(
            np.tan(lat2 / 2 + np.pi / 4) * np.sin(delta_lon),
            np.tan(lat1 / 2 + np.pi / 4)
        )

    # Spherical excess formula: Area = (sum of angles - (n-2)*pi) * radius^2
    spherical_excess = abs(total_angle - (n - 2) * np.pi)

    # Return the area on the sphere
    area = spherical_excess * (radius ** 2)

    return area