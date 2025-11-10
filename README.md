# ExoCcycle
A python based library for creating exoplanetary-like bathymetry models and conducting carbon cycle analyses with bathymetry models.

Currently, a primarily component of this library are its class/methods/functions for objectively finding communities on the sphere of an input spherical shell. These methods are useful for define current, paleo-, and exo- ocean basins for use in and constructing intermediate C cycle models. However, there are additional domains of study that can readily make use of community detection of single or multiple fields on the spherical shells.

There is jupyter notebook within the JN folder. After dowloading/cloning this repository, create a conda environment, open/run the GMD_Manuscript_Figures.ipynb examples to see use cases for the library.


Procedure (installation + GMD figure recreation):
1. Download the repo.
```
$ git clone https://github.com/Bogumil-Matthew/ExoCcycle.git
```

2. Navigate to the ExoCcycle folder (folder containing environment.yml file).
```
$ cd ExoCcycle/
```

3. Create a conda environment to run the code/JN within:
```
$ conda env create -f environment.yml
$ conda activate exoccycle-dev
```

4. Install python library:
```
$ pip install -e .
```

5. Navigate to the JN/ folder.
```
$ cd JN/
```

6. Run jupyter-notebooks and recreate GMD manuscript analysis. Make sure to select the kernel associated with the exoccycle-dev conda environment.
```
$ jupyter-notebook
```



