## eq-fault-geom

Python tools for manipulating fault geometries.

### Getting started

##### 1. Clone the repository

````
git clone https://github.com/GNS-Science/eq-fault-geom.git
````

##### 2. Setup a Python virtulenv

pre-requisites:

* python >=Python 3.7

* python pip3

* python virtualenv

##### 3. Create and activate Python virtualenv

````
mkdir eq-fault-geom/ENV
cd eq-fault-geom/ENV
virtualenv -p python3 eqf_venv
source eqf_venv/bin/activate
````
##### 4. Install required packages in virtual environment
```
pip3 install -r requirements.txt
````
[PyPA](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

##### 5. Install eq-fault_geom package
```
pip install -e .
````
#####

#### 6. Run the tests
To run all the tests:
````
python setup.py test

````
https://pythonhosted.org/foo-test/testing.html



______________________________________________________
#### Proposed initial layout of Python package.
Top-level directories:
 - data - Test data (e.g., example Tsurfs, .grd files, etc.)
 - src/geomio - Subpackage for performing I/O on fault geometry
 - src/geomops - Subpackage for performing operations on geometry (e.g., surface fitting, cutting/truncating surfaces, etc.)
 - src/meshing - Subpackage for either meshing using pytriangle or creating meshing commands for Trelis
 - src/faultmeshio - Subpackage for I/O on different mesh formats (Tsurfs, RSQSim, OpenSHA, etc.)
 - src/meshops - Subpackage for mesh operations (eg., edge extraction, transformations, etc.)

I haven't yet decided if we need a subpackage for coordinate transformations. I would assume we would
primarily rely on pyproj for that.

Anticipated dependencies:
numpy, scipy, pyproj, pytriangle, netCDF4, XML package (not sure which), shapely

Possible future dependencies:
h5py, Python XDMF writer

It's better to keep requirement.txt without combining it with setup.py 

https://caremad.io/posts/2013/07/setup-vs-requirement/

