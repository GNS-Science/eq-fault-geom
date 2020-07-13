# eq-fault-geom
python tools for manipulating fault geometries 

# Proposed initial layout of Python package.
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
