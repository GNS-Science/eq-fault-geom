#!/usr/bin/env python
"""
Very basic test of faultmeshio/tsurf.py.
Reads a Tsurf file and exports a VTK file using the meshio package.
"""

import sys
import numpy
import meshio
sys.path.insert(0, '../../src')
import pdb
pdb.set_trace()
import faultmeshio
# Files.
inFile = '../../data/Wellington_Hutt_Valley_1.ts'
outFile = 'Wellington_Hutt_Valley_1_test1.vtk'

# Read sample Tsurf file.
mesh = faultmeshio.tsurf(inFile)

# Get vertices and cells and convert to numpy arrays for VTK output.
points = numpy.array(mesh.vertices, dtype=numpy.float64)
cells = {"triangle": numpy.array(mesh.triangles, dtype=numpy.int)}

# Create new mesh.
newMesh = meshio.Mesh(points, cells)

# Write VTK file.
meshio.write(outFile, newMesh)
