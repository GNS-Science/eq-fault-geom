#!/usr/bin/env python
"""
Very basic test of faultmeshio/tsurf.py.
Reads a Tsurf file and exports a VTK file using the meshio package.
"""

import sys
import numpy
import meshio
sys.path.insert(0, '../../src')
# import pdb
# pdb.set_trace()
from eq_fault_geom import faultmeshio

# Files.
inFile = '../../data/Wellington_Hutt_Valley_1.ts'
outFile = 'Wellington_Hutt_Valley_1_test1.vtk'

# Read sample Tsurf file.
tsurf = faultmeshio.tsurf(inFile)

# Write VTK file.
meshio.write(outFile, tsurf.mesh)
