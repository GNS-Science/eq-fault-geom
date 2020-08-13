#!/usr/bin/env python
"""
Very basic test of faultmeshio/tsurf.py.
Reads and writes a Tsurf file.
"""

import sys
sys.path.insert(0, '../../src')
# import pdb
# pdb.set_trace()
from eq_fault_geom import faultmeshio

# Files.
inFile = '../../data/Wellington_Hutt_Valley_1.ts'
outFile1 = 'Wellington_Hutt_Valley_1_test1.ts'
outFile2 = 'Wellington_Hutt_Valley_1_test2.ts'
outFile3 = 'Wellington_Hutt_Valley_1_test3.ts'

# Read and write sample Tsurf file.
tsurf1 = faultmeshio.tsurf(inFile)
tsurf1.write(outFile1)

# Create and write new mesh using Tsurf and properties from original mesh.
x = tsurf1.x
y = tsurf1.y
z = tsurf1.z
triangles = tsurf1.mesh.cells
tsurf2 = faultmeshio.tsurf(x, y, z, triangles, name=tsurf1.name,
solid_color=tsurf1.solid_color, visible=tsurf1.visible, NAME=tsurf1.NAME,
AXIS_NAME=tsurf1.AXIS_NAME, AXIS_UNIT=tsurf1.AXIS_UNIT, ZPOSITIVE=tsurf1.ZPOSITIVE)

# Write the mesh.
tsurf2.write(outFile2)

# Create and write mesh using default properties.
tsurf3 = faultmeshio.tsurf(x, y, z, triangles)
tsurf3.write(outFile3)
