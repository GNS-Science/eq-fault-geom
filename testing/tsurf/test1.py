#!/usr/bin/env python
"""
Very basic test of meshio/tsurf.py.
Reads and writes a Tsurf file.
"""

import sys
sys.path.insert(0, '../../src')
# import pdb
# pdb.set_trace()
import faultmeshio
# Files.
inFile = '../../data/Wellington_Hutt_Valley_1.ts'
outFile1 = 'Wellington_Hutt_Valley_1_test1.ts'
outFile2 = 'Wellington_Hutt_Valley_1_test2.ts'
outFile3 = 'Wellington_Hutt_Valley_1_test3.ts'

# Read and write sample Tsurf file.
mesh1 = faultmeshio.tsurf(inFile)
mesh1.write(outFile1)

# Create and write new mesh using Tsurf and properties from original mesh.
x = mesh1.x
y = mesh1.y
z = mesh1.z
triangles = mesh1.triangles
mesh2 = faultmeshio.tsurf(x, y, z, triangles, name=mesh1.name,
solid_color=mesh1.solid_color, visible=mesh1.visible, NAME=mesh1.NAME,
AXIS_NAME=mesh1.AXIS_NAME, AXIS_UNIT=mesh1.AXIS_UNIT, ZPOSITIVE=mesh1.ZPOSITIVE)
# Write the mesh.
mesh2.write(outFile2)

# Create and write mesh using default properties.
mesh3 = faultmeshio.tsurf(x, y, z, triangles)
mesh3.write(outFile3)
