#!/usr/bin/env python
"""
Very basic test of meshio/tsurf.py.
Reads and writes a Tsurf file.
"""

import sys
sys.path.insert(0, '..')
# import pdb
# pdb.set_trace()
import meshio
inFile = '../data/Wellington_Hutt_Valley_1.ts'
outFile = 'Wellington_Hutt_Valley_1_test.ts'
mesh = meshio.tsurf(inFile)
mesh.write(outFile)
