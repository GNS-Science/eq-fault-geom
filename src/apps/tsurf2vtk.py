#!/usr/bin/env python3
"""
Simple app to convert either a single tsurf file or a directory with
multiple tsurf files to VTK format for visualization with Paraview or
other visualization packages.
"""

# The code requires numpy, meshio, pathlib2 (for compatibility),
# argparse and the faultmeshio package.
from pathlib2 import Path
import argparse
import sys
import numpy
import meshio
sys.path.insert(0, '../../src')
import faultmeshio
# import pdb
# pdb.set_trace()

tsurfSuffix = '.ts'
vtkSuffix = '.vtk'
tsurfSearch = '*' + tsurfSuffix

# ----------------------------------------------------------------------
def convertFile(inFile, outFile):
    """
    Function to read a TSurf file and output a VTK file.
    """
    # Read TSurf file.
    mesh = faultmeshio.tsurf(inFile)

    # Convert vertices and cells to numpy arrays.
    # Cells consists of a dictionary.
    points = numpy.array(mesh.vertices, dtype=numpy.float64)
    cells = {"triangle": numpy.array(mesh.triangles, dtype=numpy.int)}

    # Create new mesh.
    newMesh = meshio.Mesh(points, cells)

    # Write new mesh as VTK file.
    outString = str(outFile)
    meshio.write(outString, newMesh)

    return
    

def convertDir(inPath, outPath):
    """
    Function to convert a directory of Tsurf files to VTK format.
    """
    tsurfFiles = sorted(inPath.glob(tsurfSearch))
    outPath.mkdir(parents=True, exist_ok=True)
    for tsurfFile in tsurfFiles:
        stem = tsurfFile.stem
        outFile = Path.joinpath(outPath, stem).with_suffix(vtkSuffix)
        convertFile(tsurfFile, outFile)

    return

  
# ======================================================================
# The main part of the code is below.
# Get command-line arguments.

parser = argparse.ArgumentParser(description='Convert one or more TSurf files to VTK format.')
parser.add_argument("-i", "--in_files", action="store", 
                  dest="inFiles", required=True, help="input file or directory")
parser.add_argument("-o", "--out_files", action="store", 
                  dest="outFiles", required=True, help="output file or directory")

args = parser.parse_args()

inPath = Path(args.inFiles)
outPath = Path(args.outFiles)

# Case 1:  Convert single file.
if (inPath.is_file()):
    inSuff = inPath.suffix
    inStem = inPath.stem
    outDir = outPath.parent
    outDir.mkdir(parents=True, exist_ok=True)
    if (inSuff != tsurfSuffix):
        msg = 'Only tsurf (*.ts) files are allowed as input.'
        raise ValueError(msg)
    outPath = outPath.with_suffix(vtkSuffix)
    convertFile(inPath, outPath)
# Case 2:  Convert directory.
elif (inPath.is_dir()):
    convertDir(inPath, outPath)
# Case 3:  Give up.
else:
    msg = 'Unable to find %s.' % inPath
    raise ValueError(msg)
    
