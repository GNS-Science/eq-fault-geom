#!/bin/bash
python ./tsurf2vtk.py --in_files=../../../data --out_files=multifile
python ./tsurf2vtk.py --in_files=../../../data/Wellington_Hutt_Valley_1.ts  --out_files=singlefile/Wellington_Hutt_Valley_1.vtk
