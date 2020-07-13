#!/usr/bin/env bash
grd_wgs=../data/grid_exclude_wgs84_ext.grd
grd_wgs_lt50=../data/grid_exclude_wgs84_ext_lt50.grd
tif_wgs_lt50=../data/williams_0_005_wgs84.tif
tif_nztm_lt50=../data/williams_0_005_nztm.tif
gmt grdclip $grd_wgs -G$grd_wgs_lt50 -Sb-50/NaN
gdal_translate $grd_wgs_lt50 $tif_wgs_lt50
gdalwarp $tif_wgs_lt50 -s_srs epsg:4326 -t_srs epsg:2193 $tif_nztm_lt50




