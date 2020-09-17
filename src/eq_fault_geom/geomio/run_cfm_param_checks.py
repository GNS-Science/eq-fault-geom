import logging
from eq_fault_geom.geomio.cfm_faults import CfmMultiFault

logging.basicConfig(filename="errors2.txt", level=logging.WARNING)

shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/cfm_linework/NZ_CFM_v0_3_170620.shp"

data = CfmMultiFault.from_shp(shp, warning_output="save")
xml = data.to_opensha_xml()
with open("test3.xml", "wb") as f:
    f.write(xml)
