from unittest import TestCase
import geopandas as gpd
from xmlunittest import XmlTestMixin
import logging


from src.eq_fault_geom.geomio.cfm_faults import CfmMultiFault
from src.eq_fault_geom.geomio.cfm_faults import CfmFault
from src.eq_fault_geom.geomio.cfm_faults import required_fields
from src.eq_fault_geom.geomio.cfm_faults import expected_fields
from src.eq_fault_geom.geomio.cfm_faults import valid_dip_directions
from shapely.geometry.polygon import LineString

class test_cfm_faults(TestCase, XmlTestMixin):

    def setUp(self):

        self.filename = "../../../data/cfm_linework/NZ_CFM_v0_6_160221.shp"
        self.fault_geodataframe = gpd.GeoDataFrame.from_file(self.filename)
        self.cmf_faults = CfmMultiFault(self.fault_geodataframe)
        self.logger = logging.getLogger('cmf_logger')

        # Sort alphabetically by name
        self.sorted_df = self.fault_geodataframe.sort_values("Name")
        # Reset index to line up with alphabetical sorting
        self.sorted_df = self.sorted_df.reset_index(drop=True)

        self.faults = []






    def test_check_input1(self):
        df_response = self.fault_geodataframe[required_fields[:-1]].copy()
        with self.assertRaises(ValueError):
            self.cmf_faults.check_input1(df_response)

    def test_check_input2(self):
        exf = [i for i in expected_fields if i not in ['Source2', 'Source1_1', 'Name', 'Qual_Code']]
        df_reponse = self.fault_geodataframe[exf[:-1]].copy()
        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_faults.check_input2(df_reponse)
            print(cm.output)
            self.assertIn(
                "WARNING:cmf_logger:missing expected field", cm.output[0]
            )

    def test_add_fault(self):

        self.assertGreater(len(self.sorted_df), 0, "Not enough rows in the input data set")

        for i,fault in self.sorted_df.iterrows():
            length = len(self.cmf_faults.faults)
            self.cmf_faults.add_fault(fault)
            self.assertEqual(length+1, len(self.cmf_faults.faults))


        #assert False

    def test_fault_numbers(self):
        #self.assertIsNone(self.cmf_faults.fault_numbers)
        if self.assertIsNotNone(self.cmf_faults.fault_numbers):
            self.assertIsInstance(self.cmf_faults.fault_numbers, int)

        self.assertFalse(len(self.cmf_faults.fault_numbers) == 0, 'The fault number is missing')

    #assert False

    def test_from_shp(self):
        multiFault = self.cmf_faults.from_shp(self.filename)
        self.assertIsNotNone(multiFault)
        response = isinstance(multiFault, CfmMultiFault)
        self.assertTrue(response, 'supplied object is not a "src.eq_fault_geom.geomio.cfm_faults.CfmMultiFault"'
                                  ', it is a "{}"'.format( type( multiFault )))



    # def test_to_opensha_xml(self):
    #    prettyXml = self.cmf_faults.to_opensha_xml()
    #    self.assertIsNotNone(prettyXml)
    #    self.assertXmlDocument(prettyXml)
    #    #assert False




class test_cfm_fault(TestCase):
    def setUp(self):
        self.cmf_fault = CfmFault()
        self.logger = logging.getLogger('cmf_logger')

        self.filename = "../../../data/cfm_linework/NZ_CFM_v0_6_160221.shp"
        self.fault_geodataframe = gpd.GeoDataFrame.from_file(self.filename)
        self.cmf_faults = CfmMultiFault(self.fault_geodataframe)
        # Sort alphabetically by name
        self.sorted_df = self.fault_geodataframe.sort_values("Name")
        # Reset index to line up with alphabetical sorting
        self.sorted_df = self.sorted_df.reset_index(drop=True)

    # def test_depth_best(self): => This gets tested by depth_max and depth_min
    #     self.cmf_fault.depth_best = 5.5
    #     self.assertEqual(self.cmf_fault.depth_best, 5.5)
    #     self.cmf_fault._depth_best = 3.3
    #     self.assertNotEqual(self.cmf_fault.depth_best, 5.5)
    #
    #     self.depth_min = 20
    #     self.depth_max = 25.6
    #     depth = 17.4
    #     with self.assertLogs(logger=self.logger, level='WARNING') as cm:
    #         self.cmf_fault.depth_best = depth
        #     self.assertIn(
        #         "WARNING:cmf_logger:depth_best lower than depth_min", cm.output
        #     )




    def test_depth_max(self):
        self.cmf_fault.depth_max = 10.5
        self.assertEqual(self.cmf_fault.depth_max, 10.5)

        self.cmf_fault._depth_max = 8.6
        self.assertNotEqual(self.cmf_fault.depth_max, 10.5)


        with self.assertRaises(Exception):
            self.cmf_fault.depth_max = "Hello"

        # depth_min = self.cmf_fault.depth_min
        # depth_best = self.cmf_fault.depth_best
        # depth = min(depth_min, depth_best) - 1
        self.cmf_fault.depth_min = 20
        self.cmf_fault.depth_best = 20.4
        depth = 19.5

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.depth_max = depth
            self.assertIn(
                "WARNING:cmf_logger:depth_max lower than either depth_min or depth_best", cm.output
            )



    def test_depth_min(self):
        self.cmf_fault.depth_min = 30.5
        self.assertEqual(self.cmf_fault.depth_min, 30.5)

        self.cmf_fault._depth_min = 1.5
        self.assertNotEqual(self.cmf_fault.depth_min, 10.5)

        with self.assertRaises(Exception):
            self.cmf_fault.depth_min = "Hello"

        self.cmf_fault.depth_max = 50
        self.cmf_fault.depth_best = 10
        depth = 30.5


        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.depth_min = depth
            self.assertIn(
                "WARNING:cmf_logger:depth_min higher than either depth_max or depth_best", cm.output
            )


    # def test_validate_depth(self):        #no need to test this as it's used and tested in depth_max and depth_min
    #     assert False


    # def test_dip_best(self): => get tested by dip max and dip_min
    #     assert False



    def test_dip_max(self):
        self.cmf_fault.dip_max = 10.5
        self.assertEqual(self.cmf_fault.dip_max, 10.5)

        self.cmf_fault._dip_max = 8.6
        self.assertNotEqual(self.cmf_fault.dip_max, 10.5)
        #
        with self.assertRaises(Exception):
            self.cmf_fault.dip_max = "Hello"


        self.cmf_fault.dip_min = 20.6
        self.cmf_fault.dip_best = 40.1
        dip = 19.5

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.dip_max = dip
            self.assertIn(
                "WARNING:cmf_logger:dip_max is lower than dip min or dip best", cm.output
            )


    def test_dip_min(self):
        self.cmf_fault.dip_min = 10.5
        self.assertEqual(self.cmf_fault.dip_min, 10.5)

        self.cmf_fault._dip_min = 8.6
        self.assertNotEqual(self.cmf_fault.dip_min, 10.5)
        #
        with self.assertRaises(Exception):
            self.cmf_fault.dip_min = "Hello"

        self.cmf_fault.dip_max = 45.3
        self.cmf_fault.dip_best = 40.1
        dip = 50.6

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.dip_min = dip
            self.assertIn(
                "WARNING:cmf_logger:dip_min is higher than dip max or dip best", cm.output
            )



#not sure if the test beolw is correct;
    def test_dip_dir_str(self):
        dip_dir = 'NE'
        self.cmf_fault.dip_dir_str = dip_dir
        self.assertIsInstance(dip_dir, str)

        series = self.sorted_df.iloc[0]
        self.cmf_fault.nztm_trace = series['geometry']

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.validate_dip_direction()
            self.assertIn(
                "WARNING:cmf_logger:Supplied trace and dip direction are inconsistent", cm.output
            )

        dip_dir = None
        self.cmf_fault.dip_dir_str = dip_dir
        self.assertEqual(self.cmf_fault.dip_dir, 330.15406806735234)


    #still working on this
    def test_dip_sigma(self):
        self.cmf_fault._dip_sigma = 8.6
        self.assertEqual(self.cmf_fault.dip_sigma, 8.6)

        self.cmf_fault._dip_sigma = None
        self.cmf_fault.dip_min = 16
        self.cmf_fault.dip_max = 25
        self.assertEqual(self.cmf_fault.dip_sigma, 4.5)

        # I think no need to test the Valuerror here as it should satisfy the validate_dip() function
        # self.cmf_fault.dip_min = None
        # self.cmf_fault.dip_max = 25
        # with self.assertRaises(ValueError):
        #     self.cmf_fault.dip_sigma

    #
    # def test_dip_dir(self):
    #     assert False
    #

    def test_validate_dip_direction(self):
        series = self.sorted_df.iloc[0]
        self.cmf_fault.nztm_trace = series['geometry']

        dip_dir = 'SE'
        self.cmf_fault.dip_dir_str = dip_dir

        self.cmf_fault.validate_dip_direction()
        self.assertEqual(self.cmf_fault.dip_dir, 150.15406806735643)

        dip_dir = None
        self.cmf_fault.dip_dir_str = dip_dir
        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.validate_dip_direction()
            self.assertIn(
                "WARNING:cmf_logger:Insufficient information to validate dip direction", cm.output
            )

        dip_dir = 'NE'
        self.cmf_fault.dip_dir_str = dip_dir
        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.validate_dip_direction()
            self.assertIn(
                "WARNING:cmf_logger:Supplied trace and dip direction are inconsistent", cm.output
            )


    def test_validate_dip(self):
        dip = 15.6
        self.assertIsInstance(self.cmf_fault.validate_dip(dip), float)

        dip = "Hello"
        with self.assertRaises(Exception):
            self.cmf_fault.validate_dip(dip)

        dip = -20.6 #should be between 0 - 90 otherwise assert error
        with self.assertRaises(Exception):
            self.cmf_fault.validate_dip(dip)

    def test_nztm_trace(self):
        series = self.sorted_df.iloc[0]
        trace = series['geometry']
        #trace = 0.124
        self.cmf_fault.nztm_trace = trace

    def test_wgs_trace(self):
        series = self.sorted_df.iloc[0]
        trace = series['geometry']
        self.cmf_fault.nztm_trace = trace

        reponseX, reponseY = self.cmf_fault.wgs_trace.coords.xy
        response = reponseX.tolist()
        actual = [172.81975618060406, 172.78381840673984, 172.7622924223485]
        self.assertEqual(response, actual)



    def test_rake_max(self):
        self.cmf_fault.rake_max = 40.5
        self.assertEqual(self.cmf_fault.rake_max, 40.5)

        self.cmf_fault._rake_max = 20.1
        self.assertNotEqual(self.cmf_fault.rake_max, 40.5)
        # #
        with self.assertRaises(Exception):
            self.cmf_fault.rake_max = "rake"

        rake = 5.21
        self.cmf_fault.rake_min = 19.00
        self.cmf_fault.rake_best = 40.10


        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.rake_max = rake
            self.assertIn(
                "WARNING:cmf_logger:rake_max is lower than rake min or rake best", cm.output
            )
    def test_rake_min(self):
        self.cmf_fault.rake_min = 10.5
        self.assertEqual(self.cmf_fault.rake_min, 10.5)

        self.cmf_fault._rake_min = 8.6
        self.assertNotEqual(self.cmf_fault.rake_min, 10.5)
        #
        with self.assertRaises(Exception):
            self.cmf_fault.rake_min = "Hello"

        self.cmf_fault.rake_max = 45.3
        self.cmf_fault.rake_best = 40
        rake = 50

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.rake_min = rake
            self.assertIn(
                "WARNING:cmf_logger:rake_min is higher than rake max or rake best", cm.output
            )

#
    def test_rake_best(self):
        self.cmf_fault.rake_min = 30.4
        rake = 20

        with self.assertLogs(logger=self.logger, level='WARNING') as cm:
            self.cmf_fault.rake_best = rake
            self.assertIn(
                "WARNING:cmf_logger:rake_best is lower than rake_min", cm.output
            )

        self.cmf_fault.rake_max = 15.7

        with self.assertLogs(logger=self.logger, level='WARNING') as cm1:
            self.cmf_fault.rake_best = rake
            self.assertIn(
                "WARNING:cmf_logger:rake_best is greater than rake_max", cm1.output
            )

        self.cmf_fault.sense_dom = 'dextral'
        self.cmf_fault.sense_sec = 'sinistral'

        with self.assertLogs(logger=self.logger, level='WARNING') as cm2:
            self.cmf_fault.rake_best = rake
            self.assertIn(
                "WARNING:cmf_logger:Supplied rake differs from dominant slip sense", cm2.output
            )









    #
    # def test_sense_dom(self):
    #     assert False
    #
    # def test_sense_dom(self):
    #     assert False
    #
    # def test_sense_sec(self):
    #     assert False
    #
    # def test_sense_sec(self):
    #     assert False
    #
    # def test_rake_to_opensha(self):
    #     assert False
    #
    # def test_validate_rake(self):
    #     assert False
    #
    # def test_validate_rake_sense(self):
    #     assert False
    #
    # def test_sr_best(self):
    #     assert False
    #
    #
    # def test_sr_min(self):
    #     assert False
    #
    # def test_sr_max(self):
    #     assert False
    #
    #
    # def test_validate_sr(self):
    #     assert False
    #
    # def test_sr_sigma(self):
    #     assert False
    #
    #
    # def test_name(self):
    #     assert False
    #
    # def test_number(self):
    #     assert False
    #
    # def test_parent(self):
    #     assert False
    #


    # def test_from_series(self):
    #     series = self.sorted_df.iloc[0]
    #     # length = series.
    #     response = self.cmf_fault.from_series(series)






    #
    # def test_to_xml(self):
    #     assert False
