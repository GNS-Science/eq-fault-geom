import unittest
from io import StringIO
from eq_fault_geom.geomops.rupture_set.fault_section import (
    SheetFault, FaultSubSection, FaultSubSectionFactory)
import csv
import os

module_path = os.path.dirname(__file__)


class TestSubductionZoneFault(unittest.TestCase):

    def setUp(self):
        self.factory = FaultSubSectionFactory()

    def test_create_new_fault(self):
        sf = SheetFault("My First Subduction Zone")
        self.assertEqual(sf.name, "My First Subduction Zone")
        self.assertEqual(len(sf.sub_sections), 0)

    def test_load_sub_sections_from_csv(self):
        tile_param_csv = open(os.path.join(module_path,
                              'fixture/tile_param.csv'), 'r')
        sf = SheetFault("9 part Subduction Zone")\
            .build_surface_from_csv(self.factory, tile_param_csv)
        self.assertEqual(len(sf.sub_sections), 24)
        self.assertIsInstance(sf.sub_sections[0], FaultSubSection)

        # print(sf.sub_sections.values[-1])
        self.assertIs(sf, sf.sub_sections[0].parent)

    def test_load_sub_sections_from_invalid_csv_exception(self):
        with self.assertRaises((ValueError, IndexError)):
            SheetFault("24 part Subduction Zone")\
                .build_surface_from_csv(self.factory,
                                        StringIO('Sorry this is not csv_data'))


class TestFaultSubSection(unittest.TestCase):

    def setUp(self):
        self.factory = FaultSubSectionFactory()
        tile_param_csv = open(os.path.join(module_path,
                              'fixture/tile_param.csv'), 'r')
        reader = csv.DictReader(tile_param_csv)
        self.csvrows = [x for x in reader]

    def test_create_from_invalid_csvrow_exception(self):
        with self.assertRaises((KeyError,)):
            FaultSubSection.from_csv_row(self.factory,
                                         dict(x='Sorry this is not csv_data'))

    def test_create_from_csv_row(self):
        fss = FaultSubSection\
                .from_csv_row(self.factory, self.csvrows[0], parent=None)

        self.assertEqual(0, fss.id)
        self.assertAlmostEqual(-43.6606619468, fss.top_trace[0].x)
        self.assertAlmostEqual(172.550493842, fss.top_trace[0].y)

        self.assertEqual((0, 0), fss.strike_dip_index)
        self.assertAlmostEqual(17.190332526, fss.dip)

        self.assertAlmostEqual(27.77718083, fss.top_depth)
        self.assertAlmostEqual(30.73264945, fss.bottom_depth)


class TestGenerateRectangularRuptures(unittest.TestCase):

    def setUp(self):
        self.factory = FaultSubSectionFactory()
        tile_param_csv = open(os.path.join(module_path,
                                           'fixture/tile_param.csv'), 'r')
        self.sf = SheetFault("9 part Subduction Zone")\
            .build_surface_from_csv(self.factory, tile_param_csv)

    # @unittest.skip("WIP")
    def test_rupture_one_by_one(self):
        shape_spec = dict(name="1 by 1", scale=1, aspect=1)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [(0, 0)])
        self.assertEqual(ruptures[1], [(0, 1)])
        self.assertEqual(ruptures[7], [(1, 0)])
        self.assertEqual(ruptures[8], [(1, 1)])

    def test_rupture_one_by_two(self):
        shape_spec = dict(name="1 by 2", scale=1, aspect=2)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [(0, 0), (1, 0)])     # begin col 0
        self.assertEqual(ruptures[1], [(0, 1), (1, 1)])

        self.assertEqual(ruptures[7], [(1, 0), (2, 0)])     # begin col 1
        self.assertEqual(ruptures[8], [(1, 1), (2, 1)])

    def test_rupture_two_by_one(self):
        shape_spec = dict(name="1 by 2", scale=2, aspect=0.5)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [(0, 0), (0, 1)])     # begin col 0
        self.assertEqual(ruptures[5], [(0, 5), (0, 6)])

        self.assertEqual(ruptures[6], [(1, 0), (1, 1)])     # begin col 1
        self.assertEqual(ruptures[17], [(1, 11), (1, 12)])

        self.assertEqual(ruptures[-1], [(2, 2), (2, 3)])    # last

    def test_rupture_two_by_three(self):
        shape_spec = dict(name="2 by 3", scale=2, aspect=1.5)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [(0, 0), (0, 1), (1, 0),
                                       (1, 1), (2, 0), (2, 1)])  # col 0

    def test_rupture_four_by_four(self):
        shape_spec = dict(name="4x4", scale=4, aspect=1, min_fill_factor=0.7)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3)])

    def test_rupture_offset_(self):
        shape_spec = dict(name="4 by 4", scale=4,
                          aspect=1, min_fill_factor=0.55, interval=2)
        ruptures = [r for r in self.sf.get_ruptures(shape_spec)]

        self.assertEqual(ruptures[0], [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3)])
        self.assertEqual(ruptures[1], [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 2), (2, 3)])
