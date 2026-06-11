"""Tests for the standalone `stfio` Python module.

The fixture file [`test.h5`](src/pystfio/test.h5) contains a recording with:
- 4 channels
- 3 sections per channel
- 40000 data points per section
- sampling interval 0.05 ms
"""

from pathlib import Path
import unittest

import numpy as np

import stfio

TEST_DATA = Path(__file__).with_name("test.h5")
MISSING_DATA = Path(__file__).with_name("test.txt")


class DataListTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not TEST_DATA.exists():
            raise unittest.SkipTest(
                f"Missing test fixture: {TEST_DATA}. "
                "The stfio Python tests require src/pystfio/test.h5."
            )
        cls.rec = stfio.read(TEST_DATA)

    def test_read_h5(self):
        """HDF5 recordings can be read into a Recording object."""
        rec = stfio.read(TEST_DATA)
        self.assertIsInstance(rec, stfio.Recording)

    def test_read_accepts_pathlike(self):
        """`read()` accepts pathlib.Path inputs via os.PathLike support."""
        rec = stfio.read(TEST_DATA)
        self.assertIsInstance(rec, stfio.Recording)

    def test_open_supports_context_manager(self):
        """`open()` supports modern `with ... as ...` usage."""
        with stfio.open(TEST_DATA) as rec:
            self.assertIsInstance(rec, stfio.Recording)
            self.assertEqual(4, len(rec))

    def test_read_unsupported_file_raises(self):
        """Reading an unsupported or missing file raises StfIOException."""
        with self.assertRaises(stfio.StfIOException):
            stfio.read(MISSING_DATA)

    def test_write_unsupported_format_returns_false(self):
        """Writing to an unsupported target format reports failure."""
        res = self.rec.write('new.abf', 'abf')
        self.assertEqual(False, res)

    def test_number_of_channels(self):
        self.assertEqual(4, len(self.rec))

    def test_number_of_sections(self):
        self.assertEqual(3, len(self.rec[0]))

    def test_number_of_data_points(self):
        self.assertEqual(40000, len(self.rec[0][0]))

    def test_units(self):
        self.assertEqual('mV', self.rec[0].yunits)
        self.assertEqual('pA', self.rec[1].yunits)
        self.assertEqual('ms', self.rec.xunits)

    def test_array_creation(self):
        arr = np.asarray(self.rec[0][0].asarray())
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual((40000,), arr.shape)

    def test_channel_name(self):
        names = [self.rec[i].name for i in range(len(self.rec))]
        self.assertEqual(names, ['Amp1', 'Amp2', 'Amp3', 'Amp4'])

    def test_sampling_rate(self):
        self.assertAlmostEqual(0.05, self.rec.dt, places=3)

    def test_time(self):
        self.assertEqual(self.rec.time, '23:24:42')


if __name__ == '__main__':
    unittest.main()
