"""
unittest_stfio.py

Tue, 20 Jul 2010 09:38:19 +0200

The unittest for stfio module created by Christoph,
the stfio module is a Stimfit-independent python module to read/write
electrophysiological data from/to different file formats.

Note that to execute this test you need the following files:
test.h5
test.abf
test.dat

These files correspond to the same recording with different file 
extensions; the recording contains the following properties:
 
Number of channels = 4
Number of traces = 3
Sampling interval  = 0.05
 
"""
import numpy as np
import unittest

import stfio

rec = stfio.read('test.h5')
class DataListTest(unittest.TestCase):
    
    def testReadH5(self):
        """ testReadH5() Read HDF5 file system """
        try:
            rec = stfio.read('test.h5')
        except SyntaxError:
            rec = None

        # test if Recording object was created
        self.assertTrue(True, isinstance(rec, stfio.Recording))

    # def testReadCFS(self):
    #     """ testReadCFS() Read CED filling system files"""
    #     try:
    #         rec = stfio.read('test.dat')
    #     except SyntaxError:
    #         rec = None

    #     # test if Recording object was created
    #     self.assertTrue(True, isinstance(rec, stfio.Recording))

    # def testReadATF(self):
    #     """ testReadATF() Read ABF files"""
    #     try:
    #         rec = stfio.read('test.atf')
    #     except SyntaxError:
    #         rec = None

    #     # test if Recording object was created
    #     self.assertTrue(True, isinstance(rec, stfio.Recording))

    def testReadStfException(self):
        """ Raises a StfException if file format to read is not supported"""

        # should raise an StfIOException if filetype is not supported
        self.assertRaises(stfio.StfIOException, stfio.read, 'test.txt')


    def testWrite(self):
        """ testWrite() Returns False if file format
            to write is not supported"""

        # should raise an StfException if filetype is not supported
        rec = stfio.read('test.h5')
        res = rec.write('new.abf', 'abf')
        self.assertEquals(False, res)

    def testNumberofChannels(self):
        """ testNumberofChannels() returns the number of channels """
        self.assertEquals(4,len(rec))

    def testNumberofSections(self):
        """ testNumberofSections() returns the number of channels """
        self.assertEquals(3,len(rec[0]))

    def testNumberofDataPoints(self):
        """ testNumberofDataPoints() returns the number of sampling points """
        self.assertEquals(40000,len(rec[0][0]))

    def testunits(self):
        """ testunits() returns the units in the X/Y axis """
        self.assertEquals('mV',rec[0].yunits)
        self.assertEquals('pA',rec[1].yunits)
        self.assertEquals('ms',rec.xunits)

    def testArrayCreation(self):
        """ testArrayCreation() creation of a numpy array""" 
        self.assertTrue(type(rec[0][0].asarray()), type(np.empty(0)))

    def testChannelName(self):
        """ testChannelName() returns the names of the channels """
        names = [rec[i].name for i in range(len(rec))]
        mynames = ['Amp1', 'Amp2', 'Amp3', 'Amp4']
        self.assertEquals(names, mynames)

    def testSamplingRate(self):
        """ testSamplingRate() returns the sampling rate """
        self.assertAlmostEqual(0.05, rec.dt, 3)

    def testDate(self):
        """ testDate() returns the creation date """
        self.assertEquals(rec.date, '19/07/10')

    def testTime(self):
        """ testTime() returns the creation time """
        self.assertEquals(rec.time, '23:24:42')

if __name__ == '__main__':
    # test all cases
    unittest.main()
