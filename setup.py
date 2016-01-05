from distutils.core import setup, Extension
from numpy.distutils import system_info
import numpy as np

numpy_config_keys = [
    'atlas_blas_info',
    'atlas_blas_threads_info',
    'atlas_info',
    'atlas_threads_info',
    'blas_mkl_info',
    'blas_opt_info',
    'get_info',
    'lapack_mkl_info',
    'lapack_opt_info',
    'mkl_info',
    'openblas_info',
    'openblas_lapack_info'
    ]

np_define_macros = []
np_extra_compile_args = []
np_extra_link_args = []

for key in numpy_config_keys:
    try:
        np_define_macros += np.__config__.get_info(key)['define_macros']
        np_extra_compile_args += np.__config__.get_info(key)[
            'extra_compile_args']
        np_extra_link_args += np.__config__.get_info(key)['extra_link_args']
    except:
        pass

stfio_module = Extension(
    '_stfio',
    py_modules=['stfio'],
    swig_opts=['-c++'],
    libraries=['hdf5', 'hdf5_hl'] +
    system_info.get_info('fftw3')['libraries'],
    define_macros=np_define_macros,
    extra_compile_args=np_extra_compile_args,
    extra_link_args=np_extra_link_args,
    sources=[
        'src/pystfio/pystfio.i',
        'src/pystfio/pystfio.cxx',
        'src/libstfio/stfio.cpp',
        'src/libstfio/channel.cpp',
        'src/libstfio/section.cpp',
        'src/libstfio/recording.cpp',
        'src/libstfio/atf/atflib.cpp',
        'src/libstfio/abf/axon/AxAtfFio32/axatffio32.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abffiles.cpp',
        'src/libstfio/abf/axon2/abf2headr.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfheadr.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfhwave.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfutil.cpp',
        'src/libstfio/abf/axon/AxAtfFio32/fileio2.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/Oldheadr.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/msbincvt.cpp',
        'src/libstfio/abf/axon/Common/FileReadCache.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/filedesc.cpp',
        'src/libstfio/cfs/cfslib.cpp',
        'src/libstfio/cfs/cfs.c',
        'src/libstfio/abf/abflib.cpp',
        'src/libstfio/abf/axon2/ProtocolReaderABF2.cpp',
        'src/libstfio/abf/axon2/SimpleStringCache.cpp',
        'src/libstfio/abf/axon/Common/FileIO.cpp',
        'src/libstfio/axg/axglib.cpp',
        'src/libstfio/axg/AxoGraph_ReadWrite.cpp',
        'src/libstfio/axg/byteswap.cpp',
        'src/libstfio/axg/fileUtils.cpp',
        'src/libstfio/axg/stringUtils.cpp',
        'src/libstfio/hdf5/hdf5lib.cpp',
        'src/libstfio/igor/igorlib.cpp',
        'src/libstfio/igor/CrossPlatformFileIO.c',
        'src/libstfio/igor/WriteWave.c',
        'src/libstfio/abf/axon/AxAbfFio32/csynch.cpp',
        'src/libstfnum/stfnum.cpp',
        'src/libstfnum/funclib.cpp',
        'src/libstfnum/fit.cpp',
        'src/libstfnum/measure.cpp',
        'src/libstfio/abf/axon/Common/unix.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abferror.cpp',
        'src/libstfnum/levmar/lmbc.c',
        'src/libstfnum/levmar/Axb.c',
        'src/libstfnum/levmar/misc.c',
        'src/libstfnum/levmar/lm.c',
    ])

setup(name='stfio',
      version='0.14.13',
      description='stfio module',
      include_dirs=system_info.default_include_dirs + [
          np.get_include()],
      ext_modules=[stfio_module])
