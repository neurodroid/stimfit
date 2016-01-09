from distutils.core import setup, Extension

import sys
import subprocess
import shlex

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
np_libraries = []

for key in numpy_config_keys:
    try:
        np_libraries += np.__config__.get_info(key)['libraries']
    except:
        pass
    try:
        np_define_macros += np.__config__.get_info(key)['define_macros']
    except:
        pass
    try:
        np_extra_compile_args += np.__config__.get_info(key)[
            'extra_compile_args']
    except:
        pass
    try:
        np_extra_link_args += np.__config__.get_info(key)['extra_link_args']
    except:
        pass

hdf5_extra_compile_args = []
hdf5_extra_link_args = []
if 'linux' in sys.platform:
    cmd = shlex.split('pkg-config --cflags hdf5')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    pkg_config_out = p.stdout.read().decode("utf-8")[:-2]
    pkg_config_err = p.stderr.read().decode("utf-8")[:-2]
    if "No package" in pkg_config_err:
        hdf5_extra_compile_args = ["-I/usr/include/hdf5/serial"]
    else:
        hdf5_extra_compile_args = [pkg_config_out]

    cmd = shlex.split('pkg-config --libs hdf5')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    pkg_config_out = p.stdout.read().decode("utf-8")[:-2]
    pkg_config_err = p.stderr.read().decode("utf-8")[:-2]
    if "No package" in pkg_config_err:
        hdf5_extra_link_args = [
            "-L/usr/lib/x86_64-linux-gnu/",
            "-L/usr/lib/x86_64-linux-gnu/hdf5/serial"]
    else:
        hdf5_extra_link_args = [pkg_config_out]


if 'linux' not in sys.platform:
    biosig_define_macros = [('WITH_BIOSIG2', None)]
    biosig_libraries = ['biosig2']
else:
    biosig_define_macros = [('WITH_BIOSIG', None)]
    biosig_libraries = ['biosig', 'cholmod']

fftw3_libraries = ['fftw3']
if 'libraries' in system_info.get_info('fftw3').keys():
    fftw3_libraries = system_info.get_info('fftw3')['libraries']

stfio_module = Extension(
    '_stfio',
    swig_opts=['-c++'],
    libraries=['hdf5', 'hdf5_hl'] + fftw3_libraries + np_libraries +
    biosig_libraries,
    define_macros=np_define_macros + biosig_define_macros,
    extra_compile_args=np_extra_compile_args + hdf5_extra_compile_args,
    extra_link_args=np_extra_link_args + hdf5_extra_link_args,
    sources=[
        'src/libstfio/abf/abflib.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/Oldheadr.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abferror.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abffiles.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfheadr.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfhwave.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/abfutil.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/csynch.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/filedesc.cpp',
        'src/libstfio/abf/axon/AxAbfFio32/msbincvt.cpp',
        'src/libstfio/abf/axon/AxAtfFio32/axatffio32.cpp',
        'src/libstfio/abf/axon/AxAtfFio32/fileio2.cpp',
        'src/libstfio/abf/axon/Common/FileIO.cpp',
        'src/libstfio/abf/axon/Common/FileReadCache.cpp',
        'src/libstfio/abf/axon/Common/unix.cpp',
        'src/libstfio/abf/axon2/ProtocolReaderABF2.cpp',
        'src/libstfio/abf/axon2/SimpleStringCache.cpp',
        'src/libstfio/abf/axon2/abf2headr.cpp',
        'src/libstfio/atf/atflib.cpp',
        'src/libstfio/axg/AxoGraph_ReadWrite.cpp',
        'src/libstfio/axg/axglib.cpp',
        'src/libstfio/axg/byteswap.cpp',
        'src/libstfio/axg/fileUtils.cpp',
        'src/libstfio/axg/stringUtils.cpp',
        'src/libstfio/biosig/biosiglib.cpp',
        'src/libstfio/cfs/cfs.c',
        'src/libstfio/cfs/cfslib.cpp',
        'src/libstfio/channel.cpp',
        'src/libstfio/hdf5/hdf5lib.cpp',
        'src/libstfio/igor/CrossPlatformFileIO.c',
        'src/libstfio/igor/WriteWave.c',
        'src/libstfio/igor/igorlib.cpp',
        'src/libstfio/recording.cpp',
        'src/libstfio/section.cpp',
        'src/libstfio/stfio.cpp',
        'src/libstfnum/fit.cpp',
        'src/libstfnum/funclib.cpp',
        'src/libstfnum/levmar/Axb.c',
        'src/libstfnum/levmar/lm.c',
        'src/libstfnum/levmar/lmbc.c',
        'src/libstfnum/levmar/misc.c',
        'src/libstfnum/measure.cpp',
        'src/libstfnum/stfnum.cpp',
        'src/pystfio/pystfio.cxx',
        'src/pystfio/pystfio.i',
    ])


setup(name='stfio',
      version='0.14.13',
      description='stfio module',
      include_dirs=system_info.default_include_dirs + [
          np.get_include()],
      scripts=['src/pystfio/stfio.py'],
      package_dir={'stfio': 'src/pystfio'},
      packages=['stfio'],
      ext_modules=[stfio_module])
