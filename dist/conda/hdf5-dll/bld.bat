%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\bin\hdf5.dll -o%LIBRARY_BIN%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\bin\hdf5_hl.dll -o%LIBRARY_BIN%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\bin\zlib.dll -o%LIBRARY_BIN%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\bin\szip.dll -o%LIBRARY_BIN%

%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\lib\hdf5.lib -o%LIBRARY_LIB%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\lib\hdf5_hl.lib -o%LIBRARY_LIB%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\lib\zlib.lib -o%LIBRARY_LIB%
%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\lib\szip.lib -o%LIBRARY_LIB%

%LIBRARY_BIN%\7z.exe e %SRC_DIR%\HDF5-1.8.13-win%ARCH%.exe $_OUTDIR\include\*.h -o%LIBRARY_INC%
