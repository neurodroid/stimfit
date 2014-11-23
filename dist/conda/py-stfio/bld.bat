:: Need to enabled delayed variable expansion so that variables are evaluated when each line is executed, rather 
:: than when this file is initially parsed. Otherwise, SDKPLATFORM and PLATFORM will not be defined when evaluated
setlocal EnableDelayedExpansion

:: Generate the arch-specific arguments for SetEnv.cmd and msbuild
if %ARCH% equ 64 (set SDKPLATFORM=x64) else (set SDKPLATFORM=x86)
if %ARCH% equ 64 (set PLATFORM=x64) else (set PLATFORM=win32)

:: Run the Windows SDK SetEnv.cmd to setup the appropriate paths, etc. to build the VC solution
call "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.Cmd" /Release /%SDKPLATFORM%

:: Build the Release configuration of the pystfio project
msbuild %SRC_DIR%\dist\windows\VS2008\Stimfit\Stimfit.sln /t:pystfio /p:Configuration=Release /p:Platform=%PLATFORM%

:: Create the stfio site-packages directory and copy over the relevant files. The current conda practice is to put
:: the required DLLs in the same location, so these are copied too. 
mkdir %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\__init__.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio_plot.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio_neo.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio.py %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Release\_stfio.pyd %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Release\libstfio.dll %SP_DIR%\stfio

:: The hdf5-dll and biosig conda packages put their DLLs in %LIBRARY_BIN%. Since pystfio depends on these, need
:: to copy them too
copy %LIBRARY_BIN%\hdf5.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\hdf5_hl.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\zlib.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\szip.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\libbiosig2.dll %SP_DIR%\stfio

