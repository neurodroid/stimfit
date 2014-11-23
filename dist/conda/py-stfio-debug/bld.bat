:: Need to enabled delayed variable expansion so that variables are evaluated when each line is executed, rather 
:: than when this file is initially parsed. Otherwise, SDKPLATFORM and PLATFORM will not be defined when evaluated
setlocal EnableDelayedExpansion

:: Generate the arch-specific arguments for SetEnv.cmd and msbuild

if %ARCH% equ 64 (set SDKPLATFORM=x64) else (set SDKPLATFORM=x86)
if %ARCH% equ 64 (set PLATFORM=x64) else (set PLATFORM=win32)

:: Run the Windows SDK SetEnv.cmd to setup the appropriate paths, etc. to build the VC solution
call "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.Cmd" /Debug /%SDKPLATFORM%

:: When building in debug mode, pyconfig.h will insist on linking to the debug version of the python library
:: (python27_d.lib), which is not included in most distributions. This patch makes it use the release version
:: instead. Note that this only affects debugging info for the core python code. You will still be able to step
:: through the stfio code line by line
::
:: This obviously depends on "patch" being available. I believe it should be since conda needs to be able to apply
:: patches.
patch %PREFIX%\include\pyconfig.h %RECIPE_DIR%\pyconfig.patch

:: Build the Debug configuration of the pystfio project
msbuild %SRC_DIR%\dist\windows\VS2008\Stimfit\Stimfit.sln /t:pystfio /p:Configuration=Debug /p:Platform=%PLATFORM%

:: Reverse the pyconfig.h patch now that the build is done
patch -R %PREFIX%\include\pyconfig.h %RECIPE_DIR%\pyconfig.patch

:: Create the stfio site-packages directory and copy over the relevant files. The current conda practice is to put
:: the required DLLs in the same location, so these are copied too. 
mkdir %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\__init__.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio_plot.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio_neo.py %SP_DIR%\stfio
copy %SRC_DIR%\src\pystfio\stfio.py %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Debug\_stfio.pyd %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Debug\_stfio.pdb %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Debug\libstfio.dll %SP_DIR%\stfio
copy %SRC_DIR%\dist\windows\VS2008\Stimfit\%PLATFORM%\Debug\libstfio.pdb %SP_DIR%\stfio

:: The hdf5-dll and biosig conda packages put their DLLs in %LIBRARY_BIN%. Since pystfio depends on these, need
:: to copy them too
copy %LIBRARY_BIN%\hdf5.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\hdf5_hl.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\zlib.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\szip.dll %SP_DIR%\stfio
copy %LIBRARY_BIN%\libbiosig2.dll %SP_DIR%\stfio
