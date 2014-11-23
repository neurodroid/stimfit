setlocal EnableDelayedExpansion
if %ARCH% equ 64 (set SDKPLATFORM=x64) else (set SDKPLATFORM=x86)
call "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.Cmd" /Release /%SDKPLATFORM%
if %ARCH% equ 64 (set PLATFORM=x64) else (set PLATFORM=win32)

cd %SRC_DIR%\lib
lib /def:libbiosig2.def

copy %SRC_DIR%\lib\libbiosig2.lib %LIBRARY_LIB%\
copy %SRC_DIR%\lib\libbiosig2.dll %LIBRARY_BIN%\
xcopy %SRC_DIR%\include\*.h %LIBRARY_INC%\
