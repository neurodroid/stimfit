%LIBRARY_BIN%\7za.exe e %SRC_DIR%\7z920.msi _7z.exe -o%LIBRARY_BIN%
%LIBRARY_BIN%\7za.exe e %SRC_DIR%\7z920.msi _7zip.dll -o%LIBRARY_BIN%

ren %LIBRARY_BIN%\_7z.exe 7z.exe
ren %LIBRARY_BIN%\_7zip.dll 7zip.dll

