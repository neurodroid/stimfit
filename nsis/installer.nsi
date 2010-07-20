; installer.nsi
;
; NSI script for stimfit 

; This may slightly reduce the executable size, but compression is slower.
SetCompressor lzma

;--------------------------------
; Use modern interface
!include MUI2.nsh

;--------------------------------

!define PRODUCT_VERSION "0.9.3rc2"
!define EXE_NAME "stimfit"
!define REG_NAME "Stimfit 0.9"
!define PRODUCT_PUBLISHER "Christoph Schmidt-Hieber"
!define PRODUCT_WEB_SITE "http://www.stimfit.org"
!define STFDIR "D:\Users\cs\stimfit"
!define MSIDIR "D:\Users\cs\Downloads"
!define WXWDIR "D:\Users\cs\wxWidgets"
!define FFTDIR "D:\Users\cs\fftw"
!define WXPDIR "D:\Users\cs\wxPython\final.build\Python26\Lib\site-packages"
!define PRODIR "C:\Programme"
!define FULL_WELCOME "This wizard will guide you through the installation \
of ${REG_NAME} and wxPython. You can optionally \
install Python 2.6.4 and NumPy 1.4.0 \
if you don't have them on your machine."
!define UPDATE_WELCOME "This wizard will update an existing installation \
of ${REG_NAME} to version ${PRODUCT_VERSION}."
; The name of the installer
Name "${REG_NAME}"


; The file to write
!ifdef UPDATE
OutFile "${EXE_NAME}-${PRODUCT_VERSION}-update.exe"
!else
OutFile "${EXE_NAME}-${PRODUCT_VERSION}-full.exe"
!endif

; The default installation directory
InstallDir "$PROGRAMFILES\${REG_NAME}"

; Request application privileges for Windows Vista
RequestExecutionLevel admin

!define STRING_PYTHON_NOT_FOUND "Python 2.6 is not installed on this system. \
$\nPlease install Python first. \
$\nClick OK to cancel installation and remove installation files."

;--------------------------------
;Variables

Var StartMenuFolder
Var StrNoUsablePythonFound

;--------------------------------

; Pages
!ifdef UPDATE
!define MUI_WELCOMEPAGE_TEXT "${UPDATE_WELCOME}"
!else
!define MUI_WELCOMEPAGE_TEXT "${FULL_WELCOME}"
!endif
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${STFDIR}\gpl-2.0.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

;Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU" 
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\${REG_NAME}" 
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"
  
!insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
;Languages
 
!insertmacro MUI_LANGUAGE "English"

;--------------------------------

; The stuff to install
!ifndef UPDATE
Section "Python 2.6.4" 0

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR

  ; Put installer into installation dir temporarily
  File "${MSIDIR}\python-2.6.4.msi"

  ExecWait '"Msiexec.exe" /i "$INSTDIR\python-2.6.4.msi"'
  
  ; Delete installer once we are done
  Delete "$INSTDIR\python-2.6.4.msi"

SectionEnd

Section "NumPy 1.4.0" 1

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR

  ; Put installer into installation dir temporarily
  File "${MSIDIR}\numpy-1.4.0-win32-superpack-python2.6.exe"

  ExecWait '"$INSTDIR\numpy-1.4.0-win32-superpack-python2.6.exe"'
  
  ; Delete installer once we are done
  Delete "$INSTDIR\numpy-1.4.0-win32-superpack-python2.6.exe"

SectionEnd
!endif

Section "!Program files and wxPython" 2 ; Core program files and wxPython

  ;This section is required : readonly mode
  SectionIn RO
   
  ; Create default error message
  StrCpy $StrNoUsablePythonFound "${STRING_PYTHON_NOT_FOUND}"

  ClearErrors
  ReadRegStr $9 HKEY_LOCAL_MACHINE "SOFTWARE\Python\PythonCore\2.6\InstallPath" ""
    
  IfErrors 0 +3
    MessageBox MB_OK "$StrNoUsablePythonFound"
    Quit    

  ClearErrors
  DetailPrint "Found a Python 2.6 installation at '$9'"
  
  ; Add a path to the installation directory in the python site-packages folder
  FileOpen $0 $9\Lib\site-packages\stimfit.pth w
  FileWrite $0 "$INSTDIR"
  FileClose $0
  
  IfErrors 0 +3
    MessageBox MB_OK "Couldn't create path for python module"
    Quit    

  ClearErrors
  
  ; Set output path to the installation directory.
  SetOutPath $INSTDIR
  
!ifndef UPDATE
  File "${FFTDIR}\dll\libfftw3-3.dll"
  File "${WXWDIR}\lib\vc_dll\wxmsw290u_core_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxbase290u_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxmsw290u_aui_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxmsw290u_adv_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxbase290u_net_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxmsw290u_html_vc_custom.dll"
  File "${WXWDIR}\lib\vc_dll\wxmsw290u_stc_vc_custom.dll"
  File "${PRODIR}\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\msvcp90.dll"
  File "${PRODIR}\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\msvcr90.dll"
  File /r "${WXPDIR}\wx*"
!endif
  File "${STFDIR}\stimfit_VS03\stimfit_exe\Release\${EXE_NAME}.exe"
  File "${STFDIR}\stimfit_VS03\stimfit_exe\Release\stimfit.dll"
  File "${STFDIR}\stimfit_VS03\stfswig\Release\_stf.pyd"
  File "${STFDIR}\src\stfswig\stf.py"
  File "${STFDIR}\src\stfswig\ivtools.py"
  File "${STFDIR}\src\stfswig\mintools.py"
  File "${STFDIR}\src\stfswig\natools.py"
;  File "${STFDIR}\src\stfswig\plottools.py"
  File "${STFDIR}\src\stfswig\minidemo.py"
  File "${STFDIR}\src\stfswig\charlie.py"
  File "${STFDIR}\src\stfswig\hdf5tools.py"
  File "${STFDIR}\src\stfswig\spells.py"
  File "${STFDIR}\src\stfswig\embedded_init.py"
  File "${STFDIR}\src\stfswig\embedded_stf.py"
  File "${STFDIR}\src\stfswig\embedded_ipython.py"
  File /r /x .hg "${STFDIR}\src"
  
  ;Store installation folder
  WriteRegStr HKCU "Software\${REG_NAME}" "" $INSTDIR 
  WriteRegStr HKCU "Software\${REG_NAME}" "InstallLocation" $INSTDIR 
  WriteRegExpandStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegExpandStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "DisplayName" "${REG_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "DisplayIcon" "$INSTDIR\${EXE_NAME}.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"

  ; Associate files to amaya
  WriteRegStr HKCR "${REG_NAME}" "" "${REG_NAME} Files"
  WriteRegStr HKCR "${REG_NAME}\DefaultIcon" "" "$INSTDIR\${EXE_NAME}.exe"
  WriteRegStr HKCR "${REG_NAME}\shell" "" "open"
  WriteRegStr HKCR "${REG_NAME}\shell\open\command" "" '"$INSTDIR\${EXE_NAME}.exe" "/d=$INSTDIR" "%1"'

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Install for all users
  SetShellVarContext all

  ;Start Menu
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application   

    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\${REG_NAME}.lnk" "$INSTDIR\${EXE_NAME}.exe"
 
 !insertmacro MUI_STARTMENU_WRITE_END

  ; Create desktop link
  CreateShortCut "$DESKTOP\${REG_NAME}.lnk" "$INSTDIR\${EXE_NAME}.exe"
 
SectionEnd ; end the section

Section "Uninstall"

  SetDetailsPrint textonly
  DetailPrint "Deleting program files..."
  SetDetailsPrint listonly

  ;Uninstall Stimfit for all users
  SetShellVarContext all
  
  ReadRegStr $StartMenuFolder HKCU "Software\${REG_NAME}" "Start Menu Folder"
  IfFileExists "$SMPROGRAMS\$StartMenuFolder\${EXE_NAME}.lnk" stimfit_smp_installed
    Goto stimfit_smp_notinstalled
  stimfit_smp_installed:
  Delete "$SMPROGRAMS\$StartMenuFolder\${EXE_NAME}.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk"
  RMDir "$SMPROGRAMS\$StartMenuFolder"
  stimfit_smp_notinstalled:

  RMDir /r "$INSTDIR"
  Delete "$DESKTOP\${EXE_NAME}.lnk"

  SetDetailsPrint textonly
  DetailPrint "Deleting registry keys..."
  SetDetailsPrint listonly

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${REG_NAME}"
  DeleteRegKey HKLM "Software\${REG_NAME}"
  DeleteRegKey HKCR "${REG_NAME}"
  DeleteRegKey HKCU "Software\${REG_NAME}"

  ; uninstall files associations
  ; --> .dat
  ReadRegStr $R0 HKCR ".dat" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".dat" "AM_OLD_VALUE"
    WriteRegStr HKCR ".dat" "" $R0
	
  ; --> .cfs
  ReadRegStr $R0 HKCR ".cfs" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".cfs" "AM_OLD_VALUE"
    WriteRegStr HKCR ".cfs" "" $R0

  ; --> .h5
  ReadRegStr $R0 HKCR ".h5" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".h5" "AM_OLD_VALUE"
    WriteRegStr HKCR ".h5" "" $R0

  ; --> .axgd
  ReadRegStr $R0 HKCR ".axgd" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".axgd" "AM_OLD_VALUE"
    WriteRegStr HKCR ".axgd" "" $R0
	
  ; --> .axgx
  ReadRegStr $R0 HKCR ".axgx" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".axgx" "AM_OLD_VALUE"
    WriteRegStr HKCR ".axgx" "" $R0

  ; --> .abf
  ReadRegStr $R0 HKCR ".abf" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".abf" "AM_OLD_VALUE"
    WriteRegStr HKCR ".abf" "" $R0
	
  ; --> .atf
  ReadRegStr $R0 HKCR ".atf" ""
  StrCmp $R0 "${REG_NAME}" 0 +3
    ReadRegStr $R0 HKCR ".atf" "AM_OLD_VALUE"
    WriteRegStr HKCR ".atf" "" $R0

  SetDetailsPrint textonly
  DetailPrint "Successfully uninstalled stimfit"
  SetDetailsPrint listonly
	
SectionEnd

; File associations
SubSection "File associations" SecFileAss

; --> .dat
Section ".dat (CED filing system)" SecAssDAT
  ReadRegStr $R0 HKCR ".dat" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".dat" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".dat" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .cfs
Section ".cfs (CED filing system)" SecAssCFS
  ReadRegStr $R0 HKCR ".cfs" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".cfs" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".cfs" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .h5
Section ".h5 (HDF5)" SecAssH5
  ReadRegStr $R0 HKCR ".h5" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".h5" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".h5" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .axgd
Section ".axgd (Axograph digitized)" SecAssAxgd
  ReadRegStr $R0 HKCR ".axgd" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".axgd" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".axgd" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .axgx
Section ".axgx (Axograph X)" SecAssAxgx
  ReadRegStr $R0 HKCR ".axgx" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".axgx" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".axgx" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .abf
Section ".abf (Axon binary file)" SecAssABF
  ReadRegStr $R0 HKCR ".abf" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".abf" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".abf" "" "${REG_NAME}"
  already_stf:
SectionEnd

; --> .atf
Section ".atf (Axon text file)" SecAssATF
  ReadRegStr $R0 HKCR ".atf" ""
  StrCmp $R0 "${REG_NAME}" already_stf no_stf
  no_stf:
    WriteRegStr HKCR ".atf" "AM_OLD_VALUE" $R0
  WriteRegStr HKCR ".atf" "" "${REG_NAME}"
  already_stf:
SectionEnd

SubSectionEnd

;--------------------------------
;Descriptions

  ;Assign descriptions to sections
!ifndef UPDATE
  !insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT 0 "Python 2.6 is required to run stimfit. Unselect this if it's already installed on your system."
    !insertmacro MUI_DESCRIPTION_TEXT 1 "NumPy is required for efficient numeric computations in python. Unselect this if you already have NumPy on your system."
    !insertmacro MUI_DESCRIPTION_TEXT 2 "The core program files and wxPython 2.9 (mandatory)."
    !insertmacro MUI_DESCRIPTION_TEXT 3 "Selects Stimfit as the default application for files of these types."
  !insertmacro MUI_FUNCTION_DESCRIPTION_END
!endif
