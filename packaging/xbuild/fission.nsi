; Fission installer (NSIS 3). Built by xbuild.py:  makensis -DVERSION=0.6 -DSRC=<app folder> -DOUT=<setup.exe> -DASSETS=<icons> fission.nsi
Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"

!ifndef VERSION
  !define VERSION "0.0"
!endif
Name "Fission ${VERSION}"
OutFile "${OUT}"
InstallDir "$LOCALAPPDATA\Programs\Fission"
InstallDirRegKey HKCU "Software\Fission" "InstallDir"
RequestExecutionLevel user          ; per-user install: no admin prompt
SetCompressor /SOLID lzma
SetCompressorDictSize 64
BrandingText "Fission ${VERSION}"
VIProductVersion "${VERSION}.0.0"
VIAddVersionKey "ProductName" "Fission"
VIAddVersionKey "FileDescription" "Fission ${VERSION} installer"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "CompanyName" "Fission"
VIAddVersionKey "LegalCopyright" "Fission"

!define MUI_ICON "${ASSETS}\fission.ico"
!define MUI_UNICON "${ASSETS}\fission.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\Fission.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start Fission"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

!define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\Fission"

Section "Fission (required)" SecMain
  SectionIn RO
  ; replace an older version cleanly
  RMDir /r "$INSTDIR\runtime"
  RMDir /r "$INSTDIR\app"
  SetOutPath "$INSTDIR"
  File /r "${SRC}\*"
  WriteUninstaller "$INSTDIR\Uninstall Fission.exe"
  CreateShortCut "$SMPROGRAMS\Fission.lnk" "$INSTDIR\Fission.exe" "" "$INSTDIR\Fission.exe" 0
  WriteRegStr HKCU "Software\Fission" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST}" "DisplayName" "Fission"
  WriteRegStr HKCU "${UNINST}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST}" "Publisher" "Fission"
  WriteRegStr HKCU "${UNINST}" "DisplayIcon" "$INSTDIR\Fission.exe"
  WriteRegStr HKCU "${UNINST}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST}" "UninstallString" '"$INSTDIR\Uninstall Fission.exe"'
  WriteRegStr HKCU "${UNINST}" "QuietUninstallString" '"$INSTDIR\Uninstall Fission.exe" /S'
  WriteRegDWORD HKCU "${UNINST}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD HKCU "${UNINST}" "EstimatedSize" $0
SectionEnd

Section "Desktop shortcut" SecDesktop
  CreateShortCut "$DESKTOP\Fission.lnk" "$INSTDIR\Fission.exe" "" "$INSTDIR\Fission.exe" 0
SectionEnd

Section "Open .fission files with Fission" SecAssoc
  WriteRegStr HKCU "Software\Classes\.fission" "" "Fission.Design"
  WriteRegStr HKCU "Software\Classes\Fission.Design" "" "Fission design"
  WriteRegStr HKCU "Software\Classes\Fission.Design\DefaultIcon" "" "$INSTDIR\app\fission-file.ico"
  WriteRegStr HKCU "Software\Classes\Fission.Design\shell\open\command" "" '"$INSTDIR\Fission.exe" "%1"'
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd

LangString DESC_Main ${LANG_ENGLISH} "The Fission app and its own Python, Qt and OpenCascade."
LangString DESC_Desktop ${LANG_ENGLISH} "Put a Fission icon on the desktop."
LangString DESC_Assoc ${LANG_ENGLISH} "Double-click a .fission design to open it in Fission."
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} $(DESC_Main)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_Desktop)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecAssoc} $(DESC_Assoc)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  Delete "$SMPROGRAMS\Fission.lnk"
  Delete "$DESKTOP\Fission.lnk"
  RMDir /r "$INSTDIR\runtime"
  RMDir /r "$INSTDIR\app"
  Delete "$INSTDIR\Fission.exe"
  Delete "$INSTDIR\Uninstall Fission.exe"
  RMDir "$INSTDIR"
  ReadRegStr $0 HKCU "Software\Classes\Fission.Design\shell\open\command" ""
  StrCmp $0 "" +3
  DeleteRegKey HKCU "Software\Classes\Fission.Design"
  DeleteRegValue HKCU "Software\Classes\.fission" ""
  DeleteRegKey HKCU "${UNINST}"
  DeleteRegKey HKCU "Software\Fission"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd
