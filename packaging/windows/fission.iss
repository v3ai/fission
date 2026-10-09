; Inno Setup script for Fission (built by build_windows.ps1, which passes AppVersion / SourceDir / OutDir / Assets)
#ifndef AppVersion
  #define AppVersion "0.6"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Fission"
#endif
#ifndef OutDir
  #define OutDir "..\..\dist"
#endif
#ifndef Assets
  #define Assets "..\assets"
#endif

[Setup]
AppId={{6F1D2C9E-3B7A-4E52-9C1F-F15510A0C0DE}
AppName=Fission
AppVersion={#AppVersion}
AppVerName=Fission {#AppVersion}
AppPublisher=Fission
DefaultDirName={autopf}\Fission
DefaultGroupName=Fission
DisableProgramGroupPage=yes
; installs for the current user without admin rights, or for everyone if the user chooses (and has admin)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutDir}
OutputBaseFilename=Fission-{#AppVersion}-Windows-x64-Setup
SetupIconFile={#Assets}\fission.ico
UninstallDisplayIcon={app}\Fission.exe
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assoc"; Description: "Open .fission design files with Fission"; GroupDescription: "File types:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Assets}\fission-file.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Fission"; Filename: "{app}\Fission.exe"
Name: "{autodesktop}\Fission"; Filename: "{app}\Fission.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.fission"; ValueType: string; ValueName: ""; ValueData: "Fission.Design"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Fission.Design"; ValueType: string; ValueName: ""; ValueData: "Fission design"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Fission.Design\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\fission-file.ico"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Fission.Design\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Fission.exe"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\Fission.exe"; Description: "{cm:LaunchProgram,Fission}"; Flags: nowait postinstall skipifsilent
