; Inno Setup script for the HCFT Windows 64-bit installer.
;
; Normally compiled by build.ps1, which passes the version from hcft/version.py:
;   ISCC.exe /DMyAppVersion=1.1.0 packaging\windows\hcft_installer.iss
; Expects the PyInstaller output in dist\hcft and the wizard images generated
; by make_icon.py in build\windows. The installer is written to dist\.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "High-Cycle Fatigue Tool"
#define MyAppShortName "HCFT"
#define MyAppPublisher "RWTH Aachen University - Institute of Structural Concrete"
#define MyAppURL "https://github.com/bmcs-group/high_cycle_fatigue_tool"
#define MyAppExeName "hcft.exe"
#define RootDir "..\.."

[Setup]
; AppId identifies the app for upgrades and uninstall. Never change it.
AppId={{B945D6BF-9F5C-4217-94B7-9030AAD4A4CD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
VersionInfoVersion={#MyAppVersion}
VersionInfoProductName={#MyAppName}

; 64-bit only
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Install for the current user without admin rights by default; the user can
; still choose "install for all users" in the dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

LicenseFile={#RootDir}\LICENSE
SetupIconFile={#RootDir}\hcft\resources\hcft_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
WizardStyle=modern
WizardSmallImageFile={#RootDir}\build\windows\wizard_small_55.bmp,{#RootDir}\build\windows\wizard_small_69.bmp,{#RootDir}\build\windows\wizard_small_83.bmp,{#RootDir}\build\windows\wizard_small_110.bmp,{#RootDir}\build\windows\wizard_small_138.bmp

OutputDir={#RootDir}\dist
OutputBaseFilename=hcft-{#MyAppVersion}-win64-setup
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Remove the libraries of a previous version, so no outdated files are left behind
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#RootDir}\dist\hcft\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
