# Windows build (exe + installer)

Builds a 64-bit `hcft.exe` with [PyInstaller](https://pyinstaller.org) and a
setup wizard with [Inno Setup](https://jrsoftware.org/isinfo.php).

## One-time setup

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Inno Setup 6: `winget install --id JRSoftware.InnoSetup -e`

## Build

Double-click `build.bat`, or from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

Results (takes about 5 minutes):

- `dist\hcft\hcft.exe`: the app folder, runs without installing (portable)
- `dist\hcft-<version>-win64-setup.exe`: the installer

Add `-SkipInstaller` to build only the exe folder.

## Releasing a new version

1. Update `hcft/version.py`. The exe, installer name and installer
   metadata all take the version from there.
2. Run the build and test the installer.
3. Upload the installer to a GitHub release.

## Files

| File | Purpose |
|---|---|
| `build.ps1` / `build.bat` | Runs all steps below |
| `make_icon.py` | Renders `hcft/resources/hcft_icon.svg` to `.ico`, `.png` and the wizard images |
| `hcft.spec` | PyInstaller configuration |
| `hcft_launcher.py` | Entry script of the exe |
| `rthook_ets_toolkit.py` | Selects the Qt toolkit at runtime |
| `hcft_installer.iss` | Inno Setup script |

The build uses its own environment (`build\windows\venv`), created exactly
from `uv.lock`, so builds are reproducible and your development `.venv` is not
changed. To upgrade the bundled libraries, run `uv lock --upgrade` first.

To change the icon, edit `hcft/resources/hcft_icon.svg` (e.g. in Inkscape);
the build regenerates the `.ico` and `.png` files.

## Troubleshooting

The exe has no console, so startup errors appear in a dialog. If a new
dependency version breaks the build with `No module named ...`, add the module
(or its package via `collect_submodules`) to `hiddenimports` in `hcft.spec`.
