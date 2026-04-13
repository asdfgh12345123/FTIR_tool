# Release Guide

## Release tag example

Use the next semantic version tag when publishing:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Replace `vX.Y.Z` with the version you want to publish, such as `v1.0.4`, `v1.0.5`, or `v1.1.0`.
Check the latest published version here:
[Releases](https://github.com/asdfgh12345123/FTIR_tool/releases)

After the tag is pushed:

- GitHub Actions will run the `Release` workflow
- A Windows EXE will be built automatically
- `FTIR_Tool.exe` and `FTIR_Tool_Windows_Portable.zip` will be attached to the GitHub Release

Recommended user-facing guidance:

- Ask Windows users to download `FTIR_Tool_Windows_Portable.zip`
- Tell users not to download `Source code (zip)` unless they want the source code
- After extracting the zip, users should double-click `Run_FTIR_Tool.bat`

## What should go into a release

Recommended notes:

- GUI FTIR plotting tool for Windows
- Supports single spectrum and stacked multi-spectrum plotting
- Exports PNG / TIFF / CSV
- Supports Abs and %T detection
- Suitable for paper-style FTIR figures

## Versioning suggestion

- `v1.0.0`: first public usable release
- `v1.0.1`: bug fix only
- `v1.0.5`: another patch release example
- `v1.1.0`: new features without breaking old behavior
- `v2.0.0`: breaking changes
