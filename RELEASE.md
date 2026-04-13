# Release Guide

## First release recommendation

Suggested first tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

After the tag is pushed:

- GitHub Actions will run the `Release` workflow
- A Windows EXE will be built automatically
- `FTIR_Tool.exe` and `FTIR_Tool_Windows.zip` will be attached to the GitHub Release

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
- `v1.1.0`: new features without breaking old behavior
- `v2.0.0`: breaking changes
