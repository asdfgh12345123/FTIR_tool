# Release Guide

## First release recommendation

Suggested first tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```

After the tag is pushed:

- GitHub Actions will run the `Release` workflow
- A Windows EXE will be built automatically
- `FTIR 光谱绘图工具.exe` and `FTIR_Tool_Windows_请下载这个压缩包.zip` will be attached to the GitHub Release

Recommended user-facing guidance:

- Ask Windows users to download `FTIR_Tool_Windows_请下载这个压缩包.zip`
- Tell users not to download `Source code (zip)` unless they want the source code
- After extracting the zip, users should double-click `打开 FTIR 光谱绘图工具.bat`

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
