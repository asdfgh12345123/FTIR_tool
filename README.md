# FTIR Spectrum Plot Tool

A desktop FTIR plotting application for scientific use.

## Project Structure

```text
FTIR_APP/
|-- ftir_core.py
|-- ftir_gui.py
|-- output/
`-- README.md
```

## Features

- Read FTIR txt files (2 columns: Wavenumber, value)
- Auto-detect `##YUNITS=%T` / `##YUNITS=Abs`
- Auto-convert Abs to %T: `%T = 10 ** (-Abs) * 100`
- Savitzky-Golay smoothing
- Baseline correction
- Single spectrum plotting with arrow peak labels
- Multi stacked spectra plotting
- Red/blue short-line peak markers
- PNG / TIFF / CSV export
- 600 dpi output
- Paper-style figure formatting

## Install Dependencies

```bash
pip install numpy pandas matplotlib scipy
```

## Run GUI

```bash
python ftir_gui.py
```

## Input Data Format

Example txt content:

```text
##YUNITS=%T
4000 78.1
3998 78.2
...
```

or

```text
##YUNITS=Abs
4000 0.312
3998 0.315
...
```

## GUI Usage

### Single Spectrum

1. Click **Select FTIR File**.
2. Input target peaks (e.g. `3431,2921,1346,754`).
3. Click **Generate Single Spectrum**.

### Multi Spectrum

1. Click **Select Multiple FTIR Files**.
2. Input sample names (e.g. `MEL,MEL-Si`).
3. Input vertical offsets (e.g. `18,0`, or keep empty for auto).
4. Input peak spec for each sample, one line per sample:

```text
sample1:red=2997|1433|997|463;blue=1651|1043|650
sample2:red=2961|1385;blue=3445|789
```

5. Click **Generate Multi Spectrum**.

All outputs are saved in `output/`.

## Build EXE (Windows)

Install PyInstaller:

```bash
pip install pyinstaller
```

Build command:

```bash
pyinstaller --onefile --windowed ftir_gui.py
```

Optional (recommended for matplotlib reliability):

```bash
pyinstaller --onefile --windowed --collect-all matplotlib ftir_gui.py
```
