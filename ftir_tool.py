from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.signal import savgol_filter as scipy_savgol_filter
except ImportError:
    scipy_savgol_filter = None


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.color": "#8f8f8f",
            "grid.alpha": 0.65,
            "savefig.dpi": 600,
        }
    )


def _adjust_window_length(n_points: int, desired: int, polyorder: int) -> int:
    if n_points < 3:
        return 0

    window = int(max(3, desired))
    if window % 2 == 0:
        window += 1

    max_window = n_points if n_points % 2 == 1 else n_points - 1
    if max_window < 3:
        return 0
    if window > max_window:
        window = max_window

    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1

    if window < min_window:
        window = min_window
        if window > max_window:
            window = max_window

    if window % 2 == 0:
        window -= 1

    return window if window >= 3 else 0


def _savgol_filter_numpy(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=float)
    vandermonde = np.vander(x, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(vandermonde)[0]
    padded = np.pad(y, (half, half), mode="reflect")
    return np.convolve(padded, coeffs[::-1], mode="valid")


def _detect_yunits(file_path: Path) -> str:
    detected_raw = ""
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.upper().startswith("##YUNITS"):
                if "=" in line_strip:
                    detected_raw = line_strip.split("=", 1)[1].strip()
                else:
                    detected_raw = line_strip.replace("##YUNITS", "").replace(":", "").strip()
                break

    normalized = detected_raw.upper().replace(" ", "")
    if normalized in {"%T", "%TRANSMITTANCE", "TRANSMITTANCE", "TRANSMITTANCE(%)", "T"}:
        print(f"File {file_path.name} detected unit: %T")
        return "%T"
    if normalized in {"ABS", "ABSORBANCE", "A"}:
        print(f"File {file_path.name} detected unit: Abs -> converted to %T")
        return "Abs"
    if detected_raw:
        print(
            f"Warning: File {file_path.name} has unrecognized unit '{detected_raw}', defaulting to %T"
        )
    else:
        print(f"Warning: File {file_path.name} missing ##YUNITS, defaulting to %T")
    return "%T"


def _auto_curve_colors(n_curves: int) -> List[str]:
    palette = ["black", "darkred", "navy", "darkgreen", "purple", "brown"]
    return [palette[i % len(palette)] for i in range(n_curves)]


def _resolve_input_file(file_path: Union[str, Path], search_roots: Sequence[Path]) -> Optional[Path]:
    p = Path(file_path)
    if p.is_absolute():
        return p if p.exists() else None
    if p.exists():
        return p

    for root in search_roots:
        candidate = root / p
        if candidate.exists():
            return candidate

    filename = p.name
    matches: List[Path] = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_dir_path(dir_value: Union[str, Path], base_dir: Path) -> Path:
    dir_path = Path(dir_value)
    if dir_path.is_absolute():
        return dir_path
    return (base_dir / dir_path).resolve()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("ftir_config.json must contain a JSON object at top level.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FTIR plotting tool for single and stacked spectra."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to project ftir_config.json",
    )
    return parser.parse_args()


def read_ftir_txt(file_path: Union[str, Path]) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FTIR txt file not found: {file_path}")
    detected_unit = _detect_yunits(file_path)

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    if df.shape[1] < 2:
        raise ValueError(f"File does not contain at least two columns: {file_path}")

    df = df.iloc[:, :2].copy()
    df.columns = ["Wavenumber", "Transmittance"]
    df["Wavenumber"] = pd.to_numeric(df["Wavenumber"], errors="coerce")
    df["Transmittance"] = pd.to_numeric(df["Transmittance"], errors="coerce")
    df = df.dropna().drop_duplicates(subset="Wavenumber")

    if df.empty:
        raise ValueError(f"No valid numeric FTIR data found in: {file_path}")

    if detected_unit == "Abs":
        df["Transmittance"] = np.power(10.0, -df["Transmittance"].to_numpy(dtype=float)) * 100.0

    df = df.sort_values("Wavenumber").reset_index(drop=True)
    return df


def smooth_signal(y: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()

    wl = _adjust_window_length(y.size, window_length, polyorder)
    if wl < 3:
        return y.copy()

    po = min(int(polyorder), wl - 1)
    if po < 1:
        po = 1

    if scipy_savgol_filter is not None:
        return scipy_savgol_filter(y, window_length=wl, polyorder=po, mode="interp")
    return _savgol_filter_numpy(y, window_length=wl, polyorder=po)


def baseline_correction(
    y: np.ndarray,
    window_length: int = 301,
    quantile: float = 0.10,
    smooth_window: int = 151,
    polyorder: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y.copy(), np.zeros_like(y)

    q = float(np.clip(quantile, 0.0, 1.0))
    rolling_window = _adjust_window_length(y.size, window_length, 1)
    if rolling_window < 3:
        rolling_window = 3

    baseline_raw = (
        pd.Series(y)
        .rolling(window=rolling_window, center=True, min_periods=1)
        .quantile(q)
        .to_numpy()
    )
    baseline = smooth_signal(baseline_raw, window_length=smooth_window, polyorder=polyorder)

    corrected = y - baseline + float(np.nanmedian(baseline))
    return corrected, baseline


def find_peak_near_target(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
    search_half_width: float = 40.0,
) -> Optional[Tuple[float, float, int]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None

    half = abs(float(search_half_width))
    left = float(target) - half
    right = float(target) + half

    idx_candidates = np.where((x >= left) & (x <= right))[0]
    if idx_candidates.size == 0:
        idx = int(np.argmin(np.abs(x - float(target))))
        return float(x[idx]), float(y[idx]), idx

    local_idx = idx_candidates[np.argmin(y[idx_candidates])]
    idx = int(local_idx)
    return float(x[idx]), float(y[idx]), idx


def _normalize_peak_spec(peak_spec: Any, default_color: str = "red") -> List[Tuple[float, str]]:
    normalized: List[Tuple[float, str]] = []

    if peak_spec is None:
        return normalized

    if isinstance(peak_spec, dict):
        for color, peaks in peak_spec.items():
            color_name = str(color)
            if isinstance(peaks, (int, float, np.number)):
                normalized.append((float(peaks), color_name))
            else:
                for peak in peaks:
                    normalized.append((float(peak), color_name))
        return normalized

    for item in peak_spec:
        if isinstance(item, (int, float, np.number)):
            normalized.append((float(item), default_color))
            continue

        if isinstance(item, dict):
            peak_val = item.get("target", item.get("wavenumber"))
            if peak_val is None:
                continue
            color_name = str(item.get("color", default_color))
            normalized.append((float(peak_val), color_name))
            continue

        if isinstance(item, (tuple, list)):
            if len(item) == 1:
                normalized.append((float(item[0]), default_color))
            elif len(item) >= 2:
                normalized.append((float(item[0]), str(item[1])))

    return normalized


def export_peak_table(
    peak_records: Sequence[Dict[str, Union[str, int, float]]],
    output_csv: Union[str, Path],
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if peak_records:
        peak_df = pd.DataFrame(peak_records)
    else:
        peak_df = pd.DataFrame(columns=["Sample", "Peak_No", "Wavenumber_cm-1"])

    for col in ["Sample", "Peak_No", "Wavenumber_cm-1"]:
        if col not in peak_df.columns:
            peak_df[col] = np.nan

    peak_df = peak_df[["Sample", "Peak_No", "Wavenumber_cm-1"]]
    peak_df.to_csv(output_csv, index=False)
    return peak_df


def plot_single_ftir(
    file_path: Union[str, Path],
    output_dir: Union[str, Path] = "output",
    target_peaks: Sequence[float] = (3431, 2921, 1346, 754),
    search_half_width: float = 40.0,
    smooth_window_length: int = 25,
    smooth_polyorder: int = 3,
    baseline_window_length: int = 301,
    baseline_quantile: float = 0.10,
    baseline_smooth_window: int = 151,
    baseline_polyorder: int = 2,
    figsize: Tuple[float, float] = (7.4, 5.6),
    output_name: Optional[str] = None,
) -> pd.DataFrame:
    _apply_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_ftir_txt(file_path)
    x = df["Wavenumber"].to_numpy()
    y = df["Transmittance"].to_numpy()

    y_smooth = smooth_signal(y, window_length=smooth_window_length, polyorder=smooth_polyorder)
    y_corrected, _ = baseline_correction(
        y_smooth,
        window_length=baseline_window_length,
        quantile=baseline_quantile,
        smooth_window=baseline_smooth_window,
        polyorder=baseline_polyorder,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y_corrected, color="#e41a1c", linewidth=1.6)

    peak_records: List[Dict[str, Union[str, int, float]]] = []
    sample_name = Path(file_path).stem

    for i, target in enumerate(target_peaks, start=1):
        found = find_peak_near_target(x, y_corrected, float(target), search_half_width=search_half_width)
        if found is None:
            continue

        peak_x, peak_y, _ = found
        direction = 1.0 if i % 2 == 1 else -1.0
        text_x = float(np.clip(peak_x + direction * 180.0, 420.0, 3980.0))
        text_y = float(peak_y + (5.8 if i % 2 == 1 else -5.8))
        vertical_align = "bottom" if text_y >= peak_y else "top"

        ax.annotate(
            f"{peak_x:.2f} cm$^{{-1}}$",
            xy=(peak_x, peak_y),
            xytext=(text_x, text_y),
            ha="center",
            va=vertical_align,
            fontsize=11,
            fontweight="bold",
            color="black",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.1,
                "color": "black",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

        peak_records.append(
            {
                "Sample": sample_name,
                "Peak_No": i,
                "Wavenumber_cm-1": round(peak_x, 2),
            }
        )

    ax.set_xlim(4000, 400)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.grid(False)
    ax.tick_params(axis="both", direction="in", length=5, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    y_min = float(np.nanmin(y_corrected))
    y_max = float(np.nanmax(y_corrected))
    y_span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.14 * y_span)

    if output_name is None:
        output_name = f"{sample_name}_single_ftir"

    png_path = output_dir / f"{output_name}.png"
    tiff_path = output_dir / f"{output_name}.tiff"
    csv_path = output_dir / f"{output_name}_peaks.csv"

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.97)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return export_peak_table(peak_records, csv_path)


def plot_multi_ftir(
    file_list: Sequence[Union[str, Path]],
    sample_names: Sequence[str],
    vertical_offsets: Optional[Sequence[float]],
    target_peak_lists: Sequence[Any],
    curve_colors: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "output",
    search_half_width: float = 40.0,
    smooth_window_length: int = 25,
    smooth_polyorder: int = 3,
    baseline_window_length: int = 301,
    baseline_quantile: float = 0.10,
    baseline_smooth_window: int = 151,
    baseline_polyorder: int = 2,
    peak_line_height: float = 3.2,
    sample_label_x: float = 2500.0,
    sample_label_dy: float = 3.0,
    figsize: Tuple[float, float] = (10.5, 6.8),
    output_name: str = "ftir_multi_stacked",
) -> pd.DataFrame:
    _apply_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(file_list)
    if len(sample_names) != n:
        raise ValueError("sample_names length must match file_list length.")
    if len(target_peak_lists) != n:
        raise ValueError("target_peak_lists length must match file_list length.")
    if curve_colors is not None and len(curve_colors) != n:
        raise ValueError("curve_colors length must match file_list length.")

    spectra: List[Tuple[np.ndarray, np.ndarray]] = []
    ranges: List[float] = []

    for file_path in file_list:
        df = read_ftir_txt(file_path)
        x = df["Wavenumber"].to_numpy()
        y = df["Transmittance"].to_numpy()
        y_smooth = smooth_signal(y, window_length=smooth_window_length, polyorder=smooth_polyorder)
        y_corrected, _ = baseline_correction(
            y_smooth,
            window_length=baseline_window_length,
            quantile=baseline_quantile,
            smooth_window=baseline_smooth_window,
            polyorder=baseline_polyorder,
        )
        spectra.append((x, y_corrected))
        ranges.append(float(np.nanmax(y_corrected) - np.nanmin(y_corrected)))

    if vertical_offsets is None:
        step = float(np.nanmedian(ranges)) if len(ranges) > 0 else 10.0
        if not np.isfinite(step) or step <= 0:
            step = 10.0
        step *= 0.85
        vertical_offsets = [(n - 1 - i) * step for i in range(n)]
    elif len(vertical_offsets) != n:
        raise ValueError("vertical_offsets length must match file_list length.")

    if curve_colors is None:
        curve_colors = _auto_curve_colors(n)

    fig, ax = plt.subplots(figsize=figsize)
    color_map = {"red": "#e41a1c", "blue": "#1f4fff"}

    peak_records: List[Dict[str, Union[str, int, float]]] = []

    for x_y, sample_name, offset, peak_spec, curve_color in zip(
        spectra, sample_names, vertical_offsets, target_peak_lists, curve_colors
    ):
        x, y = x_y
        y_plot = y + float(offset)
        ax.plot(x, y_plot, color=curve_color, linewidth=1.2, label=sample_name)

        normalized_peaks = _normalize_peak_spec(peak_spec, default_color="red")
        peak_no = 1
        for target, color_name in normalized_peaks:
            found = find_peak_near_target(x, y_plot, target, search_half_width=search_half_width)
            if found is None:
                continue

            peak_x, peak_y, _ = found
            color_key = str(color_name).lower()
            line_color = color_map.get(color_key, color_name)

            if color_key == "blue":
                y0 = peak_y - peak_line_height
                y1 = peak_y
                text_y = y0 - 0.28
                text_va = "top"
            else:
                y0 = peak_y
                y1 = peak_y + peak_line_height
                text_y = y1 + 0.28
                text_va = "bottom"

            ax.vlines(peak_x, y0, y1, colors=line_color, linewidth=1.1)
            ax.text(
                peak_x,
                text_y,
                f"{peak_x:.0f}",
                fontsize=9,
                ha="center",
                va=text_va,
                color="black",
            )

            peak_records.append(
                {
                    "Sample": sample_name,
                    "Peak_No": peak_no,
                    "Wavenumber_cm-1": round(peak_x, 2),
                }
            )
            peak_no += 1

    legend_fontsize = 11 if n <= 4 else 10
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.1,
        borderaxespad=0.35,
    )

    ax.set_xlim(4000, 400)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("%T")
    ax.grid(False)
    ax.tick_params(axis="both", direction="in", length=4.5, width=1.0)
    ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.margins(x=0.01, y=0.08)

    png_path = output_dir / f"{output_name}.png"
    tiff_path = output_dir / f"{output_name}.tiff"
    csv_path = output_dir / f"{output_name}_peaks.csv"

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.98)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return export_peak_table(peak_records, csv_path)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    data_dir = _resolve_dir_path(config.get("data_dir", "../../raw_data"), config_dir)
    output_dir = _resolve_dir_path(config.get("output_dir", "output"), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        data_dir,
        config_dir,
        script_dir,
        Path.cwd(),
    ]

    run_single_mode = bool(config.get("run_single_mode", True))
    run_multi_mode = bool(config.get("run_multi_mode", True))

    single_cfg = config.get("single", {})
    if not isinstance(single_cfg, dict):
        single_cfg = {}
    if run_single_mode:
        single_file = single_cfg.get("file", "mel-si.txt")
        single_file_resolved = _resolve_input_file(single_file, search_roots)
        if single_file_resolved is None:
            print(f"[Warning] Single-mode file not found: {single_file}")
        else:
            plot_single_ftir(
                file_path=single_file_resolved,
                output_dir=output_dir,
                target_peaks=single_cfg.get("target_peaks", [3431, 2921, 1346, 754]),
                search_half_width=float(single_cfg.get("search_half_width", 40.0)),
                smooth_window_length=int(single_cfg.get("smooth_window_length", 25)),
                smooth_polyorder=int(single_cfg.get("smooth_polyorder", 3)),
                baseline_window_length=int(single_cfg.get("baseline_window_length", 301)),
                baseline_quantile=float(single_cfg.get("baseline_quantile", 0.10)),
                baseline_smooth_window=int(single_cfg.get("baseline_smooth_window", 151)),
                baseline_polyorder=int(single_cfg.get("baseline_polyorder", 2)),
                figsize=tuple(single_cfg.get("figsize", [7.4, 5.6])),
                output_name=single_cfg.get("output_name", "ftir_single"),
            )

    multi_cfg = config.get("multi", {})
    if not isinstance(multi_cfg, dict):
        multi_cfg = {}
    if run_multi_mode:
        multi_files = multi_cfg.get("files", ["mel.txt", "mel-si.txt"])
        sample_names = multi_cfg.get("sample_names", ["Sample-1", "Sample-2"])
        vertical_offsets = multi_cfg.get("vertical_offsets", None)
        curve_colors = multi_cfg.get("curve_colors", None)
        target_peak_lists = multi_cfg.get("target_peak_lists", [])

        resolved_multi_files: List[str] = []
        missing: List[str] = []
        for f in multi_files:
            resolved = _resolve_input_file(f, search_roots)
            if resolved is None:
                missing.append(str(f))
            else:
                resolved_multi_files.append(str(resolved))

        if missing:
            print("[Warning] Multi-mode files missing:", ", ".join(missing))
        elif len(sample_names) != len(resolved_multi_files):
            print("[Warning] sample_names length does not match files length in ftir_config.json")
        elif len(target_peak_lists) != len(resolved_multi_files):
            print("[Warning] target_peak_lists length does not match files length in ftir_config.json")
        elif vertical_offsets is not None and len(vertical_offsets) != len(resolved_multi_files):
            print("[Warning] vertical_offsets length does not match files length in ftir_config.json")
        elif curve_colors is not None and len(curve_colors) != len(resolved_multi_files):
            print("[Warning] curve_colors length does not match files length in ftir_config.json")
        else:
            plot_multi_ftir(
                file_list=resolved_multi_files,
                sample_names=sample_names,
                vertical_offsets=vertical_offsets,
                target_peak_lists=target_peak_lists,
                curve_colors=curve_colors,
                output_dir=output_dir,
                search_half_width=float(multi_cfg.get("search_half_width", 40.0)),
                smooth_window_length=int(multi_cfg.get("smooth_window_length", 25)),
                smooth_polyorder=int(multi_cfg.get("smooth_polyorder", 3)),
                baseline_window_length=int(multi_cfg.get("baseline_window_length", 301)),
                baseline_quantile=float(multi_cfg.get("baseline_quantile", 0.10)),
                baseline_smooth_window=int(multi_cfg.get("baseline_smooth_window", 151)),
                baseline_polyorder=int(multi_cfg.get("baseline_polyorder", 2)),
                peak_line_height=float(multi_cfg.get("peak_line_height", 3.2)),
                sample_label_x=float(multi_cfg.get("sample_label_x", 2500.0)),
                sample_label_dy=float(multi_cfg.get("sample_label_dy", 3.0)),
                figsize=tuple(multi_cfg.get("figsize", [10.5, 6.8])),
                output_name=multi_cfg.get("output_name", "ftir_multi"),
            )


if __name__ == "__main__":
    main()


