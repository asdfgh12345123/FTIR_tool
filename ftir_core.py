from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import savgol_filter as scipy_savgol_filter
except ImportError:
    scipy_find_peaks = None
    scipy_savgol_filter = None

try:
    from scipy.sparse import csc_matrix, diags
    from scipy.sparse.linalg import spsolve
except ImportError:
    csc_matrix = None
    diags = None
    spsolve = None


LogFunc = Optional[Callable[[str], None]]
DEFAULT_CANDIDATE_PEAKS = [3410, 2950, 1650, 1460, 1250, 1020, 810, 460]
DEFAULT_COLORS = ["black", "red", "blue", "green", "purple"]

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _log(logger: LogFunc, message: str) -> None:
    if logger is not None:
        logger(message)
    else:
        print(message)


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "SimHei", "SimSun"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
            "axes.unicode_minus": False,
            "axes.linewidth": 1.2,
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.dpi": 600,
        }
    )


def _savgol_filter_numpy(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=float)
    vandermonde = np.vander(x, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(vandermonde)[0]
    padded = np.pad(y, (half, half), mode="reflect")
    return np.convolve(padded, coeffs[::-1], mode="valid")


def _find_peaks_fallback(
    values: np.ndarray,
    prominence: float = 0.01,
    distance: int = 20,
    width: int = 3,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    y = np.asarray(values, dtype=float)
    if y.size < 3:
        empty = np.array([], dtype=int)
        return empty, {"prominences": np.array([], dtype=float), "widths": np.array([], dtype=float)}

    candidates = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if candidates.size == 0:
        empty = np.array([], dtype=int)
        return empty, {"prominences": np.array([], dtype=float), "widths": np.array([], dtype=float)}

    prominences: List[float] = []
    widths: List[float] = []
    keep_indices: List[int] = []
    span = max(int(distance), 5)

    for idx in candidates:
        left = max(0, idx - span)
        right = min(y.size, idx + span + 1)
        left_min = float(np.min(y[left: idx + 1]))
        right_min = float(np.min(y[idx:right]))
        prom = float(y[idx] - max(left_min, right_min))
        level = float(y[idx] - prom * 0.5)

        left_width = idx
        while left_width > 0 and y[left_width] > level:
            left_width -= 1
        right_width = idx
        while right_width < y.size - 1 and y[right_width] > level:
            right_width += 1
        peak_width = float(right_width - left_width)

        if prom >= float(prominence) and peak_width >= float(width):
            keep_indices.append(int(idx))
            prominences.append(prom)
            widths.append(peak_width)

    if not keep_indices:
        empty = np.array([], dtype=int)
        return empty, {"prominences": np.array([], dtype=float), "widths": np.array([], dtype=float)}

    order = np.argsort(np.asarray(prominences))[::-1]
    selected: List[int] = []
    selected_prom: List[float] = []
    selected_width: List[float] = []

    for order_idx in order:
        idx = keep_indices[order_idx]
        if any(abs(idx - chosen) < int(distance) for chosen in selected):
            continue
        selected.append(idx)
        selected_prom.append(prominences[order_idx])
        selected_width.append(widths[order_idx])

    selected_array = np.asarray(sorted(selected), dtype=int)
    prom_map = {idx: prom for idx, prom in zip(selected, selected_prom)}
    width_map = {idx: wid for idx, wid in zip(selected, selected_width)}
    return selected_array, {
        "prominences": np.asarray([prom_map[idx] for idx in selected_array], dtype=float),
        "widths": np.asarray([width_map[idx] for idx in selected_array], dtype=float),
    }


def _second_difference_penalty(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.zeros_like(x)

    main = np.full(n, 6.0, dtype=float)
    main[0] = 1.0
    main[1] = 5.0
    main[-2] = 5.0
    main[-1] = 1.0

    off1 = np.full(n - 1, -4.0, dtype=float)
    off1[0] = -2.0
    off1[-1] = -2.0
    off2 = np.ones(n - 2, dtype=float)

    out = main * x
    out[:-1] += off1 * x[1:]
    out[1:] += off1 * x[:-1]
    out[:-2] += off2 * x[2:]
    out[2:] += off2 * x[:-2]
    return out


def _solve_als_cg(
    y: np.ndarray,
    weights: np.ndarray,
    lam: float,
    tol: float = 1e-6,
    max_iter: int = 400,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    b = weights * y

    def matvec(vec: np.ndarray) -> np.ndarray:
        return weights * vec + float(lam) * _second_difference_penalty(vec)

    x = y.copy()
    r = b - matvec(x)
    p = r.copy()
    rsold = float(np.dot(r, r))
    if rsold <= tol ** 2:
        return x

    for _ in range(max_iter):
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if abs(denom) < 1e-14:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = float(np.dot(r, r))
        if rsnew <= tol ** 2:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x


def _detect_yunits(file_path: Path, lines: Sequence[str], logger: LogFunc = None) -> str:
    detected_raw = ""
    for line in lines:
        line_strip = line.strip()
        if line_strip.upper().startswith("##YUNITS"):
            if "=" in line_strip:
                detected_raw = line_strip.split("=", 1)[1].strip()
            else:
                detected_raw = line_strip.replace("##YUNITS", "").replace(":", "").strip()
            break

    normalized = detected_raw.upper().replace(" ", "")
    if normalized in {"%T", "%TRANSMITTANCE", "TRANSMITTANCE", "TRANSMITTANCE(%)", "T"}:
        _log(logger, f"File {file_path.name} detected unit: %T")
        return "%T"
    if normalized in {"ABS", "ABSORBANCE", "A"}:
        _log(logger, f"File {file_path.name} detected unit: Abs -> converted to %T")
        return "Abs"
    return ""


def _infer_unit(values: np.ndarray, file_path: Path, logger: LogFunc = None) -> str:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        _log(logger, f"Warning: File {file_path.name} missing ##YUNITS, defaulting to %T")
        return "%T"

    median_val = float(np.nanmedian(finite_values))
    max_val = float(np.nanmax(finite_values))
    if max_val <= 10.0 and median_val <= 5.0:
        _log(logger, f"Warning: File {file_path.name} missing ##YUNITS, inferred unit: Abs -> converted to %T")
        return "Abs"

    _log(logger, f"Warning: File {file_path.name} missing ##YUNITS, inferred unit: %T")
    return "%T"


def _extract_numeric_rows(lines: Sequence[str]) -> List[Tuple[float, float]]:
    number_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    rows: List[Tuple[float, float]] = []
    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith("#"):
            continue
        numbers = number_pattern.findall(line_strip)
        if len(numbers) < 2:
            continue
        rows.append((float(numbers[0]), float(numbers[1])))
    return rows


def read_ftir_file(file_path: Union[str, Path], logger: LogFunc = None) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FTIR data file not found: {file_path}")

    if file_path.suffix.lower() not in {".txt", ".csv", ".dat"}:
        raise ValueError(f"Unsupported FTIR file format: {file_path.suffix}")

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = _extract_numeric_rows(lines)
    if not rows:
        raise ValueError(f"No valid FTIR numeric data found in: {file_path}")

    df = pd.DataFrame(rows, columns=["Wavenumber", "Intensity"])
    df = df.dropna().drop_duplicates(subset="Wavenumber")
    if df.empty:
        raise ValueError(f"No valid FTIR numeric data found in: {file_path}")

    detected_unit = _detect_yunits(file_path, lines, logger=logger)
    if not detected_unit:
        detected_unit = _infer_unit(df["Intensity"].to_numpy(), file_path, logger=logger)

    values = df["Intensity"].to_numpy(dtype=float)
    if detected_unit == "Abs":
        values = np.power(10.0, -values) * 100.0

    df["Transmittance"] = values
    df = df[["Wavenumber", "Transmittance"]].sort_values("Wavenumber").reset_index(drop=True)
    return df


read_ftir_txt = read_ftir_file


def smooth_signal(y: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y.copy()

    window = int(max(5, window_length))
    if window % 2 == 0:
        window += 1
    if window >= y.size:
        window = y.size - 1 if y.size % 2 == 0 else y.size
    if window < 5:
        return y.copy()

    poly = min(int(polyorder), window - 1)
    if scipy_savgol_filter is not None:
        return scipy_savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    return _savgol_filter_numpy(y, window_length=window, polyorder=poly)


def baseline_correction(
    y: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10,
    correction_strength: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y.copy(), np.zeros_like(y)

    length = y.size
    weights = np.ones(length, dtype=float)

    if diags is not None and csc_matrix is not None and spsolve is not None:
        d = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length))
        dtd = csc_matrix(d.T @ d)
        for _ in range(int(niter)):
            w = diags(weights, 0, shape=(length, length))
            baseline = spsolve(w + lam * dtd, weights * y)
            weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)
    else:
        baseline = y.copy()
        for _ in range(int(niter)):
            baseline = _solve_als_cg(y, weights, lam)
            weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)

    baseline = np.asarray(baseline, dtype=float)
    median_baseline = float(np.nanmedian(baseline))
    corrected = y - float(correction_strength) * (baseline - median_baseline)
    return corrected, baseline


def preprocess_spectrum(
    file_path: Union[str, Path],
    logger: LogFunc = None,
    smooth_window_length: int = 11,
    smooth_polyorder: int = 3,
    baseline_lam: float = 1e6,
    baseline_p: float = 0.01,
    baseline_niter: int = 10,
    baseline_strength: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    df = read_ftir_file(file_path, logger=logger)
    x = df["Wavenumber"].to_numpy(dtype=float)
    y = df["Transmittance"].to_numpy(dtype=float)
    y_smooth = smooth_signal(y, window_length=smooth_window_length, polyorder=smooth_polyorder)
    y_corrected, _ = baseline_correction(
        y_smooth,
        lam=baseline_lam,
        p=baseline_p,
        niter=baseline_niter,
        correction_strength=baseline_strength,
    )
    return x, y_corrected


def find_peak_near_target(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
    search_half_width: float = 35.0,
) -> Optional[Tuple[float, float, int]]:
    mask = (x >= float(target) - search_half_width) & (x <= float(target) + search_half_width)
    indices = np.where(mask)[0]
    if indices.size == 0:
        return None
    local_idx = indices[np.argmin(y[indices])]
    return float(x[local_idx]), float(y[local_idx]), int(local_idx)


def _classify_feature_type(y: np.ndarray, idx: int, fallback: str = "valley", half_window: int = 2) -> str:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return fallback

    idx = int(np.clip(idx, 0, y.size - 1))
    left_idx = max(0, idx - max(1, int(half_window)))
    right_idx = min(y.size - 1, idx + max(1, int(half_window)))
    window = y[left_idx : right_idx + 1]
    center = float(y[idx])
    local_idx = idx - left_idx

    if window.size >= 3:
        neighbors = np.delete(window, local_idx)
        if neighbors.size > 0:
            if center >= float(np.max(neighbors)):
                return "peak"
            if center <= float(np.min(neighbors)):
                return "valley"
            neighbor_mean = float(np.mean(neighbors))
            return "peak" if center >= neighbor_mean else "valley"

    left = float(y[max(0, idx - 1)])
    right = float(y[min(y.size - 1, idx + 1)])
    neighbor_mean = 0.5 * (left + right)
    return "peak" if center >= neighbor_mean else "valley"


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float = 0.01,
    distance: int = 20,
    width: int = 3,
) -> List[Dict[str, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inv_y = np.nanmax(y) - y
    scale = float(np.ptp(inv_y))
    inv_norm = inv_y / scale if scale > 0 else inv_y.copy()

    if scipy_find_peaks is not None:
        peak_indices, props = scipy_find_peaks(
            inv_norm,
            prominence=prominence,
            distance=distance,
            width=width,
        )
    else:
        peak_indices, props = _find_peaks_fallback(
            inv_norm,
            prominence=prominence,
            distance=distance,
            width=width,
        )

    peaks: List[Dict[str, float]] = []
    prominences = props.get("prominences", np.zeros_like(peak_indices, dtype=float))
    widths = props.get("widths", np.zeros_like(peak_indices, dtype=float))
    for idx, prom, wid in zip(peak_indices, prominences, widths):
        peaks.append(
            {
                "peak_x": float(x[idx]),
                "peak_y": float(y[idx]),
                "prominence": float(prom),
                "width": float(wid),
            }
        )

    peaks.sort(key=lambda item: item["peak_x"], reverse=True)
    return peaks


def _select_candidate_peaks(
    x: np.ndarray,
    y: np.ndarray,
    detected_peaks: Sequence[Dict[str, float]],
    candidate_peaks: Sequence[float],
    search_half_width: float = 35.0,
) -> List[Dict[str, float]]:
    if not candidate_peaks:
        return list(detected_peaks)

    selected: List[Dict[str, float]] = []
    used_positions: List[float] = []
    for candidate in sorted(candidate_peaks, reverse=True):
        matches = [
            peak for peak in detected_peaks
            if abs(float(peak["peak_x"]) - float(candidate)) <= float(search_half_width)
        ]
        if matches:
            chosen = min(matches, key=lambda peak: abs(float(peak["peak_x"]) - float(candidate)))
        else:
            local_peak = find_peak_near_target(x, y, float(candidate), search_half_width=search_half_width)
            if local_peak is None:
                continue
            chosen = {
                "peak_x": float(local_peak[0]),
                "peak_y": float(local_peak[1]),
                "prominence": 0.0,
                "width": 0.0,
            }

        if any(abs(float(chosen["peak_x"]) - used_x) < 1.0 for used_x in used_positions):
            continue
        used_positions.append(float(chosen["peak_x"]))
        selected.append(dict(chosen))

    selected.sort(key=lambda item: item["peak_x"], reverse=True)
    return selected


def _collect_candidate_peak_details(
    x: np.ndarray,
    y: np.ndarray,
    candidate_peaks: Sequence[float],
    search_half_width: float = 35.0,
) -> List[Dict[str, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return []

    spectral_span = max(float(np.ptp(y)), 1.0)
    dx = np.diff(x)
    step = float(np.nanmedian(np.abs(dx))) if dx.size > 0 else 1.0
    window_pts = max(10, int(search_half_width / max(step, 1e-6)))
    min_strength = max(0.25, 0.012 * spectral_span)

    details: List[Dict[str, float]] = []
    for candidate in candidate_peaks:
        mask = (x >= float(candidate) - search_half_width) & (x <= float(candidate) + search_half_width)
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue

        min_idx = int(indices[np.argmin(y[indices])])
        max_idx = int(indices[np.argmax(y[indices])])

        def _feature_strength(idx: int, feature_type: str) -> Tuple[float, float]:
            left = max(0, idx - window_pts)
            right = min(len(y), idx + window_pts + 1)
            local_window = y[left:right]
            if feature_type == "valley":
                local_high = float(np.percentile(local_window, 85))
                strength = float(local_high - y[idx])
            else:
                local_low = float(np.percentile(local_window, 15))
                strength = float(y[idx] - local_low)
            distance_penalty = 1.0 + abs(float(x[idx]) - float(candidate)) / max(float(search_half_width), 1.0)
            return strength, strength / distance_penalty

        valley_strength, valley_score = _feature_strength(min_idx, "valley")
        peak_strength, peak_score = _feature_strength(max_idx, "peak")

        if max(valley_strength, peak_strength) < min_strength:
            continue

        if peak_score > valley_score:
            chosen_idx = max_idx
            feature_type = "peak"
            feature_score = peak_strength
        else:
            chosen_idx = min_idx
            feature_type = "valley"
            feature_score = valley_strength

        feature_type = _classify_feature_type(y, chosen_idx, fallback=feature_type)

        peak_x = float(x[chosen_idx])
        peak_y = float(y[chosen_idx])
        details.append(
            {
                "target_x": float(candidate),
                "peak_x": float(peak_x),
                "peak_y": float(peak_y),
                "feature_type": feature_type,
                "score": float(feature_score),
                "distance_to_target": abs(float(peak_x) - float(candidate)),
            }
        )

    details.sort(key=lambda item: item["peak_x"], reverse=True)
    return details


def _select_sparse_representative_peaks(
    peak_details: Sequence[Dict[str, float]],
    max_labels: int = 4,
    min_spacing_cm1: float = 60.0,
) -> List[Dict[str, float]]:
    ordered = sorted(
        peak_details,
        key=lambda item: (
            -float(item["score"]),
            float(item["distance_to_target"]),
        ),
    )
    selected: List[Dict[str, float]] = []
    for item in ordered:
        if any(abs(float(item["peak_x"]) - float(existing["peak_x"])) < float(min_spacing_cm1) for existing in selected):
            continue
        selected.append(dict(item))
        if len(selected) >= int(max_labels):
            break

    selected.sort(key=lambda item: item["peak_x"], reverse=True)
    return selected


def _deduplicate_shared_labels(
    per_sample_selected: Sequence[Sequence[Dict[str, float]]],
    n_spectra: int,
    shared_threshold_cm1: float = 8.0,
) -> List[Dict[str, float]]:
    all_entries: List[Dict[str, float]] = []
    for sample_idx, peaks in enumerate(per_sample_selected):
        for peak in peaks:
            entry = dict(peak)
            entry["sample_idx"] = float(sample_idx)
            all_entries.append(entry)

    if not all_entries:
        return []

    all_entries.sort(key=lambda item: float(item["peak_x"]), reverse=True)
    groups: List[List[Dict[str, float]]] = []
    current_group: List[Dict[str, float]] = [all_entries[0]]

    for entry in all_entries[1:]:
        current_center = float(np.mean([float(item["peak_x"]) for item in current_group]))
        if abs(float(entry["peak_x"]) - current_center) < float(shared_threshold_cm1):
            current_group.append(entry)
        else:
            groups.append(current_group)
            current_group = [entry]
    groups.append(current_group)

    preferred_mid = (n_spectra - 1) / 2.0
    selected: List[Dict[str, float]] = []
    for group in groups:
        best = max(
            group,
            key=lambda item: (
                float(item["score"]),
                -abs(float(item["sample_idx"]) - preferred_mid),
                -float(item["distance_to_target"]),
            ),
        )
        selected.append(best)

    selected.sort(key=lambda item: float(item["peak_x"]), reverse=True)
    return selected


def _simple_vertical_label_candidates(
    peak_x: float,
    peak_y: float,
    preferred_direction: str,
    base_line_height: float,
    base_gap: float,
    compact_mode: bool = False,
) -> List[Tuple[float, float, float, float]]:
    sign_pref = 1.0 if preferred_direction == "up" else -1.0
    if compact_mode:
        line_tip_y = peak_y + sign_pref * base_line_height
        text_y = line_tip_y + sign_pref * base_gap
        x_shifts = [0.0, -10.0, 10.0, -16.0, 16.0, -24.0, 24.0]
        return [(peak_x + x_shift, line_tip_y, text_y, 0.03 * abs(x_shift)) for x_shift in x_shifts]

    specs = [(1.00, 1.00), (1.25, 1.20), (1.55, 1.45), (1.90, 1.70), (2.25, 2.00)]
    x_shifts = [0.0, -18.0, 18.0, -30.0, 30.0, -42.0, 42.0, -55.0, 55.0]
    candidates: List[Tuple[float, float, float, float]] = []
    for line_factor, gap_factor in specs:
        line_tip_y = peak_y + sign_pref * base_line_height * line_factor
        text_y = line_tip_y + sign_pref * base_gap * gap_factor
        for x_shift in x_shifts:
            penalty = 0.022 * abs(x_shift) + 0.8 * (line_factor - 1.0)
            candidates.append((peak_x + x_shift, line_tip_y, text_y, penalty))
    return candidates


def _choose_vertical_label_layout(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: Sequence[Tuple[float, float, float, float]],
    occupied_segments: Sequence[Tuple[float, float, float, float]],
    peak_x: float,
    peak_y: float,
    preferred_direction: str,
    label_text: str,
    fontsize: float,
    base_line_height: float,
    base_gap: float,
    compact_mode: bool = False,
) -> Dict[str, float]:
    axes_box = ax.get_window_extent(renderer=renderer)
    safe_axes = (
        float(axes_box.x0 + 8.0),
        float(axes_box.y0 + 8.0),
        float(axes_box.x1 - 8.0),
        float(axes_box.y1 - 8.0),
    )
    best_layout: Optional[Dict[str, float]] = None
    best_score = -np.inf

    for text_x, line_tip_y, text_y, penalty in _simple_vertical_label_candidates(
        peak_x,
        peak_y,
        preferred_direction,
        base_line_height,
        base_gap,
        compact_mode=compact_mode,
    ):
        line_segment = ax.transData.transform(np.array([[peak_x, peak_y], [peak_x, line_tip_y]], dtype=float))
        line_box = _segment_box_from_points(
            (float(line_segment[0, 0]), float(line_segment[0, 1])),
            (float(line_segment[1, 0]), float(line_segment[1, 1])),
            pad_x=2.5,
            pad_y=2.5,
        )
        bbox = _display_bbox_for_text(ax, renderer, text_x, text_y, label_text, fontsize)
        bbox = (bbox[0] - 3.0, bbox[1] - 2.0, bbox[2] + 3.0, bbox[3] + 2.0)

        if bbox[0] < safe_axes[0] or bbox[1] < safe_axes[1] or bbox[2] > safe_axes[2] or bbox[3] > safe_axes[3]:
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=4.0, pad_y=4.0) for placed in occupied_boxes):
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=2.0, pad_y=2.0) for placed in occupied_segments):
            continue

        curve_distance = _min_curve_distance_to_box(display_curves, bbox)
        if curve_distance < max(10.0, 0.9 * fontsize):
            continue

        border_margin = min(
            bbox[0] - safe_axes[0],
            safe_axes[2] - bbox[2],
            bbox[1] - safe_axes[1],
            safe_axes[3] - bbox[3],
        )
        score = 1.8 * curve_distance + 0.15 * border_margin - 4.0 * penalty
        if score > best_score:
            best_score = score
            best_layout = {
                "layout_type": "vertical",
                "needs_leader": False,
                "text_x": float(text_x),
                "line_tip_y": float(line_tip_y),
                "text_y": float(text_y),
                "bbox_x0": float(bbox[0]),
                "bbox_y0": float(bbox[1]),
                "bbox_x1": float(bbox[2]),
                "bbox_y1": float(bbox[3]),
                "seg_x0": float(line_box[0]),
                "seg_y0": float(line_box[1]),
                "seg_x1": float(line_box[2]),
                "seg_y1": float(line_box[3]),
            }

    if best_layout is not None:
        return best_layout

    return None


def annotate_peak(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: List[Tuple[float, float, float, float]],
    occupied_segments: List[Tuple[float, float, float, float]],
    peak_x: float,
    peak_y: float,
    direction: str,
    label_text: str,
    fontsize: float = 11.0,
    line_color: str = "black",
    line_width: float = 1.1,
    base_line_height: float = 4.2,
    base_gap: float = 1.0,
    compact_mode: bool = False,
) -> Optional[Dict[str, float]]:
    layout = _choose_vertical_label_layout(
        ax=ax,
        renderer=renderer,
        display_curves=display_curves,
        occupied_boxes=occupied_boxes,
        occupied_segments=occupied_segments,
        peak_x=peak_x,
        peak_y=peak_y,
        preferred_direction=direction,
        label_text=label_text,
        fontsize=fontsize,
        base_line_height=base_line_height,
        base_gap=base_gap,
        compact_mode=compact_mode,
    )
    if layout is None:
        layout = _force_vertical_label_layout(
            ax=ax,
            renderer=renderer,
            display_curves=display_curves,
            occupied_boxes=occupied_boxes,
            occupied_segments=occupied_segments,
            peak_x=peak_x,
            peak_y=peak_y,
            preferred_direction=direction,
            label_text=label_text,
            fontsize=fontsize,
            base_line_height=base_line_height,
            base_gap=base_gap,
            compact_mode=compact_mode,
        )

    ax.vlines(
        peak_x,
        peak_y,
        float(layout["line_tip_y"]),
        colors=line_color,
        linewidth=line_width,
        zorder=4,
    )
    ax.text(
        float(layout.get("text_x", peak_x)),
        float(layout["text_y"]),
        label_text,
        fontsize=fontsize,
        ha="center",
        va="center",
        color="black",
        bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        zorder=6,
    )

    occupied_boxes.append(
        (
            float(layout["bbox_x0"]),
            float(layout["bbox_y0"]),
            float(layout["bbox_x1"]),
            float(layout["bbox_y1"]),
        )
    )
    occupied_segments.append(
        (
            float(layout["seg_x0"]),
            float(layout["seg_y0"]),
            float(layout["seg_x1"]),
            float(layout["seg_y1"]),
        )
    )
    return layout


def _force_vertical_label_layout(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: Sequence[Tuple[float, float, float, float]],
    occupied_segments: Sequence[Tuple[float, float, float, float]],
    peak_x: float,
    peak_y: float,
    preferred_direction: str,
    label_text: str,
    fontsize: float,
    base_line_height: float,
    base_gap: float,
    compact_mode: bool = False,
) -> Dict[str, float]:
    axes_box = ax.get_window_extent(renderer=renderer)
    safe_axes = (
        float(axes_box.x0 + 8.0),
        float(axes_box.y0 + 8.0),
        float(axes_box.x1 - 8.0),
        float(axes_box.y1 - 8.0),
    )
    sign = 1.0 if preferred_direction == "up" else -1.0
    if compact_mode:
        x_shifts = [0.0, -10.0, 10.0, -16.0, 16.0, -24.0, 24.0, -32.0, 32.0]
        line_tip_y = peak_y + sign * base_line_height
        text_y = line_tip_y + sign * base_gap
        scale_specs = [(line_tip_y, text_y)]
    else:
        x_shifts = [0.0, -18.0, 18.0, -30.0, 30.0, -42.0, 42.0]
        scale_specs = [(1.6, 1.6), (2.0, 1.9), (2.4, 2.2), (2.8, 2.5)]
    best_layout: Optional[Dict[str, float]] = None
    best_score = -np.inf

    for spec in scale_specs:
        if compact_mode:
            line_tip_y, text_y = spec
        else:
            line_scale, gap_scale = spec
            line_tip_y = peak_y + sign * base_line_height * line_scale
            text_y = line_tip_y + sign * base_gap * gap_scale
        for x_shift in x_shifts:
            text_x = float(peak_x + x_shift)
            bbox = _display_bbox_for_text(ax, renderer, text_x, text_y, label_text, fontsize)
            bbox = (bbox[0] - 3.0, bbox[1] - 2.0, bbox[2] + 3.0, bbox[3] + 2.0)

            if bbox[0] < safe_axes[0] or bbox[1] < safe_axes[1] or bbox[2] > safe_axes[2] or bbox[3] > safe_axes[3]:
                continue

            line_segment = ax.transData.transform(np.array([[peak_x, peak_y], [peak_x, line_tip_y]], dtype=float))
            line_box = _segment_box_from_points(
                (float(line_segment[0, 0]), float(line_segment[0, 1])),
                (float(line_segment[1, 0]), float(line_segment[1, 1])),
                pad_x=2.5,
                pad_y=2.5,
            )

            overlap_penalty = 0.0
            overlap_penalty += 1.2 * sum(_bbox_overlap(bbox, placed, pad_x=4.0, pad_y=4.0) for placed in occupied_boxes)
            overlap_penalty += 0.8 * sum(_bbox_overlap(bbox, placed, pad_x=2.0, pad_y=2.0) for placed in occupied_segments)
            curve_distance = _min_curve_distance_to_box(display_curves, bbox)
            border_margin = min(
                bbox[0] - safe_axes[0],
                safe_axes[2] - bbox[2],
                bbox[1] - safe_axes[1],
                safe_axes[3] - bbox[3],
            )
            score = 1.2 * curve_distance + 0.08 * border_margin - 0.025 * abs(x_shift) - 2.5 * overlap_penalty
            if score > best_score:
                best_score = score
                best_layout = {
                    "layout_type": "vertical",
                    "needs_leader": False,
                    "text_x": float(text_x),
                    "line_tip_y": float(line_tip_y),
                    "text_y": float(text_y),
                    "bbox_x0": float(bbox[0]),
                    "bbox_y0": float(bbox[1]),
                    "bbox_x1": float(bbox[2]),
                    "bbox_y1": float(bbox[3]),
                    "seg_x0": float(line_box[0]),
                    "seg_y0": float(line_box[1]),
                    "seg_x1": float(line_box[2]),
                    "seg_y1": float(line_box[3]),
                }
    if best_layout is not None:
        return best_layout

    if compact_mode:
        line_tip_y = peak_y + sign * base_line_height
        text_y = line_tip_y + sign * base_gap
    else:
        fallback_scale = 2.0
        line_tip_y = peak_y + sign * base_line_height * fallback_scale
        text_y = line_tip_y + sign * base_gap * fallback_scale
    bbox = _display_bbox_for_text(ax, renderer, peak_x, text_y, label_text, fontsize)
    bbox = (bbox[0] - 3.0, bbox[1] - 2.0, bbox[2] + 3.0, bbox[3] + 2.0)
    line_segment = ax.transData.transform(np.array([[peak_x, peak_y], [peak_x, line_tip_y]], dtype=float))
    line_box = _segment_box_from_points(
        (float(line_segment[0, 0]), float(line_segment[0, 1])),
        (float(line_segment[1, 0]), float(line_segment[1, 1])),
        pad_x=2.5,
        pad_y=2.5,
    )
    return {
        "layout_type": "vertical",
        "needs_leader": False,
        "text_x": float(peak_x),
        "line_tip_y": float(line_tip_y),
        "text_y": float(text_y),
        "bbox_x0": float(bbox[0]),
        "bbox_y0": float(bbox[1]),
        "bbox_x1": float(bbox[2]),
        "bbox_y1": float(bbox[3]),
        "seg_x0": float(line_box[0]),
        "seg_y0": float(line_box[1]),
        "seg_x1": float(line_box[2]),
        "seg_y1": float(line_box[3]),
    }


def _compute_required_offset_step(
    spectra: Sequence[Tuple[np.ndarray, np.ndarray]],
    min_step: float = 24.0,
) -> float:
    if not spectra:
        return float(min_step)

    valid_spectra = [
        (np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float))
        for x_values, y_values in spectra
        if np.asarray(y_values).size > 0
    ]
    if not valid_spectra:
        return float(min_step)

    amplitudes = [float(np.max(y_values) - np.min(y_values)) for _, y_values in valid_spectra]
    amp = max(amplitudes)
    if not np.isfinite(amp):
        return float(min_step)

    global_ymin = min(float(np.min(y_values)) for _, y_values in valid_spectra)
    global_ymax = max(float(np.max(y_values)) for _, y_values in valid_spectra)
    global_span = max(global_ymax - global_ymin, 1.0)

    line_len = min(max(global_span * 0.035, 3.0), 8.0)
    label_space = line_len * 0.8
    gap_space = 4.0

    offset = amp * 0.55 + line_len * 2.0 + label_space + gap_space
    offset = max(float(offset), float(min_step))

    if len(valid_spectra) >= 3:
        offset *= 1.15

    return float(offset)


def _ensure_minimum_vertical_offsets(
    offsets: Sequence[float],
    required_step: float,
) -> List[float]:
    if not offsets:
        return []

    adjusted = [float(value) for value in offsets]
    for idx in range(len(adjusted) - 2, -1, -1):
        adjusted[idx] = max(adjusted[idx], adjusted[idx + 1] + float(required_step))
    return adjusted


def _compute_peak_label_line_length(
    n_spectra: int,
    y_min: float,
    y_max: float,
    offset: Optional[float] = None,
) -> float:
    y_span = max(float(y_max) - float(y_min), 1.0)
    if int(n_spectra) <= 1:
        return max(y_span * 0.03, 0.8)

    if int(n_spectra) == 2:
        return 4.0

    offset_value = float(offset) if offset is not None else 0.0
    if offset_value <= 0:
        offset_value = max(30.0, y_span * 0.12)

    return max(offset_value * 0.20, 0.8)


def _compute_peak_label_text_gap(
    n_spectra: int,
    line_len: float,
) -> float:
    if int(n_spectra) == 2:
        return 0.8
    return line_len * 0.25


def _choose_leader_label_layout(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: Sequence[Tuple[float, float, float, float]],
    occupied_segments: Sequence[Tuple[float, float, float, float]],
    peak_x: float,
    peak_y: float,
    preferred_direction: str,
    label_text: str,
    fontsize: float,
    base_line_height: float,
    base_gap: float,
) -> Dict[str, float]:
    axes_box = ax.get_window_extent(renderer=renderer)
    safe_axes = (
        float(axes_box.x0 + 8.0),
        float(axes_box.y0 + 8.0),
        float(axes_box.x1 - 8.0),
        float(axes_box.y1 - 8.0),
    )
    sign = 1.0 if preferred_direction == "up" else -1.0
    x_shifts = [55.0, -55.0, 85.0, -85.0, 120.0, -120.0]
    y_factors = [1.4, 1.8, 2.2]

    best_layout: Optional[Dict[str, float]] = None
    best_score = -np.inf

    for x_shift in x_shifts:
        for y_factor in y_factors:
            text_x = float(peak_x + x_shift)
            text_y = float(peak_y + sign * (base_line_height + base_gap * y_factor))
            bbox = _display_bbox_for_text(ax, renderer, text_x, text_y, label_text, fontsize)
            bbox = (bbox[0] - 4.0, bbox[1] - 3.0, bbox[2] + 4.0, bbox[3] + 3.0)

            if bbox[0] < safe_axes[0] or bbox[1] < safe_axes[1] or bbox[2] > safe_axes[2] or bbox[3] > safe_axes[3]:
                continue
            if any(_bbox_overlap(bbox, placed, pad_x=5.0, pad_y=4.0) for placed in occupied_boxes):
                continue
            if any(_bbox_overlap(bbox, placed, pad_x=2.0, pad_y=2.0) for placed in occupied_segments):
                continue

            peak_disp = ax.transData.transform((peak_x, peak_y))
            text_disp = ax.transData.transform((text_x, text_y))
            line_box = _segment_box_from_points(
                (float(peak_disp[0]), float(peak_disp[1])),
                (float(text_disp[0]), float(text_disp[1])),
                pad_x=2.0,
                pad_y=2.0,
            )

            if any(_bbox_overlap(line_box, placed, pad_x=1.0, pad_y=1.0) for placed in occupied_boxes):
                continue

            curve_distance = _min_curve_distance_to_box(display_curves, bbox)
            if curve_distance < max(10.0, 0.9 * fontsize):
                continue

            border_margin = min(
                bbox[0] - safe_axes[0],
                safe_axes[2] - bbox[2],
                bbox[1] - safe_axes[1],
                safe_axes[3] - bbox[3],
            )
            score = 1.7 * curve_distance + 0.12 * border_margin - 0.01 * abs(x_shift)

            if score > best_score:
                best_score = score
                best_layout = {
                    "layout_type": "leader",
                    "needs_leader": True,
                    "text_x": text_x,
                    "text_y": text_y,
                    "bbox_x0": float(bbox[0]),
                    "bbox_y0": float(bbox[1]),
                    "bbox_x1": float(bbox[2]),
                    "bbox_y1": float(bbox[3]),
                    "seg_x0": float(line_box[0]),
                    "seg_y0": float(line_box[1]),
                    "seg_x1": float(line_box[2]),
                    "seg_y1": float(line_box[3]),
                }

    if best_layout is not None:
        return best_layout

    text_x = float(peak_x + 70.0)
    text_y = float(peak_y + sign * (base_line_height + base_gap * 1.6))
    bbox = _display_bbox_for_text(ax, renderer, text_x, text_y, label_text, fontsize)
    bbox = (bbox[0] - 4.0, bbox[1] - 3.0, bbox[2] + 4.0, bbox[3] + 3.0)
    peak_disp = ax.transData.transform((peak_x, peak_y))
    text_disp = ax.transData.transform((text_x, text_y))
    line_box = _segment_box_from_points(
        (float(peak_disp[0]), float(peak_disp[1])),
        (float(text_disp[0]), float(text_disp[1])),
        pad_x=2.0,
        pad_y=2.0,
    )
    return {
        "layout_type": "leader",
        "needs_leader": True,
        "text_x": text_x,
        "text_y": text_y,
        "bbox_x0": float(bbox[0]),
        "bbox_y0": float(bbox[1]),
        "bbox_x1": float(bbox[2]),
        "bbox_y1": float(bbox[3]),
        "seg_x0": float(line_box[0]),
        "seg_y0": float(line_box[1]),
        "seg_x1": float(line_box[2]),
        "seg_y1": float(line_box[3]),
    }


def _build_shared_peak_groups(
    per_sample_peaks: Sequence[Sequence[Dict[str, float]]],
    shared_threshold_cm1: float = 8.0,
) -> List[List[Dict[str, float]]]:
    all_entries: List[Dict[str, float]] = []
    for sample_idx, peaks in enumerate(per_sample_peaks):
        for peak in peaks:
            entry = dict(peak)
            entry["sample_idx"] = float(sample_idx)
            all_entries.append(entry)

    if not all_entries:
        return []

    all_entries.sort(key=lambda item: item["peak_x"], reverse=True)
    groups: List[List[Dict[str, float]]] = []
    current_group: List[Dict[str, float]] = [all_entries[0]]

    for entry in all_entries[1:]:
        current_center = float(np.mean([float(item["peak_x"]) for item in current_group]))
        if abs(float(entry["peak_x"]) - current_center) < float(shared_threshold_cm1):
            current_group.append(entry)
        else:
            groups.append(current_group)
            current_group = [entry]
    groups.append(current_group)

    for group in groups:
        merged: Dict[int, Dict[str, float]] = {}
        for entry in group:
            sample_idx = int(entry["sample_idx"])
            if sample_idx not in merged or float(entry["prominence"]) > float(merged[sample_idx]["prominence"]):
                merged[sample_idx] = entry
        group[:] = sorted(merged.values(), key=lambda item: int(item["sample_idx"]))

    return groups


def _group_representative_sample(
    group: Sequence[Dict[str, float]],
    n_spectra: int,
) -> int:
    if not group:
        return 0

    present_indices = [int(item["sample_idx"]) for item in group]
    strongest = max(group, key=lambda item: float(item["prominence"]))
    strongest_idx = int(strongest["sample_idx"])
    strongest_prom = float(strongest["prominence"])
    weak_prom = min(float(item["prominence"]) for item in group)
    if weak_prom <= 0.0 or strongest_prom >= 1.35 * weak_prom:
        return strongest_idx

    desired_idx = n_spectra // 2 if n_spectra % 2 == 1 else min(n_spectra - 1, n_spectra // 2)
    return min(present_indices, key=lambda idx: abs(idx - desired_idx))


def _compute_group_density(group_centers: Sequence[float], index: int, dense_gap: float = 40.0) -> int:
    center = float(group_centers[index])
    return sum(abs(center - float(other)) < dense_gap for other in group_centers) - 1


def _display_bbox_for_text(
    ax: plt.Axes,
    renderer: Any,
    x: float,
    y: float,
    label: str,
    fontsize: float,
) -> Tuple[float, float, float, float]:
    ghost = ax.text(x, y, label, fontsize=fontsize, ha="center", va="center", alpha=0.0)
    bbox = ghost.get_window_extent(renderer=renderer)
    ghost.remove()
    return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)


def _normalize_box(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _segment_box_from_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    pad_x: float = 3.0,
    pad_y: float = 3.0,
) -> Tuple[float, float, float, float]:
    x0, y0 = p0
    x1, y1 = p1
    return (
        min(x0, x1) - pad_x,
        min(y0, y1) - pad_y,
        max(x0, x1) + pad_x,
        max(y0, y1) + pad_y,
    )


def _bbox_overlap(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> bool:
    box_a = _normalize_box(box_a)
    box_b = _normalize_box(box_b)
    return not (
        box_a[2] + pad_x < box_b[0]
        or box_b[2] + pad_x < box_a[0]
        or box_a[3] + pad_y < box_b[1]
        or box_b[3] + pad_y < box_a[1]
    )


def _min_curve_distance_to_box(
    display_curves: Sequence[np.ndarray],
    box: Tuple[float, float, float, float],
) -> float:
    x0, y0, x1, y1 = box
    distances: List[float] = []
    for points in display_curves:
        dx = np.maximum.reduce([x0 - points[:, 0], np.zeros(points.shape[0]), points[:, 0] - x1])
        dy = np.maximum.reduce([y0 - points[:, 1], np.zeros(points.shape[0]), points[:, 1] - y1])
        distances.append(float(np.min(np.hypot(dx, dy))))
    return min(distances) if distances else np.inf


def _compute_unified_label_metrics(ax: plt.Axes) -> Tuple[float, float]:
    y_min, y_max = ax.get_ylim()
    line_len = min(max((float(y_max) - float(y_min)) * 0.035, 3.0), 8.0)
    text_gap = line_len * 0.25
    return float(line_len), float(text_gap)


def _build_unified_label_candidates(
    peak_x: float,
    peak_y: float,
    direction: str,
    line_len: float,
    text_gap: float,
    dense_count: int,
) -> List[Tuple[float, float, float, float]]:
    sign = 1.0 if direction == "up" else -1.0
    x_shifts = [0.0, -10.0, 10.0, -18.0, 18.0, -28.0, 28.0]
    if dense_count > 0:
        x_shifts.extend([-38.0, 38.0])

    line_scales = [1.0, 1.15]
    if dense_count > 0:
        line_scales.append(1.30)

    candidates: List[Tuple[float, float, float, float]] = []
    for line_scale in line_scales:
        line_tip_y = peak_y + sign * line_len * line_scale
        text_y = line_tip_y + sign * text_gap
        for x_shift in x_shifts:
            penalty = 0.04 * abs(x_shift) + 1.2 * (line_scale - 1.0)
            candidates.append((peak_x + x_shift, line_tip_y, text_y, penalty))
    return candidates


def _choose_unified_label_layout(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: Sequence[Tuple[float, float, float, float]],
    occupied_segments: Sequence[Tuple[float, float, float, float]],
    peak_x: float,
    peak_y: float,
    direction: str,
    label_text: str,
    fontsize: float,
    line_len: float,
    text_gap: float,
    dense_count: int,
) -> Optional[Dict[str, float]]:
    axes_box = ax.get_window_extent(renderer=renderer)
    safe_axes = (
        float(axes_box.x0 + 8.0),
        float(axes_box.y0 + 8.0),
        float(axes_box.x1 - 8.0),
        float(axes_box.y1 - 8.0),
    )
    best_layout: Optional[Dict[str, float]] = None
    best_score = -np.inf

    for text_x, line_tip_y, text_y, penalty in _build_unified_label_candidates(
        peak_x,
        peak_y,
        direction,
        line_len,
        text_gap,
        dense_count,
    ):
        line_segment = ax.transData.transform(np.array([[peak_x, peak_y], [peak_x, line_tip_y]], dtype=float))
        line_box = _segment_box_from_points(
            (float(line_segment[0, 0]), float(line_segment[0, 1])),
            (float(line_segment[1, 0]), float(line_segment[1, 1])),
            pad_x=2.5,
            pad_y=2.5,
        )
        bbox = _display_bbox_for_text(ax, renderer, text_x, text_y, label_text, fontsize)
        bbox = (bbox[0] - 3.0, bbox[1] - 2.0, bbox[2] + 3.0, bbox[3] + 2.0)

        if bbox[0] < safe_axes[0] or bbox[1] < safe_axes[1] or bbox[2] > safe_axes[2] or bbox[3] > safe_axes[3]:
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=4.0, pad_y=4.0) for placed in occupied_boxes):
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=2.0, pad_y=2.0) for placed in occupied_segments):
            continue
        if any(_bbox_overlap(line_box, placed, pad_x=1.0, pad_y=1.0) for placed in occupied_boxes):
            continue

        curve_distance = _min_curve_distance_to_box(display_curves, bbox)
        if curve_distance < max(8.0, 0.75 * fontsize):
            continue

        border_margin = min(
            bbox[0] - safe_axes[0],
            safe_axes[2] - bbox[2],
            bbox[1] - safe_axes[1],
            safe_axes[3] - bbox[3],
        )
        score = 1.8 * curve_distance + 0.12 * border_margin - 3.0 * penalty
        if score > best_score:
            best_score = score
            best_layout = {
                "text_x": float(text_x),
                "text_y": float(text_y),
                "line_tip_y": float(line_tip_y),
                "bbox_x0": float(bbox[0]),
                "bbox_y0": float(bbox[1]),
                "bbox_x1": float(bbox[2]),
                "bbox_y1": float(bbox[3]),
                "seg_x0": float(line_box[0]),
                "seg_y0": float(line_box[1]),
                "seg_x1": float(line_box[2]),
                "seg_y1": float(line_box[3]),
            }
    return best_layout


def annotate_peaks_unified(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    x_curve: np.ndarray,
    y_curve: np.ndarray,
    peak_details: Sequence[Dict[str, float]],
    sample_name: str,
    occupied_boxes: List[Tuple[float, float, float, float]],
    occupied_segments: List[Tuple[float, float, float, float]],
    fontsize: float = 11.0,
    line_width: float = 1.1,
    line_style: str = "feature",
) -> List[Dict[str, Union[str, int, float]]]:
    x_curve = np.asarray(x_curve, dtype=float)
    y_curve = np.asarray(y_curve, dtype=float)
    if x_curve.size == 0 or y_curve.size == 0:
        return []

    line_len, text_gap = _compute_unified_label_metrics(ax)
    peak_centers = [float(item["peak_x"]) for item in peak_details]
    ordered_peaks = sorted(
        peak_details,
        key=lambda item: (-float(item.get("score", 0.0)), -float(item["peak_x"])),
    )

    peak_records: List[Dict[str, Union[str, int, float]]] = []
    peak_no = 0
    for peak in ordered_peaks:
        peak_x = float(peak["peak_x"])
        peak_y = float(np.interp(peak_x, x_curve, y_curve))
        nearest_idx = int(np.argmin(np.abs(x_curve - peak_x)))
        feature_type = _classify_feature_type(y_curve, nearest_idx, fallback=str(peak.get("feature_type", "valley")))
        direction = "up" if feature_type == "peak" else "down"
        dense_count = sum(abs(peak_x - center) < 60.0 for center in peak_centers) - 1

        layout = _choose_unified_label_layout(
            ax=ax,
            renderer=renderer,
            display_curves=display_curves,
            occupied_boxes=occupied_boxes,
            occupied_segments=occupied_segments,
            peak_x=peak_x,
            peak_y=peak_y,
            direction=direction,
            label_text=f"{peak_x:.0f}",
            fontsize=fontsize,
            line_len=line_len,
            text_gap=text_gap,
            dense_count=max(0, dense_count),
        )
        if layout is None:
            continue

        if line_style == "black":
            line_color = "black"
        else:
            line_color = "#e41a1c" if direction == "up" else "#1f4fff"

        ax.vlines(
            peak_x,
            peak_y,
            float(layout["line_tip_y"]),
            colors=line_color,
            linewidth=line_width,
            zorder=4,
        )
        ax.text(
            float(layout["text_x"]),
            float(layout["text_y"]),
            f"{peak_x:.0f}",
            fontsize=fontsize,
            ha="center",
            va="center",
            color="black",
            bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            zorder=6,
        )

        occupied_boxes.append(
            (
                float(layout["bbox_x0"]),
                float(layout["bbox_y0"]),
                float(layout["bbox_x1"]),
                float(layout["bbox_y1"]),
            )
        )
        occupied_segments.append(
            (
                float(layout["seg_x0"]),
                float(layout["seg_y0"]),
                float(layout["seg_x1"]),
                float(layout["seg_y1"]),
            )
        )

        peak_no += 1
        peak_records.append(
            {
                "Sample": sample_name,
                "Peak_No": peak_no,
                "Wavenumber_cm-1": round(peak_x, 2),
            }
        )

    peak_records.sort(key=lambda item: float(item["Wavenumber_cm-1"]), reverse=True)
    for idx, record in enumerate(peak_records, start=1):
        record["Peak_No"] = idx
    return peak_records


def _group_label_candidates(
    peak_y: float,
    direction: str,
    base_height: float,
    base_gap: float,
) -> List[Tuple[float, float, float]]:
    sign_pref = 1.0 if direction == "up" else -1.0
    specs = [
        (sign_pref, 1.00, 1.00),
        (sign_pref, 1.25, 1.15),
        (sign_pref, 1.55, 1.35),
        (sign_pref, 1.90, 1.60),
        (-sign_pref, 1.15, 1.10),
        (-sign_pref, 1.40, 1.25),
    ]
    candidates: List[Tuple[float, float, float]] = []
    for sign, line_factor, gap_factor in specs:
        line_tip_y = peak_y + sign * base_height * line_factor
        text_y = line_tip_y + sign * base_gap * gap_factor
        penalty = 0.0 if sign == sign_pref else 2.8
        candidates.append((line_tip_y, text_y, penalty))
    return candidates


def _choose_group_label_layout(
    ax: plt.Axes,
    renderer: Any,
    display_curves: Sequence[np.ndarray],
    occupied_boxes: Sequence[Tuple[float, float, float, float]],
    occupied_segments: Sequence[Tuple[float, float, float, float]],
    group_x: float,
    anchor_y: float,
    direction: str,
    label_text: str,
    fontsize: float,
    base_line_height: float,
    base_text_gap: float,
) -> Dict[str, float]:
    axes_box = ax.get_window_extent(renderer=renderer)
    safe_axes = (
        float(axes_box.x0 + 8.0),
        float(axes_box.y0 + 8.0),
        float(axes_box.x1 - 8.0),
        float(axes_box.y1 - 8.0),
    )
    group_x_disp = ax.transData.transform((group_x, anchor_y))[0]
    best_layout: Optional[Dict[str, float]] = None
    best_score = -np.inf

    for line_tip_y, text_y, penalty in _group_label_candidates(
        anchor_y,
        direction,
        base_line_height,
        base_text_gap,
    ):
        line_segment = ax.transData.transform(np.array([[group_x, anchor_y], [group_x, line_tip_y]], dtype=float))
        line_box = _segment_box_from_points(
            (float(line_segment[0, 0]), float(line_segment[0, 1])),
            (float(line_segment[1, 0]), float(line_segment[1, 1])),
            pad_x=3.0,
            pad_y=3.0,
        )

        bbox = _display_bbox_for_text(ax, renderer, group_x, text_y, label_text, fontsize)
        bbox = (bbox[0] - 4.0, bbox[1] - 2.0, bbox[2] + 4.0, bbox[3] + 2.0)

        if bbox[0] < safe_axes[0] or bbox[1] < safe_axes[1] or bbox[2] > safe_axes[2] or bbox[3] > safe_axes[3]:
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=5.0, pad_y=4.0) for placed in occupied_boxes):
            continue
        if any(_bbox_overlap(bbox, placed, pad_x=3.0, pad_y=3.0) for placed in occupied_segments):
            continue
        if any(_bbox_overlap(line_box, placed, pad_x=1.0, pad_y=1.0) for placed in occupied_boxes):
            continue

        curve_distance = _min_curve_distance_to_box(display_curves, bbox)
        if curve_distance < max(10.0, 0.95 * fontsize):
            continue

        border_margin = min(
            bbox[0] - safe_axes[0],
            safe_axes[2] - bbox[2],
            bbox[1] - safe_axes[1],
            safe_axes[3] - bbox[3],
        )
        score = 2.2 * curve_distance + 0.25 * border_margin - 6.0 * penalty

        if score > best_score:
            best_score = score
            best_layout = {
                "group_x": float(group_x),
                "line_tip_y": float(line_tip_y),
                "text_y": float(text_y),
                "bbox_x0": float(bbox[0]),
                "bbox_y0": float(bbox[1]),
                "bbox_x1": float(bbox[2]),
                "bbox_y1": float(bbox[3]),
                "seg_x0": float(line_box[0]),
                "seg_y0": float(line_box[1]),
                "seg_x1": float(line_box[2]),
                "seg_y1": float(line_box[3]),
            }

    if best_layout is not None:
        return best_layout

    fallback_tip_y = anchor_y + (base_line_height if direction == "up" else -base_line_height)
    fallback_text_y = fallback_tip_y + (base_text_gap if direction == "up" else -base_text_gap)
    bbox = _display_bbox_for_text(ax, renderer, group_x, fallback_text_y, label_text, fontsize)
    bbox = (bbox[0] - 4.0, bbox[1] - 2.0, bbox[2] + 4.0, bbox[3] + 2.0)
    line_segment = ax.transData.transform(np.array([[group_x, anchor_y], [group_x, fallback_tip_y]], dtype=float))
    line_box = _segment_box_from_points(
        (float(line_segment[0, 0]), float(line_segment[0, 1])),
        (float(line_segment[1, 0]), float(line_segment[1, 1])),
        pad_x=3.0,
        pad_y=3.0,
    )
    return {
        "group_x": float(group_x),
        "line_tip_y": float(fallback_tip_y),
        "text_y": float(fallback_text_y),
        "bbox_x0": float(bbox[0]),
        "bbox_y0": float(bbox[1]),
        "bbox_x1": float(bbox[2]),
        "bbox_y1": float(bbox[3]),
        "seg_x0": float(line_box[0]),
        "seg_y0": float(line_box[1]),
        "seg_x1": float(line_box[2]),
        "seg_y1": float(line_box[3]),
    }


def export_peak_table(
    peak_records: Sequence[Dict[str, Union[str, int, float]]],
    output_csv: Union[str, Path],
    logger: LogFunc = None,
) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if peak_records:
        peak_df = pd.DataFrame(peak_records)
    else:
        peak_df = pd.DataFrame(columns=["Sample", "Peak_No", "Wavenumber_cm-1"])

    peak_df = peak_df[["Sample", "Peak_No", "Wavenumber_cm-1"]]
    peak_df.to_csv(output_csv, index=False)
    _log(logger, f"Saved peak table: {output_csv}")
    return peak_df


def plot_single_ftir(
    file_path: Union[str, Path],
    output_dir: Union[str, Path] = "output",
    target_peaks: Sequence[float] = DEFAULT_CANDIDATE_PEAKS,
    search_half_width: float = 35.0,
    figsize: Tuple[float, float] = (10.0, 6.0),
    output_name: Optional[str] = None,
    logger: LogFunc = None,
) -> pd.DataFrame:
    _apply_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = preprocess_spectrum(file_path, logger=logger)
    sample_name = Path(file_path).stem
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color="red", linewidth=2.0, label=sample_name)

    peak_records: List[Dict[str, Union[str, int, float]]] = []
    candidate_details = _collect_candidate_peak_details(
        x,
        y,
        target_peaks,
        search_half_width=search_half_width,
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
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    y_span = max(float(np.ptp(y)), 1.0)
    ax.set_ylim(float(np.min(y)) - 0.08 * y_span, float(np.max(y)) + 0.12 * y_span)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    display_curves = [ax.transData.transform(np.column_stack([x, y]))]
    occupied_boxes: List[Tuple[float, float, float, float]] = []
    occupied_segments: List[Tuple[float, float, float, float]] = []
    peak_records = annotate_peaks_unified(
        ax=ax,
        renderer=renderer,
        display_curves=display_curves,
        x_curve=x,
        y_curve=y,
        peak_details=candidate_details,
        sample_name=sample_name,
        occupied_boxes=occupied_boxes,
        occupied_segments=occupied_segments,
        fontsize=11.0,
        line_width=1.0,
        line_style="black",
    )

    output_name = output_name or f"{sample_name}_single"
    png_path = output_dir / f"{output_name}.png"
    tiff_path = output_dir / f"{output_name}.tiff"
    csv_path = output_dir / f"{output_name}_peaks.csv"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    _log(logger, f"Saved figure: {png_path}")
    _log(logger, f"Saved figure: {tiff_path}")
    return export_peak_table(peak_records, csv_path, logger=logger)


def plot_multi_ftir(
    file_list: Sequence[Union[str, Path]],
    sample_names: Sequence[str],
    vertical_offsets: Optional[Sequence[float]],
    target_peak_lists: Sequence[Any],
    curve_colors: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "output",
    offset_step: float = 18.0,
    figsize: Tuple[float, float] = (10.0, 6.0),
    output_name: str = "ftir_multi",
    logger: LogFunc = None,
    smooth_window_length: int = 11,
    smooth_polyorder: int = 3,
    baseline_lam: float = 1e6,
    baseline_p: float = 0.01,
    baseline_niter: int = 10,
    prominence: float = 0.01,
    distance: int = 20,
    width: int = 3,
    search_half_width: float = 35.0,
    shared_threshold_cm1: float = 8.0,
    dense_gap_cm1: float = 40.0,
    linewidth: float = 2.0,
    peak_label_fontsize: float = 11.0,
) -> pd.DataFrame:
    _apply_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(file_list)
    if len(sample_names) != n:
        raise ValueError("sample_names length must match file_list length.")

    candidate_peaks = [float(value) for value in target_peak_lists] if target_peak_lists else DEFAULT_CANDIDATE_PEAKS
    spectra: List[Tuple[np.ndarray, np.ndarray]] = []
    per_sample_selected: List[List[Dict[str, float]]] = []

    for file_path in file_list:
        x, y = preprocess_spectrum(
            file_path,
            logger=logger,
            smooth_window_length=smooth_window_length,
            smooth_polyorder=smooth_polyorder,
            baseline_lam=baseline_lam,
            baseline_p=baseline_p,
            baseline_niter=baseline_niter,
        )
        spectra.append((x, y))
        candidate_details = _collect_candidate_peak_details(
            x,
            y,
            candidate_peaks,
            search_half_width=search_half_width,
        )
        selected = _select_sparse_representative_peaks(
            candidate_details,
            max_labels=4,
            min_spacing_cm1=60.0,
        )
        per_sample_selected.append(selected)

    required_step = _compute_required_offset_step(spectra)

    if vertical_offsets is None:
        vertical_offsets = [(n - 1 - i) * float(required_step) for i in range(n)]
        _log(
            logger,
            "Auto-generated vertical offsets: "
            + ", ".join(str(int(offset)) if float(offset).is_integer() else f"{offset:.2f}" for offset in vertical_offsets),
        )
    elif len(vertical_offsets) != n:
        raise ValueError("vertical_offsets length must match file_list length.")
    else:
        adjusted_offsets = _ensure_minimum_vertical_offsets(vertical_offsets, required_step)
        if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(adjusted_offsets, vertical_offsets)):
            _log(
                logger,
                "Adjusted vertical offsets to prevent overlap: "
                + ", ".join(str(int(offset)) if float(offset).is_integer() else f"{offset:.2f}" for offset in adjusted_offsets),
            )
        vertical_offsets = adjusted_offsets

    if curve_colors is None:
        curve_colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    x_arrays: List[np.ndarray] = []
    y_plot_arrays: List[np.ndarray] = []

    for (x, y), offset, color, sample_name in zip(spectra, vertical_offsets, curve_colors, sample_names):
        y_plot = y + float(offset)
        x_arrays.append(x)
        y_plot_arrays.append(y_plot)
        ax.plot(x, y_plot, color=color, linewidth=linewidth, label=sample_name)

    all_y_values = np.concatenate(y_plot_arrays) if y_plot_arrays else np.array([0.0, 1.0])
    y_span = max(float(np.ptp(all_y_values)), 1.0)
    ax.set_xlim(4000, 400)
    ax.set_ylim(float(np.min(all_y_values)) - 0.08 * y_span, float(np.max(all_y_values)) + 0.18 * y_span)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.grid(False)
    ax.tick_params(axis="both", direction="in", length=4.5, width=1.0)
    ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    display_curves = [
        ax.transData.transform(np.column_stack([x_values, y_values]))
        for x_values, y_values in zip(x_arrays, y_plot_arrays)
    ]

    occupied_boxes: List[Tuple[float, float, float, float]] = []
    occupied_segments: List[Tuple[float, float, float, float]] = []
    peak_records: List[Dict[str, Union[str, int, float]]] = []

    for sample_idx, selected_peaks in enumerate(per_sample_selected):
        peak_records.extend(
            annotate_peaks_unified(
                ax=ax,
                renderer=renderer,
                display_curves=display_curves,
                x_curve=x_arrays[sample_idx],
                y_curve=y_plot_arrays[sample_idx],
                peak_details=selected_peaks,
                sample_name=sample_names[sample_idx],
                occupied_boxes=occupied_boxes,
                occupied_segments=occupied_segments,
                fontsize=peak_label_fontsize,
                line_width=1.1,
                line_style="feature",
            )
        )

    png_path = output_dir / f"{output_name}.png"
    tiff_path = output_dir / f"{output_name}.tiff"
    csv_path = output_dir / f"{output_name}_peaks.csv"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    _log(logger, f"Saved figure: {png_path}")
    _log(logger, f"Saved figure: {tiff_path}")
    return export_peak_table(peak_records, csv_path, logger=logger)
