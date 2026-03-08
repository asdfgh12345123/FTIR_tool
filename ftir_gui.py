import json
import os
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional

import ftir_core


DEFAULT_PEAK_TEXT = ",".join(str(int(v)) for v in ftir_core.DEFAULT_CANDIDATE_PEAKS)
DEFAULT_MULTI_OUTPUT_NAME = "ftir_multi_gui"
DEFAULT_OFFSET_STEP = 35.0
GUI_STATE_FILE = ".ftir_gui_state.json"
SUPPORTED_FILETYPES = [("FTIR data", "*.txt *.csv *.dat"), ("Text files", "*.txt"), ("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]


class FTIRGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FTIR 光谱绘图工具")
        self.root.geometry("1180x860")
        self.root.minsize(1020, 760)

        self.base_dir = Path(__file__).resolve().parent
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.base_dir / GUI_STATE_FILE
        self.last_open_dir = self._load_last_open_dir()
        self.last_auto_offsets_text: Optional[str] = None
        self.last_default_single_output_name = ""

        self.single_file: Optional[Path] = None
        self.multi_files: List[Path] = []

        self._configure_style()
        self._build_ui()
        self.restore_defaults(log_message=False)
        self._update_button_states()

        self.log(f"输出目录：{self.output_dir}")
        self.log("程序已启动。")

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        font_name = "Microsoft YaHei UI"
        style.configure(".", font=(font_name, 10))
        style.configure("TLabelframe.Label", font=(font_name, 11, "bold"))
        style.configure("TButton", padding=(10, 4))
        style.configure("TLabel", padding=(0, 1))

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        single_frame = ttk.LabelFrame(main, text="单谱模式", padding=12)
        single_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(single_frame, text="选择 FTIR 文件", command=self.select_single_file).grid(row=0, column=0, sticky="w")

        self.single_file_var = tk.StringVar(value="未选择文件")
        ttk.Entry(single_frame, textvariable=self.single_file_var, state="readonly").grid(
            row=0, column=1, columnspan=3, padx=(10, 0), sticky="ew"
        )

        ttk.Label(single_frame, text="目标峰位：").grid(row=1, column=0, pady=(10, 0), sticky="w")
        self.single_peaks_var = tk.StringVar()
        ttk.Entry(single_frame, textvariable=self.single_peaks_var).grid(
            row=1, column=1, padx=(10, 0), pady=(10, 0), sticky="ew"
        )

        ttk.Label(single_frame, text="单谱输出文件名：").grid(row=1, column=2, padx=(12, 0), pady=(10, 0), sticky="w")
        self.single_output_name_var = tk.StringVar()
        ttk.Entry(single_frame, textvariable=self.single_output_name_var).grid(
            row=1, column=3, padx=(10, 0), pady=(10, 0), sticky="ew"
        )

        self.single_generate_btn = ttk.Button(single_frame, text="生成单谱图", command=self.generate_single_spectrum)
        self.single_generate_btn.grid(row=2, column=0, columnspan=4, pady=(12, 0), sticky="w")

        single_frame.columnconfigure(1, weight=1)
        single_frame.columnconfigure(3, weight=1)

        multi_frame = ttk.LabelFrame(main, text="多谱模式（堆叠对比）", padding=12)
        multi_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(multi_frame, text="选择多个 FTIR 文件", command=self.select_multi_files).grid(row=0, column=0, sticky="nw")

        self.multi_files_text = ScrolledText(multi_frame, height=5, wrap="word")
        self.multi_files_text.grid(row=0, column=1, columnspan=3, padx=(10, 0), sticky="ew")
        self._set_text_readonly(self.multi_files_text)
        self._write_readonly_text(self.multi_files_text, "未选择文件")

        ttk.Label(multi_frame, text="样品名称（用英文逗号分隔）：").grid(row=1, column=0, pady=(10, 0), sticky="w")
        self.sample_names_var = tk.StringVar()
        ttk.Entry(multi_frame, textvariable=self.sample_names_var).grid(
            row=1, column=1, padx=(10, 0), pady=(10, 0), sticky="ew"
        )
        self.auto_fill_names_btn = ttk.Button(
            multi_frame,
            text="根据文件名自动填充样品名称",
            command=self.fill_sample_names_from_files,
        )
        self.auto_fill_names_btn.grid(row=1, column=2, columnspan=2, padx=(10, 0), pady=(10, 0), sticky="w")

        ttk.Label(multi_frame, text="垂直偏移量（用英文逗号分隔）：").grid(row=2, column=0, pady=(10, 0), sticky="w")
        self.offsets_var = tk.StringVar()
        ttk.Entry(multi_frame, textvariable=self.offsets_var).grid(
            row=2, column=1, padx=(10, 0), pady=(10, 0), sticky="ew"
        )

        ttk.Label(multi_frame, text="多谱输出文件名：").grid(row=2, column=2, padx=(12, 0), pady=(10, 0), sticky="w")
        self.multi_output_name_var = tk.StringVar()
        ttk.Entry(multi_frame, textvariable=self.multi_output_name_var).grid(
            row=2, column=3, padx=(10, 0), pady=(10, 0), sticky="ew"
        )

        ttk.Label(multi_frame, text="阻燃材料候选峰位池：").grid(row=3, column=0, pady=(10, 0), sticky="nw")
        self.multi_peaks_text = ScrolledText(multi_frame, height=5, wrap="word")
        self.multi_peaks_text.grid(row=3, column=1, columnspan=3, padx=(10, 0), pady=(10, 0), sticky="ew")

        ttk.Button(
            multi_frame,
            text="加载阻燃材料常见峰模板",
            command=self.fill_flame_retardant_candidates,
        ).grid(row=4, column=0, pady=(12, 0), sticky="w")

        self.multi_generate_btn = ttk.Button(multi_frame, text="生成多谱图", command=self.generate_multi_spectrum)
        self.multi_generate_btn.grid(row=4, column=1, pady=(12, 0), padx=(10, 0), sticky="w")

        multi_frame.columnconfigure(1, weight=1)
        multi_frame.columnconfigure(3, weight=1)

        tools_frame = ttk.LabelFrame(main, text="工具与日志", padding=12)
        tools_frame.pack(fill="both", expand=True)

        button_row = ttk.Frame(tools_frame)
        button_row.pack(fill="x")

        ttk.Button(button_row, text="打开输出文件夹", command=self.open_output_folder).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="恢复默认参数", command=self.restore_defaults).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="清空日志", command=self.clear_log).pack(side="left")

        self.log_text = ScrolledText(tools_frame, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(10, 0))
        self._set_text_readonly(self.log_text)

    def _set_text_readonly(self, widget: ScrolledText) -> None:
        widget.configure(state="disabled")

    def _write_readonly_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _set_multi_candidate_text(self, text: str) -> None:
        self.multi_peaks_text.delete("1.0", "end")
        self.multi_peaks_text.insert("1.0", text)

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log("日志已清空。")

    def _load_last_open_dir(self) -> Path:
        if self.state_file.exists():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                last_dir = Path(state.get("last_open_dir", ""))
                if last_dir.exists():
                    return last_dir
            except Exception:
                pass
        return self.base_dir

    def _save_last_open_dir(self, folder: Path) -> None:
        self.last_open_dir = folder
        state = {"last_open_dir": str(folder)}
        try:
            with self.state_file.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            self.log("无法保存上一次打开目录。")

    def _update_button_states(self) -> None:
        if self.single_file is None:
            self.single_generate_btn.state(["disabled"])
        else:
            self.single_generate_btn.state(["!disabled"])

        if self.multi_files:
            self.multi_generate_btn.state(["!disabled"])
            self.auto_fill_names_btn.state(["!disabled"])
        else:
            self.multi_generate_btn.state(["disabled"])
            self.auto_fill_names_btn.state(["disabled"])

    def _default_single_output_name(self) -> str:
        if self.single_file is None:
            return "ftir_single"
        return f"{self.single_file.stem}_single"

    def _default_multi_output_name(self) -> str:
        if not self.multi_files:
            return DEFAULT_MULTI_OUTPUT_NAME
        parts = [path.stem for path in self.multi_files]
        joined = "_vs_".join(parts)
        joined = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', "_", joined)
        joined = joined.strip("_")
        if not joined:
            return DEFAULT_MULTI_OUTPUT_NAME
        return f"ftir_multi_{joined}"[:120]

    def _format_offsets_text(self, offsets: List[float]) -> str:
        parts: List[str] = []
        for offset in offsets:
            if float(offset).is_integer():
                parts.append(str(int(offset)))
            else:
                parts.append(str(offset))
        return ",".join(parts)

    def _generate_default_offsets(self, n_files: int) -> List[float]:
        return [DEFAULT_OFFSET_STEP * (n_files - 1 - i) for i in range(n_files)]

    def _apply_auto_offsets(self, n_files: int, reason: Optional[str] = None) -> List[float]:
        offsets = self._generate_default_offsets(n_files)
        offsets_text = self._format_offsets_text(offsets)
        self.offsets_var.set(offsets_text)
        self.last_auto_offsets_text = offsets_text
        self.log(f"自动生成垂直偏移量：{offsets_text}")
        if reason:
            self.log(reason)
        return offsets

    def _sanitize_output_name(self, text: str, fallback: str) -> str:
        name = text.strip()
        if not name:
            name = fallback
        name = re.sub(r'[<>:"/\\|?*]+', "_", name)
        name = name.strip().strip(".")
        return name or fallback

    def _resolve_unique_output_name(self, base_name: str) -> str:
        png_path = self.output_dir / f"{base_name}.png"
        tiff_path = self.output_dir / f"{base_name}.tiff"
        csv_path = self.output_dir / f"{base_name}_peaks.csv"
        if not png_path.exists() and not tiff_path.exists() and not csv_path.exists():
            return base_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{base_name}_{timestamp}"
        self.log(f"输出文件名已存在，已自动添加时间戳：{unique_name}")
        return unique_name

    def restore_defaults(self, log_message: bool = True) -> None:
        self.single_peaks_var.set(DEFAULT_PEAK_TEXT)
        self.single_output_name_var.set(self._default_single_output_name() if self.single_file else "")

        if self.multi_files:
            self.sample_names_var.set(",".join(path.stem for path in self.multi_files))
            self._apply_auto_offsets(len(self.multi_files))
            self.multi_output_name_var.set(self._default_multi_output_name())
        else:
            self.sample_names_var.set("")
            self.offsets_var.set("")
            self.last_auto_offsets_text = None
            self.multi_output_name_var.set(DEFAULT_MULTI_OUTPUT_NAME)

        self._set_multi_candidate_text(DEFAULT_PEAK_TEXT)

        if log_message:
            self.log("已恢复默认参数。")

    def fill_sample_names_from_files(self) -> None:
        if not self.multi_files:
            messagebox.showwarning("提示", "请先选择多个 FTIR 文件。")
            return
        sample_names = ",".join(path.stem for path in self.multi_files)
        self.sample_names_var.set(sample_names)
        self.log("已根据文件名自动填充样品名称。")

    def fill_flame_retardant_candidates(self) -> None:
        self._set_multi_candidate_text(DEFAULT_PEAK_TEXT)
        self.log("已加载阻燃材料常见峰模板。")

    def select_single_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择 FTIR 文件",
            initialdir=str(self.last_open_dir),
            filetypes=SUPPORTED_FILETYPES,
        )
        if not file_path:
            return

        self.single_file = Path(file_path)
        self.single_file_var.set(str(self.single_file))
        self._save_last_open_dir(self.single_file.parent)
        self._update_button_states()

        new_default_output = self._default_single_output_name()
        current_output = self.single_output_name_var.get().strip()
        if not current_output or current_output == self.last_default_single_output_name:
            self.single_output_name_var.set(new_default_output)
        self.last_default_single_output_name = new_default_output
        self.log(f"已选择单谱文件：{self.single_file}")

    def select_multi_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择多个 FTIR 文件",
            initialdir=str(self.last_open_dir),
            filetypes=SUPPORTED_FILETYPES,
        )
        if not files:
            return

        self.multi_files = [Path(file_path) for file_path in files]
        self._write_readonly_text(self.multi_files_text, "\n".join(str(path) for path in self.multi_files))
        self._save_last_open_dir(self.multi_files[0].parent)
        self._update_button_states()
        self.fill_sample_names_from_files()

        current_offsets_text = self.offsets_var.get().strip()
        if not current_offsets_text or current_offsets_text == self.last_auto_offsets_text:
            self._apply_auto_offsets(len(self.multi_files))

        current_candidates = self.multi_peaks_text.get("1.0", "end").strip()
        if not current_candidates:
            self._set_multi_candidate_text(DEFAULT_PEAK_TEXT)
            self.log("已自动填入阻燃材料常见峰模板。")

        if not self.multi_output_name_var.get().strip() or self.multi_output_name_var.get().strip() == DEFAULT_MULTI_OUTPUT_NAME:
            self.multi_output_name_var.set(self._default_multi_output_name())

        self.log("已选择多谱文件：")
        for path in self.multi_files:
            self.log(f"  - {path}")

    @staticmethod
    def _split_csv_text(text: str) -> List[str]:
        return [item.strip() for item in re.split(r"[,，]", text.strip()) if item.strip()]

    def _parse_float_list(self, text: str, field_name: str) -> List[float]:
        parts = self._split_csv_text(text)
        if not parts:
            raise ValueError(f"{field_name}不能为空。")
        try:
            return [float(item) for item in parts]
        except ValueError as exc:
            raise ValueError(f"{field_name}格式错误，请输入数字并使用英文逗号分隔。") from exc

    def _parse_single_peaks(self) -> List[float]:
        text = self.single_peaks_var.get().strip()
        if not text:
            return self._parse_float_list(DEFAULT_PEAK_TEXT, "目标峰位")
        return self._parse_float_list(text, "目标峰位")

    def _parse_sample_names(self, n_files: int) -> List[str]:
        text = self.sample_names_var.get().strip()
        if not text:
            return [path.stem for path in self.multi_files]
        names = self._split_csv_text(text)
        if len(names) != n_files:
            raise ValueError("样品名称数量必须与已选择文件数量一致。")
        return names

    def _parse_offsets(self, n_files: int) -> Optional[List[float]]:
        text = self.offsets_var.get().strip()
        if not text or text == self.last_auto_offsets_text:
            self.log("未手动指定垂直偏移量，系统将根据谱线振幅自动计算。")
            return None

        offsets = self._parse_float_list(text, "垂直偏移量")
        if len(offsets) < n_files:
            self.log(
                f"你选择了 {n_files} 个 FTIR 文件，但只填写了 {len(offsets)} 个偏移量，系统将改为自动计算。"
            )
            messagebox.showinfo("提示", "未填写完整偏移量，系统将自动计算。")
            return None
        if len(offsets) > n_files:
            raise ValueError("垂直偏移量数量不能多于已选择文件数量。")
        return offsets

    def _parse_candidate_peaks(self) -> List[float]:
        text = self.multi_peaks_text.get("1.0", "end").strip()
        if not text:
            self._set_multi_candidate_text(DEFAULT_PEAK_TEXT)
            self.log(f"未填写候选峰位池，系统已自动填入默认模板：{DEFAULT_PEAK_TEXT}")
            text = DEFAULT_PEAK_TEXT

        raw_parts = [item.strip() for item in re.split(r"[\s,，;；|]+", text) if item.strip()]
        if not raw_parts:
            raise ValueError("阻燃材料候选峰位池不能为空。")
        try:
            values = [float(item) for item in raw_parts]
        except ValueError as exc:
            raise ValueError("候选峰位格式错误，请输入数字并使用逗号、空格或分号分隔。") from exc

        candidate_peaks: List[float] = []
        seen = set()
        for value in values:
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            candidate_peaks.append(float(value))
        return candidate_peaks

    def _show_error(self, exc: Exception, fallback_message: str) -> None:
        if isinstance(exc, (ValueError, FileNotFoundError)):
            message = str(exc)
        else:
            message = fallback_message
        self.log(f"[错误] {message}")
        messagebox.showerror("错误", message)

    def _preview_png(self, png_path: Path) -> None:
        try:
            os.startfile(str(png_path))
        except Exception as exc:
            self.log(f"自动打开图片失败：{exc}")

    def generate_single_spectrum(self) -> None:
        if self.single_file is None:
            messagebox.showerror("错误", "请先选择一个 FTIR 文件。")
            return

        try:
            target_peaks = self._parse_single_peaks()
            requested_name = self._sanitize_output_name(
                self.single_output_name_var.get(),
                self._default_single_output_name(),
            )
            output_name = self._resolve_unique_output_name(requested_name)
            png_path = self.output_dir / f"{output_name}.png"

            self.log("开始生成单谱图...")
            ftir_core.plot_single_ftir(
                file_path=self.single_file,
                output_dir=self.output_dir,
                target_peaks=target_peaks,
                output_name=output_name,
                logger=self.log,
            )
            self.log("已成功生成单谱图。")
            self.log("结果已保存到 output 文件夹。")
            self._preview_png(png_path)
            messagebox.showinfo("完成", "绘图完成，结果已保存到 output 文件夹。")
        except Exception as exc:
            self._show_error(exc, "生成单谱图失败。")

    def generate_multi_spectrum(self) -> None:
        if not self.multi_files:
            messagebox.showerror("错误", "请先选择多个 FTIR 文件。")
            return

        try:
            n_files = len(self.multi_files)
            sample_names = self._parse_sample_names(n_files)
            offsets = self._parse_offsets(n_files)
            candidate_peaks = self._parse_candidate_peaks()

            requested_name = self._sanitize_output_name(
                self.multi_output_name_var.get(),
                self._default_multi_output_name(),
            )
            output_name = self._resolve_unique_output_name(requested_name)
            png_path = self.output_dir / f"{output_name}.png"

            self.log("开始生成多谱图...")
            self.log("当前候选峰位池：" + ", ".join(str(int(round(peak))) for peak in candidate_peaks))
            ftir_core.plot_multi_ftir(
                file_list=self.multi_files,
                sample_names=sample_names,
                vertical_offsets=offsets,
                target_peak_lists=candidate_peaks,
                output_dir=self.output_dir,
                output_name=output_name,
                logger=self.log,
            )
            self.log("已成功生成多谱图。")
            self.log("结果已保存到 output 文件夹。")
            self._preview_png(png_path)
            messagebox.showinfo("完成", "绘图完成，结果已保存到 output 文件夹。")
        except Exception as exc:
            self._show_error(exc, "生成多谱图失败。")

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.output_dir))
        except Exception as exc:
            messagebox.showerror("错误", f"无法打开输出文件夹：{exc}")


def main() -> None:
    root = tk.Tk()
    FTIRGuiApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
