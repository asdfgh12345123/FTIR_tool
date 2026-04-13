# FTIR Tool

面向材料、高分子、阻燃等科研场景的 FTIR 光谱绘图工具，支持单谱绘图、多谱 `stacked spectra` 对比、峰位标注、PNG/TIFF/CSV 导出，以及 Windows 桌面 GUI 使用。

## 主要功能

- 支持 `txt`、`csv`、`dat` 光谱文件
- 自动识别 `##YUNITS=%T` / `Abs`，并在需要时自动转换为 `%T`
- Savitzky-Golay 平滑与 baseline correction
- 单谱与多谱 stacked FTIR 绘图
- 自动峰位标注与峰位表导出
- Tkinter 图形界面：`ftir_gui.py`
- 600 dpi PNG / TIFF 输出

## 项目结构

```text
FTIR_tool/
├─ ftir_core.py            # 核心数据处理与绘图逻辑
├─ ftir_gui.py             # Tkinter 图形界面入口
├─ ftir_tool.py            # 兼容脚本入口
├─ requirements.txt        # 运行依赖
├─ raw_data/               # 本地原始光谱数据（默认不建议提交）
├─ output/                 # 本地导出结果（默认忽略）
└─ .github/                # Issue / PR / CI / Release 配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 GUI

```bash
python ftir_gui.py
```

### 3. 导入数据

支持两列光谱数据：

```text
Wavenumber  Intensity
4000        78.1
3998        78.2
...
```

也支持带头信息的格式：

```text
##YUNITS=%T
4000 78.1
3998 78.2
```

或：

```text
##YUNITS=Abs
4000 0.312
3998 0.315
```

## 输出内容

程序默认导出到 `output/`：

- `*.png`
- `*.tiff`
- `*_peaks.csv`

## 开发与检查

提交前建议至少执行：

```bash
python -m py_compile ftir_core.py ftir_gui.py ftir_tool.py
```

## 打包 EXE

```bash
pyinstaller --onefile --windowed ftir_gui.py
```

如需更稳妥地包含 `matplotlib` 资源：

```bash
pyinstaller --onefile --windowed --collect-all matplotlib ftir_gui.py
```

## 许可证

本项目使用 [MIT License](LICENSE)。

## 贡献

欢迎通过 Issue 和 Pull Request 反馈问题或改进想法。提交前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 仓库说明

- `raw_data/`、`output/`、打包产物、临时状态文件默认不建议提交到 Git
- GitHub 已配置 Issue 模板、PR 模板、基础 CI 和基于 tag 的 Release 工作流
- 适合作为桌面科研绘图工具长期维护和公开协作

## 发布

推荐使用 tag 触发正式 Release：

```bash
git tag v1.0.1
git push origin v1.0.1
```

推送 tag 后，GitHub Actions 会自动：

- 构建 Windows 可执行文件 `FTIR_Tool.exe`
- 打包 `FTIR_Tool_Windows_Portable.zip`
- 自动创建 GitHub Release 并上传附件

Windows 用户请优先下载 Release 里的：

- `FTIR_Tool_Windows_Portable.zip`

不要下载 GitHub 自动生成的：

- `Source code (zip)`
- `Source code (tar.gz)`

这两个是源码，不是可直接运行的程序。
