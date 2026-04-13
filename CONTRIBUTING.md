# Contributing

感谢你愿意改进这个项目。

## 提交前建议

1. 先开一个分支，不要直接在 `main` 上开发。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 提交前至少运行：

```bash
python -m py_compile ftir_core.py ftir_gui.py ftir_tool.py
```

## 代码约定

- 尽量不要改动已经验证稳定的 FTIR 数据处理逻辑，除非问题定位明确。
- GUI 修改要保持中文界面一致性。
- 生成文件、临时状态、本地原始数据不要提交。
- 新增功能时，优先保持论文绘图风格一致。

## Pull Request 建议内容

- 改了什么
- 为什么要改
- 如何验证
- 是否影响旧行为

## 不建议提交的内容

- `output/`
- `raw_data/` 中的私人实验数据
- PyInstaller `build/` / `dist/`
- `.ftir_gui_state.json`
