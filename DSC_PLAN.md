# DSC_PLAN

## 1. 新增 DSC 自动绘图功能时，建议修改哪些模块

### 需要修改的文件

#### `ftir_core.py`
建议作为 **核心算法与绘图主模块** 扩展 DSC 能力，原因：
- 当前它已经承担 FTIR 的数据读取、预处理、峰检测、单谱/多谱绘图、结果导出。
- DSC 新功能最适合沿用同样的“读取 -> 预处理 -> 绘图 -> 导出”主链路。
- GUI 和脚本入口都可以直接复用这里的新能力。

建议新增的 DSC 能力放在这里：
- `read_dsc_file()`
- `preprocess_dsc_curve()`
- `plot_single_dsc()`
- `plot_multi_dsc()`
- `export_dsc_table()` 或 `export_dsc_transition_table()`

#### `ftir_gui.py`
建议作为 **GUI 交互层** 扩展 DSC 入口，原因：
- 当前 GUI 已有完整的文件选择、参数输入、输出名、日志、按钮状态控制。
- DSC 如果需要在桌面界面里直接使用，必须在这里增加 DSC 模式与对应控件。

建议增加：
- FTIR / DSC 模式切换
- DSC 单图文件选择
- DSC 多图文件选择
- DSC 样品名、偏移量、输出名、方向参数、平滑参数等输入控件
- 调用 `ftir_core.py` 中的 DSC 函数

#### `ftir_tool.py`
建议作为 **配置/批处理入口** 扩展 DSC 调度，原因：
- 当前它负责基于配置文件执行单谱/多谱流程。
- 若 DSC 要支持自动化批量出图，必须在这里加入 DSC 配置解析与任务分发。

建议增加：
- `mode: ftir / dsc`
- `dsc.single`
- `dsc.multi`
- 调用 `ftir_core.py` 中的 DSC 核心函数，而不是再复制一套实现

---

## 2. 对当前项目结构的判断

从现状看，这个项目已经有两层结构：

### 第一层：核心能力层
- `ftir_core.py`
- 负责 FTIR 的主业务逻辑与图像输出

### 第二层：入口层
- `ftir_gui.py`：GUI 入口
- `ftir_tool.py`：脚本 / 配置入口

这个结构本身是适合接 DSC 的。

但有一个现实问题：
- `ftir_tool.py` 内部已经存在一套相对独立的 FTIR 读取/平滑/基线/绘图逻辑
- `ftir_core.py` 里也有自己的主实现

这说明当前项目有一定程度的“核心逻辑重复”。

所以 **新增 DSC 时最重要的原则** 是：
> 不要再在 `ftir_tool.py` 里复制第三套 DSC 逻辑。

应该尽量收敛为：
- `ftir_core.py` 负责核心实现
- `ftir_gui.py` 和 `ftir_tool.py` 只负责调用

---

## 3. 推荐的模块划分方案

### 方案 A：最小改动、最适合当前项目

#### `ftir_core.py`
继续作为统一核心模块，但内部拆成两组能力：

##### FTIR 子模块职责
- FTIR 文件读取
- FTIR 平滑与基线校正
- FTIR 峰位检测
- FTIR 单谱/多谱绘图
- FTIR 峰表导出

##### DSC 子模块职责
- DSC 文件读取
- DSC 曲线平滑与基线处理
- DSC 峰/转变点检测（后续可选）
- DSC 单曲线/多曲线绘图
- DSC 转变表导出

#### `ftir_gui.py`
保持纯界面层：
- 模式切换
- 参数输入
- 文件选择
- 日志显示
- 调用核心模块

#### `ftir_tool.py`
保持纯配置/调度层：
- 读取 JSON 配置
- 判断任务类型（FTIR / DSC）
- 调用核心模块

### 方案 B：中期更优的结构
如果后续 DSC 功能会继续扩展，建议第二阶段重构为：
- `ftir_core.py`：FTIR 核心
- `dsc_core.py`：DSC 核心
- `ftir_gui.py`：统一 GUI 入口
- `ftir_tool.py`：统一脚本入口
- `spectrum_common.py`：公共工具（平滑、导出、样式、命名等）

这个结构长期更干净，但不属于“最小改动”。

**结论：当前阶段更推荐方案 A。**

---

## 4. 推荐的 DSC 功能模块拆分

### 4.1 数据读取层
负责：
- 读取 DSC 原始文件
- 自动识别两列或多列数据
- 提取温度、热流、时间等字段
- 标准化列名

建议标准输出格式：
- `Temperature`
- `HeatFlow`
- `Time`（可选）

建议函数：
- `read_dsc_file(file_path)`

### 4.2 预处理层
负责：
- 曲线平滑
- 基线校正
- 温度轴排序
- 热流方向统一（吸热向上 / 吸热向下）
- 升温段/降温段筛选（如果文件包含多段）

建议函数：
- `smooth_dsc_signal()`
- `baseline_correct_dsc()`
- `normalize_dsc_direction()`
- `preprocess_dsc_curve()`

### 4.3 绘图层
负责：
- 单个 DSC 曲线绘图
- 多个 DSC 曲线堆叠绘图
- 峰顶、起始点、终止点、Tg/Tm/Tc 标注
- 输出 PNG / TIFF
- 导出 CSV

建议函数：
- `plot_single_dsc()`
- `plot_multi_dsc()`
- `export_dsc_transition_table()`

### 4.4 入口层
GUI 与脚本都只做：
- 参数收集
- 模式切换
- 调用核心
- 错误展示

不应再承载 DSC 算法实现。

---

## 5. 最小改动实施步骤

### 第一步：先在 `ftir_core.py` 打通 DSC 基础闭环
目标：先实现“能读、能画、能导出”。

建议先做：
1. `read_dsc_file()`
2. `plot_single_dsc()`
3. `plot_multi_dsc()`

这一步先不强求：
- 自动检测 Tg
- 自动检测 Tm / Tc
- 高级基线拟合
- 焓积分

原因：
- 基础绘图链路先跑通，风险最低
- 也最容易被 GUI 和脚本入口复用

### 第二步：在 `ftir_gui.py` 中增加 DSC 模式
建议增加：
- 模式选择：FTIR / DSC
- DSC 单文件选择按钮
- DSC 多文件选择按钮
- DSC 输出名
- DSC 样品名
- DSC 偏移量
- DSC 方向参数（吸热向上/向下）
- DSC 平滑参数

这一步中 GUI 只负责调用：
- `ftir_core.plot_single_dsc()`
- `ftir_core.plot_multi_dsc()`

### 第三步：在 `ftir_tool.py` 中增加 DSC 配置模式
建议配置结构示例：

```json
{
  "mode": "dsc",
  "dsc": {
    "run_single_mode": true,
    "run_multi_mode": true,
    "single": {
      "file": "sample1.txt",
      "output_name": "dsc_single"
    },
    "multi": {
      "files": ["a.txt", "b.txt"],
      "sample_names": ["A", "B"],
      "output_name": "dsc_multi"
    }
  }
}
```

`ftir_tool.py` 中需要做的只是：
- 读配置
- 判断 `mode`
- 转发参数给核心函数

### 第四步：补自动分析能力
等基础图稳定后，再逐步加入：
- DSC 峰检测
- onset / endset 自动计算
- Tg 拐点检测
- 焓面积积分
- 自动标注规则
- 导出 transition table

这部分适合作为第二阶段迭代，不建议在第一版就全部并入。

---

## 6. 最小改动下的推荐实现路线

### 路线建议

#### 路线 1：最推荐
- `ftir_core.py`：新增 DSC 核心函数
- `ftir_gui.py`：新增 DSC GUI 入口
- `ftir_tool.py`：新增 DSC 配置调度

优点：
- 接入成本最低
- 最符合现有工程结构
- GUI 与脚本都能复用同一套 DSC 核心逻辑

缺点：
- `ftir_core.py` 会继续变大

#### 路线 2：下一阶段优化
- 先按路线 1 接入 DSC
- 等功能稳定后，再抽离 `dsc_core.py` 和公共工具模块

这个路线最符合实际工程节奏。

---

## 7. 工程化建议

### 7.1 不要在 `ftir_tool.py` 中复制 DSC 逻辑
`ftir_tool.py` 现在已经偏重，如果再复制一套 DSC，会进一步加剧维护分叉。

### 7.2 优先统一输出接口
建议 FTIR / DSC 最终都采用类似接口：
- `plot_single_xxx(...)`
- `plot_multi_xxx(...)`
- `export_xxx_table(...)`

这样 GUI 和脚本层更容易统一。

### 7.3 把方向参数做成可配置项
DSC 图常见差异：
- 吸热向上
- 吸热向下

这不应写死，建议做成参数：
- `heatflow_direction='endo_up' | 'endo_down'`

### 7.4 先实现“基础自动绘图”，再做“自动热分析”
“自动绘图”与“自动判峰/判转变”难度不是一个量级。
建议先完成：
- 文件读取
- 单图/多图绘制
- 输出文件导出

然后再迭代自动分析。

---

## 8. 最终建议

如果你现在就准备加 DSC 自动绘图，最合理的改动点是：
- `ftir_core.py`：新增 DSC 核心能力
- `ftir_gui.py`：新增 DSC 界面入口
- `ftir_tool.py`：新增 DSC 配置入口

最推荐的模块职责边界是：
- **核心逻辑放 `ftir_core.py`**
- **界面逻辑放 `ftir_gui.py`**
- **配置调度放 `ftir_tool.py`**

最小改动实施顺序是：
1. 先实现 `read_dsc_file()` / `plot_single_dsc()` / `plot_multi_dsc()`
2. 再给 GUI 加 DSC 模式
3. 再给 `ftir_tool.py` 加 DSC 配置入口
4. 最后补自动峰/转变点分析

这个路线改动最小、可控性最好，也最适合你现在这个项目结构。
