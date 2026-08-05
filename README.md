# GTJA 191 Factors Backtest — Chinese A-Share

基于国泰君安《基于短周期价量特征的多因子选股体系》中的 191 个 Alpha 因子，针对中国 A 股市场进行因子复现、IC 检验与 Portfolio Sort 分组回测。

---

## 项目简介

本项目实现国泰君安 191 个短周期量价 Alpha，并在沪深300与中证500历史成分股并集上检验因子的截面选股能力。

项目包含三个核心部分：

- 使用 Tushare 下载并整理 A 股行情、复权因子、估值、市值、指数、行业及 ST 相关数据；
- 使用统一注册表计算 Alpha001—Alpha191；
- 使用 IC Decay、五分组 Portfolio Sort、Rank IC 和多空组合 t 检验评价因子有效性。

数据来源：

- 个股量价数据：Tushare `daily`；
- 复权因子：Tushare `adj_factor`；
- 总市值、PE/PB：Tushare `daily_basic`；
- 沪深300、中证500成分及权重：Tushare `index_weight`；
- 中证800指数行情：Tushare `index_daily`；

---

## 范围及条件

1. 原始数据范围为 2022—2025 年 A 股交易日；正式回测区间默认为 2023—2025 年，2022 年数据用于滚动窗口预热。
2. 股票池为每半年更新一次的沪深300与中证500历史成分股并集，理论容量约 800 只。
3. 股票池快照锚定每年 6 月末和 12 月末；每个信号日使用不晚于该日的最近一期快照。
4. 当前正式回测默认按周换仓，每个自然周最后一个实际交易日产生信号。
5. 个股量价数据采用宽表保存：`index = trade_date`，`columns = ts_code`。
6. 行情字段包括 `close`、`open`、`high`、`low`、`vol`、`amount` 和 `adj_factor`。
7. Tushare 单位：`amount` 为千元，`vol` 为手，每手 100 股。
8. Alpha30 额外使用每日总市值、PE-TTM和中证800指数数据，在函数内部构造 MKT、SMB、HML_EP 三因子。

---

## 回测标准

建议按以下标准筛选有效因子：

1. IC 均值绝对值大于 0.02；
2. ICIR 绝对值大于 0.30；
3. IC t 统计量显著，且有效样本数大于 50 期；
4. 因子方向确定后，多空组合收益为正且 t 统计量显著；
5. Q1—Q5 分组收益具有稳定单调性；
6. 因子在不同年份和市场状态下方向基本稳定；
7. 负 IC 不直接代表因子无效：若 IC 长期显著为负，应在训练期确定反向使用，而不能根据完整回测期事后翻转。

---

## 回测结果

### 显著有效因子

010

### IC 显著但分组收益不单调

待补充。

### IC显著但是多空组合t检验不显著

030

---

## 项目结构

```text
gtja-191-factors-backtest-Chinese-A-share/
├── tushare取数 - 副本.py
│   ├── 下载个股 OHLCV、成交额和复权因子
│   ├── 下载总市值、PE-TTM、PB
│   └── 下载中证800指数行情
├── 国泰君安191因子库.py
│   ├── ALPHA_REGISTRY 因子注册表
│   ├── 通用时序与截面算子
│   ├── Alpha001—Alpha191
│   └── Alpha30 三因子构造与滚动回归
├── 量价前800portfolio sorting_自动.py
│   ├── 历史指数成分股股票池
│   ├── IC Decay
│   ├── Portfolio Sort
│   ├── Rank IC 与 t 检验
│   └── Excel 图表报告
├── data/
│   ├── close_pivot_22-25.csv
│   ├── open_pivot_22-25.csv
│   ├── high_pivot_22-25.csv
│   ├── low_pivot_22-25.csv
│   ├── vol_pivot_22-25.csv
│   ├── amount_pivot_22-25.csv
│   ├── adj_factor_pivot_22-25.csv
│   ├── market_value_22-25.csv
│   ├── pe_ttm22-25.csv
│   └── 000906_SH_22-25.csv
├── results/
│   ├── alpha001_factor_report.xlsx
│   ├── ...
│   └── alpha191_factor_report.xlsx
├── README.md
└── LICENSE
```

---

## 回测方法

```text
Tushare 数据下载
    ↓
宽表读取与字段对齐
    ↓
VWAP及前复权价格计算
    ↓
沪深300＋中证500历史股票池
    ↓
逐因子循环
    ├── run_mode=1：IC Decay分析
    └── run_mode=2：Portfolio Sort → IC → t检验 → 报告输出
```

### 一、数据下载与宽表整理

`tushare取数 - 副本.py` 提供以下数据接口：

- `fetch_price`：按交易日下载个股行情字段，并转换为日期×股票的宽表；
- `fetch_adj_factor_pivot`：下载复权因子宽表；
- `fetch_daily_basic`：下载总市值、PE-TTM或PB宽表；
- `get_index_fixdate`：下载指定宽基指数日行情；

宽表统一格式：

```text
index   = trade_date
columns = ts_code
values  = 对应行情、估值或市值字段
```

代码默认使用 Tushare 股票代码，例如：

```text
600000.SH
000001.SZ
```

### 二、数据预处理

#### 前复权价格

原始行情为未复权价格，当前代码按以下方式生成前复权序列：

```text
前复权价格 = 原始价格 × 当日 adj_factor ÷ 样本最后一期 adj_factor
```

`close`、`open`、`high`、`low` 和 `vwap` 使用相同复权方式；`volume` 和 `amount` 保持原始值。

> 注意：以完整样本最后一期复权因子归一化，会在价格绝对水平或跨股票价格比较类因子中引入未来公司行动信息。严格的无前视研究应改用点时可得复权口径，或验证目标因子对股票级缩放是否不敏感。

#### VWAP

Tushare 中 `amount` 单位为千元，`vol` 单位为手，因此：

```text
VWAP = amount × 1000 ÷ (vol × 100)
```

所得单位为元/股。

#### 多字段对齐

主程序计算所有基础行情表的公共日期和公共股票列，再按相同的 `index/columns` 顺序重排，避免 Pandas 在跨字段计算时产生隐式错位。

### 三、历史股票池

`get_semiannual_snapshots` 每年在以下两个锚点获取指数成分：

- 6 月30日前后；
- 12月31日前后。

每个锚点分别获取：

- 沪深300：`000300.SH`；
- 中证500：`000905.SH`。

两者取并集形成约800只股票的历史股票池。每个信号日通过 `get_universe_for_date` 选择不晚于该日的最近一期快照。

为避免前视偏差，正式研究应保证回测开始前已经存在至少一期历史快照，不应使用未来成分股快照回填早期股票池。

### 四、因子库与注册表

`国泰君安191因子库.py` 使用 `ALPHA_REGISTRY` 管理因子函数和输入字段。例如：

```python
ALPHA_REGISTRY = {
    "alpha001": {
        "func": lambda dfs: alpha001(...),
        "required": [...],
    },
    # ...
    "alpha191": {
        "func": lambda dfs: alpha191(...),
        "required": [...],
    },
}
```

主程序通过 `ALPHA_RANGE` 选择需要计算的因子：

```python
ALPHA_RANGE = range(1, 192)  # 全部191个因子
```

也可以仅测试指定因子：

```python
ALPHA_RANGE = [30, 75, 149]
```

### 五、Alpha30 三因子处理

Alpha30 依赖 MKT、SMB 和 HML。当前实现仅在计算 Alpha30 时调用 `build_ff3_factors`，不为其他因子重复构造三因子。

当前口径：

1. MKT 使用中证800指数 `000906.SH` 的普通日收益率；
2. 每月最后一个交易日进行独立 2×2 排序；
3. 按总市值中位数分为 Small 和 Big；
4. 按 `1 / PE-TTM` 中位数分为 High 和 Low；
5. 形成 SH、SL、BH、BL 四个组合；
6. 四组合使用前一交易日总市值加权；
7. 分组从下一个交易日起生效；
8. SMB为小盘组合相对大盘组合收益；
9. HML为高E/P组合相对低E/P组合收益。

由于价值指标使用 `1 / PE-TTM`，这里的 HML 实际是 `HML_EP`，不是标准账面市值比 HML。若改用PB数据，应传入 `1 / PB` 对应的价值口径。

随后，Alpha30 使用个股日收益对三因子进行带截距的60日滚动回归，取窗口最后一天残差；残差平方后计算20日WMA。

### 六、IC Decay分析

`run_mode=1` 时调用 `calc_ic_decay`。

对每个 `lag = 1...max_lag`：

1. 计算从信号日收盘到未来第 lag 日收盘的累计收益；
2. 每日计算因子截面值与未来收益的 Spearman Rank IC；
3. 汇总 IC均值、IC标准差、t统计量和p值；
4. 根据显著预测期限给出周度或月度换仓建议。

当前建议规则：

- 最大显著 lag 不超过5日：建议周度换仓；
- 最大显著 lag 不超过20日：建议月度换仓；
- 全部不显著：因子可能无效。

### 七、换仓日程

`build_schedule` 根据真实交易日生成：

```text
(signal_date, ret_start, ret_end)
```

支持：

- `freq='D'`：每个交易日；
- `freq='W'`：每个自然周最后一个实际交易日；
- `freq='ME'`：每个自然月最后一个实际交易日。

当前正式回测默认：

```python
freq = "W"
```

### 八、因子五分组

`factor_group_from_panel` 在每个信号日执行：

1. 取当日因子截面；
2. 过滤至当期指数股票池；
3. 删除缺失因子值；
4. 有效股票少于50只时跳过；
5. 对因子值做截面百分比排名；
6. 使用 `qcut` 均匀划分 Q1—Q5。

分组含义：

```text
Q1 = 因子值最低组
Q5 = 因子值最高组
```

若因子截面仅包含 `-1` 和 `1` 两种取值，则直接映射到Q1和Q5，跳过五分位分箱。

### 九、持仓收益与组合收益

`calc_adj_returns` 使用复权收盘价计算持仓收益：

```text
持仓收益 = ret_end复权收盘价 ÷ ret_start复权收盘价 - 1
```

其中：

- `signal_date` 为本期最后一个交易日；
- `ret_start` 为信号日后的第一个交易日；
- `ret_end` 为下一期换仓日；
- 当前实现相当于在 `ret_start` 收盘成交，而不是严格的T+1开盘成交。

`calc_quantile_returns` 对Q1—Q5分别计算组内股票等权平均收益：

```text
Q5-Q1 = Q5收益 - Q1收益
```

若一个因子长期呈显著负 IC，应单独记录其预测方向；不能仅因为 `Q5-Q1` 为负就判定无效。

### 十、主回测循环

`run_portfolio_sort` 的流程为：

```text
for (signal_date, ret_start, ret_end) in schedule:
    1. 获取信号日历史股票池
    2. 读取因子截面并分成Q1—Q5
    3. 计算下一持仓期个股收益
    4. 汇总各组等权收益
    5. 保存因子截面和未来收益面板

合并所有持仓期结果
生成Q5-Q1多空组合
```

### 十一、统计检验

#### 多空组合t检验

`calc_ttest` 对Q5-Q1收益序列执行单样本t检验，输出：

- 均值；
- 标准差；
- 年化收益；
- 年化夏普；
- t统计量；
- p值；
- 样本数。

年化参数随换仓频率变化：

```python
FREQ_PERIODS_PER_YEAR = {
    "D": 252,
    "W": 52,
    "ME": 12,
}
```

#### Rank IC

`calc_ic` 逐期计算因子值与对应持仓期个股收益之间的 Spearman Rank IC，并输出：

- IC均值；
- IC标准差；
- ICIR；
- IC大于0占比；
- IC t统计量；
- 样本数。

### 十二、报告输出

每个Alpha生成一份由 `openpyxl` 创建的工作簿，包含四个Sheet：

| Sheet | 内容 |
|---|---|
| 统计摘要 | Q5-Q1 t检验和IC汇总 |
| IC序列 | IC柱状图、12期滚动均值和累积IC |
| 分组收益 | Q1—Q5及Q5-Q1收益分布、CDF和Sharpe |
| 累计净值 | 各分组和多空组合累计净值 |

> 当前主程序使用 `openpyxl.Workbook` 生成Excel内容，但输出文件名后缀写成了 `.csv`。正式运行前应将 `factor_report.csv` 改为 `factor_report.xlsx`。

---

## 研究注意事项

1. 复权、股票池、行业分类和基本面数据都必须采用点时可得口径；
2. 回测开始前应准备足够长的预热数据，尤其是使用60日、180日或252日窗口的因子；
3. 指数成分股必须使用历史快照，不能将当前成分股回填到全部历史；
4. PE、PB等基本面衍生指标应确保对应财务数据在信号日已经公开；
5. IC符号取决于因子方向，筛选时应同时观察IC绝对值和稳定性；
6. 因子方向必须由训练期确定，不能用完整样本事后翻转；
7. 周度持仓收益目前从下一交易日收盘开始，若假设T+1开盘成交，应改用复权开盘价；
8. 当前组合收益未考虑手续费、印花税、冲击成本、涨跌停和无法成交约束；
9. 当前Q1—Q5为等权组合，未进行行业或市值中性化；
10. 部分Alpha191原始公式存在转录或算子定义歧义，应与目标平台逐因子核对。

---

## License

本项目仅用于学术研究和量化方法验证，不构成任何投资建议。使用者应自行承担数据授权、模型风险和交易风险。
