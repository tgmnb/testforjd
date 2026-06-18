# testforjd

**多合约期货回测框架 — 不使用 KQ.m@ 主力连续拼接数据**

检测主力合约切换时间表，用每个合约的独立小时K线计算MA20/MA75，按三区逻辑生成交易信号。

## 解决的问题

KQ.m@ 主力连续合约在换月时有跳空，导致用 KQ.m@ 计算的技术指标（MA20/MA60/MA75）是不准确的。本框架：

1. 用 **持仓量（OI）比率法** 从 KQ.m@ 数据中检测主力合约切换日期
2. 自动下载 **每个独立合约的完整小时K线**
3. **仅在单个合约的数据内** 计算MA指标，不跨合约拼接
4. 在 **交割月前一个月末截断** 数据（个人投资者不可交割）
5. 持仓期间使用 **持仓合约自身数据** 判断退出信号

## 使用方法

### 1. 设置环境变量

```bash
export TQ_USERNAME="你的天勤账号"
export TQ_PASSWORD="你的天勤密码"
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行回测

```bash
# 全流程：发现合约 → 下载数据 → 回测 → 输出结果
python run_multi.py --variety DCE.jd --start 2022-01-01 --end 2026-06-18

# 自定义MA参数
python run_multi.py --variety DCE.jd --ma-short 20 --ma-long 75

# 分步执行（可用于缓存数据和调试）
python run_multi.py --only-discover
python run_multi.py --only-download
python run_multi.py --only-backtest
```

### 4. 输出

- `results/{variety}_metrics.txt` — 评价指标
- `results/{variety}_trades.csv` — 交易记录
- `results/{variety}_equity.png` — 权益曲线
- `data/{variety}_schedule.csv` — 合约切换时间表

## 文件结构

```
testforjd/
├── run_multi.py            # 入口
├── config/
│   ├── strategy.py         # 策略参数
│   ├── symbols.py          # 品种配置（夜盘映射）
│   └── credentials.py      # tqsdk 认证（环境变量）
├── data/
│   ├── discovery.py        # 主力合约发现模块
│   ├── contract_data.py    # 独立合约数据下载
│   ├── schedule_csv/       # 合约切换时间表缓存
│   └── contract_cache/     # 合约K线数据缓存
├── engine/
│   └── multi.py            # 多合约回测引擎
├── strategy/
│   ├── signal.py           # 信号计算
│   └── rules.py            # 交易规则
├── risk/
│   ├── sizing.py           # 仓位管理
│   └── cost.py             # 成本计算
├── stats/
│   ├── metrics.py          # 评价指标
│   ├── plot.py             # 权益曲线图
│   └── journal.py          # 交易记录
└── results/                # 回测输出
```

## 技术细节

### 主力合约发现算法

1. 加载 KQ.m@ 日线数据（一次API调用）
2. 计算每日持仓量变化比率 `OI_ratio = OI_t / OI_{t-1}`
3. 当 `OI_ratio > 1.4` 或 `OI_ratio < 1/1.4` 时标记为**切换日**
4. 每段之间的数据对应同一个主力合约
5. 从该段时间中点推断合约月份（交割月 ≈ 时间点 + 3个月）

### 合约末期截断

- 对 `DCE.jdYYMM` 合约，个人投资者最晚交易到 `(MM-1)` 月最后一天
- 例如 JD2409（交割月2024年9月）→ 最后交易日 **2024-08-31**

### 三区交易逻辑

- `bullish` 区：`close > max(MA20, MA_long)` → 多头
- `bearish` 区：`close < min(MA20, MA_long)` → 空头
- `middle` 区：中间 → 观望
- 开仓：进入区域 + MA方向确认
- 平仓：离开区域

## 注意事项

- 依赖 [tqsdk](https://github.com/shinnytech/tqsdk) 免费版数据
- 已过期合约的 K 线数据由 tqsdk 免费版提供（data_length=5000 限制）
- 缓存目录 `data/contract_cache/` 可删除以强制重新下载
