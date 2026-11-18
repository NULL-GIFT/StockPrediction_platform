# core/config.py
"""
[配置中心]
所有硬编码参数必须在此处统一管理
业务逻辑代码只负责执行，不负责定义参数
"""


# 基础技术指标 (传统均线与动量)
BASIC_FEATURES = [
    'MA5', 'MA10', 'MA20', 'MACD', 'K', 'D', 'J', 'RSI_14', 'BIAS', 'OBV'
]

# 增强因子_度量市场波动与风险
"""
Ret_5d：过去5天的累计涨跌幅（短期动量）
Amplitude：日内振幅 (high-low)/close（反映多空分歧）
volatility_5d/20d：5日和20日波动率（波动率通道）
vol_ratio_5d/20d：5日和20日量比（成交量异常波动）
bb_upper/lower/width：布林带上下轨及带宽（价格波动边界）
atr_14 / atr_norm：真实波幅及归一化ATR（反映真实波动风险）
"""
FACTOR_FEATURES = [
    'Ret_5d', 'Amplitude', 'Relative_Strength',
    'volatility_5d', 'volatility_20d', 'vol_ratio_5d', 'vol_ratio_20d',
    'bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'atr_norm'
]

# 时序记忆因子
LAG_FEATURES = [
    'Close_Lag1', 'Close_Lag2', 'Volume_Lag1'
]

# 默认全局使用的特征全集
FEATURE_COLUMNS = BASIC_FEATURES

TARGET_COLUMNS = ['label_1d_abs', 'label_5d_abs', 'label_1d_cs', 'label_5d_cs']



# 全局时间切分
TRAIN_START = '2022-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'


# 回测与模拟盘核心参数

INITIAL_CAPITAL = 100000.0   # 默认初始本金10万
BUY_THRESHOLD = 0.55         # 模型看涨概率超过此值才买入
SELL_THRESHOLD = 0.45        # 模型看涨概率低于此值则卖出


# 摩擦成本
COMMISSION_RATE = 0.0003     # 券商佣金 (万三)
TAX_RATE = 0.001             # 印花税 (千分之一，仅卖出收取)


# 股票池与行业映射
STOCK_INDUSTRY_MAP = {
    'sh.601318': '金融', 'sh.600036': '金融', 'sh.600030': '金融', 'sz.300059': '金融', 'sh.601398': '金融',
    'sh.688981': '科技', 'sz.002371': '科技', 'sz.002415': '科技', 'sz.002475': '科技', 'sz.002230': '科技',
    'sz.300750': '新能源', 'sz.002594': '新能源', 'sh.601012': '新能源', 'sz.300274': '新能源', 'sz.300124': '新能源',
    'sh.600519': '消费', 'sz.000858': '消费', 'sh.600887': '消费', 'sz.000333': '消费', 'sh.600276': '消费'
}

# 动态生成股票池列表，防止未来修改时两边数据不一致
STOCK_POOL = list(STOCK_INDUSTRY_MAP.keys())






