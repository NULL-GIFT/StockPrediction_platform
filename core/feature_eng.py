import pandas as pd
import numpy as np
import os



def add_technical_indicators(group):
    """
    单只股票技术指标计算引擎
    输入：单只股票的 DataFrame (必须包含高开低收、成交量、日期)
    输出：DataFrame
    """

    group = group.sort_values('date')

    # 均线
    group['MA5'] = group['close'].rolling(window=5).mean()
    group['MA10'] = group['close'].rolling(window=10).mean()
    group['MA20'] = group['close'].rolling(window=20).mean()

    # MACD趋势指标
    ema12 = group['close'].ewm(span=12, adjust=False).mean()
    ema26 = group['close'].ewm(span=26, adjust=False).mean()
    group['DIF'] = ema12 - ema26
    group['DEA'] = group['DIF'].ewm(span=9, adjust=False).mean()
    group['MACD'] = (group['DIF'] - group['DEA']) * 2

    # KDJ随机指标(9天周期)
    low_min = group['low'].rolling(window=9).min()
    high_max = group['high'].rolling(window=9).max()
    rsv = (group['close'] - low_min) / (high_max - low_min) * 100
    group['K'] = rsv.ewm(com=2).mean()
    group['D'] = group['K'].ewm(com=2).mean()
    group['J'] = 3 * group['K'] - 2 * group['D']

    # RSI_14(相对强弱指标)
    delta = group['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    group['RSI_14'] = 100 - (100 / (1 + rs))

    # BIAS_20(乖离率)
    group['BIAS'] = (group['close'] - group['MA20']) / group['MA20'] * 100

    # OBV(能量潮)
    direction = np.sign(group['close'].diff()).fillna(0)
    group['OBV'] = (direction * group['volume']).cumsum()


    # 短期动量(过去5天的真实涨跌幅)
    group['Ret_5d'] = group['close'].pct_change(5)

    # 日内波动率(今天的振幅，反映多空分歧)
    group['Amplitude'] = (group['high'] - group['low']) / group['close']

    # 相对强弱(个股收益率跑赢大盘多少)
    if 'HS300_Return' in group.columns:
        group['Relative_Strength'] = group['close'].pct_change() - group['HS300_Return']
    else:
        group['Relative_Strength'] = 0


    # 时序记忆因子：给机器学习模型赋予“记忆”
    group['Close_Lag1'] = group['close'].shift(1)
    group['Close_Lag2'] = group['close'].shift(2)
    group['Volume_Lag1'] = group['volume'].shift(1)


    # 抗干扰与通道特征：波动率、量比、布林带与ATR
    # 波动率
    group['volatility_5d'] = group['close'].pct_change().rolling(window=5).std()
    group['volatility_20d'] = group['close'].pct_change().rolling(window=20).std()

    # 量比
    group['vol_ratio_5d'] = group['volume'] / (group['volume'].rolling(window=5).mean() + 1e-8)
    group['vol_ratio_20d'] = group['volume'] / (group['volume'].rolling(window=20).mean() + 1e-8)

    # 布林带与带宽
    bb_mid = group['close'].rolling(window=20).mean()
    bb_std = group['close'].rolling(window=20).std()
    group['bb_upper'] = bb_mid + 2 * bb_std
    group['bb_lower'] = bb_mid - 2 * bb_std
    group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / bb_mid

    # ATR波动幅度
    high_low = group['high'] - group['low']
    high_close = (group['high'] - group['close'].shift(1)).abs()
    low_close = (group['low'] - group['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    group['atr_14'] = tr.rolling(window=14).mean()
    group['atr_norm'] = group['atr_14'] / group['close']

    """
    动态预测标签,剔除未来函数
    T日收盘出信号->T+1日开盘买入->T+2日(或T+6日)开盘卖出
    """
    group['future_return_1'] = group['open'].shift(-2) / group['open'].shift(-1) - 1
    group['future_return_5'] = group['open'].shift(-6) / group['open'].shift(-1) - 1

    group['label_1d_abs'] = np.where(group['future_return_1'].isna(), np.nan,
                                     np.where(group['future_return_1'] > 0.005, 1, 0))
    group['label_5d_abs'] = np.where(group['future_return_5'].isna(), np.nan,
                                     np.where(group['future_return_5'] > 0.015, 1, 0))

    return group


def add_cross_sectional_labels(df):
    """
    横向切片打分引擎
    在大表上按日历日groupby，计算中位数，产出_cs后缀的标签。
    """
    print("正在按日历进行横截面排序打分 (Top 50% = 1, Bottom 50% = 0)...")

    # 每天的收益中位数
    median_1d = df.groupby(['date', 'industry'])['future_return_1'].transform('median')
    median_5d = df.groupby(['date', 'industry'])['future_return_5'].transform('median')

    # 横截面标签
    df['label_1d_cs'] = np.where(df['future_return_1'] > median_1d, 1, 0)
    df['label_5d_cs'] = np.where(df['future_return_5'] > median_5d, 1, 0)

    df.loc[df['future_return_1'].isna(), 'label_1d_cs'] = np.nan
    df.loc[df['future_return_5'].isna(), 'label_5d_cs'] = np.nan

    return df

