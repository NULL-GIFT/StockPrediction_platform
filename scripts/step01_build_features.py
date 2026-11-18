import pandas as pd
import os
import sys


current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

from core.feature_eng import add_technical_indicators, add_cross_sectional_labels
from core import config


def main():
    print("=== 开始构建全量特征数据 ===")

    stock_path = os.path.join(project_root, 'data', 'raw', 'stock_daily.csv')
    index_path = os.path.join(project_root, 'data', 'raw', 'index300_daily.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    save_path = os.path.join(processed_dir, 'features_base.csv')

    stock_df = pd.read_csv(stock_path)
    index_df = pd.read_csv(index_path)
    print(f"成功读取股票数据：{len(stock_df)} 行，大盘数据：{len(index_df)} 行。")

    print("正在计算大盘环境特征...")
    index_df['HS300_Return'] = index_df['index300_close'].pct_change()
    index_df['Market_Volatility'] = index_df['HS300_Return'].rolling(window=20).std()


    print("正在构建【日历x股票】全网格，执行严格物理对齐...")

    all_dates = index_df['date'].unique()
    all_codes = stock_df['code'].unique()
    grid_df = pd.MultiIndex.from_product([all_dates, all_codes], names=['date', 'code']).to_frame(index=False)


    padded_stock_df = pd.merge(grid_df, stock_df, on=['date', 'code'], how='left')


    industry_map = config.STOCK_INDUSTRY_MAP
    padded_stock_df['industry'] = padded_stock_df['code'].map(industry_map)

    df = pd.merge(padded_stock_df, index_df, on='date', how='left')


    df = df.sort_values(by=['code', 'date'])
    df[['open', 'high', 'low', 'close']] = df.groupby('code')[['open', 'high', 'low', 'close']].ffill()
    df['volume'] = df['volume'].fillna(0)


    df = df.groupby('code', group_keys=False).apply(add_technical_indicators)
    df = add_cross_sectional_labels(df)


    print("正在清理空值并重排时序...")
    df_clean = df.dropna().copy()
    df_clean = df_clean.sort_values(by=['date', 'code'])


    print("正在进行行业板块 One-Hot 编码...")
    industry_dummies = pd.get_dummies(df_clean['industry'], prefix='ind').astype(int)
    df_clean = pd.concat([df_clean, industry_dummies], axis=1)

    # 7. 落盘保存
    os.makedirs(processed_dir, exist_ok=True)
    df_clean.to_csv(save_path, index=False)

    print(f"\n大功告成！带有身份特征的干净数据已保存至：{save_path}")
    print(f"清洗后的数据还剩：{len(df_clean)} 行。")

if __name__ == "__main__":
    main()