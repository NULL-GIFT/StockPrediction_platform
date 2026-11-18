# fetch_real_data.py
import baostock as bs
import pandas as pd
import os


def fetch_a_share_data():
    print("=== 启动 Baostock 真实市场数据通道 ===")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    stock_pool = [
        'sh.601318', 'sh.600036', 'sh.600030', 'sz.300059', 'sh.601398',  # 大金融
        'sh.688981', 'sz.002371', 'sz.002415', 'sz.002475', 'sz.002230',  # 硬科技
        'sz.300750', 'sz.002594', 'sh.601012', 'sz.300274', 'sz.300124',  # 新能源
        'sh.600519', 'sz.000858', 'sh.600887', 'sz.000333', 'sh.600276'  # 大消费
    ]

    start_date = '2022-01-01'
    end_date = '2024-12-31'
    all_data = []

    for code in stock_pool:
        print(f"正在拉取 {code} ...")

        rs = bs.query_history_k_data_plus(code,
                                          "date,code,open,high,low,close,volume",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")

        if rs.error_code != '0':
            print(f"拉取 {code} 失败: {rs.error_msg}")
            continue

        while (rs.error_code == '0') & rs.next():
            all_data.append(rs.get_row_data())

    df = pd.DataFrame(all_data, columns=rs.fields)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'stock_daily.csv')
    df.to_csv(save_path, index=False)
    print(f"\n✅ 真实 A 股数据拉取完毕！共 {len(df)} 行标准数据。已覆盖：{save_path}")
    bs.logout()


if __name__ == '__main__':
    fetch_a_share_data()