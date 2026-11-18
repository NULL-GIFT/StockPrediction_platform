import pandas as pd
import numpy as np


class PortfolioAccount:
    """独立的资金状态机：负责各自的资金推演，隔离摩擦成本计算"""

    def __init__(self, initial_capital, commission_rate=0.0):
        self.cash = initial_capital
        self.holdings = 0.0
        self.history_assets = []
        self.commission_rate = commission_rate

    def process_signal(self, price, action):
        """接收每天的价格和交易信号"""
        # 全仓买入
        if action == 1 and self.cash > 0:
            cost = self.cash * self.commission_rate
            self.holdings = (self.cash - cost) / price
            self.cash = 0.0

        # 清仓卖出
        elif action == -1 and self.holdings > 0:
            market_value = self.holdings * price
            cost = market_value * self.commission_rate
            self.cash = market_value - cost
            self.holdings = 0.0

        # 每日结算总资产并归档
        self.history_assets.append(self.cash + self.holdings * price)


class SignalGenerator:
    """
    信号发生器
    负责根据阈值，将模型输出的0~1概率转化为明确的动作
    """

    def __init__(self, buy_threshold=0.56, sell_threshold=0.45):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate(self, prob):
        """
        返回信号：1 (买入), -1 (卖出), 0 (持有/观望)
        """
        if prob >= self.buy_threshold:
            return 1
        elif prob <= self.sell_threshold:
            return -1
        else:
            return 0


class PortfolioBacktester:
    """
    投资组合回测模拟器
    收带信号的DataFrame，模拟真实的资金流转、持仓变化、摩擦成本以及止盈止损。
    """

    def __init__(self, initial_capital=100000.0, commission_rate=0.0003, tax_rate=0.001, take_profit=None,
                 stop_loss=None, max_position_weight=0.20):
        self.initial_capital = initial_capital
        self.max_position_weight = max_position_weight  # 接收外部传来的动态仓位指令
        self.cash = initial_capital
        self.positions = {}  # 记录持仓股数
        self.avg_cost = {}  # 记录持仓均价

        # 多空对冲账本
        self.short_positions = {}  # 记录融券做空的股数
        self.short_avg_cost = {}  # 记录做空的成本均价

        # 全市场最新价格缓存字典（解决停牌/退市导致的估值黑洞）
        self.last_price = {}

        # 风控与摩擦成本配置
        self.commission_rate = commission_rate  # 券商佣金 (双边)
        self.tax_rate = tax_rate  # 印花税 (A股仅卖出单边征收)
        self.min_commission = 5.0  # 最低佣金5元
        self.take_profit = take_profit  # 止盈阈值
        self.stop_loss = stop_loss  # 止损阈值


        self.daily_records = []
        self.trade_logs = []

    def _calculate_buy_cost(self, price, amount):
        """计算买入总成本 (包含佣金)"""
        trade_value = price * amount
        commission = max(trade_value * self.commission_rate, self.min_commission)
        return trade_value + commission

    def _calculate_sell_income(self, price, amount):
        """计算卖出总净收入 (扣除佣金和印花税)"""
        trade_value = price * amount
        commission = max(trade_value * self.commission_rate, self.min_commission)
        tax = trade_value * self.tax_rate
        return trade_value - commission - tax

    def _execute_sell(self, date, code, price, amount, reason="SELL"):
        """内部卖出执行器，统筹常规卖出与止损止盈"""
        income = self._calculate_sell_income(price, amount)
        self.cash += income

        self.trade_logs.append({
            'date': date, 'code': code, 'action': reason,
            'price': price, 'amount': amount, 'cash_change': round(income, 2)
        })
        self.positions[code] = 0
        self.avg_cost[code] = 0.0

    def _execute_short(self, date, code, price, amount, reason="SHORT"):
        """【低耦合算子】融券做空：获取卖出资金，记录空头负债"""
        income = self._calculate_sell_income(price, amount)
        self.cash += income  # 做空获得现金流
        self.short_positions[code] = amount
        self.short_avg_cost[code] = price
        self.trade_logs.append({
            'date': date, 'code': code, 'action': reason,
            'price': price, 'amount': amount, 'cash_change': round(income, 2)
        })

    def _execute_cover(self, date, code, price, amount, reason="COVER"):
        """【低耦合算子】买券还款：消耗资金，平掉空头负债"""
        cost = self._calculate_buy_cost(price, amount)
        self.cash -= cost  # 买回股票还给券商
        self.short_positions[code] = 0
        self.short_avg_cost[code] = 0.0
        self.trade_logs.append({
            'date': date, 'code': code, 'action': reason,
            'price': price, 'amount': amount, 'cash_change': round(-cost, 2)
        })



    def run_backtest(self, df):
        """核心调度方法：按日历时间逐日推进，模拟交易"""
        print(f"初始资金: {self.initial_capital}元，开始执行交易模拟...")
        if self.take_profit: print(f" 止盈线已开启: +{self.take_profit * 100}%")
        if self.stop_loss: print(f" 止损线已开启: -{self.stop_loss * 100}%")

        # 提前预排序，避免每日循环里重复排序
        df = df.sort_values(by='date')

        # 预先提取需要循环的列，利用itertuples极大提升遍历速度
        grouped = df.groupby('date')


        for date, daily_data in grouped:
            daily_data_indexed = daily_data.set_index('code')
            processed_codes = set()

            # 每天开盘第一件事，更新当天的行情快照
            for row in daily_data.itertuples(index=False):
                self.last_price[row.code] = row.close

            # 将总资产盘点提前，作为全天仓位分配的物理锚点
            # 净资产 = 现金 + 多头市值 - 空头负债
            long_value = sum(
                self.positions[c] * self.last_price.get(c, self.avg_cost.get(c, 0))
                for c in self.positions if self.positions[c] > 0
            )
            short_liability = sum(
                self.short_positions[c] * self.last_price.get(c, self.short_avg_cost.get(c, 0))
                for c in self.short_positions if self.short_positions[c] > 0
            )
            # 【将总资产盘点提前，作为全天仓位分配的物理锚点
            current_total_assets = self.cash + long_value - short_liability

            # 多头风控检查
            for code, amount in list(self.positions.items()):
                if amount > 0 and code in daily_data_indexed.index:
                    if daily_data_indexed.loc[code, 'volume'] == 0: continue
                    open_price = daily_data_indexed.loc[code, 'open']
                    cost_price = self.avg_cost.get(code, open_price)
                    return_rate = (open_price - cost_price) / cost_price
                    if self.take_profit is not None and return_rate >= self.take_profit:
                        self._execute_sell(date, code, open_price, amount, reason="TAKE_PROFIT_LONG")
                        processed_codes.add(code)
                    elif self.stop_loss is not None and return_rate <= -self.stop_loss:
                        self._execute_sell(date, code, open_price, amount, reason="STOP_LOSS_LONG")
                        processed_codes.add(code)

            # 空头风控检查
            for code, amount in list(self.short_positions.items()):
                if amount > 0 and code in daily_data_indexed.index:
                    if daily_data_indexed.loc[code, 'volume'] == 0: continue
                    open_price = daily_data_indexed.loc[code, 'open']
                    cost_price = self.short_avg_cost.get(code, open_price)
                    return_rate = (cost_price - open_price) / cost_price  # 反向收益率
                    if self.take_profit is not None and return_rate >= self.take_profit:
                        self._execute_cover(date, code, open_price, amount, reason="TAKE_PROFIT_SHORT")
                        processed_codes.add(code)
                    elif self.stop_loss is not None and return_rate <= -self.stop_loss:
                        self._execute_cover(date, code, open_price, amount, reason="STOP_LOSS_SHORT")
                        processed_codes.add(code)

            # 跑输中位数平多仓，反手开空仓
            sell_signals = daily_data[(daily_data['signal'] == -1) & (daily_data['volume'] > 0)]
            for row in sell_signals.itertuples(index=False):
                code, price = row.code, row.open
                if code in processed_codes: continue

                # 发现持有多头，斩仓卖出
                if self.positions.get(code, 0) > 0:
                    self._execute_sell(date, code, price, self.positions[code], reason="SELL")

                # 立刻反手融券做空
                if self.short_positions.get(code, 0) == 0:
                    target_invest = current_total_assets * self.max_position_weight
                    affordable_shares = int(target_invest / (price * (1 + self.commission_rate)))
                    actual_shares = (affordable_shares // 100) * 100
                    if actual_shares >= 100:
                        self._execute_short(date, code, price, actual_shares, reason="SHORT")
                processed_codes.add(code)

            # 跑赢中位数平空仓，反手开多仓
            buy_signals = daily_data[(daily_data['signal'] == 1) & (~daily_data['code'].isin(processed_codes)) & (daily_data['volume'] > 0)].sort_values(by='prob', ascending=False)
            for row in buy_signals.itertuples(index=False):
                code, price = row.code, row.open

                # 发现持有空头，买券平仓
                if self.short_positions.get(code, 0) > 0:
                    self._execute_cover(date, code, price, self.short_positions[code], reason="COVER")

                # 立刻反手买入多头
                target_invest = min(self.cash, current_total_assets * self.max_position_weight)
                if target_invest < 2000: break  # 现金枯竭阻断
                affordable_shares = int(target_invest / (price * (1 + self.commission_rate)))
                actual_shares = (affordable_shares // 100) * 100
                if actual_shares >= 100:
                    cost = self._calculate_buy_cost(price, actual_shares)
                    if self.cash >= cost:
                        curr_amt = self.positions.get(code, 0)
                        curr_cost = curr_amt * self.avg_cost.get(code, 0)
                        self.positions[code] = curr_amt + actual_shares
                        self.avg_cost[code] = (curr_cost + cost) / self.positions[code]
                        self.cash -= cost
                        self.trade_logs.append({'date': date, 'code': code, 'action': 'BUY', 'price': price, 'amount': actual_shares, 'cash_change': round(-cost, 2)})

            # 每日收盘结算资产净值
            long_value_close = sum(
                self.positions[c] * self.last_price.get(c, self.avg_cost.get(c, 0))
                for c in self.positions if self.positions[c] > 0
            )
            short_liability_close = sum(
                self.short_positions[c] * self.last_price.get(c, self.short_avg_cost.get(c, 0))
                for c in self.short_positions if self.short_positions[c] > 0
            )

            market_value = long_value_close - short_liability_close
            total_assets = self.cash + market_value





            self.daily_records.append({
                'date': date, 'total_assets': round(total_assets, 2),
                'market_value': round(market_value, 2), 'cash': round(self.cash, 2)
            })

        print("回测结束！")
        return pd.DataFrame(self.daily_records), pd.DataFrame(self.trade_logs)


class PerformanceAnalyzer:
    """绩效分析引擎 (保持不变)"""

    @staticmethod
    def calculate(daily_df, initial_capital, benchmark_df=None):
        final_assets = daily_df.iloc[-1]['total_assets']
        total_return = (final_assets - initial_capital) / initial_capital

        daily_df['roll_max'] = daily_df['total_assets'].cummax()
        daily_df['drawdown'] = (daily_df['total_assets'] - daily_df['roll_max']) / daily_df['roll_max']
        max_drawdown = daily_df['drawdown'].min()

        daily_returns = daily_df['total_assets'].pct_change().dropna()
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0.0

        benchmark_return = 0.0
        alpha = 0.0
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df = benchmark_df.sort_values('date')
            first_close = benchmark_df.iloc[0]['index300_close']
            last_close = benchmark_df.iloc[-1]['index300_close']
            benchmark_return = (last_close - first_close) / first_close
            alpha = total_return - benchmark_return

        return {
            'Total_Return': total_return,
            'Max_Drawdown': max_drawdown,
            'Sharpe_Ratio': sharpe_ratio,
            'Benchmark_Return': benchmark_return,
            'Alpha': alpha
        }