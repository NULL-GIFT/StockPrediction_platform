import os
import sys
import pandas as pd
import joblib
import json



current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

from core.backtester import SignalGenerator, PortfolioBacktester, PerformanceAnalyzer
from core import config


def run_backtest_api(user_params=None):
    user_params = user_params or {}
    print("\n" + "=" * 50)
    print("=== 🚀 启动量化回测验证流水线 ===")
    print("=" * 50)

    train_start = user_params.get('train_start', config.TRAIN_START)
    train_end = user_params.get('train_end', config.TRAIN_END)
    test_start = user_params.get('test_start', config.TEST_START)
    test_end = user_params.get('test_end', config.TEST_END)

    buy_thresh = user_params.get('buy_threshold', config.BUY_THRESHOLD)
    sell_thresh = user_params.get('sell_threshold', config.SELL_THRESHOLD)
    init_capital = user_params.get('initial_capital', config.INITIAL_CAPITAL)
    comm_rate = user_params.get('commission_rate', config.COMMISSION_RATE)
    tax_rate = user_params.get('tax_rate', config.TAX_RATE)
    model_name = user_params.get('model_name', 'XGBoost_Enhanced')

    take_profit = user_params.get('take_profit', None)
    stop_loss = user_params.get('stop_loss', None)

    target_code = user_params.get('target_code', 'ALL')
    predict_window = user_params.get('predict_window', '5d')
    label_type = user_params.get('label_type', 'cs')


    target_col = f"label_{predict_window}_{label_type}"


    raw_features = user_params.get('selected_features', [])
    clean_features = list(set(raw_features)) if raw_features else ['base']

    if 'base' not in clean_features:
        clean_features.append('base')

    clean_features.sort()

    dynamic_features = []
    if 'base' in clean_features: dynamic_features.extend(config.BASIC_FEATURES)
    if 'factor' in clean_features: dynamic_features.extend(config.FACTOR_FEATURES)
    if 'lag' in clean_features: dynamic_features.extend(config.LAG_FEATURES)

    data_path = os.path.join(project_root, 'data', 'processed', 'features_base.csv')
    df = pd.read_csv(data_path)

    identity_features = [col for col in df.columns if col.startswith('ind_')]

    if 'industry' in clean_features:
        final_features = dynamic_features + identity_features
    else:
        final_features = dynamic_features

    time_tag = f"{train_start.replace('-', '')}_{train_end.replace('-', '')}"
    time_folder_name = f"Train_{time_tag}"

    feature_folder_name = "Feat_" + "_".join([f.capitalize() for f in clean_features])

    model_dir = os.path.join(project_root, 'data', 'models', time_folder_name, feature_folder_name)
    figure_dir = os.path.join(project_root, 'output', 'figures', time_folder_name, feature_folder_name)
    table_dir = os.path.join(project_root, 'output', 'tables', time_folder_name, feature_folder_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    target_model_path = os.path.join(model_dir, f'{model_name}_{target_col}.pkl')
    target_scaler_path = os.path.join(model_dir, f'global_scaler_{target_col}.pkl')

    if not os.path.exists(target_model_path):
        print(f"\n⚠️  [Cache Miss] 系统检测到全新训练区间: [{train_start} 至 {train_end}]")
        print(f"🔥 现场炼丹开始...")

        train_mask = (df['date'] >= str(train_start)) & (df['date'] <= str(train_end))
        custom_train_df = df[train_mask].dropna(subset=final_features + [target_col]).copy()


        from sklearn.preprocessing import StandardScaler
        custom_scaler = StandardScaler()

        X_train_array = custom_scaler.fit_transform(custom_train_df[final_features])
        X_train_custom = pd.DataFrame(X_train_array, columns=final_features, index=custom_train_df.index)

        y_train_custom = custom_train_df[target_col]
        joblib.dump(custom_scaler, target_scaler_path)

        from core.models.logistic import LogisticModel
        from core.models.svm import SVMModel
        from core.models.bp_nn import BPNNModel
        from core.models.xgboost_clf import XGBoostModel

        model_map = {
            "Logistic_Regression": LogisticModel(),
            "SVM_GridSearch": SVMModel(),
            "BP_NeuralNetwork": BPNNModel(),
            "XGBoost_Enhanced": XGBoostModel()
        }
        live_model = model_map[model_name]
        live_model.train(X_train_custom, y_train_custom)

        joblib.dump(live_model.model, target_model_path)

        if "XGBoost" in model_name and hasattr(live_model, 'get_feature_importance'):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            fi_data = live_model.get_feature_importance()

            if isinstance(fi_data, dict):
                fi_df = pd.DataFrame({'Feature': list(fi_data.keys()), 'Importance': list(fi_data.values())})
            else:
                fi_df = pd.DataFrame({'Feature': final_features, 'Importance': fi_data})

            fi_df = fi_df.sort_values(by='Importance', ascending=False)
            fi_df.to_csv(os.path.join(table_dir, f'xgboost_fi_{target_col}.csv'), index=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', hue='Feature', legend=False)
            plt.title(f'XGBoost 现场特征贡献度排名 ({time_tag})', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f'xgboost_importance_{target_col}.png'), dpi=300)
            plt.close()

        print("✅ 现场训练完毕，模型与图表已隔离保存！")
    else:
        print(f"⚡ [Cache Hit] 从专属沙盒极速装载模型: {os.path.basename(target_model_path)}")


    print(f"正在装载测试集数据...")
    index_path = os.path.join(project_root, 'data', 'raw', 'index300_daily.csv')
    benchmark_df = pd.read_csv(index_path)
    benchmark_test = benchmark_df[
        (benchmark_df['date'] >= str(test_start)) & (benchmark_df['date'] <= str(test_end))].copy()

    sorted_dates = sorted(df['date'].unique())
    valid_start_dates = [d for d in sorted_dates if d >= str(test_start)]
    if not valid_start_dates:
        raise ValueError("测试集起始日期超出数据范围！")
    real_test_start = valid_start_dates[0]

    #获取前一个交易日
    start_idx = sorted_dates.index(real_test_start)
    lookback_date = sorted_dates[start_idx - 1] if start_idx > 0 else real_test_start

    #按日历切片（包含前一个交易日）
    test_df_with_history = df[(df['date'] >= lookback_date) & (df['date'] <= str(test_end))].copy()

    if target_code != 'ALL':
        test_df_with_history = test_df_with_history[test_df_with_history['code'] == target_code].copy()

    test_df_with_history = test_df_with_history.dropna(subset=final_features)

    #统一标准化并执行模型预测
    scaler = joblib.load(target_scaler_path)
    X_test_scaled = pd.DataFrame(scaler.transform(test_df_with_history[final_features]),
                                 columns=final_features,
                                 index=test_df_with_history.index)

    quant_model = joblib.load(target_model_path)
    test_df_with_history['prob'] = quant_model.predict_proba(X_test_scaled)[:, 1]

    signal_gen = SignalGenerator(buy_threshold=buy_thresh, sell_threshold=sell_thresh)
    test_df_with_history['signal'] = test_df_with_history['prob'].apply(signal_gen.generate)

    #平移，现在shift(1)承接了T-1的状态
    test_df_with_history['signal'] = test_df_with_history.groupby('code')['signal'].shift(1).fillna(0)

    #裁剪掉借来的那一天
    test_df = test_df_with_history[test_df_with_history['date'] >= real_test_start].copy()




    backtest_input = test_df[['date', 'code', 'open', 'close', 'volume', 'signal', 'prob']]


    dynamic_weight = 0.20 if target_code == 'ALL' else 1.0

    real_backtester = PortfolioBacktester(
        initial_capital=init_capital, commission_rate=comm_rate, tax_rate=tax_rate,
        take_profit=take_profit, stop_loss=stop_loss,
        max_position_weight=dynamic_weight  # 注入动态仓位
    )
    real_daily_df, _ = real_backtester.run_backtest(backtest_input)

    ideal_backtester = PortfolioBacktester(
        initial_capital=init_capital, commission_rate=0.0, tax_rate=0.0,
        take_profit=take_profit, stop_loss=stop_loss,
        max_position_weight=dynamic_weight  # 注入动态仓位
    )


    ideal_daily_df, _ = ideal_backtester.run_backtest(backtest_input)


    metrics = PerformanceAnalyzer.calculate(real_daily_df, init_capital, benchmark_df=benchmark_test)
    dates = real_daily_df['date'].tolist()
    total_assets = real_daily_df['total_assets'].tolist()
    ideal_assets = ideal_daily_df['total_assets'].tolist()  # 【新增提取】


    benchmark_test['base_100k'] = init_capital * (
                benchmark_test['index300_close'] / benchmark_test.iloc[0]['index300_close'])
    benchmark_assets = benchmark_test['base_100k'].round(2).tolist()

    test_df['normalized_close'] = test_df.groupby('code')['close'].transform(lambda x: x / x.iloc[0])
    bnh_series = test_df.groupby('date')['normalized_close'].mean() * init_capital
    bnh_dict = bnh_series.round(2).to_dict()
    bnh_assets = [bnh_dict.get(d, init_capital) for d in dates]

    kline_df = df[df['date'].isin(dates)].copy()


    if target_code == 'ALL':
        if not kline_df.empty:
            # 【正统量化解法】：按天聚合，将池内所有股票的 OHLC 求均值
            # 现场捏出一根代表“整个股票池整体波动与情绪”的真实 K 线！
            kline_sub = kline_df.groupby('date').agg({
                'open': 'mean',
                'close': 'mean',
                'high': 'mean',
                'low': 'mean'
            }).reset_index().sort_values('date')
        else:
            kline_sub = pd.DataFrame()
    else:
        kline_sub = kline_df[kline_df['code'] == target_code].sort_values('date')

    if not kline_sub.empty:
        kline_sub['MA5'] = kline_sub['close'].rolling(window=5, min_periods=1).mean()
        kline_sub['MA10'] = kline_sub['close'].rolling(window=10, min_periods=1).mean()
        kline_sub['MA20'] = kline_sub['close'].rolling(window=20, min_periods=1).mean()


    from sklearn.metrics import confusion_matrix
    eval_df = test_df[test_df[target_col].notna()].copy()
    y_true = eval_df[target_col].astype(int).tolist()
    y_prob = eval_df['prob'].values
    y_pred = (eval_df['prob'] >= 0.5).astype(int).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_data = [
        [0, 0, int(cm[0][0])], [1, 0, int(cm[0][1])],
        [0, 1, int(cm[1][0])], [1, 1, int(cm[1][1])]
    ]

    from core.evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    eval_name = f"{model_name}_{target_col}"
    metrics_dict = evaluator.evaluate(
        model_name=eval_name,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=test_df['prob'].values,
        fig_save_dir=figure_dir
    )

    comparison_path = os.path.join(table_dir, 'model_comparison.csv')
    model_names, model_accuracies, model_aucs = [], [], []
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        comp_df = comp_df[comp_df['Target'] == target_col] if 'Target' in comp_df.columns else comp_df

        model_names = comp_df['Model'].tolist()
        model_accuracies = comp_df['Accuracy'].fillna(0).tolist()
        model_aucs = [float(x) if pd.notna(x) and str(x).lower() != "nan" else 0.5 for x in comp_df['AUC'].tolist()]
    else:
        model_names = [model_name]
        model_accuracies = [metrics_dict['Accuracy']]
        model_aucs = [float(metrics_dict['AUC']) if str(metrics_dict['AUC']).lower() != "nan" else 0.5]


    result_data = {
        "metrics": {
            "total_return": f"{metrics['Total_Return'] * 100:.2f}%",
            "alpha": f"{metrics['Alpha'] * 100:.2f}%",
            "max_drawdown": f"{metrics['Max_Drawdown'] * 100:.2f}%",
            "sharpe_ratio": f"{metrics['Sharpe_Ratio']:.2f}"
        },
        "academic_metrics": {
            "accuracy": f"{metrics_dict['Accuracy']:.4f}",
            "precision": f"{metrics_dict['Precision']:.4f}",
            "recall": f"{metrics_dict['Recall']:.4f}",
            "f1_score": f"{metrics_dict['F1_Score']:.4f}",
            "auc": f"{metrics_dict['AUC']}"
        },
        "chart_data": {
            "dates": dates,
            "strategy_assets": total_assets,
            "strategy_assets_ideal": ideal_assets,
            "benchmark_assets": benchmark_assets,
            "buy_and_hold_assets": bnh_assets
        },
        "kline_data": {
            "dates": kline_sub['date'].tolist(),
            "values": kline_sub[['open', 'close', 'low', 'high']].values.tolist(),
            "ma5": kline_sub['MA5'].round(2).tolist() if 'MA5' in kline_sub else [],
            "ma10": kline_sub['MA10'].round(2).tolist() if 'MA10' in kline_sub else [],
            "ma20": kline_sub['MA20'].round(2).tolist() if 'MA20' in kline_sub else []
        },
        "matrix_data": matrix_data,
        "comparison_data": {
            "models": model_names, "accuracy": model_accuracies, "auc": model_aucs
        }
    }

    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    return result_data