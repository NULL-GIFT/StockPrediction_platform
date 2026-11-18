import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from core import config

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)


def run_offline_regime_eval(params):

    print("=== 🔬 启动大盘行情交叉体检 (TimeSeriesSplit) ===")

    # 1. 还原寻址路径，去沙盒里提取已经炼好的四大金刚模型
    target_col = f"label_{params['predict_window']}_{params['label_type']}"
    time_tag = f"{params['train_start'].replace('-', '')}_{params['train_end'].replace('-', '')}"

    feat_tags = []
    if 'base' in params['selected_features']: feat_tags.append("Base")
    if 'factor' in params['selected_features']: feat_tags.append("Factor")
    if 'lag' in params['selected_features']: feat_tags.append("Lag")
    if 'industry' in params['selected_features']: feat_tags.append("Industry")

    feat_tags.sort()

    feature_folder_name = "Feat_" + "_".join(feat_tags)

    model_dir = os.path.join(project_root, 'data', 'models', f"Train_{time_tag}", feature_folder_name)
    scaler_path = os.path.join(model_dir, f'global_scaler_{target_col}.pkl')

    if not os.path.exists(scaler_path):
        raise ValueError("尚未找到该参数组合的训练档案，请先在主界面执行一次回测触发训练！")

    scaler = joblib.load(scaler_path)

    # 准备数据
    df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'features_base.csv'))
    index_df = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'index300_daily.csv'))
    index_df['date'] = pd.to_datetime(index_df['date'])

    # 获取特征全集
    dynamic_features = []
    if 'base' in params['selected_features']: dynamic_features.extend(config.BASIC_FEATURES)
    if 'factor' in params['selected_features']: dynamic_features.extend(config.FACTOR_FEATURES)
    if 'lag' in params['selected_features']: dynamic_features.extend(config.LAG_FEATURES)
    identity_features = [col for col in df.columns if col.startswith('ind_')]
    final_features = dynamic_features + identity_features if 'industry' in params[
        'selected_features'] else dynamic_features


    df['date'] = pd.to_datetime(df['date'])
    test_start = params.get('test_start', '2024-01-01')
    test_end = params.get('test_end', '2024-12-31')

    test_df = df[(df['date'] >= pd.to_datetime(test_start)) &
                 (df['date'] <= pd.to_datetime(test_end))].dropna(
        subset=final_features + [target_col]).sort_values('date')

    eval_target_df = test_df

    unique_dates = eval_target_df['date'].unique()
    tscv = TimeSeriesSplit(n_splits=3)

    regime_tags = []
    model_names = ["Logistic_Regression", "BP_NeuralNetwork", "SVM_GridSearch", "XGBoost_Enhanced"]

    scores_dict = {m: [] for m in model_names}

    fold = 1
    for train_idx, test_idx in tscv.split(unique_dates):
        test_dates = unique_dates[test_idx]

        if len(test_dates) == 0:
            continue

        fold_start, fold_end = test_dates[0], test_dates[-1]

        idx_sub = index_df[(index_df['date'] >= fold_start) & (index_df['date'] <= fold_end)]
        if len(idx_sub) > 1:
            market_return = (idx_sub.iloc[-1]['index300_close'] - idx_sub.iloc[0]['index300_close']) / idx_sub.iloc[0][
                'index300_close']
        else:
            market_return = 0.0


        if market_return > 0.02:
            regime = f"Fold {fold}\n(牛市 {market_return * 100:.1f}%)"
        elif market_return < -0.02:
            regime = f"Fold {fold}\n(熊市 {market_return * 100:.1f}%)"
        else:
            regime = f"Fold {fold}\n(震荡 {market_return * 100:.1f}%)"
        print(f"[调试]Fold {fold} 真实大盘区间收益率: {market_return * 100:.2f}%")
        regime_tags.append(regime)

        fold_test_df = eval_target_df[eval_target_df['date'].isin(test_dates)]




        X_test = pd.DataFrame(scaler.transform(fold_test_df[final_features]), columns=final_features,
                              index=fold_test_df.index)
        y_test = fold_test_df[target_col].astype(int)


        for m_name in model_names:
            model_path = os.path.join(model_dir, f"{m_name}_{target_col}.pkl")
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                y_pred = clf.predict(X_test)

                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                scores_dict[m_name].append(round(f1, 4))
            else:
                scores_dict[m_name].append(0.0)
        fold += 1

    output_dir = os.path.join(project_root, 'output', 'regime_evaluator')
    os.makedirs(output_dir, exist_ok=True)

    # 存CSV
    res_df = pd.DataFrame(scores_dict, index=[tag.replace('\n', ' ') for tag in regime_tags])
    res_df.to_csv(os.path.join(output_dir, f'regime_scores_{target_col}.csv'))

    # 画图并保存
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    res_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title(f'多模型在不同大盘行情下的表现 ({target_col})')
    plt.ylabel('综合指标 (Macro F1-Score)')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'regime_comparison_{target_col}.png'), dpi=300)
    plt.close()


    series_data = []
    color_map = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71']
    for idx, (m_name, scores) in enumerate(scores_dict.items()):
        series_data.append({
            "name": m_name,
            "type": "bar",
            "data": scores,
            "itemStyle": {"color": color_map[idx % len(color_map)]}
        })


    window_str = "1日" if params['predict_window'] == '1d' else "5日"
    label_str = "横截面打标(cs)" if params['label_type'] == 'cs' else "绝对收益打标(abs)"
    eval_target_title = f"【本次体检标的】：{window_str}收益预测 - {label_str}模型"

    print("=== 🔬 体检完毕，报告已成功落盘 ===")
    return {
        "eval_target": eval_target_title,
        "regime_tags": regime_tags,
        "models": model_names,
        "series_data": series_data
    }