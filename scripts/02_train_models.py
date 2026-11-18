import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)

from core.models.logistic import LogisticModel
from core.models.bp_nn import BPNNModel
from core.models.svm import SVMModel
from core.models.xgboost_clf import XGBoostModel
from core.evaluator import ModelEvaluator
from core import config


def split_dataset_by_date(df, feature_cols, target_col, train_range, test_range):
    print(f"正在按指定日期切分数据...")
    df['date'] = pd.to_datetime(df['date'])
    train_start = pd.to_datetime(train_range[0])
    train_end = pd.to_datetime(train_range[1])
    test_start = pd.to_datetime(test_range[0])
    test_end = pd.to_datetime(test_range[1])

    train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
    test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)

    train_df = df[train_mask].dropna(subset=feature_cols + [target_col])
    test_df = df[test_mask].dropna(subset=feature_cols + [target_col])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test


def main():
    print("=== 启动炼丹炉：开始训练与评估多周期双轨模型 ===")

    data_path = os.path.join(project_root, 'data', 'processed', 'features_base.csv')
    df = pd.read_csv(data_path)

    TRAIN_PERIOD = (config.TRAIN_START, config.TRAIN_END)
    TEST_PERIOD = (config.TEST_START, config.TEST_END)

    base_features = config.FEATURE_COLUMNS
    identity_features = [col for col in df.columns if col.startswith('ind_')]
    final_features = base_features + identity_features
    print(f"全局基座模型将使用以下 {len(final_features)} 个特征进行训练...")



    base_time_tag = f"{config.TRAIN_START.replace('-', '')}_{config.TRAIN_END.replace('-', '')}"
    time_folder_name = f"Train_{base_time_tag}"

    feat_tags = []
    if any(f in final_features for f in config.BASIC_FEATURES): feat_tags.append("Base")
    if any(f in final_features for f in config.FACTOR_FEATURES): feat_tags.append("Factor")
    if any(f in final_features for f in config.LAG_FEATURES): feat_tags.append("Lag")
    if any(f.startswith('ind_') for f in final_features): feat_tags.append("Industry")
    feat_tags.sort()
    feature_folder_name = "Feat_" + "_".join(feat_tags)
    model_dir = os.path.join(project_root, 'data', 'models', time_folder_name, feature_folder_name)
    figure_dir = os.path.join(project_root, 'output', 'figures', time_folder_name, feature_folder_name)
    table_dir = os.path.join(project_root, 'output', 'tables', time_folder_name, feature_folder_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    evaluator = ModelEvaluator()
    results = []

    for target_col in config.TARGET_COLUMNS:
        print(f"\n" + "=" * 50)
        print(f"🌟 正在开启【{target_col}】维度的模型训练流水线")
        print("=" * 50)

        X_train, y_train, X_test, y_test = split_dataset_by_date(
            df, final_features, target_col, TRAIN_PERIOD, TEST_PERIOD
        )

        print("正在执行特征标准化 (StandardScaler)...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Scaler 强制进收纳盒
        scaler_save_path = os.path.join(model_dir, f'global_scaler_{target_col}.pkl')
        joblib.dump(scaler, scaler_save_path)

        models = {
            "Logistic": LogisticModel(),
            "BP_NN": BPNNModel(),
            "SVM": SVMModel(),
            "XGBoost": XGBoostModel()
        }

        for _, model in models.items():
            print(f"--> 正在训练并评估: {model.model_name} (针对 {target_col}) ...")
            model.train(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

            eval_name = f"{model.model_name}_{target_col}"

            metrics = evaluator.evaluate(eval_name, y_test, y_pred, y_prob, fig_save_dir=figure_dir)

            metrics['Target'] = target_col
            results.append(metrics)


            save_path = os.path.join(model_dir, f"{model.model_name}_{target_col}.pkl")
            joblib.dump(model.model, save_path)

            if "XGBoost" in model.model_name:
                if hasattr(model, 'get_feature_importance'):
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.rcParams['font.sans-serif'] = ['SimHei']
                    plt.rcParams['axes.unicode_minus'] = False

                    fi_data = model.get_feature_importance()

                    if isinstance(fi_data, dict):
                        fi_df = pd.DataFrame({
                            'Feature': list(fi_data.keys()),
                            'Importance': list(fi_data.values())
                        })
                    else:
                        fi_df = pd.DataFrame({
                            'Feature': final_features,
                            'Importance': fi_data
                        })

                    fi_df = fi_df.sort_values(by='Importance', ascending=False)


                    fi_df.to_csv(os.path.join(table_dir, f'xgboost_fi_{target_col}.csv'), index=False)

                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', hue='Feature', legend=False)
                    plt.title(f'XGBoost 特征贡献度排名 ({target_col} | {base_time_tag})', fontsize=12)
                    plt.tight_layout()


                    fig_save_path = os.path.join(figure_dir, f'xgboost_importance_{target_col}.png')
                    plt.savefig(fig_save_path, dpi=300)
                    plt.close()

    print("\n=== 最终多周期期末成绩单 ===")
    results_df = pd.DataFrame(results)
    cols = ['Target', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
    results_df = results_df[cols]
    print(results_df.to_markdown(index=False))

    table_output_path = os.path.join(table_dir, 'model_comparison.csv')
    results_df.to_csv(table_output_path, index=False)
    print(f"\n全部完成！三大件已全部隔离收纳至带有 {base_time_tag} 后缀的专属目录。")


if __name__ == "__main__":
    main()
