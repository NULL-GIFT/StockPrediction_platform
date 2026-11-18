import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve


plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """
    统一评价与可视化引擎
    计算指标与绘制图表
    """

    def __init__(self):
        # 动态获取项目根目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        self.fig_dir = os.path.join(self.project_root, 'output', 'figures')
        self.table_dir = os.path.join(self.project_root, 'output', 'tables')

        # 确保输出目录存在
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)





    def evaluate(self, model_name, y_true, y_pred, y_prob, fig_save_dir=None):
        """打分器：计算5大学术指标"""
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 防范单类别导致的AUC崩溃
        if y_prob is not None:
            # 检查真实标签中是否同时包含0和1
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                print(f"[警告] {model_name} 测试集极端异常，仅包含单一类别标签！已跳过 AUC 计算。")
                auc = np.nan
        else:
            auc = np.nan

        metrics = {
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1, 4),
            'AUC': round(auc, 4) if not np.isnan(auc) else "NaN"
        }

        self._plot_confusion_matrix(model_name, y_true, y_pred, fig_save_dir)

        if y_prob is not None and not np.isnan(auc):
            self._plot_roc_curve(model_name, y_true, y_prob, auc, fig_save_dir)

        return metrics






    def _plot_confusion_matrix(self, model_name, y_true, y_pred, fig_save_dir=None):
        """内部方法：画混淆矩阵热力图"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        target_dir = fig_save_dir if fig_save_dir else self.fig_dir
        save_path = os.path.join(target_dir, f"{model_name}_confusion_matrix.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(self, model_name, y_true, y_prob, auc_score, fig_save_dir=None):
        """内部方法：画ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title(f'{model_name} - ROC曲线')
        plt.legend(loc="lower right")

        target_dir = fig_save_dir if fig_save_dir else self.fig_dir
        save_path = os.path.join(target_dir, f"{model_name}_ROC_curve.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


