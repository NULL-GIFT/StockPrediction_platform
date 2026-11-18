from xgboost import XGBClassifier
import pandas as pd
from core.models.base_model import BaseQuantModel


class XGBoostModel(BaseQuantModel):
    """
    XGBoost增强模型
    """

    def __init__(self):
        super().__init__(model_name="XGBoost_Enhanced")
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        )
        # 用于记录输入矩阵的列名，方便后期可视化
        self.feature_names = None

    def train(self, X_train, y_train):
        print(f"正在训练 {self.model_name} ...")
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # 动态样本权重平衡
        import numpy as np
        if len(np.unique(y_train)) > 1:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            # 计算负样本与正样本的比例，注入给XGBoost
            scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
            self.model.set_params(scale_pos_weight=scale_weight)

        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError(f"{self.model_name} 模型尚未训练！")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """
            提取特征重要性权重
        """
        if self.model is None:
            raise ValueError(f"{self.model_name} 尚未训练，无法提取特征重要性！")

        importance_array = self.model.feature_importances_
        if self.feature_names is not None:
            # 将特征名与重要性分数组装成字典返回
            return dict(zip(self.feature_names, importance_array))
        return importance_array

