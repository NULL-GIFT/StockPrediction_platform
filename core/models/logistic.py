from sklearn.linear_model import LogisticRegression
from core.models.base_model import BaseQuantModel

class LogisticModel(BaseQuantModel):
    """
    逻辑回归基准模型
    """
    def __init__(self):
        # 初始化基类，赋予模型名称
        super().__init__(model_name="Logistic_Regression")
        # 实例化 sklearn 的算法对象，设置最大迭代次数防报错
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    def train(self, X_train, y_train):
        print(f"正在训练 {self.model_name} ...")
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("模型未训练！")
        # 返回看涨 (Label=1) 的概率分布
        return self.model.predict_proba(X)[:, 1]