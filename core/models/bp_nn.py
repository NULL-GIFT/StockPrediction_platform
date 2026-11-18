from sklearn.neural_network import MLPClassifier
from core.models.base_model import BaseQuantModel

class BPNNModel(BaseQuantModel):
    """
    BP神经网络模型
    """
    def __init__(self):
        super().__init__(model_name="BP_NeuralNetwork")
        # 严格限制神经元规模，防止训练卡死
        self.model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)

    def train(self, X_train, y_train):
        print(f"正在训练 {self.model_name} ...")
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError(f"{self.model_name} 模型尚未训练！")
        # 返回看涨 (Label=1) 的概率
        return self.model.predict_proba(X)[:, 1]