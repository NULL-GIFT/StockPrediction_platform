from abc import ABC, abstractmethod
import joblib
import os


class BaseQuantModel(ABC):
    """
    量化模型的抽象基类。
    所有继承的子类必须实现train和predict_proba方法，保证极低的耦合度。
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        """训练模型"""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """预测概率，用于计算AUC和策略回测"""
        pass

    def predict(self, X):
        """预测分类标签"""
        if self.model is None:
            raise ValueError(f"{self.model_name} 模型尚未训练！")
        return self.model.predict(X)

    def save_model(self, save_dir='data/models/'):
        """通用的模型序列化保存方法"""
        if self.model is None:
            raise ValueError("没有可保存的已训练模型！")

        # 确保目录存在
        # 注意：这里的相对路径是在执行主脚本时生效的，后续会在流水线脚本中传入绝对路径
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.model_name}.pkl")
        joblib.dump(self.model, save_path)
        print(f"[{self.model_name}] 模型已保存至: {save_path}")

    def load_model(self, model_path):
        """通用的模型加载方法"""
        self.model = joblib.load(model_path)
        print(f"[{self.model_name}] 模型已从 {model_path} 加载。")