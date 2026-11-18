from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from core.models.base_model import BaseQuantModel


class SVMModel(BaseQuantModel):
    """支持向量机模型 (纯净版)"""

    def __init__(self):
        super().__init__(model_name="SVM_GridSearch")

        self.base_estimator = SVC(probability=True, random_state=42, class_weight='balanced')


        self.param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
        }
        self.model = None

    def train(self, X_train, y_train):
        print(f"正在训练 {self.model_name} 并执行参数寻优 (耗时可能较长，请耐心等待)...")

        self.model = GridSearchCV(
            self.base_estimator,
            self.param_grid,
            cv=3,
            n_jobs=2,
            verbose=3
        )
        self.model.fit(X_train, y_train)
        print(f"[{self.model_name}] 最佳超参数组合锁定为: {self.model.best_params_}")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError(f"{self.model_name} 模型尚未训练！")
        return self.model.predict_proba(X)[:, 1]