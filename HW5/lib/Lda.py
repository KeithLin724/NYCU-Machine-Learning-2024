from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


class LdaTestAbc(ABC):

    @abstractmethod
    def fit(self, data_in: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def acc(self, x_df: pd.DataFrame) -> float:
        raise NotImplementedError

    @abstractmethod
    def func(self) -> dict | list:
        raise NotImplementedError

    @abstractmethod
    def to_df_record(self) -> dict:
        raise NotImplementedError


class Lda(LdaTestAbc):
    def __init__(
        self, positive_class: str, negative_class: str, c1: int = 1, c2: int = 1
    ):
        self._w: np.ndarray = None
        self._cov_matrix: np.ndarray = None
        self._b = None

        self._positive_class: str = positive_class
        self._negative_class: str = negative_class
        self._display_class = [self._negative_class, self._positive_class]
        self._c1 = c1
        self._c2 = c2
        return

    @property
    def w(self) -> np.ndarray:
        return self._w

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c1, self._c2

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @staticmethod
    def _build_mean_and_cov(data_in: np.ndarray):
        mean = np.mean(data_in, axis=0)
        cov = np.cov(data_in.T)

        if len(cov.shape) == 0:
            cov = np.array([[cov]])

        return mean, cov

    def fit(self, data_in: pd.DataFrame, column_name: str = "Label") -> None:
        positive_data = data_in[data_in[column_name] == self._positive_class]
        negative_data = data_in[data_in[column_name] == self._negative_class]

        positive_data = positive_data.drop(columns=[column_name]).to_numpy()
        negative_data = negative_data.drop(columns=[column_name]).to_numpy()

        positive_mean, positive_cov = Lda._build_mean_and_cov(positive_data)
        negative_mean, negative_cov = Lda._build_mean_and_cov(negative_data)

        positive_len, negative_len = len(positive_data), len(negative_data)

        total = positive_len + negative_len
        p1, p2 = positive_len / total, negative_len / total

        # cov matrix
        self._cov_matrix = p1 * positive_cov + p2 * negative_cov

        inv_cov = np.linalg.pinv(self._cov_matrix)

        # weight
        self._w = (positive_mean - negative_mean).T @ inv_cov

        # b
        self._b = -(1 / 2) * (positive_mean - negative_mean).T @ inv_cov @ (
            positive_mean + negative_mean
        ) - np.log((self._c1 * p2) / (self._c2 * p1))

        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._w.T @ x + self._b

    def predict_with_df(
        self, x_df: pd.DataFrame, column_name: str = "Label"
    ) -> np.ndarray:
        np_array = x_df.drop(columns=[column_name]).to_numpy().T
        result = self.predict(np_array)
        arr_item = np.array([self._display_class[int(item)] for item in result > 0])
        return arr_item

    def acc(self, x_df: pd.DataFrame, column_name: str = "Label") -> float:
        predict_out = self.predict_with_df(x_df, column_name=column_name)
        true_label = x_df[column_name].to_numpy()

        return np.mean(predict_out == true_label)

    def acc_roc_auc(self, x_df: pd.DataFrame, mapping_dict: dict) -> dict:
        predict_out = self.predict_with_df(x_df)
        true_label = x_df["Label"].to_numpy()

        acc = np.mean(predict_out == true_label)

        predict_out_num = np.vectorize(mapping_dict.get)(predict_out)
        true_label_num = np.vectorize(mapping_dict.get)(true_label)

        fpr, tpr, thresholds = roc_curve(true_label_num, predict_out_num)

        roc_auc = auc(fpr, tpr)

        return {
            "acc": acc,
            "roc": (fpr, tpr),
            "auc": roc_auc,
        }

    def func(self) -> dict:
        return {"w": self._w, "b": self._b}

    def to_df_record(self) -> dict:
        return {
            "Name": f"Pos:{self._positive_class},Neg:{self._negative_class}",
            "Weight Vector": f"[{','.join(map(lambda x : f'{x:.2f}',  self._w))}]",
            "Bias": f"{self._b:.2f}",
        }

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def __repr__(self) -> str:
        if self._w is None or self._b is None or self._cov_matrix is None:
            return "Model is not fitted"

        return f"Pos:{self._positive_class}, Neg: {self._negative_class}, W : {self._w} B : {self._b:.2f} Cov:{self._cov_matrix}"

    def __str__(self) -> str:
        if self._w is None or self._b is None or self._cov_matrix is None:
            return "Model is not fitted"

        return f"Pos:{self._positive_class}, Neg: {self._negative_class}\nW : {self._w}\nB : {self._b:.2f}\nCov:\n{self._cov_matrix}"
