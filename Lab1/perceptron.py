from typing import List

import numpy as np

from data_reader import TrainData


class Perceptron:

    def __init__(self, size: int, h: float = 0.75) -> None:
        self.h = h
        self.weights = np.random.uniform(-1, 1, size)

    def train(self, train_data: List[TrainData], nu: float = 000.1, epochs: int = 10) -> None:
        for epoch in range(epochs):
            for train_obj in train_data:
                res = 0
                sum_local = 0
                for i in range(len(train_obj.data)):
                    sum_local += train_obj.data[i] * self.weights[i]
                if sum_local > 1:
                    res = 1
                else:
                    res = 0
                error = train_obj.target - res
                for i in range(len(self.weights)):
                    self.weights[i] += error * nu * train_obj.data[i]

    def predict(self, test_data: TrainData) -> int:
        sum_local = 0
        for i in range(len(test_data.data)):
            sum_local += test_data.data[i] * self.weights[i]
        return 1 if sum_local > 1 else 0
