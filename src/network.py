# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from typing import Any


from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss



class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(
        self,
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=0.5,
        weight_init_std=0.01,
        hidden_sizes=(512, 256),
    ):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.

        # hidden size가 2개인게 맞아? 
        # train에서 줌? 
        input_size = 784
        output_size = 10
        self.hidden_sizes = tuple(hidden_sizes)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        self.params = {}
        layer_sizes = (input_size,) + self.hidden_sizes + (output_size,)

        for idx in range(1, len(layer_sizes)):
            prev_size = layer_sizes[idx - 1]
            current_size = layer_sizes[idx]
            self.params[f"W{idx}"] = weight_init_std * np.random.randn(prev_size, current_size)
            self.params[f"b{idx}"] = np.zeros(current_size)

            if idx <= len(self.hidden_sizes) and self.use_batchnorm:
                self.params[f"gamma{idx}"] = np.ones(current_size)
                self.params[f"beta{idx}"] = np.zeros(current_size)

        self.layers = OrderedDict()

        for idx in range(1, len(self.hidden_sizes) + 1):
            self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])

            if self.use_batchnorm:
                self.layers[f"BatchNorm{idx}"] = BatchNorm(
                    self.params[f"gamma{idx}"],
                    self.params[f"beta{idx}"],
                )

            self.layers[f"Relu{idx}"] = ReLU()

            if self.use_dropout:
                self.layers[f"Dropout{idx}"] = Dropout(dropout_ratio)

        output_idx = len(self.hidden_sizes) + 1
        self.layers[f"Affine{output_idx}"] = Affine(
            self.params[f"W{output_idx}"],
            self.params[f"b{output_idx}"],
        )

        self.lastLayer = Softmax()
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
            
        return self.lastLayer.forward(x)


    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        output_idx = len(self.hidden_sizes) + 1

        for idx in range(1, output_idx + 1):
            grads[f"W{idx}"] = self.layers[f"Affine{idx}"].dW
            grads[f"b{idx}"] = self.layers[f"Affine{idx}"].db

            if idx <= len(self.hidden_sizes) and self.use_batchnorm:
                grads[f"gamma{idx}"] = self.layers[f"BatchNorm{idx}"].dgamma
                grads[f"beta{idx}"] = self.layers[f"BatchNorm{idx}"].dbeta

        self.grads = grads
        return self.grads


    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
