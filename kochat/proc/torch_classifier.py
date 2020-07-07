"""
@auther Hyunwoong
@since 6/28/2020
@see https://github.com/gusdnd852
"""
from abc import ABCMeta, abstractmethod
from time import time
from typing import List

import torch
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam

from kochat.decorators import intent
from kochat.model.fallback.ann import ANN
from kochat.proc.torch_processor import TorchProcessor


@intent
class TorchClassifier(TorchProcessor, metaclass=ABCMeta):

    def __init__(self, model: nn.Module, parameters: Parameter or List[Parameter]):
        model = self.__add_classifier(model)
        self.ood_dataset = None
        self.label_dict = model.label_dict
        self.fallback_detector = ANN(self.label_dict, self.d_model).to(self.device)
        self._initialize_weights(self.fallback_detector)
        self.fallback_loss = nn.CrossEntropyLoss()
        self.fallback_opt = Adam(params=self.fallback_detector.parameters(),
                                 lr=self.loss_lr,
                                 weight_decay=self.weight_decay)

        super().__init__(model, parameters)

    def fit(self, dataset: tuple):
        """
        Pytorch 모델을 학습/테스트하고
        모델의 출력값을 다양한 방법으로 시각화합니다.
        최종적으로 학습된 모델을 저장합니다.

        :param dataset: 학습할 데이터셋
        :param test: 테스트 여부
        """

        # 데이터 셋 unpacking
        self.train_data = dataset[0]
        self.test_data = dataset[1]

        if len(dataset) > 2:
            self.ood_dataset = dataset[2]

        for i in range(self.epochs + 1):
            eta = time()
            loss, label, predict = self._test_epoch(i)
            self._visualize(loss, label, predict, mode='test')
            # testing epoch + visualization

            loss, label, predict = self._train_epoch(i)
            self._visualize(loss, label, predict, mode='train')
            # training epoch + visualization

            if i > self.lr_scheduler_warm_up:
                self.lr_scheduler.step(loss)

            if i % self.save_epoch == 0:
                self._save_model()

            self._print('Epoch : {epoch}, ETA : {eta} sec '
                        .format(epoch=i, eta=round(time() - eta, 4)))

    @abstractmethod
    def _calibrate_msg(self, *args):
        raise NotImplementedError

    def __add_classifier(self, model):
        sample = torch.randn(1, self.max_len, self.vector_size)
        sample = sample.to(self.device)
        output_size = model.to(self.device)(sample)

        features = nn.Linear(output_size.shape[1], self.d_loss)
        classifier = nn.Linear(self.d_loss, len(model.label_dict))
        setattr(model, 'features', features.to(self.device))
        setattr(model, 'classifier', classifier.to(self.device))
        return model

    def __ood_visualize(self, labels, predicts):
        self.metrics.evaluate(labels, predicts, mode='ood')
        report, _ = self.metrics.report(['in_dist', 'out_dist'], mode='ood')
        report = report.drop(columns=['macro avg'])
        self.visualizer.draw_report(report, mode='ood')
