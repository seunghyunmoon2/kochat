# Copyright 2020 Kochat. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from abc import ABCMeta

from torch import nn

from kochat.loss.base_loss import BaseLoss


class DistanceLoss(BaseLoss, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def predict(self, feats):
        batch_size = feats.size(0)
        centers = self.centers.unsqueeze(0).repeat(batch_size, 1, 1)
        feats = feats.unsqueeze(1)

        metric = torch.norm(feats - centers, dim=2)
        metric = self.softmax(-metric)

        _, predict = metric.topk(k=1, largest=True)
        return predict, metric
