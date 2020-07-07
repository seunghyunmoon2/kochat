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
from torch import nn


class ANN(nn.Module):

    def __init__(self, label_dict, d_model):
        super().__init__()
        self.relu = nn.ReLU()
        self.stem = nn.Linear(len(label_dict), d_model)
        self.hidden = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        return x
