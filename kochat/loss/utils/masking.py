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
from torch import Tensor
from torch import nn

from kochat.decorators import loss


@loss
class Masking(nn.Module):

    def __init__(self):
        """
        시퀀스 길이를 받아서 max_len 길이의 마스킹 벡터들의 집합을 만듭니다.
        e.g. sequence_length = 3,
        [True, True, True, False, False , ..., False]
        """

        super().__init__()

    def forward(self, sequence_length: Tensor) -> Tensor:
        batch_size = sequence_length.size(0)
        masks = []

        for i in range(batch_size):
            mask = torch.zeros(self.max_len, dtype=torch.uint8).to(self.device)
            # 전부다 0으로 된 마스킹 벡터 생성

            for j in range(sequence_length[i]):
                # seq length까지만 1로 만들어줌
                mask[j] = 1

            masks.append(mask.unsqueeze(0))
            # 마스크 배열에 넣어줌

        return torch.cat(masks, dim=0)
        # batchwise concatenation
