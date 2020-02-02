import torch
from torch import nn
from fastai2.layers import MishJit, AdaptiveConcatPool2d


class MishHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_ps=0.5):
        super().__init__()

        layers = [
            nn.Linear(input_size * 2, 512),
            MishJit(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_ps),
            nn.Linear(512, output_size)
        ]

        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


HEAD_MAP = {
    'mish_head': MishHead
}
