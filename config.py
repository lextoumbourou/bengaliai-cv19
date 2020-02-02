from typing import Dict
from dataclasses import dataclass


@dataclass
class ExperimentParams:
    name: str
    template: str
    params: Dict


EXPERIMENTS = [
    ExperimentParams(
        name='more-ln-head-plus-mish',
        template='fastai2.ipynb',
        params=dict(MODEL_HEAD='mish_head')
    ),
    ExperimentParams(
        name='fastai2-baseline',
        template='fastai2.ipynb',
        params={}
    )
]

EXPERIMENT_MAP = {e.name: e for e in EXPERIMENTS}
