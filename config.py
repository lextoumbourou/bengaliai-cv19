from typing import Dict
from dataclasses import dataclass


@dataclass
class ExperimentParams:
    name: str
    template: str
    params: Dict


EXPERIMENTS = [
    ExperimentParams(
        name='effb0_cutmix',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=256, USE_FP16=True, USE_CUTMIX=True)
    ),
    ExperimentParams(
        name='eff0_finetune_with_224',
        template='fastai-stage2-finetune.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=128, LOAD_EXPERIMENT='lex/bengaliai-cv19/zgfjwgwj')
    ),
    ExperimentParams(
        name='effb0_move_to_fp16',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=256, USE_FP16=True)
    ),
    ExperimentParams(
        name='effb0_change_norm_std',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=192)
    ),
    ExperimentParams(
        name='effb0_with_class_weights',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=192)
    ),
    ExperimentParams(
        name='effb0_reduce_reg',
        template='fastai-mixup-cutup.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=192)
    ),
    ExperimentParams(
        name='effb0_gem_lb_smooth_cutmix',
        template='fastai-mixup-cutup.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=128)
    ),
    ExperimentParams(
        name='effb0_baseline',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=128)
    ),
    ExperimentParams(
        name='effb0_with_ohem_on_grapheme_root',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=128)
    ),
    ExperimentParams(
        name='effb0_with_gem',
        template='fastai.ipynb',
        params=dict(ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=128)
    ),
    ExperimentParams(
        name='se_resnext50_32x4d',
        template='fastai.ipynb',
        params=dict(
            LABEL_SMOOTHING_EPS=0.0, ENCODER_ARCH='se_resnext50_32x4d',
            BATCH_SIZE=64
        )
    ),
    ExperimentParams(
        name='mixnet_xl',
        template='fastai.ipynb',
        params=dict(
            LABEL_SMOOTHING_EPS=0.0, ENCODER_ARCH='mixnet_xl', BATCH_SIZE=64)
    ),
    ExperimentParams(
        name='label-smoothing-v2',
        template='fastai.ipynb',
        params=dict(LABEL_SMOOTHING_EPS=0.05)
    ),
    ExperimentParams(
        name='label-smoothing',
        template='fastai.ipynb',
        params=dict()
    ),
    ExperimentParams(
        name='progressive-sprinkles',
        template='fastai.ipynb',
        params=dict(PROG_SPRINKLES=True)
    ),
    ExperimentParams(
        name='mixup-cutup-until-convergence',
        template='fastai.ipynb',
        params=dict()
    ),
    ExperimentParams(
        name='mixup-cutup-50epochs',
        template='fastai.ipynb',
        params=dict(BATCH_SIZE=128, NUM_EPOCHS=50)
    ),
    ExperimentParams(
        name='more-epochs',
        template='fastai2.ipynb',
        params=dict(BATCH_SIZE=128, NUM_EPOCHS=20)
    ),
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
