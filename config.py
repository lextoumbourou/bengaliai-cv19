from typing import Dict
from dataclasses import dataclass


@dataclass
class ExperimentParams:
    name: str
    template: str
    params: Dict


EXPERIMENTS = [
    ExperimentParams(
        name='se_resnext50_drop_sched_to_mish',
        template='fastai-mish-and-no-max-pool-last-layers.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=768,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5,
            GRAPHEME_ROOT_WEIGHT=2/4,
            VOWEL_DIACRITIC_WEIGHT=1/4,
            CONSONANT_DIACRITIC_WEIGHT=1/4
        )
    ),
    ExperimentParams(
        name='se_resnext50_to_mish_and_only_gem_pooling',
        template='fastai-mish-and-no-max-pool-last-layers.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=768,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5,
            GRAPHEME_ROOT_WEIGHT=2/4,
            VOWEL_DIACRITIC_WEIGHT=1/4,
            CONSONANT_DIACRITIC_WEIGHT=1/4
        )
    ),
    ExperimentParams(
        name='se_resnext50_change_grapheme_root_head',
        template='fastai-grapheme-root-fc.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=896,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5,
            GRAPHEME_ROOT_WEIGHT=2/4,
            VOWEL_DIACRITIC_WEIGHT=1/4,
            CONSONANT_DIACRITIC_WEIGHT=1/4
        )
    ),
    ExperimentParams(
        name='se_resnext50_only_gem_pooling',
        template='fastai-no-max-pool-last-layers.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=896,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5,
            GRAPHEME_ROOT_WEIGHT=2/4,
            VOWEL_DIACRITIC_WEIGHT=1/4,
            CONSONANT_DIACRITIC_WEIGHT=1/4
        )
    ),
    ExperimentParams(
        name='se_resnext50_change_head_weights',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=896,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5,
            GRAPHEME_ROOT_WEIGHT=2/4,
            VOWEL_DIACRITIC_WEIGHT=1/4,
            CONSONANT_DIACRITIC_WEIGHT=1/4
        )
    ),
    ExperimentParams(
        name='se_resnext50_lower_reduce_factor',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d',
            IMG_SIZE=64,
            BATCH_SIZE=896,
            USE_FP16=True,
            USE_CUTMIX=True,
            USE_MIXUP=True,
            REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5
        )
    ),
    ExperimentParams(
        name='se_resnext50_mixup_and_cutmix_more_training',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, USE_MIXUP=True, REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5, LOAD_EXPERIMENT='lex/bengaliai-cv19/xde8r372',
            LR=0.0001
        )
    ),
    ExperimentParams(
        name='se_resnext50_mixup_and_cutmix_pretrain_first_channel_lower_lr',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, USE_MIXUP=True, REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.5, LOAD_EXPERIMENT='lex/bengaliai-cv19/2ps2d1kz',
            LR=0.00075
        )
    ),
    ExperimentParams(
        name='se_resnext50_mixup_and_cutmix_pretrain_first_channel',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, USE_MIXUP=True, REDUCE_LR_PATIENCE=5,
            REDUCE_LR_FACTOR=0.8
        )
    ),
    ExperimentParams(
        name='se_resnext50_finetune_mixup_224',
        template='fastai-stage2-finetune.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=64,
            IMG_SIZE=224, USE_MIXUP=True, LOAD_EXPERIMENT = "lex/bengaliai-cv19/l18p1wge",
            USE_FP16=True, USE_CUTMIX=False, REDUCE_LR_PATIENCE=5,
            LR=1e-4
        )
    ),
    ExperimentParams(
        name='se_resnext50_mixup_and_cutmix_more_patience',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, USE_MIXUP=True, REDUCE_LR_PATIENCE=6
        )
    ),
    ExperimentParams(
        name='se_resnext50_discrim_lr_cutmix_more_patience',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, REDUCE_LR_PATIENCE=6,
            LR=[3e-3, 1e-2]
        )
    ),
    ExperimentParams(
        name='se_resnext50_cutmix_more_patience',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='se_resnext50_32x4d', BATCH_SIZE=192,
            USE_FP16=True, USE_CUTMIX=True, REDUCE_LR_PATIENCE=6
        )
    ),
    ExperimentParams(
        name='effb0_cutmix_more_patience',
        template='fastai.ipynb',
        params=dict(
            ENCODER_ARCH='efficientnet-b0', BATCH_SIZE=256,
            USE_FP16=True, USE_CUTMIX=True, REDUCE_LR_PATIENCE=6
        )
    ),
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
