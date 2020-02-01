import os
import argparse
from pathlib import Path

import papermill as pm

from config import EXPERIMENT_MAP, EXPERIMENTS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', nargs='?')
    parser.add_argument('--shutdown', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.exp_name not in [e.name for e in EXPERIMENTS]:
        print(f'Experiments:')
        for exp in EXPERIMENTS:
            print(f'-- {exp.name}')

        exit(1)

    exp = EXPERIMENT_MAP[args.exp_name]

    params = exp.params
    params['WANDB_MODE'] = 'run'
    params['NAME'] = exp.name
    params['OUTPUT_VAL_SIZE'] = None

    pm.execute_notebook(
        input_path=str(Path('nbs/templates') / exp.template),
        output_path=str(Path('nbs/output') / f'{exp.name}.ipynb'),
        parameters=params, log_output=True)

    if args.shutdown:
        os.system('sudo shutdown -h now')
