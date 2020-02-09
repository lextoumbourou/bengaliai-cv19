# Bengaliai CV19

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install fastai
pip install git+https://github.com/fastai/fastai2
pip install efficientnet-pytorch
pip install jupyter papermill flake8 jupyter_contrib_nbextensions matplotlib
pip install iterative-stratification
pip install kaggle
pip install "pillow<7"
pip install python-snappy
pip install fastparquet
pip install opencv-python
```

## Get data

```
kaggle datasets download -d iafoss/grapheme-imgs-128x128 --path=data/grapheme-imgs-128x128
kaggle kernels output  yiheng/iterative-stratification --path=data/iterative-stratification
```
