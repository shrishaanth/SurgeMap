.PHONY: install preprocess-265 train-265 train-multi-265 test smoke

PYTHON ?= python
DATA_DIR_265 ?= real_processed_265
RAW_INPUT ?= data/yellow_tripdata_2024-01.csv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

preprocess-265:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_INPUT)" --out "$(DATA_DIR_265)" --month 2024-01 --all-zones

train-265:
	$(PYTHON) scripts/train_stgnn_torch.py --data-dir "$(DATA_DIR_265)" --epochs 50 --window 48 --horizon 5 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out stgnn_265_checkpoint.pt

train-multi-265:
	$(PYTHON) scripts/train_multihorizon_torch.py --data-dir "$(DATA_DIR_265)" --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_stgnn_checkpoint_265.pt

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -c "import numpy, pandas, torch, scipy; print('core dependencies import successfully')"
