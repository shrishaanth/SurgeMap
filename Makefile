.PHONY: install preprocess-20 preprocess-265 train-265 train-multi-265 eval-265 test smoke docker-build docker-run

PYTHON ?= python
DATA_DIR_20 ?= real_processed_fixed
DATA_DIR_265 ?= real_processed_265
RAW_INPUT ?= data/yellow_tripdata_2024-01.csv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

preprocess-20:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_INPUT)" --out "$(DATA_DIR_20)" --month 2024-01 --top-k 20

preprocess-265:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_INPUT)" --out "$(DATA_DIR_265)" --month 2024-01 --all-zones
	$(PYTHON) scripts/build_travel_matrix.py --data-dir "$(DATA_DIR_265)"

train-265:
	$(PYTHON) scripts/train_stgnn_torch.py --data-dir "$(DATA_DIR_265)" --epochs 50 --window 48 --horizon 5 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out stgnn_265_checkpoint.pt

train-multi-265:
	$(PYTHON) scripts/train_multihorizon_torch.py --data-dir "$(DATA_DIR_265)" --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_stgnn_checkpoint_265.pt

eval-265:
	$(PYTHON) scripts/run_evaluation_265.py --data-dir "$(DATA_DIR_265)" --fleet-sizes "500,1000,1500,2000" --horizon 288 --seeds "0,1,2" --forecast-horizon 3 --out results_265.json

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -c "import numpy, pandas, torch, scipy; print('core dependencies import successfully')"

docker-build:
	docker build -t fleet-system .

docker-run:
	docker run --rm -v "$$(pwd):/app" fleet-system
