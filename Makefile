.PHONY: install preprocess-265 train-265 train-multi-265 export evaluate plots app test smoke

PYTHON ?= python
DATA_DIR_265 ?= real_processed_265
RAW_INPUT ?= data/yellow_tripdata_2024-01.parquet

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

preprocess-265:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_INPUT)" --out "$(DATA_DIR_265)" --month 2024-01 --all-zones

train-265:
	$(PYTHON) scripts/train_stgnn_torch.py --data-dir "$(DATA_DIR_265)" --epochs 50 --window 48 --horizon 5 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out stgnn_265_checkpoint.pt

SEED ?= 7

train-multi-265:
	$(PYTHON) scripts/train_multihorizon_torch.py --data-dir "$(DATA_DIR_265)" --window 48 --epochs 80 --hidden 64 --batch-size 32 --lr 0.001 --patience 12 --shuffle --loss poisson --lag-features --seed $(SEED) --out model/poisson_lag_seed$(SEED).pt

export:
	$(PYTHON) scripts/export_predictions.py --data-dir "$(DATA_DIR_265)" --checkpoint model --out-dir artifacts
	$(PYTHON) scripts/accuracy_checks.py --merge
	$(PYTHON) scripts/calibrate_forecasts.py --checkpoint model

evaluate:
	$(PYTHON) scripts/run_evaluation.py --data-dir "$(DATA_DIR_265)" --predictions artifacts/predictions.npz --out results/evaluation.json

plots:
	$(PYTHON) scripts/plot_results.py --results results/evaluation.json --out-dir results

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -c "import numpy, pandas, torch, scipy; print('core dependencies import successfully')"
