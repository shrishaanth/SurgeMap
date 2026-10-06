.PHONY: install preprocess-265 preprocess-4mo train export evaluate plots app test smoke

PYTHON ?= python
DATA_DIR_265 ?= real_processed_265
DATA_DIR_4MO ?= real_processed_4mo
RAW_INPUT ?= data/yellow_tripdata_2024-01.parquet
RAW_4MO ?= data/yellow_tripdata_2023-10.parquet,data/yellow_tripdata_2023-11.parquet,data/yellow_tripdata_2023-12.parquet,data/yellow_tripdata_2024-01.parquet
SEED ?= 7

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# January 2024 only: the dataset the simulator and the app run on.
preprocess-265:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_INPUT)" --out "$(DATA_DIR_265)" --month 2024-01 --all-zones

# October 2023 to January 2024 with the same zones, validation days and test window:
# the dataset the shipped networks and the baselines are fitted on. Needs preprocess-265 first.
preprocess-4mo:
	$(PYTHON) scripts/preprocess.py --input "$(RAW_4MO)" --out "$(DATA_DIR_4MO)" --start 2023-10-01 --end 2024-02-01 --zone-ids-from "$(DATA_DIR_265)" --train-end "2024-01-22 16:45" --val-end "2024-01-27 08:20"

# One network; the shipped model averages SEED = 7, 1, 2, 3, 4.
train:
	$(PYTHON) scripts/train_multihorizon_torch.py --data-dir "$(DATA_DIR_4MO)" --window 48 --epochs 70 --hidden 64 --batch-size 32 --lr 0.001 --patience 8 --shuffle --loss poisson --lag-features --seed $(SEED) --out model/four_month_seed$(SEED).pt

# Forecasts of the networks in model/, the baselines fitted on the same data, calibration,
# and re-indexing onto the January dataset.
export:
	$(PYTHON) scripts/export_predictions.py --data-dir "$(DATA_DIR_4MO)" --checkpoint model --out-dir artifacts_4mo
	$(PYTHON) scripts/accuracy_checks.py --data-dir "$(DATA_DIR_4MO)" --predictions artifacts_4mo/predictions.npz --merge --out results/accuracy_checks.json
	$(PYTHON) scripts/calibrate_forecasts.py --data-dir "$(DATA_DIR_4MO)" --checkpoint model --predictions artifacts_4mo/predictions.npz --gbm-cache results/gbm_val_test.npz
	$(PYTHON) scripts/align_predictions.py --predictions artifacts_4mo/predictions.npz --data-dir "$(DATA_DIR_265)" --out artifacts/predictions.npz
	$(PYTHON) -c "import shutil; shutil.copy('artifacts_4mo/forecast_metrics.json', 'artifacts/forecast_metrics.json')"

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
