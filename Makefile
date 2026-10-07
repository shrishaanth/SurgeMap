.PHONY: install download preprocess-january preprocess-four-months train export simulate plots analyses app test

PYTHON ?= python
JANUARY ?= data/processed/january_2024
FOUR_MONTHS ?= data/processed/four_months
STAGING ?= outputs/forecasts_four_months
RAW_JANUARY ?= data/raw/yellow_tripdata_2024-01.parquet
RAW_FOUR_MONTHS ?= data/raw/yellow_tripdata_2023-10.parquet,data/raw/yellow_tripdata_2023-11.parquet,data/raw/yellow_tripdata_2023-12.parquet,data/raw/yellow_tripdata_2024-01.parquet
SEED ?= 7

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# The four monthly trip files (about 50 MB each) from the NYC TLC site, into data/raw/.
download:
	$(PYTHON) -m surgemap download

# January 2024 only: the dataset the simulator and the app run on.
preprocess-january:
	$(PYTHON) -m surgemap preprocess --input "$(RAW_JANUARY)" --out "$(JANUARY)" --month 2024-01 --all-zones

# October 2023 to January 2024 with the same zones, validation days and test window:
# the dataset the shipped networks and the baselines are fitted on. Needs preprocess-january first.
preprocess-four-months:
	$(PYTHON) -m surgemap preprocess --input "$(RAW_FOUR_MONTHS)" --out "$(FOUR_MONTHS)" --start 2023-10-01 --end 2024-02-01 --zone-ids-from "$(JANUARY)" --train-end "2024-01-22 16:45" --val-end "2024-01-27 08:20"

# One network (needs a GPU); the shipped model averages SEED = 7, 1, 2.
train:
	$(PYTHON) -m surgemap train --data-dir "$(FOUR_MONTHS)" --window 48 --epochs 45 --batch-size 32 --lr 0.001 --patience 6 --shuffle --loss poisson --lag-features --arch gwnet --amp --seed $(SEED) --out models/graph_wavenet_seed$(SEED).pt

# Forecasts of the networks in models/, the baselines fitted on the same data, calibration,
# and re-indexing onto the January dataset.
export:
	$(PYTHON) -m surgemap export --data-dir "$(FOUR_MONTHS)" --checkpoint models --out-dir "$(STAGING)"
	$(PYTHON) -m surgemap baselines --data-dir "$(FOUR_MONTHS)" --predictions "$(STAGING)/predictions.npz" --merge
	$(PYTHON) -m surgemap calibrate --data-dir "$(FOUR_MONTHS)" --checkpoint models --predictions "$(STAGING)/predictions.npz"
	$(PYTHON) -m surgemap align --predictions "$(STAGING)/predictions.npz" --data-dir "$(JANUARY)" --out outputs/forecasts/predictions.npz
	$(PYTHON) -c "import shutil; shutil.copy('$(STAGING)/forecast_metrics.json', 'outputs/forecasts/forecast_metrics.json')"

simulate:
	$(PYTHON) -m surgemap simulate --seeds 0,1,2,3,4

plots:
	$(PYTHON) -m surgemap plots

analyses:
	$(PYTHON) -m surgemap headroom
	$(PYTHON) -m surgemap regimes

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest -q
