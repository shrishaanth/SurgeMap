# Fleet system run guide

## 20-zone pipeline (legacy, still supported)

```powershell
cd 'C:\path\to\ML project\fleet_system'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts\preprocess.py --input ..\yellow_tripdata_2024-01.csv --out real_processed_fixed --month 2024-01 --top-k 20

python scripts\train_stgnn_torch.py --data-dir real_processed_fixed --epochs 50 --window 12 --horizon 5 --hidden 64 --batch-size 128 --lr 0.001 --patience 8

python scripts\train_multihorizon_torch.py --data-dir real_processed_fixed --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_stgnn_checkpoint.pt

python scripts\multihorizon_baseline.py --data-dir real_processed_fixed --window 48 --horizons 1,3,6,12 --out multihorizon_baselines.json

python scripts\hotspot_eval.py --checkpoint multihorizon_stgnn_checkpoint.pt --data-dir real_processed_fixed --horizons 1,3,6,12 --topk 3,5 --out hotspot_metrics.json

python -m pytest -q
```

## 265-zone pipeline (Option A)

```powershell
cd 'C:\path\to\ML project\fleet_system'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM 1. Preprocess ALL 265 zones
python scripts\preprocess.py --input ..\yellow_tripdata_2024-01.csv --out real_processed_265 --month 2024-01 --all-zones

REM 2. Build travel-time matrix
python scripts\build_travel_matrix.py --data-dir real_processed_265

REM 3. Train multi-horizon ST-GNN on 265 zones
python scripts\train_multihorizon_torch.py --data-dir real_processed_265 --window 48 --epochs 50 --hidden 64 --batch-size 128 --lr 0.001 --patience 8 --out multihorizon_stgnn_checkpoint_265.pt

REM 4. Run evaluation (noop, reactive, LP+forecast)
python scripts\run_evaluation_265.py --data-dir real_processed_265 --fleet-sizes "500,1000,1500,2000" --horizon 288 --seeds "0,1,2" --forecast-horizon 3 --out results_265.json

REM 5. Run tests
python -m pytest -q
```

## Makefile targets

```powershell
make preprocess-265    # Preprocess 265 zones + build travel matrix
make train-multi-265   # Train ST-GNN for 265 zones
make eval-265          # Evaluate all policies on 265-zone simulator
make test              # Run test suite
```

## Tests and simulator status

`tests/test_preprocess.py` runs against a small in-memory CSV fixture and
checks the corrected month filtering, complete 5-minute grid, feature shape,
and directed graph orientation.

`tests/test_simulator_contract.py` is a contract suite for the per-vehicle
Gymnasium simulator (`scripts/fleet_env.py`). The 265-zone evaluation uses
`scripts/simulator265.py` (zone-level, no Gymnasium dependency).

`tests/test_multihorizon_eval.py` verifies the multi-horizon windowing and
baseline evaluation logic.

## Key insight

The original 20-zone experiment limited spatial variation because all 20 zones
were high-demand Manhattan neighborhoods. With all 265 NYC zones, the demand
distribution ranges from ~12,000 pickups/day in Midtown to ~10 in remote
Staten Island zones. This creates genuine spatial mismatch, making fleet
repositioning a real optimization problem rather than a trivial no-op.
