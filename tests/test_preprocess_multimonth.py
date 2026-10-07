import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_trips(start, days, n, seed):
    rng = np.random.default_rng(seed)
    pickup = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days * 24 * 60 - 40, n), unit="min")
    dropoff = pickup + pd.to_timedelta(rng.integers(5, 25, n), unit="min")
    return pd.DataFrame({"tpep_pickup_datetime": pickup, "tpep_dropoff_datetime": dropoff,
                         "PULocationID": rng.choice([3, 5, 7], n), "DOLocationID": rng.choice([3, 5, 7], n)})


def run(args):
    subprocess.run([sys.executable, "-m", "surgemap", "preprocess"] + args, check=True, capture_output=True, cwd=ROOT)


def test_several_files_a_date_range_fixed_zones_and_explicit_split(tmp_path):
    first, second = make_trips("2024-01-01", 2, 400, 0), make_trips("2024-01-03", 2, 400, 1)
    first.to_csv(tmp_path / "a.csv", index=False)
    second.to_csv(tmp_path / "b.csv", index=False)
    zones_dir = tmp_path / "zones"
    zones_dir.mkdir()
    np.save(zones_dir / "zone_ids.npy", np.array([5, 3, 9], dtype=np.int32))      # zone 9 never appears

    both = tmp_path / "both"
    run(["--input", f"{tmp_path / 'a.csv'},{tmp_path / 'b.csv'}", "--out", str(both),
         "--start", "2024-01-01", "--end", "2024-01-05", "--zone-ids-from", str(zones_dir),
         "--train-end", "2024-01-03 00:00", "--val-end", "2024-01-04 00:00"])
    demand = np.load(both / "demand.npy")
    meta = json.loads((both / "metadata.json").read_text())
    assert demand.shape == (3, 4 * 288)
    np.testing.assert_array_equal(np.load(both / "zone_ids.npy"), [5, 3, 9])
    assert demand[2].sum() == 0
    assert meta["split"] == {"train": [0, 576], "validation": [576, 864], "test": [864, 1152],
                             "train_fraction": 0.7, "scaler_fit_end": 576}
    assert np.load(both / "features_clipped.npy").shape == (3, 1152, 7)
    for frame, zone, row in ((pd.concat([first, second]), 5, 0), (pd.concat([first, second]), 3, 1)):
        assert demand[row].sum() == (frame["PULocationID"] == zone).sum()

    # the later days must be identical to preprocessing the second file alone
    alone = tmp_path / "alone"
    run(["--input", str(tmp_path / "b.csv"), "--out", str(alone), "--start", "2024-01-03", "--end", "2024-01-05",
         "--zone-ids-from", str(zones_dir)])
    np.testing.assert_array_equal(demand[:, 576:], np.load(alone / "demand.npy"))
    times = np.load(both / "times.npy")
    assert str(times[0])[:16] == "2024-01-01T00:00" and str(times[-1])[:16] == "2024-01-04T23:55"


def test_single_month_behaviour_is_unchanged(tmp_path):
    trips = make_trips("2024-01-01", 31, 600, 2)
    trips.to_csv(tmp_path / "jan.csv", index=False)
    out = tmp_path / "jan"
    run(["--input", str(tmp_path / "jan.csv"), "--out", str(out), "--month", "2024-01", "--all-zones"])
    meta = json.loads((out / "metadata.json").read_text())
    assert np.load(out / "demand.npy").shape == (3, 8928)
    assert meta["split"]["train"] == [0, 6249] and meta["split"]["validation"] == [6249, 7588]
    np.testing.assert_array_equal(np.load(out / "zone_ids.npy"), [3, 5, 7])
