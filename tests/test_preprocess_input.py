import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from preprocess import DOLOC, DROPOFF, PICKUP, PULOC, iter_chunks

COLS = [PICKUP, DROPOFF, PULOC, DOLOC]


def _trips():
    return pd.DataFrame({
        PICKUP: pd.to_datetime(["2024-01-01 00:01:00", "2024-01-01 00:07:00", "2024-01-02 10:00:00"]),
        DROPOFF: pd.to_datetime(["2024-01-01 00:20:00", "2024-01-01 00:30:00", "2024-01-02 10:15:00"]),
        PULOC: [4, 7, 12],
        DOLOC: [7, 12, 4],
        "fare": [1.0, 2.0, 3.0],
    })


def test_parquet_and_csv_inputs_give_the_same_rows(tmp_path):
    trips = _trips()
    trips.to_csv(tmp_path / "trips.csv", index=False)
    trips.to_parquet(tmp_path / "trips.parquet", index=False)
    from_csv = pd.concat(iter_chunks(str(tmp_path / "trips.csv"), COLS, 2), ignore_index=True)
    from_parquet = pd.concat(iter_chunks(str(tmp_path / "trips.parquet"), COLS, 2), ignore_index=True)
    pd.testing.assert_frame_equal(from_csv[COLS], from_parquet[COLS], check_dtype=False)
    assert list(from_parquet.columns) == COLS


def test_chunks_respect_the_requested_size(tmp_path):
    _trips().to_parquet(tmp_path / "trips.parquet", index=False)
    sizes = [len(c) for c in iter_chunks(str(tmp_path / "trips.parquet"), COLS, 2)]
    assert sizes == [2, 1]
