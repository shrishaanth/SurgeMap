"""Download the monthly yellow-taxi trip files from the NYC Taxi & Limousine Commission."""
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request

from surgemap import paths

SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = ("2023-10", "2023-11", "2023-12", "2024-01")     # what the shipped models are trained on


def file_name(month: str) -> str:
    return f"yellow_tripdata_{month}.parquet"


def url_for(month: str) -> str:
    return f"{SOURCE}/{file_name(month)}"


def download(month: str, out_dir: str, opener=urllib.request.urlopen) -> tuple[str, bool]:
    """Fetch one month into out_dir unless it is already there. Returns (path, downloaded).

    The file is written under a temporary name and renamed when complete, so an interrupted
    download never leaves a truncated file that looks finished.
    """
    target = os.path.join(out_dir, file_name(month))
    if os.path.exists(target):
        return target, False
    os.makedirs(out_dir, exist_ok=True)
    partial = target + ".part"
    with opener(url_for(month)) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(partial, target)
    return target, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default=",".join(MONTHS), help="comma-separated, as YYYY-MM")
    parser.add_argument("--out", default=paths.RAW_DIR)
    args = parser.parse_args()
    for month in [m.strip() for m in args.months.split(",") if m.strip()]:
        print(f"[download] {url_for(month)}", flush=True)
        path, fetched = download(month, args.out)
        size = os.path.getsize(path) / 1e6
        print(f"[download]   {'saved' if fetched else 'already present'}: {path.replace(os.sep, '/')} ({size:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
