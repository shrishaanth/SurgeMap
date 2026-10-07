import io

import pytest

from surgemap.data.download import MONTHS, download, file_name, url_for


def test_urls_point_at_the_official_monthly_files():
    assert url_for("2024-01") == "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    assert MONTHS == ("2023-10", "2023-11", "2023-12", "2024-01")


def test_download_saves_the_file_and_skips_it_the_second_time(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        return io.BytesIO(b"trips")

    path, fetched = download("2023-10", str(tmp_path / "raw"), opener)
    assert fetched and open(path, "rb").read() == b"trips" and path.endswith(file_name("2023-10"))
    path_again, fetched_again = download("2023-10", str(tmp_path / "raw"), opener)
    assert path_again == path and not fetched_again and len(calls) == 1


def test_an_interrupted_download_leaves_no_finished_looking_file(tmp_path):
    class Broken(io.BytesIO):
        def read(self, *args):
            raise ConnectionError("dropped")

    with pytest.raises(ConnectionError):
        download("2023-11", str(tmp_path), lambda url: Broken())
    assert not (tmp_path / file_name("2023-11")).exists()
