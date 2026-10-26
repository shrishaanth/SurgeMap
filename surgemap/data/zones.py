from __future__ import annotations

import argparse
import csv
import json
import os

from surgemap import paths


'''
out = [] — creates an empty list to store the cleaned points.
for lon, lat in ring: — loops through each point, unpacking its longitude and latitude.
round(lon, digits) — rounds longitude to the specified number of decimal places. The same happens to latitude.
if not out — if no point has been added yet, accept the first point.
point != out[-1] — otherwise, check whether the new point differs from the previous point.
out.append(point) — add it only if it isn't a consecutive duplicate.
return out — return the cleaned list.
'''

'''
    overall this file is for cleaning the geodata and producing zone lookup table
'''

def compact_ring(ring, digits: int):
    out = []
    for lon, lat in ring:
        point = [round(lon, digits), round(lat, digits)]
        if not out or point != out[-1]:
            out.append(point)
    return out


def compact_geometry(geometry: dict, digits: int = 4) -> dict:
    """Round coordinates (4 digits is about 10 m) and drop repeated points."""
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    rings = [[compact_ring(ring, digits) for ring in polygon] for polygon in polygons]
    rings = [[ring for ring in polygon if len(ring) >= 4] for polygon in rings]
    rings = [polygon for polygon in rings if polygon]
    return {"type": "MultiPolygon", "coordinates": rings}



'''
    takes the data , cleans it up , outputs the geojson , mapping of locations IDs to zone and boroughs

'''
def main() -> None:
    parser = argparse.ArgumentParser(description="Trim the NYC Open Data 'NYC Taxi Zones' GeoJSON for the app.")
    parser.add_argument("--raw", required=True, help="GeoJSON exported from NYC Open Data dataset 8meu-9t5y")
    parser.add_argument("--lookup", default=paths.RAW_DIR + "/taxi_zone_lookup.csv")
    parser.add_argument("--out-dir", default="app/assets")
    args = parser.parse_args()

    with open(args.raw, encoding="utf-8") as f:
        raw = json.load(f)
    features = [{"type": "Feature",
                 "properties": {"location_id": int(x["properties"]["locationid"]),
                                "zone": x["properties"]["zone"], "borough": x["properties"]["borough"]},
                 "geometry": compact_geometry(x["geometry"])}
                for x in raw["features"]]
    names = {}
    with open(args.lookup, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            names[int(row["LocationID"])] = [row["Zone"], row["Borough"]]

    os.makedirs(args.out_dir, exist_ok=True)
    geo_path = os.path.join(args.out_dir, "taxi_zones.geojson")
    with open(geo_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, separators=(",", ":"))
    with open(os.path.join(args.out_dir, "zone_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, separators=(",", ":"))
    print(f"wrote {geo_path} ({os.path.getsize(geo_path) / 1e6:.2f} MB, {len(features)} zones) "
          f"and zone_names.json ({len(names)} names)")


if __name__ == "__main__":
    main()
