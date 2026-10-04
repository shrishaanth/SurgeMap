import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

from logic import (anchor_index, diverging_colors, geometry_centroid, hotspot_table, moves_to_arcs,
                   ranks_desc, sequential_colors)


def test_sequential_colors_are_lighter_for_small_values_and_valid_rgba():
    colors = sequential_colors(np.array([0.0, 1.0, 10.0, 100.0, 500.0]), vmax=100.0)
    assert colors.shape == (5, 4) and colors.dtype == np.uint8
    brightness = colors[:, :3].sum(axis=1)
    assert brightness[0] > brightness[1] > brightness[2] > brightness[3]
    np.testing.assert_array_equal(colors[3], colors[4])          # values above vmax are clipped


def test_diverging_colors_are_neutral_at_zero_and_red_blue_at_the_extremes():
    colors = diverging_colors(np.array([-5.0, 0.0, 5.0]), vmax=5.0)
    assert colors[0, 2] > colors[0, 0]          # negative -> blue
    assert colors[2, 0] > colors[2, 2]          # positive -> red
    assert tuple(colors[1, :3]) == (247, 247, 247)


def test_anchor_index_maps_a_time_to_a_five_minute_slot_and_clips():
    first = pd.Timestamp("2024-01-27 08:20")
    assert anchor_index(first, pd.Timestamp("2024-01-27 08:35"), 100) == 3
    assert anchor_index(first, pd.Timestamp("2024-01-20"), 100) == 0
    assert anchor_index(first, pd.Timestamp("2024-02-20"), 100) == 99


def test_ranks_desc_orders_largest_first_and_breaks_ties_by_position():
    np.testing.assert_array_equal(ranks_desc(np.array([5.0, 9.0, 5.0, 1.0])), [2, 1, 3, 4])


def test_hotspot_table_counts_hits_against_the_true_top_k():
    zone_ids = np.array([10, 20, 30, 40])
    names = {10: ["A", "Queens"], 20: ["B", "Bronx"], 30: ["C", "Manhattan"], 40: ["D", "Brooklyn"]}
    pred = np.array([9.0, 8.0, 1.0, 7.0])
    actual = np.array([9.0, 1.0, 8.0, 7.0])
    table, hits = hotspot_table(pred, actual, zone_ids, names, k=2)
    assert list(table["Zone"]) == ["A", "B"]
    assert list(table["Hit"]) == [True, False]
    assert hits == 1


def test_geometry_centroid_of_a_square():
    square = {"type": "MultiPolygon", "coordinates": [[[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]]}
    lon, lat = geometry_centroid(square)
    assert 0.0 < lon < 2.0 and 0.0 < lat < 2.0


def test_moves_to_arcs_skips_zones_without_a_polygon():
    zone_ids = np.array([1, 2, 3])
    names = {1: ["One", "X"], 2: ["Two", "X"], 3: ["Three", "X"]}
    centroids = {1: (-74.0, 40.7), 2: (-73.9, 40.8)}
    moves = np.zeros((3, 3), dtype=int)
    moves[0, 1], moves[0, 2] = 4, 2
    arcs = moves_to_arcs(moves, zone_ids, centroids, names)
    assert len(arcs) == 1 and arcs.iloc[0]["vehicles"] == 4 and arcs.iloc[0]["to_zone"] == "Two"
    assert moves_to_arcs(np.zeros((3, 3), dtype=int), zone_ids, centroids, names).empty
