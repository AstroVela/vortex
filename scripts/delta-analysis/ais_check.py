# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Does delta-of-delta pay on smooth trajectories? AIS vessel tracks, sorted per vessel.

This is the shape delta-of-delta is supposed to want: position sampled from something with
continuous velocity, so the second difference is acceleration.
"""

import numpy as np
import pandas as pd

from study import truth, BLOCK

cols = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
df = pd.read_csv("data/AIS_2023_01_01.csv", usecols=cols, nrows=6_000_000)
df = df.sort_values(["MMSI", "BaseDateTime"], kind="stable")
print(f"{len(df)} rows, {df.MMSI.nunique()} vessels, "
      f"median points/vessel {df.groupby('MMSI').size().median():.0f}")

frame = pd.DataFrame({
    "mmsi": df.MMSI.astype("int64"),
    "time": pd.to_datetime(df.BaseDateTime).astype("int64") // 10**9,
    "lat_e6": np.rint(df.LAT.astype("float64") * 1e6).astype("int64"),
    "lon_e6": np.rint(df.LON.astype("float64") * 1e6).astype("int64"),
    "sog_e1": np.rint(df.SOG.fillna(0).astype("float64") * 10).astype("int64"),
    "cog_e1": np.rint(df.COG.fillna(0).astype("float64") * 10).astype("int64"),
    "heading": np.rint(df.Heading.fillna(511).astype("float64")).astype("int64"),
})

rows = []
for name in frame.columns:
    values = frame[name].to_numpy()
    unsigned = values.astype("uint64")
    for block in range(min(len(values) // BLOCK, 24)):
        chunk = slice(block * BLOCK, (block + 1) * BLOCK)
        t = truth(unsigned[chunk], values[chunk], 8, True)
        rows.append(dict(column=name, block=block, **t))

res = pd.DataFrame(rows)
res["dod_gain"] = res.t_delta_bpv - res.t_dod_bpv
res["dod_residual_gain"] = (res.t_delta_bpv - 1 / 8) - (res.t_dod_bpv - 2 / 8)
pd.set_option("display.width", 250)
print(res.groupby("column")[
    ["t_plain_bpv", "t_delta_bpv", "t_dod_bpv", "t_delta_w", "t_dod_w", "dod_residual_gain"]
].mean().round(3).to_string())
print(f"\nblocks where DoD beats delta outright: {(res.dod_gain > 0).sum()} / {len(res)}")
print(f"blocks where DoD residuals alone are narrower (bases free): "
      f"{(res.dod_residual_gain > 0).sum()} / {len(res)}")
print(f"max residual bytes/value saved by second differencing: {res.dod_residual_gain.max():.4f}")

# The second delta layer's bases hold *residuals*, not values, so the cascade compresses them
# to about delta_w bits each rather than the full 64. Price them that way.
res["dod_realistic"] = (res.t_dod_bpv - 2 / 8) + 1 / 8 + (res.t_delta_w / 64) / 8
res["win"] = res.t_delta_bpv - res.dod_realistic
print("\n--- second-layer bases priced at their compressed width ---")
print(res.groupby("column")[["t_delta_bpv", "dod_realistic", "win"]].mean().round(4).to_string())
print(f"blocks where DoD wins: {(res.win > 0).sum()} / {len(res)}")
print(f"mean win where it wins: {res.win[res.win > 0].mean():.4f} B/value "
      f"({100 * (res.win[res.win > 0] / res.t_delta_bpv[res.win > 0]).mean():.1f}% of the column)")
