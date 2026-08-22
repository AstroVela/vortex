#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Downloads the real-world corpus used by the delta-encoding study into ./data.
# Everything here is a public dataset; nothing is generated.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data
cd data

fetch() {
  local url=$1 out=$2
  if [[ -f $out ]]; then
    echo "have $out"
    return
  fi
  echo "fetching $out"
  curl -sSL --max-time 1800 -o "$out" "$url"
}

# Web analytics: ClickBench "hits", one partition of 1M rows, 105 columns.
fetch https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet hits_0.parquet

# Sensor telemetry with Kafka offsets and nanosecond arrival timestamps.
fetch https://pcodec-public.s3.amazonaws.com/devinrsmith-air-quality.20220714.zstd.parquet airquality.parquet

# r/place 2022 canvas events: 160M rows of sorted timestamps and pixel coordinates.
fetch https://pcodec-public.s3.amazonaws.com/reddit_2022_place_numerical.parquet rplace.parquet

# NYC yellow taxi trips: near-sorted pickup/dropoff timestamps.
fetch https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet taxi.parquet

# Exchange data: fixed-interval klines (the periodic-timestamp case delta-of-delta targets)
# and raw trades (sequential ids, irregular millisecond timestamps).
fetch https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-2024-01.zip btc1s.zip
fetch https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip btc1m.zip
fetch https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip btctrades.zip

# UCI household power consumption: one-minute smart-meter readings over four years.
fetch https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip power.zip

for z in btc1s btc1m btctrades power; do
  unzip -o -q "$z.zip"
done

cd ..
uv run --with pyarrow --with pandas --with numpy python convert.py
