# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Convert the raw real-world CSVs into integer-typed parquet, the way a columnar
store would actually hold them (fixed-point prices, epoch-millis timestamps)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base_volume", "taker_quote_volume", "ignore",
]


def fixed(series, scale=100_000_000):
    return np.rint(series.astype("float64") * scale).astype("int64")


def klines(src, dst, nrows=None):
    df = pd.read_csv(src, header=None, names=KLINE_COLS, nrows=nrows)
    out = pd.DataFrame(
        {
            "open_time": df.open_time.astype("int64"),
            "close_time": df.close_time.astype("int64"),
            "open": fixed(df.open),
            "high": fixed(df.high),
            "low": fixed(df.low),
            "close": fixed(df.close),
            "volume": fixed(df.volume),
            "quote_volume": fixed(df.quote_volume),
            "trades": df.trades.astype("int32"),
            "taker_base_volume": fixed(df.taker_base_volume),
            "taker_quote_volume": fixed(df.taker_quote_volume),
        }
    )
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst)
    print(dst, out.shape)


def trades(src, dst, nrows=8_000_000):
    df = pd.read_csv(
        src,
        header=None,
        names=["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"],
        nrows=nrows,
    )
    out = pd.DataFrame(
        {
            "id": df.id.astype("int64"),
            "time": df.time.astype("int64"),
            "price": fixed(df.price),
            "qty": fixed(df.qty),
            "quote_qty": fixed(df.quote_qty),
        }
    )
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst)
    print(dst, out.shape)


def power(src, dst):
    df = pd.read_csv(src, sep=";", na_values=["?"], low_memory=False)
    ts = pd.to_datetime(df.Date + " " + df.Time, format="%d/%m/%Y %H:%M:%S")
    out = pd.DataFrame({"timestamp": ts.astype("int64") // 1_000_000_000})
    for col, scale in [
        ("Global_active_power", 1000),
        ("Global_reactive_power", 1000),
        ("Voltage", 1000),
        ("Global_intensity", 1000),
        ("Sub_metering_1", 1),
        ("Sub_metering_2", 1),
        ("Sub_metering_3", 1),
    ]:
        v = df[col].astype("float64").ffill().fillna(0.0)
        out[col] = np.rint(v * scale).astype("int64")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst)
    print(dst, out.shape)


if __name__ == "__main__":
    klines("data/BTCUSDT-1s-2024-01.csv", "data/btc_1s.parquet")
    klines("data/BTCUSDT-1m-2024-01.csv", "data/btc_1m.parquet")
    trades("data/BTCUSDT-trades-2024-01.csv", "data/btc_trades.parquet")
    power("data/household_power_consumption.txt", "data/power.parquet")
