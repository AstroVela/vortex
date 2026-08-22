# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Empirical study: cheap sampled stats as predictors of Delta / Delta-of-Delta benefit.

Ground-truth cost model mirrors Vortex:

  * FastLanes Delta residuals are exactly lag-1 differences in the original order
    (verified from the fastlanes transpose/iterate permutation); the 1024/T bases
    per 1024-element chunk cost exactly 1 bit per value.
  * The residual child cascade is FoR + BitPacking (ZigZag + BitPacking is
    equivalent in width), whose width is chosen by `best_bit_width`: minimise
    packed_cost + exception_cost with exceptions costing (byte_width + 4) bytes.
  * Signedness matters: Vortex deltas in the *unsigned* domain, so for an unsigned
    column a negative delta wraps to a huge value and FoR cannot recover it. For a
    signed column the residual array is signed and FoR/ZigZag see the true span.
    We model both: `t_*` is Vortex as it is today, `a_*` is what is achievable if
    residuals are always interpreted as signed.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BLOCK = 65536  # compression unit
U64 = np.uint64
BASE_BPV = 1 / 8  # one delta layer stores 1 bit/value of bases


def bit_length(u: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(u, dtype=np.uint64).copy()
    bl = np.zeros(u.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = u >= (U64(1) << U64(shift))
        bl[mask] += shift
        u[mask] >>= U64(shift)
    bl += (u > 0).astype(np.int64)
    return bl


def hist_of(bl: np.ndarray, width: int) -> np.ndarray:
    if bl.size == 0:
        return np.zeros(width + 1, dtype=np.int64)
    return np.bincount(np.minimum(bl, width), minlength=width + 1)[: width + 1]


def best_width_cost(hist: np.ndarray, byte_width: int):
    """Vortex `best_bit_width`: (width, total_bytes) minimising packed + exceptions."""
    n = int(hist.sum())
    if n == 0:
        return 0, 0
    bpe = byte_width + 4
    best_cost, best_w, packed = n * bpe, 0, 0
    for w, freq in enumerate(hist):
        packed += int(freq)
        cost = (n - packed) * bpe + (w * n + 7) // 8
        if cost < best_cost:
            best_cost, best_w = cost, w
    return best_w, best_cost


def mask_of(width: int) -> np.uint64:
    return U64(0xFFFFFFFFFFFFFFFF) if width >= 64 else (U64(1) << U64(width)) - U64(1)


def for_cost_unsigned(vals_u: np.ndarray, byte_width: int):
    """FoR + BitPack over an unsigned-domain array: (bytes/value, width)."""
    if vals_u.size == 0:
        return 0.0, 0
    w, cost = best_width_cost(bit_hist_unsigned(vals_u, byte_width * 8), byte_width)
    return cost / vals_u.size, w


def bit_hist_unsigned(vals_u: np.ndarray, width: int) -> np.ndarray:
    return hist_of(bit_length(vals_u - vals_u.min()), width)


def for_cost_signed(vals_i: np.ndarray, byte_width: int):
    """Best of {FoR, ZigZag} + BitPack over a signed array: (bytes/value, width).

    Both are available to the cascade below a Delta layer, and they fail on opposite
    inputs: FoR is ruined by one extreme outlier (it shifts every value up by |min|),
    ZigZag leaves outliers as cheap bitpacking exceptions.
    """
    if vals_i.size == 0:
        return 0.0, 0
    v = vals_i.astype(np.int64)
    span = (v - v.min()).astype(np.uint64)
    w_for, c_for = best_width_cost(hist_of(bit_length(span), byte_width * 8), byte_width)
    w_zz, c_zz = best_width_cost(hist_of(zz_bits(v), byte_width * 8), byte_width)
    return (c_for / v.size, w_for) if c_for <= c_zz else (c_zz / v.size, w_zz)


def wrapping_delta_u(vals_u: np.ndarray, width: int) -> np.ndarray:
    return ((vals_u[1:] - vals_u[:-1]) & mask_of(width)).astype(np.uint64)


def signed_delta(vals_i: np.ndarray) -> np.ndarray:
    return np.diff(vals_i.astype(np.int64))


def zz_bits(vals_i: np.ndarray) -> np.ndarray:
    """Zigzag bit width: magnitude bits + sign bit. Needs no min/max, one pass."""
    v = vals_i.astype(np.int64)
    mag = np.abs(v).astype(np.uint64)
    bl = bit_length(mag)
    return np.where(bl > 0, bl + 1, 0)


def stratified_runs(n: int, size: int, count: int, seed=1234567890):
    if size * count >= n:
        return [(0, n)]
    rng = np.random.default_rng(seed)
    bounds = np.linspace(0, n, count + 1).astype(int)
    return [
        (int(s), int(s) + size)
        for i in range(count)
        for s in [rng.integers(bounds[i], bounds[i + 1] - size + 1)]
    ]


def truth(vals_u: np.ndarray, vals_i: np.ndarray, byte_width: int, signed: bool):
    width = byte_width * 8
    plain_bpv, plain_w = (
        for_cost_signed(vals_i, byte_width) if signed else for_cost_unsigned(vals_u, byte_width)
    )

    # Vortex today: residuals live in the array's own signedness domain.
    if signed:
        d1 = signed_delta(vals_i)
        d1_bpv, d1_w = for_cost_signed(d1, byte_width)
        d2 = signed_delta(d1)
        d2_bpv, d2_w = for_cost_signed(d2, byte_width)
    else:
        d1u = wrapping_delta_u(vals_u, width)
        d1_bpv, d1_w = for_cost_unsigned(d1u, byte_width)
        d2u = wrapping_delta_u(d1u, width)
        d2_bpv, d2_w = for_cost_unsigned(d2u, byte_width)

    # Achievable: always interpret residuals as signed (zigzag / reinterpret-cast).
    a1 = signed_delta(vals_i)
    a1_bpv, a1_w = for_cost_signed(a1, byte_width)
    a2 = signed_delta(a1)
    a2_bpv, a2_w = for_cost_signed(a2, byte_width)

    return dict(
        t_plain_bpv=plain_bpv,
        t_plain_w=plain_w,
        t_delta_bpv=d1_bpv + BASE_BPV,
        t_delta_w=d1_w,
        t_dod_bpv=d2_bpv + 2 * BASE_BPV,
        t_dod_w=d2_w,
        a_delta_bpv=a1_bpv + BASE_BPV,
        a_delta_w=a1_w,
        a_dod_bpv=a2_bpv + 2 * BASE_BPV,
        a_dod_w=a2_w,
    )


def cheap_stats(vals_i: np.ndarray, byte_width: int, size, count, seed=1234567890):
    """Cheap stats over a stratified sample of contiguous runs.

    Everything here is one accumulation pass per sampled run: a 65-entry bit-width
    histogram for lag-1 and lag-2 residuals plus three counters.
    """
    runs = [vals_i[s:e] for s, e in stratified_runs(vals_i.size, size, count, seed)]
    samp = np.concatenate(runs)
    plain_bpv, plain_w = for_cost_signed(samp, byte_width)

    d1 = np.concatenate([signed_delta(r) for r in runs if r.size > 1])
    d2 = np.concatenate([signed_delta(signed_delta(r)) for r in runs if r.size > 2])

    w1, c1 = best_width_cost(hist_of(zz_bits(d1), byte_width * 8), byte_width)
    w2, c2 = best_width_cost(hist_of(zz_bits(d2), byte_width * 8), byte_width)

    return dict(
        s_plain_bpv=plain_bpv,
        s_plain_w=plain_w,
        s_delta_w=w1,
        s_delta_bpv=c1 / max(d1.size, 1) + BASE_BPV,
        s_dod_w=w2,
        s_dod_bpv=c2 / max(d2.size, 1) + 2 * BASE_BPV,
        s_nonneg_frac=float((d1 >= 0).mean()) if d1.size else 1.0,
        s_delta_zero_frac=float((d1 == 0).mean()) if d1.size else 0.0,
        s_dod_zero_frac=float((d2 == 0).mean()) if d2.size else 0.0,
        n_sampled=int(samp.size),
    )


def column_arrays(col):
    """Return (unsigned view, signed view, byte_width, signed?, null_frac) or None."""
    t = col.type
    if pa.types.is_timestamp(t) or pa.types.is_date(t) or pa.types.is_time(t):
        col, bw, signed = col.cast(pa.int64()), 8, True
    elif pa.types.is_integer(t):
        bw, signed = t.bit_width // 8, pa.types.is_signed_integer(t)
    else:
        return None
    n = len(col)
    if n == 0 or col.null_count == n:
        return None
    null_frac = col.null_count / n
    a = np.asarray(col.fill_null(0).to_numpy(zero_copy_only=False))
    if a.dtype.kind == "f":
        a = np.nan_to_num(a).astype(np.int64)
    signed_view = a.astype({1: "int8", 2: "int16", 4: "int32", 8: "int64"}[bw], copy=False)
    unsigned_view = signed_view.astype({1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}[bw])
    return (
        unsigned_view.astype("uint64") & mask_of(bw * 8),
        signed_view.astype("int64"),
        bw,
        signed,
        null_frac,
    )


def run(dataset, path, max_rows=None, max_blocks=16):
    pf = pq.ParquetFile(path)
    batches, got = [], 0
    for b in pf.iter_batches(batch_size=1 << 17):
        batches.append(b)
        got += b.num_rows
        if max_rows and got >= max_rows:
            break
    tbl = pa.Table.from_batches(batches)
    rows = []
    for name in tbl.schema.names:
        r = column_arrays(tbl.column(name).combine_chunks())
        if r is None:
            continue
        u, i, bw, signed, null_frac = r
        for bi in range(min(max(1, u.size // BLOCK), max_blocks)):
            lo, hi = bi * BLOCK, (bi + 1) * BLOCK
            if u[lo:hi].size < 4096:
                continue
            t = truth(u[lo:hi], i[lo:hi], bw, signed)
            for size, count, tag in ((64, 16, "1k"), (64, 64, "4k"), (256, 16, "4k_wide"), (1024, 8, "8k_wide")):
                rows.append(
                    dict(
                        dataset=dataset,
                        column=name,
                        block=bi,
                        byte_width=bw,
                        signed=signed,
                        null_frac=null_frac,
                        sample=tag,
                        **t,
                        **cheap_stats(i[lo:hi], bw, size, count),
                    )
                )
    return rows


if __name__ == "__main__":
    import pandas as pd

    rows = []
    rows += run("hits", "data/hits_0.parquet", max_rows=1 << 20)
    rows += run("airquality", "data/airquality.parquet")
    rows += run("taxi", "data/taxi.parquet")
    rows += run("rplace", "data/rplace.parquet", max_rows=1 << 21)
    rows += run("btc_1s", "data/btc_1s.parquet")
    rows += run("btc_1m", "data/btc_1m.parquet")
    rows += run("btc_trades", "data/btc_trades.parquet", max_rows=1 << 21)
    rows += run("power", "data/power.parquet")
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print(df.shape, df.dataset.value_counts().to_dict())
