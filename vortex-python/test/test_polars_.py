# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import polars as pl
import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.polars_ import polars_to_vortex


@pytest.mark.parametrize(
    "polars, vortex",
    [
        # Comparisons and logic.
        (pl.col("AdvEngineID") != 0, ve.column("AdvEngineID") != 0),
        (pl.col("MobilePhoneModel") != "", ve.column("MobilePhoneModel") != ""),
        (pl.col("UserID") == 435090932899640449, ve.column("UserID") == 435090932899640449),
        (pl.col("c") > 10000, ve.column("c") > 10000),
        # Arithmetic.
        (pl.col("a") + 1, ve.column("a") + 1),
        (pl.col("a") - 1, ve.column("a") - 1),
        (pl.col("a") * 2, ve.column("a") * 2),
        (pl.col("a") / 2, ve.column("a") / 2),
        ((pl.col("a") + pl.col("b")) * 2 > 10, (ve.column("a") + ve.column("b")) * 2 > 10),
        # Null checks and negation.
        (pl.col("a").is_null(), ve.is_null(ve.column("a"))),
        (pl.col("a").is_not_null(), ve.is_not_null(ve.column("a"))),
        (~(pl.col("a") == 1), ve.not_(ve.column("a") == 1)),
        (~pl.col("a"), ~ve.column("a")),
        # is_between.
        (pl.col("a").is_between(1, 5), ve.between(ve.column("a"), 1, 5)),
        (pl.col("a").is_between(1, 5, closed="left"), ve.between(ve.column("a"), 1, 5, upper_strict=True)),
        (pl.col("a").is_between(1, 5, closed="right"), ve.between(ve.column("a"), 1, 5, lower_strict=True)),
        (
            pl.col("a").is_between(1, 5, closed="none"),
            ve.between(ve.column("a"), 1, 5, lower_strict=True, upper_strict=True),
        ),
        # String matching (clickbench-style filters).
        (pl.col("URL").str.contains("google", literal=True), ve.like(ve.column("URL"), "%google%")),
        (pl.col("URL").str.starts_with("http"), ve.like(ve.column("URL"), "http%")),
        (pl.col("URL").str.ends_with(".html"), ve.like(ve.column("URL"), "%.html")),
        (
            pl.col("Title").str.contains("50%_off", literal=True),
            ve.like(ve.column("Title"), "%50\\%\\_off%"),
        ),
        (
            (
                (pl.col("Title").str.contains("Google", literal=True))
                & (~pl.col("URL").str.contains(".google.", literal=True))
                & (pl.col("SearchPhrase") != "")
            ),
            (
                (ve.like(ve.column("Title"), "%Google%"))
                & (ve.not_(ve.like(ve.column("URL"), "%.google.%")))
                & (ve.column("SearchPhrase") != "")
            ),
        ),
        # is_in over small literal sets.
        (pl.col("a").is_in([1, 2]), ve.or_(ve.column("a") == 1, ve.column("a") == 2)),
        (
            pl.col("a").is_in([1, 2, 3]),
            ve.or_(ve.or_(ve.column("a") == 1, ve.column("a") == 2), ve.column("a") == 3),
        ),
        (pl.col("s").is_in(["x", "y"]), ve.or_(ve.column("s") == "x", ve.column("s") == "y")),
        # Casts.
        (pl.col("a").cast(pl.Int64), ve.cast(ve.column("a"), vx.int_(64, nullable=True))),
        (pl.col("a").cast(pl.Float32), ve.cast(ve.column("a"), vx.float_(32, nullable=True))),
        (pl.col("a").cast(pl.String), ve.cast(ve.column("a"), vx.utf8(nullable=True))),
        # Temporal and decimal literals.
        (
            pl.col("EventDate") >= date(2013, 7, 1),
            ve.column("EventDate") >= ve.literal(vx.date("days"), (date(2013, 7, 1) - date(1970, 1, 1)).days),
        ),
        (
            pl.col("t") > time(12, 30),
            ve.column("t") > ve.literal(vx.time("ns"), 45_000_000_000_000),
        ),
        (
            pl.col("dec") > pl.lit(Decimal("1.50")),
            ve.column("dec") > ve.literal(vx.decimal(precision=38, scale=2), 150),
        ),
        (
            pl.col("ts") > datetime(2020, 1, 1),
            ve.column("ts")
            > ve.cast(
                ve.literal(vx.timestamp("us", tz="UTC"), 1577836800000000),
                vx.timestamp("us", nullable=True),
            ),
        ),
    ],
)
def test_exprs(polars: pl.Expr, vortex: ve.Expr):
    assert str(polars_to_vortex(polars)) == str(vortex)


def test_empty_is_in():
    assert str(polars_to_vortex(pl.col("a").is_in([]))) == str(ve.literal(vx.bool_(), False))


@pytest.mark.parametrize(
    "polars",
    [
        pl.col("s").str.contains("goo.*"),  # regex patterns are unsupported
        pl.col("dur") > timedelta(days=1),  # Vortex has no duration type
        pl.col("a").is_in([1, None]),  # null values in is_in are unsupported
    ],
)
def test_unsupported_exprs(polars: pl.Expr):
    with pytest.raises(NotImplementedError):
        _ = polars_to_vortex(polars)


@pytest.fixture(scope="module")
def vxf(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "polars_test.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(1_000_000)])
        vx.io.write(vx.compress(vx.array(a)), str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname), without_segment_cache=True)  # pyright: ignore[reportUnknownArgumentType]


@pytest.fixture(scope="module")
def vxf_strings(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "polars_strings_test.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    names = ["alice", "bob", "carol", "dave_smith", "eve", None, "100%_legit", "bobby"]
    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.table(
            {
                "id": pa.array(range(len(names)), type=pa.int64()),
                "name": pa.array(names, type=pa.string()),
            }
        )
        vx.io.write(vx.array(a), str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname), without_segment_cache=True)  # pyright: ignore[reportUnknownArgumentType]


def test_to_polars_with_limit(vxf: vx.VortexFile):
    df = vxf.to_polars().limit(100).collect()
    assert len(df) == 100


def test_to_polars_with_filter(vxf: vx.VortexFile):
    df = vxf.to_polars().filter(pl.col("index") < 500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(500))


def test_to_polars_with_is_between(vxf: vx.VortexFile):
    df = vxf.to_polars().filter(pl.col("index").is_between(100, 200)).collect()
    assert df["index"].to_list() == list(range(100, 201))


def test_to_polars_with_is_in(vxf: vx.VortexFile):
    df = vxf.to_polars().filter(pl.col("index").is_in([3, 999, 500_000])).collect()
    assert df["index"].to_list() == [3, 999, 500_000]


def test_to_polars_with_arithmetic_filter(vxf: vx.VortexFile):
    df = vxf.to_polars().filter(pl.col("index") * 2 + 1 < 10).collect()
    assert df["index"].to_list() == list(range(5))


def test_to_polars_with_projection(vxf: vx.VortexFile):
    df = vxf.to_polars().select("index").limit(10).collect()
    assert df.columns == ["index"]
    assert len(df) == 10


def test_to_polars_with_projection_and_filter(vxf: vx.VortexFile):
    df = vxf.to_polars().select("index", "value").filter(pl.col("index") < 100).collect()
    assert df.columns == ["index", "value"]
    assert len(df) == 100


def test_to_polars_with_str_contains(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").str.contains("bob", literal=True)).collect()
    assert df["name"].to_list() == ["bob", "bobby"]


def test_to_polars_with_str_starts_with(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").str.starts_with("bob")).collect()
    assert df["name"].to_list() == ["bob", "bobby"]


def test_to_polars_with_str_ends_with(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").str.ends_with("e")).collect()
    assert df["name"].to_list() == ["alice", "eve"]


def test_to_polars_with_like_escaping(vxf_strings: vx.VortexFile):
    # `%` and `_` in the search string must match literally, not as LIKE wildcards.
    df = vxf_strings.to_polars().filter(pl.col("name").str.contains("%_", literal=True)).collect()
    assert df["name"].to_list() == ["100%_legit"]


def test_to_polars_with_is_null(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").is_null()).collect()
    assert df["id"].to_list() == [5]


def test_to_polars_with_is_not_null(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").is_not_null()).collect()
    assert len(df) == 7


def test_to_polars_with_str_is_in(vxf_strings: vx.VortexFile):
    df = vxf_strings.to_polars().filter(pl.col("name").is_in(["alice", "eve"])).collect()
    assert df["name"].to_list() == ["alice", "eve"]
