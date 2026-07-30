# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import json
import operator
from collections.abc import Callable, Iterator
from typing import Any

import polars as pl

import vortex.expr as ve

from ._lib import dtype as _dtype  # pyright: ignore[reportMissingModuleSource]


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))  # pyright: ignore[reportAny]
    assert isinstance(data, dict)
    return _polars_to_vortex(data)  # pyright: ignore[reportUnknownArgumentType]


def decompose_predicate(predicate: pl.Expr) -> tuple[ve.Expr | None, pl.Expr | None]:
    """Split a Polars predicate into a pushable Vortex expression and a Polars residual.

    The predicate is decomposed into its top-level AND conjuncts. Conjuncts that convert to
    Vortex are ANDed into a single Vortex expression; the rest are ANDed into a residual
    Polars expression that the caller must apply itself, e.g. per batch with
    ``DataFrame.filter``. A row passes an AND conjunction exactly when every conjunct is
    true, so filtering by the two parts in sequence is equivalent to filtering by the
    original predicate.

    Returns ``(vortex_expr | None, residual_polars_expr | None)``. At least one side is
    always non-None for a non-trivial predicate; both parts must be applied when present.
    """
    data = json.loads(predicate.meta.serialize(format="json"))  # pyright: ignore[reportAny]
    assert isinstance(data, dict)

    pushed: ve.Expr | None = None
    residual: pl.Expr | None = None
    for conjunct in _conjuncts(data):  # pyright: ignore[reportUnknownArgumentType]
        try:
            converted = _polars_to_vortex(conjunct)
        except Exception:
            # Unsupported conjuncts fall back to evaluation by Polars rather than failing
            # the whole scan.
            expr = pl.Expr.deserialize(io.BytesIO(json.dumps(conjunct).encode()), format="json")
            residual = expr if residual is None else residual & expr
        else:
            pushed = converted if pushed is None else pushed & converted
    return pushed, residual


def _conjuncts(expr: dict[str, Any]) -> Iterator[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Iterate the top-level AND conjuncts of a serialized Polars expression."""
    if "BinaryExpr" in expr and expr["BinaryExpr"].get("op") in ("And", "LogicalAnd"):  # pyright: ignore[reportAny]
        yield from _conjuncts(expr["BinaryExpr"]["left"])  # pyright: ignore[reportAny]
        yield from _conjuncts(expr["BinaryExpr"]["right"])  # pyright: ignore[reportAny]
    else:
        yield expr


_OPS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtEq": operator.le,
    "Gt": operator.gt,
    "GtEq": operator.ge,
    "And": operator.and_,
    "Or": operator.or_,
    "LogicalAnd": operator.and_,
    "LogicalOr": operator.or_,
}


_LITERAL_TYPES: dict[str, Callable[[Any | None], _dtype.DType]] = {  # pyright: ignore[reportExplicitAny]
    "Boolean": lambda v: _dtype.bool_(nullable=v is None),
    "Int": lambda v: _dtype.int_(64, nullable=v is None),
    "Int8": lambda v: _dtype.int_(8, nullable=v is None),
    "Int16": lambda v: _dtype.int_(16, nullable=v is None),
    "Int32": lambda v: _dtype.int_(32, nullable=v is None),
    "Int64": lambda v: _dtype.int_(64, nullable=v is None),
    "UInt8": lambda v: _dtype.uint(8, nullable=v is None),
    "UInt16": lambda v: _dtype.uint(16, nullable=v is None),
    "UInt32": lambda v: _dtype.uint(32, nullable=v is None),
    "UInt64": lambda v: _dtype.uint(64, nullable=v is None),
    "Float32": lambda v: _dtype.float_(32, nullable=v is None),
    "Float64": lambda v: _dtype.float_(64, nullable=v is None),
    "Null": lambda v: _dtype.null(),
    "String": lambda v: _dtype.utf8(nullable=v is None),
    "Binary": lambda v: _dtype.binary(nullable=v is None),
}


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]  # pyright: ignore[reportAny]
        lhs = _polars_to_vortex(expr["left"])  # pyright: ignore[reportAny]
        rhs = _polars_to_vortex(expr["right"])  # pyright: ignore[reportAny]
        op = expr["op"]  # pyright: ignore[reportAny]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return _OPS[op](lhs, rhs)  # pyright: ignore[reportAny]

    if "Column" in expr:
        return ve.column(expr["Column"])  # pyright: ignore[reportAny]

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        scalar = expr["Scalar"]  # pyright: ignore[reportAny]

        if "Null" in scalar:
            value = None
            dtype = "Null"
        elif "String" in scalar:
            value = scalar["String"]  # pyright: ignore[reportAny]
            dtype = "String"
        elif "Int" in scalar:
            value = scalar["Int"]  # pyright: ignore[reportAny]
            dtype = "Int64"
        elif "Float" in scalar:
            value = scalar["Float"]  # pyright: ignore[reportAny]
            dtype = "Float64"
        elif "Float32" in scalar:
            value = scalar["Float32"]  # pyright: ignore[reportAny]
            dtype = "Float32"
        elif "Float64" in scalar:
            value = scalar["Float64"]  # pyright: ignore[reportAny]
            dtype = "Float64"
        elif "Int32" in scalar:
            value = scalar["Int32"]  # pyright: ignore[reportAny]
            dtype = "Int32"
        elif "Int64" in scalar:
            value = scalar["Int64"]  # pyright: ignore[reportAny]
            dtype = "Int64"
        else:
            raise ValueError(f"Cannot convert to Vortex: unsupported Polars scalar value type {scalar}")

        return ve.literal(_LITERAL_TYPES[dtype](value), value)

    if "Literal" in expr:
        expr = expr["Literal"]  # pyright: ignore[reportAny]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _polars_to_vortex(expr)

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times
        if literal_type == "DateTime":
            (value, unit, tz) = expr[literal_type]  # pyright: ignore[reportAny, reportAny]
            if unit == "Nanoseconds":
                unit = "ns"
            elif unit == "Microseconds":
                unit = "us"
            elif unit == "Milliseconds":
                unit = "ms"
            elif unit == "Seconds":
                unit = "s"
            else:
                raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")

            dtype = _dtype.timestamp(unit, tz=tz, nullable=value)  # pyright: ignore[reportAny]
            return ve.literal(dtype, value)  # pyright: ignore[reportAny]

        # Unwrap 'Dyn' scalars, whose type hasn't been established yet.
        # (post https://github.com/pola-rs/polars/pull/21849)
        if literal_type == "Dyn":
            expr = expr["Dyn"]  # pyright: ignore[reportAny]
            literal_type = next(iter(expr.keys()), None)

        if literal_type not in _LITERAL_TYPES:
            raise NotImplementedError(f"Unsupported Polars literal type: {literal_type}")
        value = expr[literal_type]  # pyright: ignore[reportAny]
        return ve.literal(_LITERAL_TYPES[literal_type](value), value)  # pyright: ignore[reportAny]

    if "Function" in expr:
        expr = expr["Function"]  # pyright: ignore[reportAny]
        _inputs = [_polars_to_vortex(e) for e in expr["input"]]  # pyright: ignore[reportAny]

        fn = expr["function"]  # pyright: ignore[reportAny]
        if "Boolean" in fn:
            fn = fn["Boolean"]  # pyright: ignore[reportAny]

            if "IsIn" in fn:
                fn = fn["IsIn"]  # pyright: ignore[reportAny]
                if fn["nulls_equal"]:
                    raise ValueError(f"Unsupported nulls_equal argument in fn {expr}")

                # Vortex doesn't support is-in, so we need to construct a series of ORs?

        if "StringExpr" in fn:
            fn = fn["StringExpr"]  # pyright: ignore[reportAny]
            if "Contains" in fn:
                raise ValueError("Unsupported Polars StringExpr.Contains")

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
