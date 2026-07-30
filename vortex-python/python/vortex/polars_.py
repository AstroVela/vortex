# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import operator
from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import pyarrow as pa

import vortex.expr as ve

from ._lib import dtype as _dtype  # pyright: ignore[reportMissingModuleSource]


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))  # pyright: ignore[reportAny]
    assert isinstance(data, dict)
    return _polars_to_vortex(data)  # pyright: ignore[reportUnknownArgumentType]


_OPS: dict[str, Callable[[ve.Expr, ve.Expr], ve.Expr]] = {
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
    "Plus": operator.add,
    "Minus": operator.sub,
    "Multiply": operator.mul,
    "TrueDivide": operator.truediv,
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
    "Float": lambda v: _dtype.float_(64, nullable=v is None),
    "Float32": lambda v: _dtype.float_(32, nullable=v is None),
    "Float64": lambda v: _dtype.float_(64, nullable=v is None),
    "Null": lambda v: _dtype.null(),
    "String": lambda v: _dtype.utf8(nullable=v is None),
    "Binary": lambda v: _dtype.binary(nullable=v is None),
    "Date": lambda v: _dtype.date("days", nullable=v is None),
    "Time": lambda v: _dtype.time("ns", nullable=v is None),
}


_TIME_UNITS: dict[str, Literal["s", "ms", "us", "ns"]] = {
    "Nanoseconds": "ns",
    "Microseconds": "us",
    "Milliseconds": "ms",
    "Seconds": "s",
}


def _timezone(tz: object) -> str | None:
    """Normalize a serialized Polars timezone, which may be wrapped as ``{"inner": tz}``."""
    if tz is None:
        return None
    if isinstance(tz, str):
        return tz
    if isinstance(tz, dict):
        inner = tz.get("inner")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(inner, str):
            return inner
    raise NotImplementedError(f"Unsupported Polars timezone: {tz}")


def _timestamp_literal(value: int | None, unit: object, tz: object) -> ve.Expr:
    if not isinstance(unit, str) or unit not in _TIME_UNITS:
        raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
    dtype = _dtype.timestamp(_TIME_UNITS[unit], tz=_timezone(tz), nullable=value is None)
    return ve.literal(dtype, value)


def _scalar_to_vortex(scalar: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a serialized Polars ``Scalar`` value to a Vortex literal expression."""
    if "Null" in scalar:
        return ve.literal(_dtype.null(), None)

    scalar_type = next(iter(scalar.keys()), None)
    if scalar_type is None:
        raise ValueError(f"Cannot convert to Vortex: empty Polars scalar {scalar}")
    value = scalar[scalar_type]  # pyright: ignore[reportAny]

    if scalar_type in ("Datetime", "DateTime"):
        (value, unit, tz) = value  # pyright: ignore[reportAny]
        return _timestamp_literal(value, unit, tz)  # pyright: ignore[reportAny]

    if scalar_type == "Duration":
        raise NotImplementedError("Vortex has no duration type to represent a Polars Duration literal")

    if scalar_type == "Decimal":
        (value, precision, scale) = value  # pyright: ignore[reportAny]
        dtype = _dtype.decimal(precision=precision, scale=scale, nullable=value is None)  # pyright: ignore[reportAny]
        return ve.literal(dtype, value)  # pyright: ignore[reportAny]

    if scalar_type == "Binary":
        return ve.literal(_dtype.binary(nullable=value is None), bytes(value))  # pyright: ignore[reportAny]

    if scalar_type in _LITERAL_TYPES:
        return ve.literal(_LITERAL_TYPES[scalar_type](value), value)  # pyright: ignore[reportAny]

    raise NotImplementedError(f"Cannot convert to Vortex: unsupported Polars scalar value type {scalar}")


def _polars_dtype_to_vortex(dtype: object) -> _dtype.DType:
    """Convert a serialized Polars data type to a Vortex data type.

    Polars values are always nullable, so the resulting Vortex types are nullable too.
    """
    if isinstance(dtype, str):
        simple: dict[str, Callable[[], _dtype.DType]] = {
            "Boolean": lambda: _dtype.bool_(nullable=True),
            "Int8": lambda: _dtype.int_(8, nullable=True),
            "Int16": lambda: _dtype.int_(16, nullable=True),
            "Int32": lambda: _dtype.int_(32, nullable=True),
            "Int64": lambda: _dtype.int_(64, nullable=True),
            "UInt8": lambda: _dtype.uint(8, nullable=True),
            "UInt16": lambda: _dtype.uint(16, nullable=True),
            "UInt32": lambda: _dtype.uint(32, nullable=True),
            "UInt64": lambda: _dtype.uint(64, nullable=True),
            "Float32": lambda: _dtype.float_(32, nullable=True),
            "Float64": lambda: _dtype.float_(64, nullable=True),
            "String": lambda: _dtype.utf8(nullable=True),
            "Binary": lambda: _dtype.binary(nullable=True),
            "Date": lambda: _dtype.date("days", nullable=True),
            "Time": lambda: _dtype.time("ns", nullable=True),
            "Null": lambda: _dtype.null(),
        }
        if dtype in simple:
            return simple[dtype]()
        raise NotImplementedError(f"Unsupported Polars data type: {dtype}")

    if isinstance(dtype, dict):
        if "Datetime" in dtype:
            (unit, tz) = dtype["Datetime"]  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(unit, str) or unit not in _TIME_UNITS:
                raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
            return _dtype.timestamp(_TIME_UNITS[unit], tz=_timezone(tz), nullable=True)  # pyright: ignore[reportUnknownArgumentType]
        if "Decimal" in dtype:
            (precision, scale) = dtype["Decimal"]  # pyright: ignore[reportUnknownVariableType]
            if precision is None:
                precision = 38
            if scale is None:
                scale = 0
            if not isinstance(precision, int) or not isinstance(scale, int):
                raise NotImplementedError(f"Unsupported Polars data type: {dtype}")
            return _dtype.decimal(precision=precision, scale=scale, nullable=True)
        if "List" in dtype:
            return _dtype.list_(_polars_dtype_to_vortex(dtype["List"]), nullable=True)  # pyright: ignore[reportUnknownArgumentType]

    raise NotImplementedError(f"Unsupported Polars data type: {dtype}")


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so `value` matches literally within a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _string_literal(expr: dict[str, Any]) -> str:  # pyright: ignore[reportExplicitAny]
    """Extract a Python string from a serialized Polars string literal expression."""
    literal = expr.get("Literal")
    if isinstance(literal, dict):
        scalar = literal.get("Scalar", literal)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(scalar, dict):
            for key in ("String", "StrOwned", "Str"):
                value = scalar.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if isinstance(value, str):
                    return value
    raise NotImplementedError(f"Expected a string literal, got: {expr}")


def _or_collect(exprs: list[ve.Expr]) -> ve.Expr:
    """Combine boolean expressions with OR using a balanced tree."""
    assert len(exprs) > 0
    while len(exprs) > 1:
        pairs = [ve.or_(exprs[i], exprs[i + 1]) for i in range(0, len(exprs) - 1, 2)]
        if len(exprs) % 2 == 1:
            pairs.append(exprs[-1])
        exprs = pairs
    return exprs[0]


_IS_IN_TYPES = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_string,
    pa.types.is_large_string,
    pa.types.is_string_view,
    pa.types.is_binary,
    pa.types.is_large_binary,
    pa.types.is_binary_view,
)


def _is_in_to_vortex(child: ve.Expr, values_expr: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a Polars ``is_in`` to an OR of equalities over the literal values."""
    literal = values_expr.get("Literal")
    if not isinstance(literal, dict):
        raise NotImplementedError(f"Unsupported Polars is_in values: {values_expr}")
    scalar = literal.get("Scalar", literal)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(scalar, dict) or "List" not in scalar:
        raise NotImplementedError(f"Unsupported Polars is_in values: {values_expr}")

    # Polars serializes the list of values as an Arrow IPC stream.
    table = pa.ipc.open_stream(bytes(scalar["List"])).read_all()  # pyright: ignore[reportUnknownArgumentType]
    if table.num_columns != 1:
        raise NotImplementedError(f"Unsupported Polars is_in values: {values_expr}")
    column = table.column(0)
    if len(column) == 0:
        return ve.literal(_dtype.bool_(), False)

    if not any(check(column.type) for check in _IS_IN_TYPES):  # pyright: ignore[reportAny]
        raise NotImplementedError(f"Unsupported Polars is_in value type: {column.type}")  # pyright: ignore[reportAny]

    values = column.to_pylist()
    if any(v is None for v in values):
        raise NotImplementedError("Unsupported null value in Polars is_in values")

    return _or_collect([child == value for value in values])


def _function_to_vortex(expr: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a serialized Polars function expression to a Vortex expression."""
    inputs: list[dict[str, Any]] = expr["input"]  # pyright: ignore[reportExplicitAny, reportAny]
    fn = expr["function"]  # pyright: ignore[reportAny]

    if "Boolean" in fn:
        fn = fn["Boolean"]  # pyright: ignore[reportAny]

        if fn == "IsNull":
            return ve.is_null(_polars_to_vortex(inputs[0]))
        if fn == "IsNotNull":
            return ve.is_not_null(_polars_to_vortex(inputs[0]))
        if fn == "Not":
            return ve.not_(_polars_to_vortex(inputs[0]))

        if isinstance(fn, dict) and "IsBetween" in fn:
            closed = fn["IsBetween"]["closed"]  # pyright: ignore[reportUnknownVariableType]
            return ve.between(
                _polars_to_vortex(inputs[0]),
                _polars_to_vortex(inputs[1]),
                _polars_to_vortex(inputs[2]),
                lower_strict=closed in ("Right", "None"),
                upper_strict=closed in ("Left", "None"),
            )

        if isinstance(fn, dict) and "IsIn" in fn:
            if fn["IsIn"]["nulls_equal"]:
                raise NotImplementedError(f"Unsupported nulls_equal argument in fn {expr}")
            return _is_in_to_vortex(_polars_to_vortex(inputs[0]), inputs[1])

        raise NotImplementedError(f"Unsupported Polars boolean function: {fn}")

    if "StringExpr" in fn:
        fn = fn["StringExpr"]  # pyright: ignore[reportAny]

        if fn == "StartsWith":
            return ve.like(_polars_to_vortex(inputs[0]), _like_escape(_string_literal(inputs[1])) + "%")
        if fn == "EndsWith":
            return ve.like(_polars_to_vortex(inputs[0]), "%" + _like_escape(_string_literal(inputs[1])))

        if isinstance(fn, dict) and "Contains" in fn:
            if not fn["Contains"]["literal"]:
                raise NotImplementedError("Unsupported regex pattern in Polars StringExpr.Contains")
            return ve.like(_polars_to_vortex(inputs[0]), "%" + _like_escape(_string_literal(inputs[1])) + "%")

        raise NotImplementedError(f"Unsupported Polars string function: {fn}")

    raise NotImplementedError(f"Unsupported Polars function: {fn}")


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]  # pyright: ignore[reportAny]
        lhs = _polars_to_vortex(expr["left"])  # pyright: ignore[reportAny]
        rhs = _polars_to_vortex(expr["right"])  # pyright: ignore[reportAny]
        op = expr["op"]  # pyright: ignore[reportAny]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return _OPS[op](lhs, rhs)

    if "Column" in expr:
        return ve.column(expr["Column"])  # pyright: ignore[reportAny]

    if "Cast" in expr:
        expr = expr["Cast"]  # pyright: ignore[reportAny]
        child = _polars_to_vortex(expr["expr"])  # pyright: ignore[reportAny]

        dtype = expr["dtype"]  # pyright: ignore[reportAny]
        # Post https://github.com/pola-rs/polars/pull/21797 the target is a DataTypeExpr.
        if isinstance(dtype, dict) and "Literal" in dtype:
            dtype = dtype["Literal"]  # pyright: ignore[reportUnknownVariableType]
        return ve.cast(child, _polars_dtype_to_vortex(dtype))  # pyright: ignore[reportUnknownArgumentType]

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        return _scalar_to_vortex(expr["Scalar"])  # pyright: ignore[reportAny]

    if "Literal" in expr:
        expr = expr["Literal"]  # pyright: ignore[reportAny]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _scalar_to_vortex(expr["Scalar"])  # pyright: ignore[reportAny]

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times (pre https://github.com/pola-rs/polars/pull/21849)
        if literal_type in ("DateTime", "Datetime"):
            (value, unit, tz) = expr[literal_type]  # pyright: ignore[reportAny]
            return _timestamp_literal(value, unit, tz)  # pyright: ignore[reportAny]

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
        return _function_to_vortex(expr["Function"])  # pyright: ignore[reportAny]

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
