// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::operators::Operator;

use crate::dtype::PyDType;
use crate::install_module;
use crate::scalar::factory::scalar_helper;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "expr")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.expr", &m)?;

    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(root, &m)?)?;
    m.add_function(wrap_pyfunction!(literal, &m)?)?;
    m.add_function(wrap_pyfunction!(not_, &m)?)?;
    m.add_function(wrap_pyfunction!(and_, &m)?)?;
    m.add_function(wrap_pyfunction!(or_, &m)?)?;
    m.add_function(wrap_pyfunction!(cast, &m)?)?;
    m.add_function(wrap_pyfunction!(is_null, &m)?)?;
    m.add_function(wrap_pyfunction!(is_not_null, &m)?)?;
    m.add_function(wrap_pyfunction!(between, &m)?)?;
    m.add_function(wrap_pyfunction!(like, &m)?)?;
    m.add_function(wrap_pyfunction!(fill_null, &m)?)?;
    m.add_function(wrap_pyfunction!(get_item, &m)?)?;
    m.add_function(wrap_pyfunction!(select, &m)?)?;
    m.add_function(wrap_pyfunction!(select_exclude, &m)?)?;
    m.add_function(wrap_pyfunction!(pack, &m)?)?;
    m.add_function(wrap_pyfunction!(merge, &m)?)?;
    m.add_function(wrap_pyfunction!(case_when, &m)?)?;
    m.add_function(wrap_pyfunction!(zip_, &m)?)?;
    m.add_function(wrap_pyfunction!(mask, &m)?)?;
    m.add_function(wrap_pyfunction!(list_contains, &m)?)?;
    m.add_function(wrap_pyfunction!(list_length, &m)?)?;
    m.add_function(wrap_pyfunction!(list_sum, &m)?)?;
    m.add_function(wrap_pyfunction!(byte_length, &m)?)?;
    m.add_class::<PyExpr>()?;

    Ok(())
}

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///    :func:`.column`
#[pyclass(name = "Expr", module = "vortex", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyExpr {
    inner: Expression,
}

impl From<Expression> for PyExpr {
    fn from(value: Expression) -> Self {
        Self { inner: value }
    }
}

impl Deref for PyExpr {
    type Target = Expression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl PyExpr {
    pub fn inner(&self) -> &Expression {
        &self.inner
    }

    pub fn into_inner(self) -> Expression {
        self.inner
    }
}

fn py_binary_operator<'py>(
    left: PyRef<'py, PyExpr>,
    operator: Operator,
    right: Bound<'py, PyExpr>,
) -> PyResult<Bound<'py, PyExpr>> {
    Bound::new(
        left.py(),
        PyExpr {
            inner: Binary.new_expr(operator, [left.inner.clone(), right.borrow().inner.clone()]),
        },
    )
}

fn coerce_expr<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let nonnull = Nullability::NonNullable;
    if let Ok(value) = value.cast::<PyExpr>() {
        Ok(value.clone())
    } else if let Ok(value) = value.cast::<PyNone>() {
        scalar(DType::Null, value)
    } else if let Ok(value) = value.cast::<PyInt>() {
        scalar(DType::Primitive(PType::I64, nonnull), value)
    } else if let Ok(value) = value.cast::<PyFloat>() {
        scalar(DType::Primitive(PType::F64, nonnull), value)
    } else if let Ok(value) = value.cast::<PyString>() {
        scalar(DType::Utf8(nonnull), value)
    } else if let Ok(value) = value.cast::<PyBytes>() {
        scalar(DType::Binary(nonnull), value)
    } else {
        Err(PyValueError::new_err(format!(
            "expected None, int, float, str, or bytes but found: {value}"
        )))
    }
}

#[pymethods]
impl PyExpr {
    pub fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Eq, coerce_expr(right)?)
    }

    fn __ne__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::NotEq, coerce_expr(right)?)
    }

    fn __gt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gt, coerce_expr(right)?)
    }

    fn __ge__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gte, coerce_expr(right)?)
    }

    fn __lt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lt, coerce_expr(right)?)
    }

    fn __le__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lte, coerce_expr(right)?)
    }

    fn __and__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::And, coerce_expr(right)?)
    }

    fn __or__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Or, coerce_expr(right)?)
    }

    fn __add__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Add, coerce_expr(right)?)
    }

    fn __sub__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Sub, coerce_expr(right)?)
    }

    fn __mul__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Mul, coerce_expr(right)?)
    }

    fn __truediv__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Div, coerce_expr(right)?)
    }

    fn __invert__(self_: PyRef<'_, Self>) -> PyResult<PyExpr> {
        Ok(PyExpr {
            inner: not(self_.inner.clone()),
        })
    }

    // Special methods docstrings cannot be defined in Rust. Write a docstring in the corresponding
    // rST file. https://github.com/PyO3/pyo3/issues/4326
    fn __getitem__(self_: PyRef<'_, Self>, field: String) -> PyResult<PyExpr> {
        get_item(field, self_.clone())
    }
}

/// Create an expression that represents a literal value.
///
/// Parameters
/// ----------
/// dtype : :class:`vortex.DType`
///     The data type of the literal value.
/// value : :class:`Any`
///     The literal value.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.literal(vx.int_(), 42)
/// <vortex.Expr object at ...>
/// ```
// TODO(ngates): make dtype optional, casting if necessary.
#[pyfunction]
pub fn literal<'py>(
    dtype: &Bound<'py, PyDType>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    scalar(dtype.borrow().inner().clone(), value)
}

/// Create an expression that refers to the identity scope.
///
/// That is, it returns the full input that the extension is run against.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.root()
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn root() -> PyExpr {
    PyExpr {
        inner: expr::root(),
    }
}

/// Create an expression that refers to a column by its name.
///
/// Parameters
/// ----------
/// name : :class:`str`
///     The name of the column.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.column("age")
/// <vortex.Expr object at ...>
/// ```
///
/// .. seealso::
///
///    Use :meth:`.vortex.expr.Expr.__getitem__` to retrieve a field of a struct array.
#[pyfunction]
pub fn column<'py>(name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyExpr>> {
    let py = name.py();
    let name: String = name.extract()?;
    Bound::new(
        py,
        PyExpr {
            inner: expr::get_item(name, expr::root()),
        },
    )
}

pub fn scalar<'py>(dtype: DType, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let py = value.py();
    Bound::new(
        py,
        PyExpr {
            inner: lit(scalar_helper(value, Some(&dtype))?),
        },
    )
}

/// Extract a named field from a struct expression.
///
/// Parameters
/// ----------
/// field : :class:`str`
///     The name of the field.
/// child : :class:`Expr`
///     An expression evaluating to a struct.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn get_item(field: String, child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: GetItem.new_expr(field.into(), [child.inner]),
    })
}

/// Negate a Boolean expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.not_(ve.literal(vx.int_(), 42) == ve.literal(vx.int_(), 42))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn not_(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: not(child.inner),
    })
}

/// True if both arguments are true.
///
/// Parameters
/// ----------
/// left : :class:`Expr`
///     A boolean expression.
///
/// right : :class:`Expr`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.and_(ve.literal(vx.bool_(), True), ve.literal(vx.bool_(), True))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn and_(left: PyExpr, right: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: and(left.inner, right.inner),
    })
}

/// Cast an expression to a compatible type.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     The expression to cast.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// Cast to a wider integer type:
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.cast(ve.literal(vx.int_(8), 1), vx.int_(16))
/// <vortex.Expr object at ...>
/// ```
///
/// Cast to a wider floating-point type:
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.cast(ve.literal(vx.float_(16), 3.145), vx.float_(64))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn cast(child: PyExpr, dtype: PyDType) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::cast(child.into_inner(), dtype.into_inner()),
    })
}

/// Checks which elements of its child are null.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     Any expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn is_null(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::is_null(child.into_inner()),
    })
}

/// Creates an expression that checks for non-null values.
///
/// Parameters
/// ----------
/// child : :class:`vortex.Expr`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn is_not_null(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::is_not_null(child.into_inner()),
    })
}

/// True if either argument is true.
///
/// Parameters
/// ----------
/// left : :class:`Expr`
///     A boolean expression.
///
/// right : :class:`Expr`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn or_(left: PyExpr, right: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::or(left.inner, right.inner),
    })
}

/// Checks whether values lie between a lower and an upper bound.
///
/// By default both bounds are inclusive, matching SQL ``BETWEEN`` semantics. Either bound
/// can be made exclusive with `lower_strict` / `upper_strict`.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     The expression whose values are tested against the bounds.
/// lower : :class:`Expr` | :class:`int` | :class:`float` | :class:`str` | :class:`bytes` | :obj:`None`
///     The lower bound.
/// upper : :class:`Expr` | :class:`int` | :class:`float` | :class:`str` | :class:`bytes` | :obj:`None`
///     The upper bound.
/// lower_strict : :class:`bool`
///     When :obj:`True`, the lower bound is exclusive (``<``) instead of inclusive (``<=``).
/// upper_strict : :class:`bool`
///     When :obj:`True`, the upper bound is exclusive (``<``) instead of inclusive (``<=``).
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (child, lower, upper, *, lower_strict = false, upper_strict = false))]
pub fn between<'py>(
    child: PyExpr,
    lower: &Bound<'py, PyAny>,
    upper: &Bound<'py, PyAny>,
    lower_strict: bool,
    upper_strict: bool,
) -> PyResult<PyExpr> {
    fn strictness(strict: bool) -> StrictComparison {
        if strict {
            StrictComparison::Strict
        } else {
            StrictComparison::NonStrict
        }
    }

    let options = BetweenOptions {
        lower_strict: strictness(lower_strict),
        upper_strict: strictness(upper_strict),
    };
    Ok(PyExpr {
        inner: expr::between(
            child.inner,
            coerce_expr(lower)?.get().inner.clone(),
            coerce_expr(upper)?.get().inner.clone(),
            options,
        ),
    })
}

/// Creates a SQL ``LIKE`` expression.
///
/// In the pattern, ``%`` matches any sequence of characters, ``_`` matches exactly one
/// character, and ``\`` escapes the next character.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     A string expression to match against the pattern.
/// pattern : :class:`Expr` | :class:`str`
///     The ``LIKE`` pattern.
/// negated : :class:`bool`
///     When :obj:`True`, the match is negated (``NOT LIKE``).
/// case_insensitive : :class:`bool`
///     When :obj:`True`, matching is case-insensitive (``ILIKE``).
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (child, pattern, *, negated = false, case_insensitive = false))]
pub fn like<'py>(
    child: PyExpr,
    pattern: &Bound<'py, PyAny>,
    negated: bool,
    case_insensitive: bool,
) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: Like.new_expr(
            LikeOptions {
                negated,
                case_insensitive,
            },
            [child.inner, coerce_expr(pattern)?.get().inner.clone()],
        ),
    })
}

/// Replaces null values with a fill value.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     The expression whose nulls are replaced.
/// fill_value : :class:`Expr` | :class:`int` | :class:`float` | :class:`str` | :class:`bytes`
///     The value used in place of nulls.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn fill_null<'py>(child: PyExpr, fill_value: &Bound<'py, PyAny>) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::fill_null(child.inner, coerce_expr(fill_value)?.get().inner.clone()),
    })
}

/// Selects (includes) specific fields from a struct expression.
///
/// Parameters
/// ----------
/// fields : :class:`list` of :class:`str`
///     The names of the fields to keep.
/// child : :class:`Expr`
///     An expression evaluating to a struct.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn select(fields: Vec<String>, child: PyExpr) -> PyResult<PyExpr> {
    let fields: Vec<FieldName> = fields.into_iter().map(FieldName::from).collect();
    Ok(PyExpr {
        inner: expr::select(fields, child.inner),
    })
}

/// Selects all but the specified fields from a struct expression.
///
/// Parameters
/// ----------
/// fields : :class:`list` of :class:`str`
///     The names of the fields to exclude.
/// child : :class:`Expr`
///     An expression evaluating to a struct.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn select_exclude(fields: Vec<String>, child: PyExpr) -> PyResult<PyExpr> {
    let fields: Vec<FieldName> = fields.into_iter().map(FieldName::from).collect();
    Ok(PyExpr {
        inner: expr::select_exclude(fields, child.inner),
    })
}

/// Packs expressions into a struct with named fields.
///
/// Parameters
/// ----------
/// fields : :class:`dict`
///     A mapping from field names to expressions.
/// nullable : :class:`bool`
///     When :obj:`True`, the resulting struct is nullable.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (fields, *, nullable = false))]
pub fn pack<'py>(fields: &Bound<'py, PyDict>, nullable: bool) -> PyResult<PyExpr> {
    let elements = fields
        .iter()
        .map(|(name, value)| {
            let name = name.extract::<String>()?;
            let value = value.extract::<PyExpr>()?;
            Ok((FieldName::from(name), value.inner))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyExpr {
        inner: expr::pack(elements, Nullability::from(nullable)),
    })
}

/// Merges struct expressions into a single struct.
///
/// Combines fields from all input expressions. If field names are duplicated, later
/// expressions win.
///
/// Parameters
/// ----------
/// children : :class:`list` of :class:`Expr`
///     Expressions evaluating to structs.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn merge(children: Vec<PyExpr>) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::merge(children.into_iter().map(PyExpr::into_inner)),
    })
}

/// Creates a ``CASE WHEN`` expression with one WHEN/THEN pair and an optional ELSE value.
///
/// Parameters
/// ----------
/// condition : :class:`Expr`
///     A boolean expression.
/// then : :class:`Expr`
///     The value produced where `condition` is true.
/// otherwise : :class:`Expr` | :obj:`None`
///     The value produced where `condition` is false or null. When omitted, those
///     positions are null.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (condition, then, otherwise = None))]
pub fn case_when(condition: PyExpr, then: PyExpr, otherwise: Option<PyExpr>) -> PyResult<PyExpr> {
    let inner = match otherwise {
        Some(otherwise) => expr::case_when(condition.inner, then.inner, otherwise.inner),
        None => expr::case_when_no_else(condition.inner, then.inner),
    };
    Ok(PyExpr { inner })
}

/// Conditionally selects values from one of two expressions.
///
/// Parameters
/// ----------
/// condition : :class:`Expr`
///     A boolean expression.
/// if_true : :class:`Expr`
///     The value produced where `condition` is true.
/// if_false : :class:`Expr`
///     The value produced where `condition` is false.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn zip_(condition: PyExpr, if_true: PyExpr, if_false: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::zip_expr(condition.inner, if_true.inner, if_false.inner),
    })
}

/// Applies a boolean mask to an expression, nulling out unselected positions.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     The expression to mask.
/// mask : :class:`Expr`
///     A non-nullable boolean expression; where it is true the input value is retained,
///     and where it is false the output is null.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn mask(child: PyExpr, mask: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::mask(child.inner, mask.inner),
    })
}

/// Checks whether each list contains a value.
///
/// Parameters
/// ----------
/// list : :class:`Expr`
///     An expression evaluating to lists.
/// value : :class:`Expr` | :class:`int` | :class:`float` | :class:`str` | :class:`bytes`
///     The value to search for.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn list_contains<'py>(list: PyExpr, value: &Bound<'py, PyAny>) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::list_contains(list.inner, coerce_expr(value)?.get().inner.clone()),
    })
}

/// Computes the number of elements in each list.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     An expression evaluating to lists.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn list_length(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::list_length(child.inner),
    })
}

/// Sums the elements of each list, following SQL ``SUM`` semantics.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     An expression evaluating to lists of numeric values.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn list_sum(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::list_sum(child.inner),
    })
}

/// Computes the byte length of each element, akin to ANSI SQL ``OCTET_LENGTH()``.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     A string or binary expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn byte_length(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::byte_length(child.inner),
    })
}
