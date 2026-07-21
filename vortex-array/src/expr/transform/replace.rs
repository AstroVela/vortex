// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::col;
use crate::expr::is_not_null;
use crate::expr::mask;
use crate::expr::pack;
use crate::expr::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
pub fn replace(expr: Expression, needle: &Expression, replacement: Expression) -> Expression {
    expr.transform_down(|node| {
        if &node == needle {
            Ok(Transformed {
                value: replacement.clone(),
                // If there is a match with a needle there can be no more matches in that subtree.
                order: TraversalOrder::Skip,
                changed: true,
            })
        } else {
            Ok(Transformed::no(node))
        }
    })
    .vortex_expect("ReplaceVisitor should not fail")
    .into_inner()
}

/// Expand the `root` expression with a pack of the given struct fields.
pub fn replace_root_fields(expr: Expression, fields: &StructFields) -> Expression {
    replace(expr, &root(), root_fields_expansion(fields))
}

/// Expand the `root` expression of a struct scope into an expression over its coordinates.
///
/// For a non-nullable struct scope this is a `pack` of every field, as in
/// [`replace_root_fields`]. For a nullable struct scope, the scope has one more coordinate
/// than it has fields — its own validity — so the expansion is the identity
///
/// ```text
/// $ == mask(pack(f1: $.f1, ..., fn: $.fn), is_not_null($))
/// ```
///
/// where the surviving `$.f` references denote the fields *without* the struct's own
/// validity applied (`pack` reassembles the values, `mask` re-applies the struct validity).
/// This keeps the validity coordinate in the expression term, so downstream analyses
/// (e.g. partitioning) can route it like any other coordinate instead of re-applying it
/// out-of-band.
pub fn replace_root_scope(expr: Expression, scope: &DType) -> Expression {
    let fields = scope
        .as_struct_fields_opt()
        .vortex_expect("replace_root_scope requires a struct scope");
    let expansion = match scope.nullability() {
        Nullability::NonNullable => root_fields_expansion(fields),
        Nullability::Nullable => mask(root_fields_expansion(fields), is_not_null(root())),
    };
    replace(expr, &root(), expansion)
}

fn root_fields_expansion(fields: &StructFields) -> Expression {
    pack(
        fields
            .names()
            .iter()
            .map(|name| (name.clone(), col(name.clone()))),
        Nullability::NonNullable,
    )
}

#[cfg(test)]
mod test {
    use super::replace;
    use super::replace_root_scope;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::is_not_null;
    use crate::expr::lit;
    use crate::expr::mask;
    use crate::expr::pack;
    use crate::expr::root;

    #[test]
    fn test_replace_full_tree() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let needle = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement.clone());
        assert_eq!(&replaced_expr, &replacement);
    }

    #[test]
    fn test_replace_leaf() {
        let e = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let needle = lit(2);
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement);
        assert_eq!(replaced_expr.to_string(), "pack(a: 1i32, b: 42i32)");
    }

    #[test]
    fn test_replace_root_scope_non_nullable() {
        let dtype = DType::Struct(
            StructFields::from_iter([("a", I32.into()), ("b", DType::from(I32))]),
            NonNullable,
        );
        let expanded = replace_root_scope(root(), &dtype);
        let expected = pack([("a", col("a")), ("b", col("b"))], NonNullable);
        assert_eq!(&expanded, &expected);
    }

    #[test]
    fn test_replace_root_scope_nullable() {
        let dtype = DType::Struct(
            StructFields::from_iter([("a", I32.into()), ("b", DType::from(I32))]),
            Nullable,
        );
        let expanded = replace_root_scope(root(), &dtype);
        // A nullable struct scope has n + 1 coordinates: its fields, and its own validity.
        let expected = mask(
            pack([("a", col("a")), ("b", col("b"))], NonNullable),
            is_not_null(root()),
        );
        assert_eq!(&expanded, &expected);
    }
}
