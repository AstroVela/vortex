// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the legacy `Expression` optimizer with the rule-driven `BoundExpression` optimizer.
//!
//! Expressions are constructed and bound outside the timed region. `builtins` varies both tree
//! size and the number of nodes that the built-in rules can rewrite. `rule_dispatch` isolates the
//! bound optimizer and varies the number of rules checked before a successful rewrite.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::BoundExpressionOptimizer;
use vortex_array::expr::BoundExpressionRewriteRule;
use vortex_array::expr::Expression;
use vortex_array::expr::ExpressionId;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::or_collect;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

fn struct_scope() -> DType {
    DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    )
}

#[derive(Clone, Copy, Debug)]
struct RewriteCase {
    terms: usize,
    rewrite_sites: usize,
}

impl RewriteCase {
    fn node_count(self) -> usize {
        // Each term is `eq(get_item("x", root()), literal)`, the terms are joined by OR nodes,
        // and each rewrite site adds `and(term, true)`.
        5 * self.terms - 1 + 2 * self.rewrite_sites
    }
}

impl Display for RewriteCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "nodes={}, rewrites={}",
            self.node_count(),
            self.rewrite_sites
        )
    }
}

const REWRITE_CASES: &[RewriteCase] = &[
    RewriteCase {
        terms: 1,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 1,
        rewrite_sites: 1,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 4,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 16,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 32,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 128,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 128,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 512,
    },
];

const NO_REWRITE_CASES: &[RewriteCase] = &[
    REWRITE_CASES[0],
    REWRITE_CASES[2],
    REWRITE_CASES[5],
    REWRITE_CASES[8],
];

fn build_expression(case: RewriteCase) -> Expression {
    or_collect((0..case.terms).map(|idx| {
        let term = eq(col("x"), lit(i32::try_from(idx).unwrap()));
        if idx < case.rewrite_sites {
            and(term, lit(true))
        } else {
            term
        }
    }))
    .unwrap()
}

mod builtins {
    use super::*;

    #[divan::bench(args = REWRITE_CASES)]
    fn expression(bencher: Bencher, case: &RewriteCase) {
        let scope = struct_scope();
        let expr = build_expression(*case);

        bencher
            .counter(ItemsCount::new(case.node_count()))
            .bench(|| black_box(expr.optimize_recursive(&scope).unwrap()));
    }

    #[divan::bench(args = REWRITE_CASES)]
    fn bound_expression(bencher: Bencher, case: &RewriteCase) {
        let scope = struct_scope();
        let unbound = build_expression(*case);
        let expr = unbound.bind(&scope).unwrap();
        let optimizer = BoundExpressionOptimizer::default();

        let expected = unbound
            .optimize_recursive(&scope)
            .unwrap()
            .bind(&scope)
            .unwrap();
        assert_eq!(optimizer.optimize(&expr).unwrap(), expected);

        bencher
            .counter(ItemsCount::new(case.node_count()))
            .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
    }

    #[divan::bench(args = NO_REWRITE_CASES)]
    fn bound_expression_empty(bencher: Bencher, case: &RewriteCase) {
        let scope = struct_scope();
        let expr = build_expression(*case).bind(&scope).unwrap();
        let optimizer = BoundExpressionOptimizer::empty();

        bencher
            .counter(ItemsCount::new(case.node_count()))
            .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
    }
}

/// A binary rule that either declines every node or simplifies `value AND true`.
#[derive(Debug)]
struct AndTrueRule {
    enabled: bool,
}

impl BoundExpressionRewriteRule for AndTrueRule {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if !self.enabled || expr.as_opt::<Binary>() != Some(&Operator::And) {
            return Ok(None);
        }
        let rhs_is_true = expr
            .child(1)
            .as_opt::<Literal>()
            .and_then(|scalar| scalar.as_bool_opt())
            .is_some_and(|value| value.value() == Some(true));
        Ok(rhs_is_true.then(|| expr.child(0).clone()))
    }
}

#[derive(Clone, Copy, Debug)]
struct RuleDispatchCase {
    terms: usize,
    candidate_rules: usize,
}

impl RuleDispatchCase {
    fn rewrite_case(self) -> RewriteCase {
        RewriteCase {
            terms: self.terms,
            rewrite_sites: self.terms,
        }
    }
}

impl Display for RuleDispatchCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "nodes={}, candidate_rules={}",
            self.rewrite_case().node_count(),
            self.candidate_rules
        )
    }
}

const RULE_DISPATCH_CASES: &[RuleDispatchCase] = &[
    RuleDispatchCase {
        terms: 16,
        candidate_rules: 1,
    },
    RuleDispatchCase {
        terms: 16,
        candidate_rules: 4,
    },
    RuleDispatchCase {
        terms: 16,
        candidate_rules: 16,
    },
    RuleDispatchCase {
        terms: 16,
        candidate_rules: 64,
    },
    RuleDispatchCase {
        terms: 128,
        candidate_rules: 1,
    },
    RuleDispatchCase {
        terms: 128,
        candidate_rules: 4,
    },
    RuleDispatchCase {
        terms: 128,
        candidate_rules: 16,
    },
    RuleDispatchCase {
        terms: 128,
        candidate_rules: 64,
    },
    RuleDispatchCase {
        terms: 512,
        candidate_rules: 1,
    },
    RuleDispatchCase {
        terms: 512,
        candidate_rules: 4,
    },
    RuleDispatchCase {
        terms: 512,
        candidate_rules: 16,
    },
    RuleDispatchCase {
        terms: 512,
        candidate_rules: 64,
    },
];

#[divan::bench(args = RULE_DISPATCH_CASES)]
fn rule_dispatch(bencher: Bencher, case: &RuleDispatchCase) {
    let rewrite_case = case.rewrite_case();
    let scope = struct_scope();
    let expr = build_expression(rewrite_case).bind(&scope).unwrap();
    let mut optimizer = BoundExpressionOptimizer::empty();
    for _ in 1..case.candidate_rules {
        optimizer.register_rule(AndTrueRule { enabled: false });
    }
    optimizer.register_rule(AndTrueRule { enabled: true });

    bencher
        .counter(ItemsCount::new(rewrite_case.node_count()))
        .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
}
