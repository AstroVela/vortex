// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A/B benchmarks for primitive comparison implementations.
//!
//! The local matrix is split into filterable groups. `throughput` covers every physical type,
//! comparison operator, and operand shape at a large row count. `scaling`, `validity`, and
//! `boundaries` isolate the dimensions that make a full Cartesian product unnecessarily large.
//! Filter a group with, for example, `cargo bench -p vortex-array --bench compare_primitive --
//! scaling`.

#![expect(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::test_harness::compare_primitive_columnar;
use vortex_array::test_harness::compare_primitive_rows;
use vortex_array::validity::Validity as ArrayValidity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const PTYPES: &[PType] = &[
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::F16,
    PType::F32,
    PType::F64,
];
const OPERATORS: &[CompareOperator] = &[
    CompareOperator::Eq,
    CompareOperator::NotEq,
    CompareOperator::Gt,
    CompareOperator::Gte,
    CompareOperator::Lt,
    CompareOperator::Lte,
];
const SHAPES: &[OperandShape] = &[
    OperandShape::ArrayArray,
    OperandShape::ArrayConstant,
    OperandShape::ConstantArray,
    OperandShape::ConstantConstant,
];

fn main() {
    divan::main();
}

#[derive(Clone, Copy)]
struct ComparisonCase {
    ptype: PType,
    op: CompareOperator,
    shape: OperandShape,
    validity: ValidityCase,
    row_count: usize,
}

impl ComparisonCase {
    const fn new(
        ptype: PType,
        op: CompareOperator,
        shape: OperandShape,
        validity: ValidityCase,
        row_count: usize,
    ) -> Self {
        Self {
            ptype,
            op,
            shape,
            validity,
            row_count,
        }
    }
}

impl fmt::Display for ComparisonCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}/len={}",
            self.ptype,
            operator_name(self.op),
            self.shape,
            self.validity,
            self.row_count
        )
    }
}

#[derive(Clone, Copy)]
enum OperandShape {
    ArrayArray,
    ArrayConstant,
    ConstantArray,
    ConstantConstant,
}

impl fmt::Display for OperandShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ArrayArray => "aa",
            Self::ArrayConstant => "ac",
            Self::ConstantArray => "ca",
            Self::ConstantConstant => "cc",
        })
    }
}

#[derive(Clone, Copy)]
enum ValidityCase {
    NonNullable,
    NullableAllValid,
    NullableLhs,
    NullableRhs,
    NullableBoth,
    Masked,
}

impl fmt::Display for ValidityCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonNullable => "nonnull",
            Self::NullableAllValid => "nullable_all_valid",
            Self::NullableLhs => "nullable_lhs",
            Self::NullableRhs => "nullable_rhs",
            Self::NullableBoth => "nullable_both",
            Self::Masked => "masked",
        })
    }
}

#[derive(Clone, Copy)]
enum OperandValidity {
    NonNullable,
    NullableAllValid,
    NullableSparse,
    Masked,
}

type CompareFn =
    fn(&ArrayRef, &ArrayRef, CompareOperator, &mut ExecutionCtx) -> VortexResult<ArrayRef>;

fn throughput_cases() -> Vec<ComparisonCase> {
    let mut cases = Vec::with_capacity(PTYPES.len() * OPERATORS.len() * SHAPES.len());
    for &ptype in PTYPES {
        for &op in OPERATORS {
            for &shape in SHAPES {
                cases.push(ComparisonCase::new(
                    ptype,
                    op,
                    shape,
                    ValidityCase::NonNullable,
                    65_536,
                ));
            }
        }
    }
    cases
}

fn scaling_cases() -> Vec<ComparisonCase> {
    const ROW_COUNTS: &[usize] = &[128, 1_024, 4_096, 8_192, 16_384, 65_536, 1_048_576];
    const SCALING_PTYPES: &[PType] = &[PType::U8, PType::I64, PType::F64];
    const SCALING_OPERATORS: &[CompareOperator] = &[CompareOperator::Eq, CompareOperator::Gte];

    let mut cases = Vec::new();
    for &ptype in SCALING_PTYPES {
        for &op in SCALING_OPERATORS {
            for &shape in SHAPES {
                for &row_count in ROW_COUNTS {
                    cases.push(ComparisonCase::new(
                        ptype,
                        op,
                        shape,
                        ValidityCase::NonNullable,
                        row_count,
                    ));
                }
            }
        }
    }
    cases
}

fn validity_cases() -> Vec<ComparisonCase> {
    const VALIDITY_PTYPES: &[PType] = &[PType::I64, PType::F64];
    const VALIDITY_OPERATORS: &[CompareOperator] = &[CompareOperator::Eq, CompareOperator::Gte];
    const VALIDITIES: &[ValidityCase] = &[
        ValidityCase::NullableAllValid,
        ValidityCase::NullableLhs,
        ValidityCase::NullableRhs,
        ValidityCase::NullableBoth,
        ValidityCase::Masked,
    ];
    const ROW_COUNTS: &[usize] = &[128, 8_192, 65_536];

    let mut cases = Vec::new();
    for &ptype in VALIDITY_PTYPES {
        for &op in VALIDITY_OPERATORS {
            for &shape in SHAPES {
                for &validity in VALIDITIES {
                    for &row_count in ROW_COUNTS {
                        cases.push(ComparisonCase::new(ptype, op, shape, validity, row_count));
                    }
                }
            }
        }
    }
    cases
}

fn boundary_cases() -> Vec<ComparisonCase> {
    const BOUNDARY_PTYPES: &[PType] = &[PType::U8, PType::I64, PType::F64];
    const BOUNDARY_OPERATORS: &[CompareOperator] = &[CompareOperator::Eq, CompareOperator::Gte];
    const ROW_COUNTS: &[usize] = &[0, 1, 63, 64, 65, 127, 128, 129];

    let mut cases = Vec::new();
    for &ptype in BOUNDARY_PTYPES {
        for &op in BOUNDARY_OPERATORS {
            for &shape in SHAPES {
                for &row_count in ROW_COUNTS {
                    cases.push(ComparisonCase::new(
                        ptype,
                        op,
                        shape,
                        ValidityCase::NonNullable,
                        row_count,
                    ));
                }
            }
        }
    }
    cases
}

fn cpu_feature_cases() -> Vec<ComparisonCase> {
    [
        (PType::I64, CompareOperator::Eq, OperandShape::ArrayArray),
        (PType::I64, CompareOperator::Gte, OperandShape::ArrayArray),
        (
            PType::I64,
            CompareOperator::Gte,
            OperandShape::ArrayConstant,
        ),
        (
            PType::I64,
            CompareOperator::Gte,
            OperandShape::ConstantArray,
        ),
        (PType::U64, CompareOperator::Gte, OperandShape::ArrayArray),
        (PType::F64, CompareOperator::Gte, OperandShape::ArrayArray),
    ]
    .into_iter()
    .map(|(ptype, op, shape)| {
        ComparisonCase::new(ptype, op, shape, ValidityCase::NonNullable, 65_536)
    })
    .collect()
}

#[cfg(not(codspeed))]
mod throughput {
    use super::*;

    #[divan::bench(args = throughput_cases())]
    fn row(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_rows);
    }

    #[divan::bench(args = throughput_cases())]
    fn columnar(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_columnar);
    }
}

#[cfg(not(codspeed))]
mod scaling {
    use super::*;

    #[divan::bench(args = scaling_cases())]
    fn row(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_rows);
    }

    #[divan::bench(args = scaling_cases())]
    fn columnar(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_columnar);
    }
}

#[cfg(not(codspeed))]
mod validity {
    use super::*;

    #[divan::bench(args = validity_cases())]
    fn row(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_rows);
    }

    #[divan::bench(args = validity_cases())]
    fn columnar(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_columnar);
    }
}

#[cfg(not(codspeed))]
mod boundaries {
    use super::*;

    #[divan::bench(args = boundary_cases())]
    fn row(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_rows);
    }

    #[divan::bench(args = boundary_cases())]
    fn columnar(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_columnar);
    }
}

mod cpu_features {
    use super::*;

    #[vortex_bench_support::cpu_features]
    #[divan::bench(args = cpu_feature_cases())]
    fn row(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_rows);
    }

    #[vortex_bench_support::cpu_features]
    #[divan::bench(args = cpu_feature_cases())]
    fn columnar(bencher: Bencher, case: ComparisonCase) {
        bench_case(bencher, case, compare_primitive_columnar);
    }
}

fn bench_case(bencher: Bencher, case: ComparisonCase, compare: CompareFn) {
    let (lhs, rhs) = make_inputs(case).unwrap();
    let session = vortex_array::array_session();

    bencher
        .counter(ItemsCount::new(case.row_count))
        .with_inputs(|| (&lhs, &rhs, session.create_execution_ctx()))
        .bench_refs(|(lhs, rhs, ctx)| {
            let result = compare(lhs, rhs, case.op, ctx).unwrap();

            result.execute::<Canonical>(ctx)
        });
}

fn make_inputs(case: ComparisonCase) -> VortexResult<(ArrayRef, ArrayRef)> {
    match_each_native_ptype!(case.ptype, |T| { make_typed_inputs::<T>(case) })
}

fn make_typed_inputs<T>(case: ComparisonCase) -> VortexResult<(ArrayRef, ArrayRef)>
where
    T: NativePType + Into<PValue> + Into<Scalar>,
{
    let (lhs_validity, rhs_validity) = match case.validity {
        ValidityCase::NonNullable => (OperandValidity::NonNullable, OperandValidity::NonNullable),
        ValidityCase::NullableAllValid => (
            OperandValidity::NullableAllValid,
            OperandValidity::NullableAllValid,
        ),
        ValidityCase::NullableLhs => (
            OperandValidity::NullableSparse,
            OperandValidity::NonNullable,
        ),
        ValidityCase::NullableRhs => (
            OperandValidity::NonNullable,
            OperandValidity::NullableSparse,
        ),
        ValidityCase::NullableBoth => (
            OperandValidity::NullableSparse,
            OperandValidity::NullableSparse,
        ),
        ValidityCase::Masked => (OperandValidity::Masked, OperandValidity::Masked),
    };

    let lhs = match case.shape {
        OperandShape::ArrayArray | OperandShape::ArrayConstant => {
            make_array::<T>(case.row_count, 1, lhs_validity)?
        }
        OperandShape::ConstantArray | OperandShape::ConstantConstant => {
            make_constant::<T>(case.row_count, lhs_validity)?
        }
    };
    let rhs = match case.shape {
        OperandShape::ArrayArray | OperandShape::ConstantArray => {
            make_array::<T>(case.row_count, 17, rhs_validity)?
        }
        OperandShape::ArrayConstant | OperandShape::ConstantConstant => {
            make_constant::<T>(case.row_count, rhs_validity)?
        }
    };

    Ok((lhs, rhs))
}

fn make_array<T>(len: usize, offset: usize, validity: OperandValidity) -> VortexResult<ArrayRef>
where
    T: NativePType,
{
    let values = (0..len)
        .map(|index| {
            let value = ((index.wrapping_mul(31) + offset) % 127) as i64;
            T::from_i64(value).unwrap()
        })
        .collect::<Buffer<_>>();

    Ok(match validity {
        OperandValidity::NonNullable => values.into_array(),
        OperandValidity::NullableAllValid => {
            PrimitiveArray::new(values, ArrayValidity::AllValid).into_array()
        }
        OperandValidity::NullableSparse => {
            PrimitiveArray::new(values, sparse_validity(len, offset)).into_array()
        }
        OperandValidity::Masked => {
            MaskedArray::try_new(values.into_array(), sparse_validity(len, offset))?.into_array()
        }
    })
}

fn make_constant<T>(len: usize, validity: OperandValidity) -> VortexResult<ArrayRef>
where
    T: NativePType + Into<PValue> + Into<Scalar>,
{
    let value = T::from_i64(63).unwrap();

    Ok(match validity {
        OperandValidity::NonNullable => ConstantArray::new(value, len).into_array(),
        OperandValidity::NullableAllValid => {
            ConstantArray::new(Scalar::primitive(value, Nullability::Nullable), len).into_array()
        }
        OperandValidity::NullableSparse | OperandValidity::Masked => MaskedArray::try_new(
            ConstantArray::new(value, len).into_array(),
            sparse_validity(len, 3),
        )?
        .into_array(),
    })
}

fn sparse_validity(len: usize, offset: usize) -> ArrayValidity {
    ArrayValidity::from_iter((0..len).map(|index| !(index + offset).is_multiple_of(10)))
}

const fn operator_name(op: CompareOperator) -> &'static str {
    match op {
        CompareOperator::Eq => "eq",
        CompareOperator::NotEq => "not_eq",
        CompareOperator::Gt => "gt",
        CompareOperator::Gte => "gte",
        CompareOperator::Lt => "lt",
        CompareOperator::Lte => "lte",
    }
}
